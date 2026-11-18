"""Tests for the coordinate precision guard and coordinate grid derivation.

Reproduce everything with::

    python -m pytest tests/test_coordinate_precision.py -q

Grid derivation (the three branches: ordinary range, narrow range far from
zero, and a range that cannot fit the error budget)::

    python -m pytest tests/test_coordinate_precision.py -q -k "grid or range or tolerance or bounds"

End-to-end write-then-close-then-append with another source grid::

    python -m pytest tests/test_coordinate_precision.py -q -k "append"

Rejection of a batch whose last point overflows or exceeds the budget,
followed by a valid batch in the same session::

    python -m pytest tests/test_coordinate_precision.py -q -k "bad_last_point"

Backwards compatibility (guard off by default)::

    python -m pytest tests/test_coordinate_precision.py -q -k "off_by_default"

These tests focus on uncompressed LAS files, which is the format the guard
must get right first.
"""

import io

import numpy as np
import pytest

import laspy
from laspy import (
    CoordinateGrid,
    CoordinateIntegerOverflow,
    CoordinateOutOfBounds,
    CoordinatePrecisionExceeded,
    CoordinateRangeTooPrecise,
    IncompatiblePointFormat,
    InvalidCoordinateBounds,
    InvalidCoordinateTolerance,
    InvalidSourceGrid,
    InvalidTargetGrid,
    MissingSourceGrid,
    NonFiniteCoordinate,
    ScaleAwarePointRecord,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_source_record(
    points_xyz, *, point_format=None, scales=None, offsets=None, extra=None
):
    """Builds a ScaleAwarePointRecord for the given (x, y, z) triplets."""
    point_format = point_format or laspy.PointFormat(3)
    scales = np.array(scales if scales is not None else [0.01, 0.01, 0.01])
    offsets = np.array(offsets if offsets is not None else [0.0, 0.0, 0.0])

    record = ScaleAwarePointRecord.zeros(
        len(points_xyz),
        point_format=point_format,
        scales=scales,
        offsets=offsets,
    )
    record.x = [p[0] for p in points_xyz]
    record.y = [p[1] for p in points_xyz]
    record.z = [p[2] for p in points_xyz]
    for name, values in (extra or {}).items():
        record[name] = values
    return record


def decoded_xyz(points):
    return np.column_stack(
        [np.asarray(points["x"]), np.asarray(points["y"]), np.asarray(points["z"])]
    )


def new_header(scales=(0.01, 0.01, 0.01), offsets=(0.0, 0.0, 0.0), point_format=3):
    header = laspy.LasHeader(version="1.2", point_format=point_format)
    header.scales = np.array(scales, dtype=np.float64)
    header.offsets = np.array(offsets, dtype=np.float64)
    return header


def write_and_read(points, header, **open_kwargs):
    stream = io.BytesIO()
    with laspy.open(stream, mode="w", header=header, closefd=False, **open_kwargs) as w:
        w.write_points(points)
    stream.seek(0)
    return laspy.read(stream)


# ---------------------------------------------------------------------------
# grid derivation
# ---------------------------------------------------------------------------


def test_ordinary_range_grid_is_deterministic():
    first = laspy.make_coordinate_grid([0.0, 0.0, 0.0], [10.0, 20.0, 5.0], 0.001)
    second = laspy.make_coordinate_grid([0.0, 0.0, 0.0], [10.0, 20.0, 5.0], 0.001)

    assert isinstance(first, CoordinateGrid)
    assert np.array_equal(first.scales, second.scales)
    assert np.array_equal(first.offsets, second.offsets)
    assert np.all(np.isfinite(first.scales)) and np.all(first.scales > 0)
    assert np.all(np.isfinite(first.offsets))
    # The midpoint keeps the integer coordinates small.
    assert np.array_equal(first.offsets, [5.0, 10.0, 2.5])


def test_range_crossing_zero():
    grid = laspy.make_coordinate_grid([-5.0, -100.0, -0.5], [5.0, 100.0, 0.5], 0.001)
    assert grid.offsets[0] == 0.0
    assert grid.offsets[1] == 0.0

    coordinates = np.linspace(-5.0, 5.0, 1001)
    integers = np.rint((coordinates - grid.offsets[0]) / grid.scales[0])
    assert np.all(np.abs(integers) <= np.iinfo(np.int32).max)
    decoded = integers * grid.scales[0] + grid.offsets[0]
    assert np.max(np.abs(decoded - coordinates)) <= 0.001


def test_narrow_range_far_from_zero():
    lo, hi = 1_000_000_000.0 - 2.0, 1_000_000_000.0 + 2.0
    grid = laspy.make_coordinate_grid([lo, lo, lo], [hi, hi, hi], 1e-4)

    assert np.all(np.isfinite(grid.scales)) and np.all(grid.scales > 0)
    assert grid.offsets[0] == 1_000_000_000.0

    coordinates = np.linspace(lo, hi, 4001)
    integers = np.rint((coordinates - grid.offsets[0]) / grid.scales[0])
    # A grid centred on the interval uses only a few integers, even though the
    # coordinates themselves are huge.
    assert np.max(np.abs(integers)) <= 20_001
    decoded = integers * grid.scales[0] + grid.offsets[0]
    assert np.max(np.abs(decoded - coordinates)) <= 1e-4


def test_single_valued_axis_is_stored_exactly():
    grid = laspy.make_coordinate_grid([7.0, 0.0, -3.5], [7.0, 1.0, 3.5], 0.01)
    assert grid.offsets[0] == 7.0
    # The only allowed coordinate maps to integer 0 and comes back exactly.
    assert np.rint((7.0 - grid.offsets[0]) / grid.scales[0]) == 0.0
    assert 0.0 * grid.scales[0] + grid.offsets[0] == 7.0


def test_full_integer_range_is_usable_and_margin_kept():
    # (2**32 - 4) steps do not fit, (2**32 - 5) do.
    maxs_ok = (2**32 - 5) / 2**32
    grid = laspy.make_coordinate_grid([0.0, 0.0, 0.0], [maxs_ok] * 3, 1.0 / 2**32)
    assert np.all(grid.scales > 0)

    for axis in range(3):
        lo_index = int(np.rint((0.0 - grid.offsets[axis]) / grid.scales[axis]))
        hi_index = int(np.rint((maxs_ok - grid.offsets[axis]) / grid.scales[axis]))
        assert np.iinfo(np.int32).min <= lo_index < hi_index <= np.iinfo(np.int32).max
        # Neither endpoint sits on the extreme representable value.
        assert abs(lo_index) < 2**31 and abs(hi_index) < 2**31


@pytest.mark.parametrize(
    "mins, maxs",
    [
        ([0.0, 0.0, 0.0], [1.0, -1.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 1.0, 1.0]),
        ([0.0, np.nan, 0.0], [1.0, 1.0, 1.0]),
        ([0.0, 0.0, 0.0], [np.inf, 1.0, 1.0]),
        ([0.0, 0.0], [1.0, 1.0, 1.0]),
    ],
)
def test_invalid_bounds_are_rejected(mins, maxs):
    with pytest.raises(InvalidCoordinateBounds):
        laspy.make_coordinate_grid(mins, maxs, 0.01)


@pytest.mark.parametrize("tolerance", [0.0, -0.01, [0.01, 0.0, 0.01], np.nan, np.inf])
def test_invalid_tolerances_are_rejected(tolerance):
    with pytest.raises(InvalidCoordinateTolerance):
        laspy.make_coordinate_grid([0.0] * 3, [1.0] * 3, tolerance)


@pytest.mark.parametrize(
    "mins, maxs, tolerance",
    [
        # 1e12 grid steps do not fit into a signed 32 bit integer.
        ([0.0, 0.0, 0.0], [1e6, 1e6, 1e6], 1e-6),
        # Width fits, but float64 at that magnitude cannot meet the budget.
        ([1e9, 1e9, 1e9], [1e9 + 100.0, 1e9 + 100.0, 1e9 + 100.0], 1e-9),
        # 1e12 magnitude with a micrometre budget.
        ([1e12, 1e12, 1e12], [1e12 + 1.0, 1e12 + 1.0, 1e12 + 1.0], 1e-6),
    ],
)
def test_ranges_that_do_not_fit_the_budget_are_rejected(mins, maxs, tolerance):
    with pytest.raises(CoordinateRangeTooPrecise):
        laspy.make_coordinate_grid(mins, maxs, tolerance)


def test_skip_known_float_limits_at_extremes():
    # A degenerate interval exactly at the value and with any tolerance works.
    grid = laspy.make_coordinate_grid([1e9] * 3, [1e9] * 3, 1e-9)
    assert np.array_equal(grid.offsets, [1e9] * 3)


def test_grid_scales_are_the_tolerances_and_offsets_are_midpoints():
    mins = np.array([-12.0, 100.0, 0.25])
    maxs = np.array([8.0, 300.0, 0.75])
    tolerance = np.array([1e-3, 1e-2, 1e-4])
    grid = laspy.make_coordinate_grid(mins, maxs, tolerance)

    assert np.array_equal(grid.scales, tolerance)
    assert np.allclose(grid.offsets, (mins + maxs) / 2.0, rtol=0, atol=0)
    assert np.array_equal(grid.mins, mins)
    assert np.array_equal(grid.maxs, maxs)


# ---------------------------------------------------------------------------
# write path
# ---------------------------------------------------------------------------


def test_write_requantizes_source_grid_and_keeps_everything_else():
    grid = laspy.make_coordinate_grid([0.0] * 3, [100.0] * 3, 0.001)

    header = new_header(point_format=3)
    header.add_extra_dim(laspy.ExtraBytesParams(name="confidence", type="u2"))

    source = make_source_record(
        [(1.0, 2.0, 3.0), (4.5, 5.25, 6.75), (7.0, 8.0, 9.0)],
        point_format=header.point_format,
        scales=[0.5, 0.5, 0.5],
        offsets=[1000.0, 1000.0, 1000.0],
        extra={"confidence": [11, 12, 13]},
    )
    source.intensity = [1, 2, 3]
    source.return_number = [1, 2, 1]
    source.classification = [2, 3, 4]

    original_header_scales = header.scales.copy()
    original_header_offsets = header.offsets.copy()
    original_array = source.array.copy()
    original_scales = source.scales.copy()
    original_offsets = source.offsets.copy()

    las = write_and_read(source, header, coordinate_guard=grid)

    # The writer defined the file header grid from the CoordinateGrid.
    assert np.array_equal(las.header.scales, grid.scales)
    assert np.array_equal(las.header.offsets, grid.offsets)

    # Every decoded source coordinate was stored within the per-axis budget.
    expected = decoded_xyz(source)
    actual = las.xyz
    assert np.all(np.abs(actual - expected) <= grid.max_errors)

    # Point order and the non-coordinate fields survived.
    assert np.array_equal(np.asarray(las.intensity), [1, 2, 3])
    assert np.array_equal(np.asarray(las.return_number), [1, 2, 1])
    assert np.array_equal(np.asarray(las.classification), [2, 3, 4])
    assert np.array_equal(np.asarray(las["confidence"]), [11, 12, 13])

    # Header statistics match the accepted records.
    assert las.header.point_count == 3
    assert np.array_equal(np.asarray(las.header.mins), actual.min(axis=0))
    assert np.array_equal(np.asarray(las.header.maxs), actual.max(axis=0))
    assert list(las.header.number_of_points_by_return[:3]) == [2, 1, 0]

    # Neither the input record nor the input header were modified.
    assert np.array_equal(header.scales, original_header_scales)
    assert np.array_equal(header.offsets, original_header_offsets)
    assert np.array_equal(source.array, original_array)
    assert np.array_equal(source.scales, original_scales)
    assert np.array_equal(source.offsets, original_offsets)


def test_write_guard_true_uses_header_scale_as_tolerance():
    header = new_header(scales=(0.01, 0.01, 0.01), offsets=(0.0, 0.0, 0.0))
    source = make_source_record(
        [(1.234, 2.345, 3.456)],
        scales=[0.25, 0.25, 0.25],
        offsets=[10.0, 10.0, 10.0],
    )
    las = write_and_read(source, header, coordinate_guard=True)
    assert np.all(np.abs(las.xyz - decoded_xyz(source)) <= header.scales)


def test_write_guard_accepts_explicit_tolerance():
    header = new_header(scales=(0.01, 0.01, 0.01))
    source = make_source_record([(1.234, 2.345, 3.456)], scales=[0.25] * 3)
    las = write_and_read(source, header, coordinate_guard=0.02)
    assert np.all(np.abs(las.xyz - decoded_xyz(source)) <= 0.02)


def test_single_valued_axis_roundtrips_write():
    grid = laspy.make_coordinate_grid([7.0, 0.0, 0.0], [7.0, 100.0, 100.0], 0.001)
    header = new_header()
    source = make_source_record([(7.0, 1.0, 2.0), (7.0, 3.0, 4.0)], scales=[0.25] * 3)
    las = write_and_read(source, header, coordinate_guard=grid)
    assert np.array_equal(las.xyz[:, 0], np.array([7.0, 7.0]))


# ---------------------------------------------------------------------------
# append path
# ---------------------------------------------------------------------------


def test_append_uses_the_grid_stored_in_the_file(tmp_path):
    path = tmp_path / "chunks.las"
    grid = laspy.make_coordinate_grid([0.0] * 3, [100.0] * 3, 0.001)
    header = new_header()
    header.add_extra_dim(laspy.ExtraBytesParams(name="confidence", type="u2"))

    batch_one = make_source_record(
        [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        point_format=header.point_format,
        scales=[0.5] * 3,
        offsets=[1000.0] * 3,
        extra={"confidence": [1, 2]},
    )
    batch_one.return_number = [1, 2]

    with laspy.open(path, mode="w", header=header, coordinate_guard=grid) as writer:
        writer.write_points(batch_one)

    # Snapshot what is on disk after the first batch.
    before = laspy.read(path)
    assert before.header.point_count == 2
    before_array = before.points.array.copy()
    before_header_grid = (
        before.header.scales.copy(),
        before.header.offsets.copy(),
    )

    batch_two = make_source_record(
        [(7.0, 8.0, 9.0), (10.0, 11.0, 12.0), (13.0, 14.0, 15.0)],
        point_format=header.point_format,
        scales=[0.25, 0.125, 0.5],
        offsets=[-50.0, 20.0, 0.0],
        extra={"confidence": [3, 4, 5]},
    )
    batch_two.return_number = [1, 3, 1]

    with laspy.open(path, mode="a", coordinate_guard=True) as appender:
        appender.append_points(batch_two)

    after = laspy.read(path)

    # The header grid was not replaced by the append session.
    assert np.array_equal(after.header.scales, before_header_grid[0])
    assert np.array_equal(after.header.offsets, before_header_grid[1])
    assert np.array_equal(after.header.scales, grid.scales)

    # Previously stored records were not modified, byte for byte.
    assert np.array_equal(after.points.array[:2], before_array)

    # New records went to the right place, within each axis' budget.
    assert np.all(
        np.abs(after.xyz[2:] - decoded_xyz(batch_two)) <= grid.max_errors
    )
    assert np.array_equal(np.asarray(after["confidence"]), [1, 2, 3, 4, 5])
    assert np.array_equal(np.asarray(after.return_number), [1, 2, 1, 3, 1])

    # Header statistics reflect every accepted record.
    assert after.header.point_count == 5
    assert np.array_equal(np.asarray(after.header.mins), after.xyz.min(axis=0))
    assert np.array_equal(np.asarray(after.header.maxs), after.xyz.max(axis=0))
    assert list(after.header.number_of_points_by_return[:4]) == [3, 1, 1, 0]


def test_append_rejects_a_grid_that_differs_from_the_file(tmp_path):
    path = tmp_path / "grid.las"
    grid = laspy.make_coordinate_grid([0.0] * 3, [100.0] * 3, 0.001)
    header = new_header()
    source = make_source_record([(1.0, 2.0, 3.0)], scales=[0.5] * 3)

    with laspy.open(path, mode="w", header=header, coordinate_guard=grid) as writer:
        writer.write_points(source)

    other_grid = laspy.make_coordinate_grid([0.0] * 3, [100.0] * 3, 0.01)
    with pytest.raises(InvalidTargetGrid):
        laspy.open(path, mode="a", coordinate_guard=other_grid)


# ---------------------------------------------------------------------------
# atomic rejection of a batch
# ---------------------------------------------------------------------------


def _make_bad_last_point(kind):
    """Returns (good_points, bad_point) for the requested failure kind."""
    good = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    if kind == "overflow":
        # With a 1.0 source scale, 2e9 decodes fine from the source but does
        # not fit in the target grid's int32 range.
        bad = (2e9, 2.0, 3.0)
    elif kind == "precision":
        # Target scale is 0.01 and the guard budget is 1e-9: this coordinate
        # does not land on a grid step closely enough.
        bad = (1.234567890123, 2.0, 3.0)
    else:  # pragma: no cover - programming error
        raise AssertionError(kind)
    return good, bad


@pytest.mark.parametrize("kind", ["overflow", "precision"])
@pytest.mark.parametrize("mode", ["write", "append"])
def test_bad_last_point_rejects_the_whole_batch_and_session_continues(
    tmp_path, kind, mode
):
    path = tmp_path / f"atomic_{kind}_{mode}.las"
    header = new_header(scales=(0.01, 0.01, 0.01))
    # A numeric tolerance gives no closed interval: the overflow check, not
    # the bounds check, is what rejects the batch.  The precision case uses a
    # tolerance far tighter than the 0.01 target scale.
    tolerance = 1e-9 if kind == "precision" else 0.001

    good, bad = _make_bad_last_point(kind)
    # The source grid must be fine enough to actually hold the offending
    # coordinate: with a coarse 1.0 scale the source itself would round
    # 1.234... to 1.0 before the guard ever sees it.
    source_scales = [1.0, 1.0, 1.0] if kind == "overflow" else [1e-6] * 3
    bad_batch = make_source_record(
        good + [bad], scales=source_scales, offsets=[0.0, 0.0, 0.0]
    )
    bad_batch.return_number = [1, 1, 1]
    good_batch = make_source_record(
        [(2.0, 3.0, 4.0)], scales=[0.25] * 3, offsets=[0.0] * 3
    )
    good_batch.return_number = [3]

    expected_error = {
        "overflow": CoordinateIntegerOverflow,
        "precision": CoordinatePrecisionExceeded,
    }[kind]

    def check_nothing_changed(owner, count, stats, mins, maxs):
        assert owner.header.point_count == count
        assert np.array_equal(owner.header.number_of_points_by_return, stats)
        assert np.array_equal(owner.header.mins, mins)
        assert np.array_equal(owner.header.maxs, maxs)

    if mode == "write":
        stream = io.BytesIO()
        with laspy.open(
            stream,
            mode="w",
            header=header,
            closefd=False,
            coordinate_guard=tolerance,
        ) as writer:
            # First a valid batch, so accumulated statistics exist before the
            # rejected one.
            writer.write_points(good_batch)
            count = writer.header.point_count
            stats = writer.header.number_of_points_by_return.copy()
            mins = writer.header.mins.copy()
            maxs = writer.header.maxs.copy()

            with pytest.raises(expected_error):
                writer.write_points(bad_batch)

            check_nothing_changed(writer, count, stats, mins, maxs)

            # The session can still accept a valid batch afterwards.
            writer.write_points(good_batch)
        stream.seek(0)
        result = laspy.read(stream)
        expected_points = 2
    else:
        with laspy.open(
            path, mode="w", header=header, coordinate_guard=tolerance
        ) as writer:
            writer.write_points(good_batch)
        # The guard tolerances are not part of the header, so the append
        # session gets them explicitly again.
        with laspy.open(path, mode="a", coordinate_guard=tolerance) as appender:
            count = appender.header.point_count
            stats = appender.header.number_of_points_by_return.copy()
            mins = appender.header.mins.copy()
            maxs = appender.header.maxs.copy()

            with pytest.raises(expected_error):
                appender.append_points(bad_batch)

            check_nothing_changed(appender, count, stats, mins, maxs)

            appender.append_points(good_batch)
        result = laspy.read(path)
        expected_points = 2

    assert result.header.point_count == expected_points
    assert len(result.points) == expected_points
    # The rejected point (and the two preceding it in its batch) never made it
    # to the file, and no wrapped-around integer is present.
    assert not np.any(np.asarray(result.x) == bad[0])
    assert np.array_equal(np.asarray(result.return_number), [3, 3])
    # Header statistics reflect exactly the two accepted records.
    assert list(result.header.number_of_points_by_return[:4]) == [0, 0, 2, 0]


@pytest.mark.parametrize("mode", ["write", "append"])
def test_bare_packed_record_without_source_grid_is_rejected(tmp_path, mode):
    path = tmp_path / f"bare_{mode}.las"
    grid = laspy.make_coordinate_grid([0.0] * 3, [100.0] * 3, 0.001)
    header = new_header()
    source = make_source_record([(1.0, 2.0, 3.0)], scales=[0.25] * 3)
    bare = laspy.PackedPointRecord(source.array.copy(), source.point_format)

    if mode == "write":
        stream = io.BytesIO()
        with laspy.open(
            stream, mode="w", header=header, closefd=False, coordinate_guard=grid
        ) as writer:
            with pytest.raises(MissingSourceGrid):
                writer.write_points(bare)
            writer.write_points(source)
        stream.seek(0)
        result = laspy.read(stream)
    else:
        with laspy.open(path, mode="w", header=header, coordinate_guard=grid) as writer:
            writer.write_points(source)
        with laspy.open(path, mode="a", coordinate_guard=True) as appender:
            with pytest.raises(MissingSourceGrid):
                appender.append_points(bare)
            appender.append_points(
                make_source_record([(2.0, 3.0, 4.0)], scales=[0.25] * 3)
            )
        result = laspy.read(path)

    expected = 1 if mode == "write" else 2
    assert result.header.point_count == expected


@pytest.mark.parametrize("mode", ["write", "append"])
def test_point_format_mismatch_is_reported_separately(tmp_path, mode):
    path = tmp_path / f"fmt_{mode}.las"
    grid = laspy.make_coordinate_grid([0.0] * 3, [100.0] * 3, 0.001)
    header = new_header(point_format=3)
    wrong = make_source_record(
        [(1.0, 2.0, 3.0)], point_format=laspy.PointFormat(0), scales=[0.25] * 3
    )
    good = make_source_record(
        [(1.0, 2.0, 3.0)], point_format=laspy.PointFormat(3), scales=[0.25] * 3
    )

    if mode == "write":
        stream = io.BytesIO()
        with laspy.open(
            stream, mode="w", header=header, closefd=False, coordinate_guard=grid
        ) as writer:
            with pytest.raises(IncompatiblePointFormat):
                writer.write_points(wrong)
            writer.write_points(good)
        stream.seek(0)
        result = laspy.read(stream)
    else:
        with laspy.open(path, mode="w", header=header, coordinate_guard=grid) as writer:
            writer.write_points(good)
        with laspy.open(path, mode="a", coordinate_guard=True) as appender:
            with pytest.raises(IncompatiblePointFormat):
                appender.append_points(wrong)
            appender.append_points(good)
        result = laspy.read(path)

    expected = 1 if mode == "write" else 2
    assert result.header.point_count == expected


def test_non_finite_source_coordinate_is_rejected():
    grid = laspy.make_coordinate_grid([0.0] * 3, [100.0] * 3, 0.001)
    header = new_header()

    # Build a record whose integer coordinate is fine, but whose scale makes
    # the decoded coordinate infinite.
    record = ScaleAwarePointRecord.zeros(
        2, point_format=laspy.PointFormat(3), scales=[1.0] * 3, offsets=[0.0] * 3
    )
    record.array["X"] = [1, np.iinfo(np.int32).max]
    record.scales = np.array([1e308, 1.0, 1.0])
    with pytest.raises(NonFiniteCoordinate):
        write_and_read(record, header, coordinate_guard=grid)


def test_invalid_source_grid_is_rejected():
    header = new_header()
    record = make_source_record([(1.0, 2.0, 3.0)], scales=[0.25] * 3)
    # A scale of zero cannot define a source grid.
    record.scales = np.array([0.0, 0.25, 0.25])
    stream = io.BytesIO()
    with laspy.open(
        stream, mode="w", header=header, closefd=False, coordinate_guard=True
    ) as writer:
        with pytest.raises(InvalidSourceGrid):
            writer.write_points(record)


def test_out_of_bounds_coordinate_is_rejected():
    grid = laspy.make_coordinate_grid([0.0] * 3, [10.0] * 3, 0.001)
    header = new_header()
    inside = make_source_record([(1.0, 2.0, 3.0)], scales=[0.25] * 3)
    outside = make_source_record([(50.0, 2.0, 3.0)], scales=[0.25] * 3)

    stream = io.BytesIO()
    with laspy.open(
        stream, mode="w", header=header, closefd=False, coordinate_guard=grid
    ) as writer:
        writer.write_points(inside)
        with pytest.raises(CoordinateOutOfBounds):
            writer.write_points(outside)
            pytest.fail("the out-of-bounds batch must be rejected")
        writer.write_points(inside)
    stream.seek(0)
    result = laspy.read(stream)
    assert result.header.point_count == 2


# ---------------------------------------------------------------------------
# backwards compatibility
# ---------------------------------------------------------------------------


def test_guard_is_off_by_default():
    header = new_header()
    # A bare PackedPointRecord of raw integers still goes through untouched.
    raw = laspy.PackedPointRecord.zeros(2, header.point_format)
    raw.array["X"] = [10, 20]
    raw.array["Y"] = [-30, 40]
    raw.array["Z"] = [50, -60]

    las = write_and_read(raw, header)
    assert np.array_equal(las.points.array["X"], [10, 20])
    assert np.array_equal(las.points.array["Y"], [-30, 40])
    assert np.array_equal(las.points.array["Z"], [50, -60])


def test_guard_is_rejected_in_read_mode():
    stream = io.BytesIO()
    header = new_header()
    with laspy.open(stream, mode="w", header=header, closefd=False) as writer:
        writer.write_points(
            make_source_record([(1.0, 2.0, 3.0)], scales=[0.01] * 3)
        )
    stream.seek(0)
    with pytest.raises(laspy.LaspyException):
        laspy.open(stream, mode="r", coordinate_guard=True)
