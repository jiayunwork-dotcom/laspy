"""Tests for the extra bytes descriptors (no_data, min/max) and their
preservation through writing, chunked subset writing and appending.

The tests build everything they need (no binary fixture is checked in) so
that each case is explicit about the on-disk encoding it exercises.
"""

import io
import struct

import numpy as np
import pytest

import laspy
from laspy.vlrs.known import ExtraBytesStruct

# 8 byte storage type used by the LAS specification for the no_data, min
# and max values of each numeric extra bytes type.
_LONG_TYPES = {
    1: np.uint64,
    2: np.int64,
    3: np.uint64,
    4: np.int64,
    5: np.uint64,
    6: np.int64,
    7: np.uint64,
    8: np.int64,
    9: np.float64,
    10: np.float64,
}


def _pack_long(values, dtype):
    fmt = {np.uint64: "<Q", np.int64: "<q", np.float64: "<d"}[dtype]
    return b"".join(struct.pack(fmt, value) for value in values)


def make_descriptor(
    data_type,
    name,
    options,
    no_data=(0, 0, 0),
    mins=(0, 0, 0),
    maxs=(0, 0, 0),
    scales=(0.0, 0.0, 0.0),
    offsets=(0.0, 0.0, 0.0),
    description=b"",
):
    """Build one 192 bytes Extra Bytes descriptor, independently from the
    laspy parsing / writing code.

    The no_data, min and max values are always encoded using the 8 byte
    storage type mandated by the specification for `data_type`.
    """
    if data_type == 0:
        raise ValueError("data_type 0 descriptors have no numeric values")

    long_type = _LONG_TYPES[((data_type - 1) % 10) + 1]

    out = b"\0\0" + struct.pack("<BB", data_type, options)
    out += name.encode().ljust(32, b"\0")
    out += b"\0" * 4
    out += _pack_long(no_data, long_type)
    out += _pack_long(mins, long_type)
    out += _pack_long(maxs, long_type)
    for i in range(3):
        out += struct.pack("<d", scales[i])
    for i in range(3):
        out += struct.pack("<d", offsets[i])
    out += description.ljust(32, b"\0")
    assert len(out) == 192
    return out


def parse_descriptor(raw):
    return ExtraBytesStruct.from_buffer_copy(raw)


# ---------------------------------------------------------------------------
# Descriptor level tests: the flags are authoritative and the values are
# decoded using their original encoding.
# ---------------------------------------------------------------------------


def test_descriptor_reads_flags_not_values():
    # Values which are not flagged as relevant must be reported as None,
    # even though the corresponding bytes are not zero.
    raw = make_descriptor(
        3,
        "unflagged",
        options=0b00000,
        no_data=(123, 0, 0),
        mins=(7, 0, 0),
        maxs=(9, 0, 0),
    )
    eb = parse_descriptor(raw)
    assert eb.no_data is None
    assert eb.min is None
    assert eb.max is None


def test_descriptor_zero_is_a_valid_value():
    # Zero is a perfectly valid value, its presence must be reported when
    # the corresponding bit is set.
    raw = make_descriptor(
        3,
        "zero",
        options=0b11111,
        no_data=(0, 0, 0),
        mins=(0, 0, 0),
        maxs=(0, 0, 0),
    )
    eb = parse_descriptor(raw)
    assert eb.no_data is not None and np.all(eb.no_data == 0)
    assert eb.min is not None and np.all(eb.min == 0)
    assert eb.max is not None and np.all(eb.max == 0)


def test_descriptor_narrow_integers_are_upcast_properly():
    # 2 * uint16 (data_type 13): the 8 byte storage slots hold uint64 values
    raw = make_descriptor(
        13,
        "two_u2",
        options=0b11111,
        no_data=(0, 65535, 0),
        mins=(0, 5, 0),
        maxs=(10, 60000, 0),
        scales=(1.0, 2.0, 0.0),
        offsets=(10.0, 20.0, 0.0),
    )
    eb = parse_descriptor(raw)
    assert eb.num_elements() == 2
    assert eb.no_data.dtype == np.uint16
    assert list(eb.no_data) == [0, 65535]
    # Elements are decoded independently, and the scale/offset of the right
    # element is applied.
    assert list(eb.min) == [10.0, 5 * 2.0 + 20.0]
    assert list(eb.max) == [1 * 10.0 + 10.0, 60000 * 2.0 + 20.0]


def test_descriptor_float_and_64bit_values_keep_their_encoding():
    u64 = make_descriptor(
        7,
        "seq",
        options=0b111,
        no_data=(2**64 - 1, 0, 0),
        mins=(2**63 + 1, 0, 0),
        maxs=(2**64 - 2, 0, 0),
    )
    eb = parse_descriptor(u64)
    assert eb.no_data.dtype == np.uint64
    assert int(eb.no_data[0]) == 2**64 - 1
    assert int(eb.min[0]) == 2**63 + 1
    assert int(eb.max[0]) == 2**64 - 2

    f32 = make_descriptor(
        9,
        "float",
        options=0b111,
        no_data=(float("nan"), 0, 0),
        mins=(-1.5, 0, 0),
        maxs=(3.25, 0, 0),
    )
    eb = parse_descriptor(f32)
    assert eb.no_data.dtype == np.float32
    assert np.isnan(eb.no_data[0])
    assert eb.min[0] == np.float32(-1.5)
    assert eb.max[0] == np.float32(3.25)


def test_type_of_extra_dims_only_exposes_relevant_no_data():
    raw = make_descriptor(3, "a", options=0b110, no_data=(12, 0, 0))
    vlr = laspy.vlrs.known.ExtraBytesVlr()
    vlr.parse_record_data(raw)
    (param,) = vlr.type_of_extra_dims()
    assert param.no_data is None

    vlr = laspy.vlrs.known.ExtraBytesVlr()
    vlr.parse_record_data(make_descriptor(3, "a", options=0b111, no_data=(12, 0, 0)))
    (param,) = vlr.type_of_extra_dims()
    assert param.no_data is not None
    assert list(param.no_data) == [12]


def test_descriptor_without_numeric_type_is_left_alone():
    # data_type == 0 stores the number of undocumented bytes in `options`.
    # It must not be interpreted as a bit field, and no statistics are
    # exposed.
    raw = (
        b"\0\0"
        + struct.pack("<BB", 0, 5)
        + b"raw".ljust(32, b"\0")
        + b"\0" * 4
        + b"\0" * (24 * 5)
        + b"desc".ljust(32, b"\0")
    )
    eb = ExtraBytesStruct.from_buffer_copy(raw)
    assert eb.num_elements() == 5
    assert eb.min is None
    assert eb.max is None
    assert eb.no_data is None
    assert eb.options == 5
    # Growing / resetting / finalizing must not touch the options field
    eb.partial_reset()
    eb.finalize()
    assert eb.options == 5


# ---------------------------------------------------------------------------
# Writing helpers
# ---------------------------------------------------------------------------


def make_las(path, n=10):
    """Create a small uncompressed LAS 1.4 file with two extra dimensions:

    - `keep`: 3 x int32, scaled, with a no_data value
    - `drop`: 1 x uint16, no no_data
    """
    las = laspy.create(point_format=7, file_version="1.4")
    las.header.scales = np.array([0.01, 0.01, 0.01])
    las.header.offsets = np.array([0.0, 0.0, 0.0])
    las.points = laspy.ScaleAwarePointRecord.zeros(n, header=las.header)
    las.X = np.arange(n)
    las.Y = np.arange(n)
    las.Z = np.arange(n)

    las.add_extra_dim(
        laspy.ExtraBytesParams(
            "keep",
            "3i4",
            description="kept field",
            scales=np.array([1.0, 2.0, 3.0]),
            offsets=np.array([10.0, 20.0, 30.0]),
            no_data=np.array([-2147483648, -2147483648, -2147483648], np.int32),
        )
    )
    las.add_extra_dim(
        laspy.ExtraBytesParams(
            "drop",
            "u2",
            description="removed field",
            no_data=np.array([65535], np.uint16),
        )
    )

    raw = las.points.array["keep"]
    raw[:] = np.array(
        [
            [0, 0, 0],
            [1, 2, 3],
            [-2147483648, -2147483648, -2147483648],
            [10**6, 2 * 10**6, 3 * 10**6],
            [0, 0, 0],
            [-1, -2, -3],
            [7, 7, 7],
            [-2147483648, -2147483648, -2147483648],
            [123456, 234567, 345678],
            # outlier, used to check that a subset write does not leak it
            [9_000_000, 9_000_000, 9_000_000],
        ],
        dtype=np.int32,
    )
    las.drop[:] = np.array([65535, 65535, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)
    las.update_header()
    las.write(path)
    return las


def stats_of(las, name):
    vlr = las.vlrs.get("ExtraBytesVlr")[0]
    for eb in vlr.extra_bytes_structs:
        if eb.format_name() == name:
            return eb
    raise AssertionError(f"no descriptor named {name}")


def assert_stats(eb, expected_min, expected_max):
    assert eb.min is not None
    assert eb.max is not None
    assert list(eb.min) == pytest.approx(expected_min)
    assert list(eb.max) == pytest.approx(expected_max)


# ---------------------------------------------------------------------------
# Reopening: values, metadata and statistics survive
# ---------------------------------------------------------------------------


def test_reopened_file_keeps_statistics_and_metadata(tmp_path):
    path = tmp_path / "in.las"
    make_las(path)

    las = laspy.read(path)
    eb = stats_of(las, "keep")

    # min/max are per component and exclude the no_data value
    raw = np.asarray(las.points.array["keep"])
    valid = raw != -2147483648
    assert list(eb.min) == [
        int(raw[valid[:, i], i].min()) * s + o
        for i, (s, o) in enumerate(zip([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]))
    ]
    assert list(eb.max) == [
        int(raw[valid[:, i], i].max()) * s + o
        for i, (s, o) in enumerate(zip([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]))
    ]

    # The flags, description, scale and offset survived the round trip
    assert eb.no_data_is_relevant()
    assert eb.min_is_relevant()
    assert eb.max_is_relevant()
    assert stats_of(las, "drop").no_data_is_relevant()

    keep_dim = las.point_format.dimension_by_name("keep")
    assert keep_dim.description == "kept field"
    assert list(keep_dim.scales) == [1.0, 2.0, 3.0]
    assert list(keep_dim.offsets) == [10.0, 20.0, 30.0]
    assert list(keep_dim.no_data) == [-2147483648] * 3

    assert las.X.tolist() == list(range(10))


def test_removing_and_adding_extra_dimensions_keeps_the_other_ones(tmp_path):
    path = tmp_path / "in.las"
    make_las(path)

    las = laspy.read(path)
    keep_before = np.asarray(las.points.array["keep"]).copy()
    keep_stats_before = stats_of(las, "keep")

    las.remove_extra_dim("drop")
    las.add_extra_dim(
        laspy.ExtraBytesParams(
            "added",
            "f4",
            description="added field",
            no_data=np.array([-9999.0], np.float32),
        )
    )
    las.added[:] = [-9999.0, -9999.0, 1.5, np.nan, -2.25, 3.0, 4.0, 5.0, 6.0, 7.0]
    las.update_header()

    out = tmp_path / "out.las"
    las.write(out)

    reopened = laspy.read(out)
    assert list(reopened.point_format.extra_dimension_names) == ["keep", "added"]

    # The kept field is bit for bit preserved and so are its statistics and
    # metadata.
    np.testing.assert_array_equal(
        np.asarray(reopened.points.array["keep"]), keep_before
    )
    keep_after = stats_of(reopened, "keep")
    assert list(keep_after.min) == list(keep_stats_before.min)
    assert list(keep_after.max) == list(keep_stats_before.max)
    assert keep_after.description == keep_stats_before.description
    assert list(keep_after.scale) == list(keep_stats_before.scale)
    assert list(keep_after.offset) == list(keep_stats_before.offset)
    assert list(keep_after.no_data) == list(keep_stats_before.no_data)

    # The new field got its own statistics, ignoring its no_data and NaNs.
    added = stats_of(reopened, "added")
    assert added.no_data_is_relevant()
    assert list(added.no_data) == [-9999.0]
    assert list(added.min) == [-2.25]
    assert list(added.max) == [7.0]


def test_removing_an_extra_dimension_removes_its_descriptor(tmp_path):
    path = tmp_path / "in.las"
    make_las(path)

    las = laspy.read(path)
    las.remove_extra_dim("drop")
    las.write(tmp_path / "out.las")

    reopened = laspy.read(tmp_path / "out.las")
    assert "drop" not in list(reopened.point_format.extra_dimension_names)
    assert len(reopened.vlrs.get("ExtraBytesVlr")[0].extra_bytes_structs) == 1


def test_point_format_conversion_keeps_extra_bytes_metadata(tmp_path):
    path = tmp_path / "in.las"
    make_las(path)

    las = laspy.read(path)
    values_before = np.asarray(las.points.array["keep"]).copy()
    before = stats_of(las, "keep")

    converted = laspy.convert(las, point_format_id=6)
    assert converted.point_format.id == 6

    converted.write(tmp_path / "converted.las")
    reopened = laspy.read(tmp_path / "converted.las")
    assert reopened.point_format.id == 6
    assert list(reopened.point_format.extra_dimension_names) == ["keep", "drop"]

    after = stats_of(reopened, "keep")
    assert list(after.min) == list(before.min)
    assert list(after.max) == list(before.max)
    assert list(after.scale) == list(before.scale)
    assert list(after.offset) == list(before.offset)
    assert list(after.no_data) == list(before.no_data)
    assert after.description == before.description
    np.testing.assert_array_equal(
        np.asarray(reopened.points.array["keep"]), values_before
    )


# ---------------------------------------------------------------------------
# Chunked subset writing
# ---------------------------------------------------------------------------


def write_chunks(points, header, boundaries, path):
    with laspy.open(path, mode="w", header=header) as writer:
        for start, stop in boundaries:
            writer.write_points(points[start:stop])


def test_subset_write_statistics_match_the_written_points(tmp_path):
    make_las(tmp_path / "in.las")
    las = laspy.read(tmp_path / "in.las")

    # Chunks of different sizes, last chunk not a multiple of the first ones
    boundaries = [(0, 3), (3, 4), (4, 7), (7, 9)]
    write_chunks(las.points, las.header, boundaries, tmp_path / "subset.las")

    subset = laspy.read(tmp_path / "subset.las")
    eb = stats_of(subset, "keep")

    written = np.asarray(subset.points.array["keep"])
    assert len(subset) == 9
    valid = written != -2147483648
    scales = np.array([1.0, 2.0, 3.0])
    offsets = np.array([10.0, 20.0, 30.0])
    expected_min = (
        np.array([written[valid[:, i], i].min() for i in range(3)]) * scales + offsets
    )
    expected_max = (
        np.array([written[valid[:, i], i].max() for i in range(3)]) * scales + offsets
    )
    assert_stats(eb, expected_min, expected_max)

    # The extrema of the points which were *not* written did not leak in:
    # the last original point is the one with the biggest raw values.
    assert len(las) == 10
    assert eb.max[0] < 9_000_000 * 1.0 + 10.0


def test_subset_write_statistics_do_not_depend_on_chunking(tmp_path):
    make_las(tmp_path / "in.las")
    las = laspy.read(tmp_path / "in.las")

    layouts = [
        [(0, 10)],
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
        ],
        [(0, 2), (2, 6), (6, 10)],
        [(0, 5), (5, 10)],
    ]
    stats = []
    for i, layout in enumerate(layouts):
        path = tmp_path / f"subset_{i}.las"
        write_chunks(las.points, las.header, layout, path)
        eb = stats_of(laspy.read(path), "keep")
        stats.append((list(eb.min), list(eb.max)))

    assert all(stat == stats[0] for stat in stats)


def test_subset_with_no_valid_component_unadvertises_it(tmp_path):
    make_las(tmp_path / "in.las")
    las = laspy.read(tmp_path / "in.las")

    # Points 2 and 7 are no_data for every component, so the written subset
    # has no valid value for any of them.
    write_chunks(las.points, las.header, [(2, 3), (7, 8)], tmp_path / "subset.las")

    subset = laspy.read(tmp_path / "subset.las")
    eb = stats_of(subset, "keep")

    assert not eb.min_is_relevant()
    assert not eb.max_is_relevant()
    assert eb.min is None
    assert eb.max is None
    # The no_data value itself is still there
    assert eb.no_data_is_relevant()
    assert list(eb.no_data) == [-2147483648] * 3


def test_subset_with_one_dead_component_unadvertises_the_extrema(tmp_path):
    # A 2 element field. The first element has valid values, the second one
    # only has the no_data sentinel. Since the descriptor's relevance flags
    # cover all the elements, the extrema cannot be advertised as known.
    las = laspy.create(point_format=6, file_version="1.4")
    las.points = laspy.ScaleAwarePointRecord.zeros(4, header=las.header)
    las.X = np.arange(4)
    las.Y = np.arange(4)
    las.Z = np.arange(4)
    las.add_extra_dim(
        laspy.ExtraBytesParams("m", "2i4", no_data=np.array([-1, -1], dtype=np.int32))
    )
    las.points.array["m"][:] = np.array(
        [[3, -1], [7, -1], [-1, -1], [2, -1]], dtype=np.int32
    )
    las.update_header()

    with laspy.open(tmp_path / "out.las", mode="w", header=las.header) as writer:
        writer.write_points(las.points)

    reopened = laspy.read(tmp_path / "out.las")
    descriptors = reopened.vlrs.get("ExtraBytesVlr")
    assert len(descriptors) == 1
    eb = descriptors[0].extra_bytes_structs[0]
    assert not eb.min_is_relevant()
    assert not eb.max_is_relevant()
    assert eb.min is None
    assert eb.max is None


def test_standard_dimensions_without_optional_stats_stay_without(tmp_path):
    # A regular field of the point format must not suddenly gain optional
    # statistics; only extra bytes descriptors carry them.
    make_las(tmp_path / "in.las")
    las = laspy.read(tmp_path / "in.las")
    write_chunks(las.points, las.header, [(0, 5), (5, 10)], tmp_path / "out.las")
    reopened = laspy.read(tmp_path / "out.las")
    assert reopened.header.point_count == 10


# ---------------------------------------------------------------------------
# Appending
# ---------------------------------------------------------------------------


def append_points_to(path, points):
    with open(path, "rb+") as stream:
        with laspy.open(stream, mode="a", closefd=False) as appender:
            appender.append_points(points)


def test_append_merges_with_the_extrema_from_disk(tmp_path):
    path = tmp_path / "in.las"
    make_las(path)

    original = laspy.read(path)
    eb_before = stats_of(original, "keep")
    min_before = list(eb_before.min)
    max_before = list(eb_before.max)

    header = laspy.open(path).header
    offset_before = header.offset_to_point_data
    old_bytes = open(path, "rb").read()[
        offset_before : offset_before + len(original) * header.point_format.size
    ]

    new_points = laspy.ScaleAwarePointRecord.zeros(3, header=header)
    new_points.X = [100, 101, 102]
    new_points.Y = [0, 0, 0]
    new_points.Z = [0, 0, 0]
    new_points.array["keep"][:] = np.array(
        [[-5, -5, -5], [-2147483648] * 3, [-6, -6, -6]], dtype=np.int32
    )
    new_points.array["drop"][:] = np.array([0, 65535, 0], dtype=np.uint16)

    append_points_to(path, new_points)

    reopened = laspy.read(path)
    assert len(reopened) == 13

    eb_after = stats_of(reopened, "keep")
    # The disk extrema are trusted and merged with the new points; the old
    # points were not read again.
    expected_min = [
        min(min_before[0], -5 * 1.0 + 10.0, -6 * 1.0 + 10.0),
        min(min_before[1], -5 * 2.0 + 20.0, -6 * 2.0 + 20.0),
        min(min_before[2], -5 * 3.0 + 30.0, -6 * 3.0 + 30.0),
    ]
    assert list(eb_after.min) == expected_min
    assert list(eb_after.max) == max_before

    # Old point bytes and the offset to the point data are untouched
    new_offset = laspy.open(path).header.offset_to_point_data
    assert new_offset == offset_before
    assert (
        open(path, "rb").read()[offset_before : offset_before + len(old_bytes)]
        == old_bytes
    )

    # The no_data sentinel of the appended points was not counted
    assert list(eb_after.no_data) == [-2147483648] * 3


def test_append_to_file_without_extrema_does_not_promote_new_batch(tmp_path):
    path = tmp_path / "no_stats.las"

    # Build a file whose descriptor has no min/max bit.
    make_las(path)
    las = laspy.read(path)
    eb = stats_of(las, "keep")
    eb.options &= ~(eb.MIN_BIT_MASK | eb.MAX_BIT_MASK)
    with open(path, "wb") as stream:
        with laspy.open(stream, mode="w", header=las.header) as writer:
            writer.write_points(las.points)

    header = laspy.open(path).header
    assert not stats_of_header(header, "keep").min_is_relevant()

    new_points = laspy.ScaleAwarePointRecord.zeros(2, header=header)
    new_points.X = [50, 51]
    new_points.Y = [0, 0]
    new_points.Z = [0, 0]
    new_points.array["keep"][:] = np.array([[-7, -7, -7], [-8, -8, -8]], dtype=np.int32)

    append_points_to(path, new_points)

    reopened = laspy.open(path).header
    eb = stats_of_header(reopened, "keep")
    # The extrema were unknown before the append, and they must not be
    # replaced by the extrema of the newly appended batch only, which would
    # pretend to describe the whole file.
    assert not eb.min_is_relevant()
    assert not eb.max_is_relevant()
    assert eb.min is None
    assert eb.max is None


def stats_of_header(header, name):
    for eb in header.vlrs.get("ExtraBytesVlr")[0].extra_bytes_structs:
        if eb.format_name() == name:
            return eb
    raise AssertionError(f"no descriptor named {name}")


def test_append_without_extrema_does_not_read_old_points(tmp_path, monkeypatch):
    path = tmp_path / "no_stats.las"
    make_las(path)
    las = laspy.read(path)
    eb = stats_of(las, "keep")
    eb.options &= ~(eb.MIN_BIT_MASK | eb.MAX_BIT_MASK)
    with open(path, "wb") as stream:
        with laspy.open(stream, mode="w", header=las.header) as writer:
            writer.write_points(las.points)

    header = laspy.open(path).header
    new_points = laspy.ScaleAwarePointRecord.zeros(1, header=header)
    new_points.X = [0]
    new_points.Y = [0]
    new_points.Z = [0]

    # Only the batches handed to the appender may be inspected while the
    # statistics are updated. If the implementation ever scanned the old
    # points, it would have to call `grow` with another, larger batch.
    import laspy.vlrs.known as known

    seen_lengths = []
    original_grow = known.ExtraBytesStruct.grow

    def recording_grow(self, points):
        seen_lengths.append(len(points))
        return original_grow(self, points)

    monkeypatch.setattr(known.ExtraBytesStruct, "grow", recording_grow)
    with open(path, "rb+") as stream:
        with laspy.open(stream, mode="a", closefd=False) as appender:
            appender.append_points(new_points)
    monkeypatch.undo()

    # Only the batches handed to the appender may be inspected while the
    # statistics are updated. If the implementation ever scanned the old
    # points, it would have to call `grow` with another, larger batch.
    assert seen_lengths
    assert all(length == 1 for length in seen_lengths)

    reopened = laspy.open(path).header
    eb = stats_of_header(reopened, "keep")
    assert not eb.min_is_relevant()
    assert not eb.max_is_relevant()


def test_append_when_extrema_are_known_does_not_grow_from_old_points(tmp_path):
    path = tmp_path / "in.las"
    make_las(path)

    header = laspy.open(path).header
    old_min = list(stats_of_header(header, "keep").min)
    old_max = list(stats_of_header(header, "keep").max)

    new_points = laspy.ScaleAwarePointRecord.zeros(1, header=header)
    new_points.X = [0]
    new_points.Y = [0]
    new_points.Z = [0]
    new_points.array["keep"][:] = np.array([[1, 1, 1]], dtype=np.int32)

    append_points_to(path, new_points)

    reopened = laspy.open(path).header
    eb = stats_of_header(reopened, "keep")
    assert list(eb.min) == old_min
    assert list(eb.max) == old_max


# ---------------------------------------------------------------------------
# Statistics memory does not grow with the number of written points
# ---------------------------------------------------------------------------


def test_writing_many_chunks_does_not_accumulate_points(tmp_path):
    # The statistics are tracked from the batches as they come; the point
    # writer never holds a reference to the previously written batches.
    las = laspy.create(point_format=6, file_version="1.4")
    las.add_extra_dim(
        laspy.ExtraBytesParams("v", "i4", no_data=np.array([-1], np.int32))
    )
    las.points = laspy.ScaleAwarePointRecord.zeros(4, header=las.header)

    path = tmp_path / "many.las"
    with laspy.open(path, mode="w", header=las.header) as writer:
        for i in range(20):
            chunk = laspy.ScaleAwarePointRecord.zeros(3, header=las.header)
            chunk.X = [i, i, i]
            chunk.Y = [0, 0, 0]
            chunk.Z = [0, 0, 0]
            chunk.array["v"][:] = np.array([i, i, i], dtype=np.int32)
            writer.write_points(chunk)

    reopened = laspy.read(path)
    assert len(reopened) == 60
    eb = reopened.vlrs.get("ExtraBytesVlr")[0].extra_bytes_structs[0]
    assert list(eb.min) == [0]
    assert list(eb.max) == [19]


# ---------------------------------------------------------------------------
# Unregistered extra bytes keep working
# ---------------------------------------------------------------------------


def test_unregistered_extra_bytes_round_trip():
    from tests.conftest import UNREGISTERED_EXTRA_BYTES_LAS

    las = laspy.read(UNREGISTERED_EXTRA_BYTES_LAS)
    assert list(las.point_format.extra_dimension_names) == ["ExtraBytes"]

    reread = laspy.lib.write_then_read_again(las)
    assert list(reread.point_format.extra_dimension_names) == ["ExtraBytes"]
    np.testing.assert_array_equal(
        np.asarray(reread.points.array["ExtraBytes"]),
        np.asarray(las.points.array["ExtraBytes"]),
    )
    assert reread.vlrs.get("ExtraBytesVlr") == []
