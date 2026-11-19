"""Offline tests for the strict selective-decompression read mode.

These tests never touch the network. They build small LAZ files in memory
with distinct, partly-zero values in every field, then check that strict
mode distinguishes fields that were actually decoded from fields that were
skipped by the :class:`laspy.DecompressionSelection`.

Coverage:

* both LAZ backends (lazrs and laszip) on a normal LAZ,
* real zeros stay readable while skipped fields raise even when their
  placeholder bytes happen to be zero,
* packed sub-fields, extra dimensions and the scaled ``z`` view,
* slicing / boolean masking / copying / chunked reads / seek,
* empty results and reader closing,
* COPC 2D and 3D window queries (using the repository sample),
* uncompressed LAS and old point formats as controls.
"""

import io
from pathlib import Path

import numpy as np
import pytest

import laspy
from laspy.errors import FieldNotReadableError
from laspy.point import dims as point_dims

SIMPLE_COPC_FILE = Path(__file__).parent / "data" / "simple.copc.laz"
LAZ_1_4_FILE = Path(__file__).parent / "data" / "1_4_w_evlr.laz"
UNCOMPRESSED_LAS = Path(__file__).parent / "data" / "simple.las"
OLD_LAZ = Path(__file__).parent / "data" / "simple.laz"


LAZRS = pytest.param(
    laspy.LazBackend.Lazrs,
    marks=pytest.mark.skipif(
        not laspy.LazBackend.Lazrs.is_available(), reason="lazrs not installed"
    ),
)
LASZIP = pytest.param(
    laspy.LazBackend.Laszip,
    marks=pytest.mark.skipif(
        not laspy.LazBackend.Laszip.is_available(), reason="laszip not installed"
    ),
)
LAZRS_PARALLEL = pytest.param(
    laspy.LazBackend.LazrsParallel,
    marks=pytest.mark.skipif(
        not laspy.LazBackend.Lazrs.is_available(), reason="lazrs not installed"
    ),
)


def _build_populated_laz(point_format_id=7, n=300, with_extra=True):
    """Creates an in-memory LAZ whose every field holds distinct values,
    including real zeros in some of them (intensity, blue, ...)."""
    header = laspy.LasHeader(point_format=point_format_id, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([100.0, 200.0, 300.0])
    if with_extra:
        header.add_extra_dims(
            [
                laspy.ExtraBytesParams(name="my_extra", type="uint16"),
                laspy.ExtraBytesParams(
                    name="my_scaled",
                    type="int32",
                    scales=np.array([0.5]),
                    offsets=np.array([1.0]),
                ),
            ]
        )

    cloud = laspy.LasData(header)
    idx = np.arange(n)
    cloud.X = (idx * 10).astype(np.int32)
    cloud.Y = (idx * 20 + 1).astype(np.int32)
    cloud.Z = (idx * 30 + 2).astype(np.int32)
    # intensity contains genuine zeros
    cloud.intensity = (idx % 7).astype(np.uint16)
    cloud.return_number = (idx % 3 + 1).astype(np.uint8)
    cloud.number_of_returns = (idx % 2 + 1).astype(np.uint8)
    cloud.scanner_channel = (idx % 2).astype(np.uint8)
    cloud.scan_direction_flag = (idx % 2).astype(np.uint8)
    cloud.edge_of_flight_line = (idx % 2).astype(np.uint8)
    cloud.classification = (idx % 5).astype(np.uint8)
    cloud.synthetic = (idx % 2).astype(np.uint8)
    cloud.key_point = (idx % 2).astype(np.uint8)
    cloud.withheld = (idx % 3 == 0).astype(np.uint8)
    cloud.overlap = (idx % 4 == 0).astype(np.uint8)
    cloud.user_data = (idx % 9).astype(np.uint8)
    cloud.scan_angle = (idx - 150).astype(np.int16)
    cloud.point_source_id = (idx + 1).astype(np.uint16)
    cloud.gps_time = idx.astype(np.float64) * 0.25
    if point_format_id in (2, 3, 7, 8, 10):
        cloud.red = idx.astype(np.uint16)
        cloud.green = (idx + 100).astype(np.uint16)
        # blue is a real, decoded zero
        cloud.blue = np.zeros(n, dtype=np.uint16)
    if point_format_id in (8, 10):
        cloud.nir = (idx + 7).astype(np.uint16)
    if with_extra:
        cloud.my_extra = (idx * 3).astype(np.uint16)
        cloud.my_scaled = (idx - 50).astype(np.int32)

    buffer = io.BytesIO()
    cloud.write(buffer, do_compress=True)
    return buffer.getvalue()


def _raw_value(record, name):
    """Reads a (possibly packed sub-) field straight from the structured
    array, bypassing the public readability guards."""
    sub = record.sub_fields_dict.get(name)
    if sub is not None:
        composed, sub_field = sub
        lsb = point_dims.packing.least_significant_bit_set(sub_field.mask)
        return (record.array[composed] & sub_field.mask) >> lsb
    return record.array[name]


@pytest.mark.parametrize("backend", [LAZRS, LASZIP, LAZRS_PARALLEL])
def test_strict_mode_flags_skipped_fields_and_keeps_decoded_ones(backend):
    data = _build_populated_laz()
    full = laspy.read(io.BytesIO(data), laz_backend=backend)
    selection = (
        laspy.DecompressionSelection.base().decompress_intensity().decompress_z()
    )
    partial = laspy.read(
        io.BytesIO(data),
        laz_backend=backend,
        decompression_selection=selection,
        strict=True,
    )

    # The point format is unchanged.
    assert partial.point_format == full.point_format

    # Readable fields decode to the exact, fully-decompressed values.
    for name in partial.readable_fields():
        np.testing.assert_array_equal(np.asarray(partial[name]), np.asarray(full[name]))

    # Z and its scaled view z are both readable, x/y too.
    assert partial.is_field_readable("Z")
    np.testing.assert_allclose(np.asarray(partial.z), np.asarray(full.z))
    np.testing.assert_array_equal(partial.xyz, full.xyz)

    # Real zeros that were decoded stay readable.
    assert partial.is_field_readable("intensity")
    assert int((partial.intensity == 0).sum()) > 0

    # Skipped standard fields, colors and extra dimensions raise with the
    # field name in the error, including blue whose real value is zero.
    for name in (
        "red",
        "green",
        "blue",
        "gps_time",
        "scan_angle",
        "user_data",
        "point_source_id",
        "my_extra",
        "my_scaled",
    ):
        assert not partial.is_field_readable(name)
        with pytest.raises(FieldNotReadableError) as info:
            partial[name]
        assert info.value.field_name == name
        with pytest.raises(FieldNotReadableError):
            getattr(partial, name)
        # writing is guarded as well
        with pytest.raises(FieldNotReadableError):
            partial[name] = 0

    # Packed sub-fields of the skipped flags item are guarded individually.
    for name in ("withheld", "overlap", "synthetic", "key_point"):
        assert not partial.is_field_readable(name)
        with pytest.raises(FieldNotReadableError) as info:
            partial[name]
        assert info.value.field_name == name

    # The base item's own sub-fields stay readable.
    for name in ("return_number", "number_of_returns", "scanner_channel"):
        assert partial.is_field_readable(name)
        np.testing.assert_array_equal(partial[name], full[name])


def test_scaled_views_and_legacy_aliases_follow_their_packed_fields():
    data = _build_populated_laz(point_format_id=6, with_extra=False)
    with_z = laspy.read(
        io.BytesIO(data),
        decompression_selection=laspy.DecompressionSelection.base().decompress_z(),
        strict=True,
    )
    # The lowercase scaled view is readable exactly when its packed field is.
    assert with_z.is_field_readable("Z")
    assert with_z.is_field_readable("z")
    np.testing.assert_allclose(
        np.asarray(with_z.z),
        np.asarray(with_z.Z) * with_z.header.z_scale + with_z.header.z_offset,
    )

    without_z = laspy.read(
        io.BytesIO(data),
        decompression_selection=laspy.DecompressionSelection.base(),
        strict=True,
    )
    assert not without_z.is_field_readable("Z")
    assert not without_z.is_field_readable("z")
    for accessor in ("z", "Z"):
        with pytest.raises(FieldNotReadableError):
            without_z[accessor]
    with pytest.raises(FieldNotReadableError):
        without_z.z = np.zeros(len(without_z))
    with pytest.raises(FieldNotReadableError):
        without_z.xyz

    # Legacy laspy names resolve to the same readability as their new names.
    assert without_z.is_field_readable("return_num")
    assert not without_z.is_field_readable("pt_src_id")
    with pytest.raises(FieldNotReadableError) as info:
        without_z.pt_src_id
    assert info.value.field_name == "pt_src_id"


@pytest.mark.parametrize("backend", [LAZRS, LASZIP])
def test_strict_mode_matches_what_backend_actually_decodes(backend):
    # Every field reported unreadable must really differ from the fully
    # decoded data on disk (the sample values are all non-constant).
    data = _build_populated_laz(point_format_id=6, with_extra=False)
    full = laspy.read(io.BytesIO(data), laz_backend=backend)
    partial = laspy.read(
        io.BytesIO(data),
        laz_backend=backend,
        decompression_selection=laspy.DecompressionSelection.base(),
        strict=True,
    )
    for name in partial.points.readability.unreadable_fields():
        assert not np.array_equal(
            _raw_value(full.points, name), _raw_value(partial.points, name)
        ), name
    for name in partial.readable_fields():
        np.testing.assert_array_equal(
            _raw_value(full.points, name), _raw_value(partial.points, name)
        )

    # Enabling the FLAGS item unlocks every flags sub-field.
    with_flags = laspy.read(
        io.BytesIO(data),
        laz_backend=backend,
        decompression_selection=laspy.DecompressionSelection.base().decompress_flags(),
        strict=True,
    )
    for name in (
        "scan_direction_flag",
        "edge_of_flight_line",
        "overlap",
        "withheld",
        "synthetic",
        "key_point",
        "scanner_channel",
    ):
        assert with_flags.is_field_readable(name)
        np.testing.assert_array_equal(with_flags[name], full[name])
    assert not with_flags.is_field_readable("classification")
    assert not with_flags.is_field_readable("Z")


@pytest.mark.parametrize("backend", [LAZRS, LASZIP])
def test_extra_bytes_selection_and_unknown_field_rules(backend):
    data = _build_populated_laz()
    full = laspy.read(io.BytesIO(data), laz_backend=backend)
    selection = laspy.DecompressionSelection.base().decompress_all_extra_bytes()
    partial = laspy.read(
        io.BytesIO(data),
        laz_backend=backend,
        decompression_selection=selection,
        strict=True,
    )
    assert partial.is_field_readable("my_extra")
    assert partial.is_field_readable("my_scaled")
    np.testing.assert_array_equal(partial.my_extra, full.my_extra)
    np.testing.assert_allclose(
        np.asarray(partial["my_scaled"]), np.asarray(full["my_scaled"])
    )
    # Standard skipped fields remain guarded.
    assert not partial.is_field_readable("red")
    assert not partial.is_field_readable("Z")

    # Fields absent from the point format keep the historical error type.
    with pytest.raises(ValueError):
        partial.points["does_not_exist"]
    with pytest.raises(AttributeError):
        partial.does_not_exist

    # The raw structured array remains directly operable.
    partial.points.array["Z"][0] = 12345


def test_readable_fields_query_does_not_trigger_reading():
    selection = laspy.DecompressionSelection.base()
    with laspy.open(
        LAZ_1_4_FILE, decompression_selection=selection, strict=True
    ) as reader:
        before = reader.points_read
        readable = reader.readable_fields()
        assert reader.is_field_readable("X")
        assert not reader.is_field_readable("Z")
        # No point was read just to answer the query.
        assert reader.points_read == before
        assert "X" in readable and "Z" not in readable

        points = reader.read_points(20)
        assert points.readable_fields() == readable
        # An empty record still carries the information.
        reader.seek(0)
        empty = reader.read_points(0)
        assert len(empty) == 0
        assert empty.readable_fields() == readable
        assert not empty.is_field_readable("Z")
        with pytest.raises(FieldNotReadableError):
            empty.Z


@pytest.mark.parametrize("backend", [LAZRS, LASZIP])
def test_views_masks_chunks_seek_and_close_preserve_readability(backend):
    data = _build_populated_laz()
    selection = laspy.DecompressionSelection.base().decompress_intensity()

    with laspy.open(
        io.BytesIO(data),
        laz_backend=backend,
        decompression_selection=selection,
        strict=True,
    ) as reader:
        first = reader.read_points(120)
        reader.seek(200)
        second = reader.read_points(50)
        chunk = reader.chunk_iterator(40)
        chunks = list(chunk)

    # Valid after closing the reader.
    for record in (first, second, *chunks):
        assert record.is_field_readable("X")
        assert not record.is_field_readable("Z")
        with pytest.raises(FieldNotReadableError):
            record.Z

    # Slicing, integer indexing, boolean masking and copying.
    sliced = first[10:40]
    indexed = first[5]
    masked = first[first.X > first.X.min()]
    copied = first.copy()
    for view in (sliced, indexed, masked, copied):
        assert view.is_field_readable("X")
        assert not view.is_field_readable("Z")
    with pytest.raises(FieldNotReadableError):
        sliced.gps_time

    # Same guarantees at the LasData level.
    las = laspy.read(
        io.BytesIO(data),
        laz_backend=backend,
        decompression_selection=selection,
        strict=True,
    )
    las_slice = las[10:40]
    las_mask = las[las.X > np.median(las.X)]
    las_empty = las[np.zeros(len(las), dtype=bool)]
    for view in (las_slice, las_mask, las_empty):
        assert view.is_field_readable("X")
        assert not view.is_field_readable("Z")
    assert las_empty.readable_fields() == las.readable_fields()
    with pytest.raises(FieldNotReadableError):
        las_slice.xyz


def test_two_independent_reads_keep_their_own_selection():
    selection = laspy.DecompressionSelection.base()
    first = laspy.read(LAZ_1_4_FILE, decompression_selection=selection, strict=True)
    second = laspy.read(
        LAZ_1_4_FILE,
        decompression_selection=laspy.DecompressionSelection.all(),
        strict=True,
    )
    assert not first.is_field_readable("Z")
    assert second.is_field_readable("Z")
    # The first reader's record is unaffected by the second read.
    with pytest.raises(FieldNotReadableError):
        first.Z
    second.Z  # does not raise


@pytest.mark.parametrize("path", [UNCOMPRESSED_LAS, OLD_LAZ])
def test_uncompressed_and_old_formats_expose_everything(path):
    # Selective decompression is unsupported there: strict mode must not
    # hide any field.
    cloud = laspy.read(
        path,
        decompression_selection=laspy.DecompressionSelection.base(),
        strict=True,
    )
    expected = set(cloud.point_format.dimension_names) | set(
        cloud.points.array.dtype.names
    )
    assert cloud.readable_fields() == expected
    assert cloud.is_field_readable("Z")
    np.testing.assert_array_equal(cloud.Z, laspy.read(path).Z)


def test_strict_mode_defaults_to_historical_behaviour():
    selection = laspy.DecompressionSelection.base()
    cloud = laspy.read(LAZ_1_4_FILE, decompression_selection=selection)
    # No metadata, skipped fields remain accessible (placeholder values).
    assert cloud.points.readability is None
    _ = cloud.Z
    with laspy.open(LAZ_1_4_FILE, decompression_selection=selection) as reader:
        assert reader.readability is None


@pytest.mark.skipif(
    not laspy.LazBackend.Lazrs.is_available(), reason="lazrs not installed"
)
class TestCopcStrictMode:
    @staticmethod
    def _z_window():
        # A 3D window narrow enough to actually drop points on the sample.
        full = laspy.read(SIMPLE_COPC_FILE)
        z_min, z_max = float(full.z.min()), float(full.z.max())
        mid = (z_min + z_max) / 2
        header_mins, header_maxs = full.header.mins, full.header.maxs
        return laspy.Bounds(
            mins=np.array([header_mins[0], header_mins[1], mid]),
            maxs=np.array([header_maxs[0], header_maxs[1], z_max]),
        )

    @staticmethod
    def _xy_window():
        return laspy.Bounds(
            mins=np.array([636000.0, 849000.0]),
            maxs=np.array([637500.0, 851000.0]),
        )

    def test_reader_exposes_readability_without_querying(self):
        selection = laspy.DecompressionSelection.base()
        with laspy.CopcReader.open(
            SIMPLE_COPC_FILE, decompression_selection=selection, strict=True
        ) as reader:
            assert reader.is_field_readable("X")
            assert not reader.is_field_readable("Z")
            assert "Z" not in reader.readable_fields()

    def test_xy_window_skips_z_but_returns_same_points_and_order(self):
        bounds = self._xy_window()
        with laspy.CopcReader.open(SIMPLE_COPC_FILE) as reader:
            reference = reader.query(bounds)

        selection = laspy.DecompressionSelection.base()
        with laspy.CopcReader.open(
            SIMPLE_COPC_FILE, decompression_selection=selection, strict=True
        ) as reader:
            partial = reader.query(bounds)

        assert len(partial) == len(reference)
        np.testing.assert_array_equal(partial.X, reference.X)
        np.testing.assert_array_equal(partial.Y, reference.Y)
        assert not partial.is_field_readable("Z")
        assert not partial.is_field_readable("intensity")
        with pytest.raises(FieldNotReadableError):
            partial.Z
        # Readability survives views and an empty selection.
        view = partial[partial.X > partial.X.min()]
        assert not view.is_field_readable("Z")

    def test_3d_window_filters_on_real_z_even_when_z_skipped(self):
        bounds = self._z_window()
        with laspy.CopcReader.open(SIMPLE_COPC_FILE) as reader:
            reference = reader.query(bounds)
            xy_only = reader.query(
                laspy.Bounds(mins=bounds.mins[:2], maxs=bounds.maxs[:2])
            )
        # The elevation bounds must really exclude some points.
        assert len(reference) < len(xy_only)

        selection = laspy.DecompressionSelection.base()
        with laspy.CopcReader.open(
            SIMPLE_COPC_FILE, decompression_selection=selection, strict=True
        ) as reader:
            partial = reader.query(bounds)

        # Same points and order as a fully decompressed query.
        assert len(partial) == len(reference)
        np.testing.assert_array_equal(partial.X, reference.X)
        np.testing.assert_array_equal(partial.Y, reference.Y)
        # Z was decoded internally for the filter but stays unreadable.
        assert not partial.is_field_readable("Z")
        with pytest.raises(FieldNotReadableError):
            partial.z
        # Unselected, filtering-unrelated columns remain skipped.
        assert not partial.is_field_readable("red")
        assert not partial.is_field_readable("intensity")

    def test_3d_window_with_z_selected_exposes_z(self):
        bounds = self._z_window()
        selection = laspy.DecompressionSelection.base().decompress_z()
        with laspy.CopcReader.open(
            SIMPLE_COPC_FILE, decompression_selection=selection, strict=True
        ) as reader:
            partial = reader.query(bounds)
        with laspy.CopcReader.open(SIMPLE_COPC_FILE) as reader:
            reference = reader.query(bounds)
        assert partial.is_field_readable("Z")
        np.testing.assert_array_equal(partial.Z, reference.Z)

    def test_empty_query_keeps_readability_and_closes_cleanly(self):
        selection = laspy.DecompressionSelection.base()
        nowhere = laspy.Bounds(
            mins=np.array([0.0, 0.0, 0.0]), maxs=np.array([1.0, 1.0, 1.0])
        )
        with laspy.CopcReader.open(
            SIMPLE_COPC_FILE, decompression_selection=selection, strict=True
        ) as reader:
            readable = reader.readable_fields()
            empty = reader.query(nowhere)
        assert len(empty) == 0
        assert empty.readable_fields() == readable
        with pytest.raises(FieldNotReadableError):
            empty.Z
