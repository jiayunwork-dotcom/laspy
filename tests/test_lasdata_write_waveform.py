from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

import laspy
from laspy.lasdata import LasData
from laspy.point.record import LazyWaveformState
from laspy.vlrs.vlr import VLR
from laspy.waveform.descriptor import WaveformPacketDescriptorRegistry
from laspy.waveform.utils import (
    EVLR_HEADER_SIZE,
    WAVEFORM_PACKET_EVLR_RECORD_ID,
    WAVEFORM_PACKET_EVLR_USER_ID,
)

FULLWAVE_LAZ_PATH = Path(__file__).parent / "data" / "fullwave.laz"
FULLWAVE_WDP_PATH = FULLWAVE_LAZ_PATH.with_suffix(".wdp")


@pytest.fixture()
def fullwave_path() -> Path:
    if not FULLWAVE_LAZ_PATH.exists() or not FULLWAVE_WDP_PATH.exists():
        pytest.skip("Missing fullwave test data")
    if len(laspy.LazBackend.detect_available()) == 0:
        pytest.skip("No Laz Backend")
    return FULLWAVE_LAZ_PATH


def set_internal_waveform_flags(las: LasData) -> None:
    las.header.global_encoding.waveform_data_packets_external = False
    las.header.global_encoding.waveform_data_packets_internal = True


def assert_waveform_write_flags_updated(las: LasData) -> None:
    assert las.header.global_encoding.waveform_data_packets_external is True
    assert las.header.global_encoding.waveform_data_packets_internal is False
    if las.header.version.minor >= 3:
        assert las.header.start_of_waveform_data_packet_record == 0


def assert_internal_waveform_write_flags(las: LasData) -> None:
    assert las.header.global_encoding.waveform_data_packets_internal is True
    assert las.header.global_encoding.waveform_data_packets_external is False


def get_waveform_evlr(las: LasData) -> VLR:
    assert las.evlrs is not None
    matches = las.evlrs.get_by_id(
        WAVEFORM_PACKET_EVLR_USER_ID, (WAVEFORM_PACKET_EVLR_RECORD_ID,)
    )
    assert len(matches) == 1
    return matches[0]


def assert_waveform_roundtrip_matches(expected: LasData, actual: LasData) -> None:
    assert np.array_equal(actual.points.array, expected.points.array)
    assert np.array_equal(actual.points["waveform"], expected.points["waveform"])


def test_lasdata_write_without_waveforms_does_not_create_wdp(
    simple_las_path: Path, tmp_path: Path
) -> None:
    las = laspy.read(simple_las_path)
    wf_las = LasData(las.header, las.points)

    assert wf_las["X"].shape[0] == len(wf_las)

    out_path = tmp_path / "no_waveforms.las"
    wf_las.write(str(out_path))
    assert out_path.exists()
    assert not out_path.with_suffix(".wdp").exists()


@pytest.mark.parametrize("extension", ["laz", "las"])
def test_waveform_lasdata_write_eager_internal_roundtrip(
    fullwave_path: Path, tmp_path: Path, extension: str
) -> None:
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()

    out_path = tmp_path / f"fullwave_eager_internal.{extension}"
    set_internal_waveform_flags(las)
    las.write(str(out_path))

    assert out_path.exists()
    assert not out_path.with_suffix(".wdp").exists()
    assert_internal_waveform_write_flags(las)
    assert (
        int(las.wavepacket_offset[las.wavepacket_index != 0].min()) == EVLR_HEADER_SIZE
    )

    with laspy.open(out_path, fullwave="never") as reader:
        on_disk_header = reader.header
        assert on_disk_header.number_of_evlrs == 1
        wave_evlr_position = on_disk_header.start_of_waveform_data_packet_record
        assert wave_evlr_position == on_disk_header.start_of_first_evlr
        assert wave_evlr_position > 0

    with laspy.open(out_path, fullwave="eager") as reader:
        roundtrip = reader.read()

    assert_waveform_roundtrip_matches(las, roundtrip)
    assert_internal_waveform_write_flags(roundtrip)
    assert len(get_waveform_evlr(roundtrip).record_data) > 0


def test_waveform_lasdata_write_lazy_internal_roundtrip(
    fullwave_path: Path, tmp_path: Path
) -> None:
    with laspy.open(fullwave_path, fullwave="lazy") as reader:
        las = reader.read()

    with laspy.open(fullwave_path, fullwave="eager") as reader:
        expected = reader.read()

    out_path = tmp_path / "fullwave_lazy_internal.laz"
    set_internal_waveform_flags(las)
    las.write(str(out_path))

    assert out_path.exists()
    assert not out_path.with_suffix(".wdp").exists()
    assert_internal_waveform_write_flags(las)

    with laspy.open(out_path, fullwave="eager") as reader:
        roundtrip = reader.read()

    assert np.array_equal(roundtrip.points.array, las.points.array)
    assert np.array_equal(roundtrip.points["waveform"], expected.points["waveform"])


def test_internal_and_external_roundtrips_have_identical_waveforms(
    fullwave_path: Path, tmp_path: Path
) -> None:
    internal_path = tmp_path / "internal.laz"
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        internal_las = reader.read()
    set_internal_waveform_flags(internal_las)
    internal_las.write(internal_path)

    external_path = tmp_path / "external.laz"
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        external_las = reader.read()
    external_las.write(external_path)
    assert external_path.with_suffix(".wdp").exists()

    with laspy.open(internal_path, fullwave="eager") as reader:
        internal_rt = reader.read()
    with laspy.open(external_path, fullwave="eager") as reader:
        external_rt = reader.read()

    assert np.array_equal(
        internal_rt.points["waveform"], external_rt.points["waveform"]
    )
    internal_sizes = internal_rt.points.array["wavepacket_size"]
    assert np.array_equal(internal_sizes, external_rt.points.array["wavepacket_size"])
    assert np.array_equal(
        internal_rt.points.array["wavepacket_index"],
        external_rt.points.array["wavepacket_index"],
    )


def test_waveform_lasdata_write_both_flags_still_writes_wdp(
    fullwave_path: Path, tmp_path: Path
) -> None:
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()

    las.header.global_encoding.waveform_data_packets_internal = True
    # external flag is still set from the source file
    assert las.header.global_encoding.waveform_data_packets_external is True

    out_path = tmp_path / "both_flags.laz"
    las.write(str(out_path))

    assert out_path.with_suffix(".wdp").exists()
    assert_waveform_write_flags_updated(las)


def test_waveform_lasdata_write_internal_lazy_subset_deduplicates_waveforms(
    fullwave_path: Path, tmp_path: Path
) -> None:
    with laspy.open(fullwave_path, fullwave="lazy") as reader:
        las = reader.read()
        indices = np.arange(0, len(las.points), 7)
        subset = las[indices]
        registry = WaveformPacketDescriptorRegistry.from_vlrs(subset.header.vlrs)
        wave_dtype = registry.dtype()
        assert wave_dtype is not None
        expected_packet_count = np.unique(subset.points.array["wavepacket_offset"]).size
        expected_data_size = expected_packet_count * wave_dtype.itemsize

        set_internal_waveform_flags(subset)
        out_path = tmp_path / "subset_fullwave_internal.laz"
        subset.write(str(out_path))

    assert not out_path.with_suffix(".wdp").exists()
    assert len(get_waveform_evlr(subset).record_data) == expected_data_size

    with laspy.open(out_path, fullwave="eager") as roundtrip_reader:
        roundtrip = roundtrip_reader.read()

    with laspy.open(fullwave_path, fullwave="eager") as expected_reader:
        expected = expected_reader.read()

    assert np.array_equal(roundtrip.points.array, subset.points.array)
    assert np.array_equal(
        roundtrip.points["waveform"],
        expected.points["waveform"][indices],
    )
    assert_internal_waveform_write_flags(subset)


def test_waveform_lasdata_write_internal_preserves_points_without_waveforms(
    fullwave_path: Path, tmp_path: Path
) -> None:
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()

    set_internal_waveform_flags(las)
    las.points.array["wavepacket_index"][0] = 0
    las.points.array["wavepacket_size"][0] = 0
    las.points.array["wavepacket_offset"][0] = 0

    out_path = tmp_path / "missing_first_waveform_internal.laz"
    las.write(str(out_path))

    assert not out_path.with_suffix(".wdp").exists()
    with laspy.open(out_path, fullwave="eager") as reader:
        points = reader.read_points(4)

    assert points.array["wavepacket_index"][0] == 0
    assert points.array["wavepacket_size"][0] == 0
    assert points.array["wavepacket_offset"][0] == 0
    assert np.all(points["waveform"][0] == 0)
    assert_internal_waveform_write_flags(las)


def test_waveform_lasdata_internal_write_preserves_other_evlrs_and_position(
    fullwave_path: Path, tmp_path: Path
) -> None:
    internal_path = tmp_path / "internal_with_evlrs.laz"
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()
    set_internal_waveform_flags(las)
    las.write(internal_path)

    # Reorganize EVLRs so the waveform EVLR is not the first one
    with laspy.open(internal_path, fullwave="never") as reader:
        reorganized = reader.read()
    assert reorganized.evlrs is not None
    wave_evlrs = [vlr for vlr in reorganized.evlrs if vlr.record_id == 65535]
    assert len(wave_evlrs) == 1
    dummy = VLR("MyUserId", 4242, "dummy", b"dummy-evlr-data")
    reorganized.evlrs[:] = (
        [dummy]
        + [vlr for vlr in reorganized.evlrs if vlr.record_id != 65535]
        + wave_evlrs
    )

    out_path = tmp_path / "reordered_internal.laz"
    reorganized.write(out_path, waveforms=False)

    with laspy.open(out_path, fullwave="never") as reader:
        header = reader.header
        ids = [(vlr.user_id, vlr.record_id) for vlr in header.evlrs]
        assert ("MyUserId", 4242) in ids
        assert (WAVEFORM_PACKET_EVLR_USER_ID, WAVEFORM_PACKET_EVLR_RECORD_ID) in ids
        expected_wave_position = (
            header.start_of_first_evlr + 60 + len(dummy.record_data)
        )
        assert header.start_of_waveform_data_packet_record == expected_wave_position

    with laspy.open(out_path, fullwave="lazy") as reader:
        roundtrip = reader.read()

    assert isinstance(roundtrip.points._waveform_state, LazyWaveformState)
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        expected = reader.read()
    assert np.array_equal(roundtrip.points["waveform"], expected.points["waveform"])


def test_waveform_lasdata_internal_write_boolean_mask_subset(
    fullwave_path: Path, tmp_path: Path
) -> None:
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()

    mask = np.asarray(las.return_number == 1)
    subset = las[mask]
    set_internal_waveform_flags(subset)
    out_path = tmp_path / "mask_subset_internal.laz"
    subset.write(out_path)

    with laspy.open(out_path, fullwave="eager") as reader:
        roundtrip = reader.read()

    assert np.array_equal(roundtrip.points["waveform"], las.points["waveform"][mask])
    assert np.array_equal(roundtrip.points.array, subset.points.array)


def test_waveform_lasdata_internal_then_external_roundtrip(
    fullwave_path: Path, tmp_path: Path
) -> None:
    internal_path = tmp_path / "internal.laz"
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()
    set_internal_waveform_flags(las)
    las.write(internal_path)

    with laspy.open(internal_path, fullwave="eager") as reader:
        internal_las = reader.read()
    internal_las.header.global_encoding.waveform_data_packets_external = True
    internal_las.header.global_encoding.waveform_data_packets_internal = False
    external_path = tmp_path / "back_to_external.laz"
    internal_las.write(external_path)

    assert external_path.with_suffix(".wdp").exists()
    with laspy.open(external_path, fullwave="eager") as reader:
        roundtrip = reader.read()
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        expected = reader.read()
    assert np.array_equal(roundtrip.points["waveform"], expected.points["waveform"])
    assert not any(
        vlr.record_id == WAVEFORM_PACKET_EVLR_RECORD_ID for vlr in roundtrip.evlrs
    )


def test_waveform_lasdata_write_without_valid_waveforms_drops_internal_evlr(
    fullwave_path: Path, tmp_path: Path
) -> None:
    internal_path = tmp_path / "internal.laz"
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()
    set_internal_waveform_flags(las)
    las.write(internal_path)

    with laspy.open(internal_path, fullwave="eager") as reader:
        cleared = reader.read()
    cleared.points.array["wavepacket_index"][:] = 0
    cleared.points.array["wavepacket_size"][:] = 123
    cleared.points.array["wavepacket_offset"][:] = 456

    out_path = tmp_path / "eager_all_no_waveforms.laz"
    cleared.write(out_path)

    assert out_path.exists()
    assert not out_path.with_suffix(".wdp").exists()
    assert np.all(cleared.points.array["wavepacket_size"] == 0)
    assert np.all(cleared.points.array["wavepacket_offset"] == 0)
    assert cleared.header.global_encoding.waveform_data_packets_external is False
    assert cleared.header.global_encoding.waveform_data_packets_internal is False
    assert not any(
        vlr.user_id == WAVEFORM_PACKET_EVLR_USER_ID
        and vlr.record_id == WAVEFORM_PACKET_EVLR_RECORD_ID
        for vlr in cleared.evlrs
    )

    roundtrip = laspy.read(out_path)
    assert np.all(roundtrip.points.array["wavepacket_index"] == 0)
    assert np.all(roundtrip.points.array["wavepacket_size"] == 0)
    assert np.all(roundtrip.points.array["wavepacket_offset"] == 0)


@pytest.mark.parametrize("fullwave", ["eager", "lazy"])
def test_read_internal_missing_waveform_evlr_raises(
    fullwave_path: Path, tmp_path: Path, fullwave: str
) -> None:
    internal_path = tmp_path / "internal.laz"
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()
    set_internal_waveform_flags(las)
    las.write(internal_path)

    with laspy.open(internal_path, fullwave="never") as reader:
        broken = reader.read()
    broken.evlrs.clear()
    out_path = tmp_path / "missing_evlr.laz"
    broken.write(out_path, waveforms=False)

    with laspy.open(out_path, fullwave=fullwave) as reader:
        with pytest.raises(ValueError, match="no EVLR"):
            reader.read_points(2)


@pytest.mark.parametrize("fullwave", ["eager", "lazy"])
def test_read_internal_duplicate_waveform_evlrs_raises(
    fullwave_path: Path, tmp_path: Path, fullwave: str
) -> None:
    internal_path = tmp_path / "internal.laz"
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()
    set_internal_waveform_flags(las)
    las.write(internal_path)

    with laspy.open(internal_path, fullwave="never") as reader:
        broken = reader.read()
    wave_evlr = next(vlr for vlr in broken.evlrs if vlr.record_id == 65535)
    broken.evlrs.append(
        VLR(WAVEFORM_PACKET_EVLR_USER_ID, WAVEFORM_PACKET_EVLR_RECORD_ID, "dup", b"x")
    )
    out_path = tmp_path / "duplicate_evlr.laz"
    broken.write(out_path, waveforms=False)

    with laspy.open(out_path, fullwave=fullwave) as reader:
        with pytest.raises(ValueError, match="more than one EVLR"):
            reader.read_points(2)


@pytest.mark.parametrize("fullwave", ["eager", "lazy"])
def test_read_internal_truncated_waveform_evlr_raises(
    fullwave_path: Path, tmp_path: Path, fullwave: str
) -> None:
    internal_path = tmp_path / "internal.las"
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()
    set_internal_waveform_flags(las)
    las.write(internal_path)

    header = laspy.read(internal_path, fullwave="never").header
    raw = bytearray(internal_path.read_bytes())
    position = header.start_of_waveform_data_packet_record
    declared_length = int.from_bytes(raw[position + 20 : position + 28], "little")
    raw[position + 20 : position + 28] = (declared_length - 100_000).to_bytes(
        8, "little"
    )
    out_path = tmp_path / "truncated_evlr.las"
    out_path.write_bytes(raw)

    with laspy.open(out_path, fullwave=fullwave) as reader:
        with pytest.raises(ValueError, match="outside of"):
            reader.read_points(-1)


def test_lazy_internal_waveforms_accessible_after_reader_closed(
    fullwave_path: Path, tmp_path: Path
) -> None:
    internal_path = tmp_path / "internal.laz"
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()
    set_internal_waveform_flags(las)
    las.write(internal_path)

    with laspy.open(internal_path, fullwave="lazy") as reader:
        lazy_las = reader.read()
    assert isinstance(lazy_las.points._waveform_state, LazyWaveformState)
    assert np.array_equal(lazy_las.points["waveform"], las.points["waveform"])


def test_waveform_lasdata_write_rejects_binaryio(fullwave_path: Path) -> None:
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()

    with pytest.raises(NotImplementedError):
        las.write(io.BytesIO())


def test_waveform_lasdata_write_rejects_unaligned_lazy_offsets(
    fullwave_path: Path, tmp_path: Path
) -> None:
    with laspy.open(fullwave_path, fullwave="lazy") as reader:
        las = reader.read()

    las.points.array["wavepacket_offset"][0] += 1

    with pytest.raises(NotImplementedError, match="byte offset"):
        las.write(tmp_path / "unaligned_offset.laz")


### Private API tests ###


def test_lazy_write_dedup_all_no_waveforms(fullwave_path: Path, tmp_path: Path) -> None:
    with laspy.open(fullwave_path, fullwave="lazy") as reader:
        las = reader.read()
        lazy_state = las.points._waveform_state
        assert isinstance(lazy_state, LazyWaveformState)
        wave_size = lazy_state.reader.wave_size_bytes
        las.points.array["wavepacket_index"][:] = 0
        las.points.array["wavepacket_size"][:] = 0
        las.points.array["wavepacket_offset"][:] = 123
        has_waveform_mask = np.asarray(
            las.points.array["wavepacket_index"] != 0, dtype=bool
        )

        out_path = tmp_path / "dedup_all_no_waveforms.laz"
        las._write_wdp_lazy(
            destination=out_path,
            lazy_state=lazy_state,
            has_waveform_mask=has_waveform_mask,
            waveform_size=wave_size,
            chunksize=1024,
        )

    out_wdp = out_path.with_suffix(".wdp")
    assert out_wdp.exists()
    assert out_wdp.stat().st_size == 0
    assert np.all(las.points.array["wavepacket_offset"] == 0)
    assert np.all(las.points.array["wavepacket_size"] == 0)
    assert las.header.global_encoding.waveform_data_packets_external is False


def test_write_waveforms_without_wavepacket_index_dimension_raises(
    simple_las_path: Path, tmp_path: Path
) -> None:
    las = laspy.read(simple_las_path)
    out_path = tmp_path / "no_wavepacket_index.las"
    with pytest.raises(ValueError, match="wavepacket_index"):
        las._write_waveforms(out_path, waveform_chunksize=1024)
    assert not out_path.with_suffix(".wdp").exists()


def test_write_wdp_no_waveforms_noop(tmp_path: Path) -> None:
    path = tmp_path / "noop.wdp"
    LasData._write_wdp(path, None)
    assert not path.exists()


def test_write_wdp_lazy_errors(fullwave_path: Path, tmp_path: Path) -> None:
    with laspy.open(fullwave_path, fullwave="lazy") as reader:
        las = reader.read()
        lazy_state = las.points._waveform_state
        assert isinstance(lazy_state, LazyWaveformState)
        wave_size = lazy_state.reader.wave_size_bytes

        with pytest.raises(ValueError):
            las._write_wdp_lazy(
                destination=tmp_path / "bad_chunk.laz",
                lazy_state=lazy_state,
                has_waveform_mask=np.asarray(
                    las.points.array["wavepacket_index"] != 0, dtype=bool
                ),
                waveform_size=wave_size,
                chunksize=0,
            )


def test_write_wdp_lazy_inconsistent_sizes(fullwave_path: Path, tmp_path: Path) -> None:
    with laspy.open(fullwave_path, fullwave="lazy") as reader:
        las = reader.read()
        lazy_state = las.points._waveform_state
        assert isinstance(lazy_state, LazyWaveformState)
        wave_size = lazy_state.reader.wave_size_bytes
        las.points.array["wavepacket_size"][0] = wave_size + 1
        with pytest.raises(ValueError):
            las._write_wdp_lazy(
                destination=tmp_path / "bad_size.laz",
                lazy_state=lazy_state,
                has_waveform_mask=np.asarray(
                    las.points.array["wavepacket_index"] != 0, dtype=bool
                ),
                waveform_size=wave_size,
                chunksize=1024,
            )


def test_write_wdp_lazy_empty_points(fullwave_path: Path, tmp_path: Path) -> None:
    with laspy.open(fullwave_path, fullwave="lazy") as reader:
        las = reader.read()
        empty = las[:0]
        lazy_state = empty.points._waveform_state
        assert isinstance(lazy_state, LazyWaveformState)
        wave_size = lazy_state.reader.wave_size_bytes

        out_path = tmp_path / "empty.laz"
        empty._write_wdp_lazy(
            destination=out_path,
            lazy_state=lazy_state,
            has_waveform_mask=np.asarray(
                empty.points.array["wavepacket_index"] != 0, dtype=bool
            ),
            waveform_size=wave_size,
            chunksize=1024,
        )

    out_wdp = out_path.with_suffix(".wdp")
    assert out_wdp.exists()
    assert out_wdp.stat().st_size == 0
