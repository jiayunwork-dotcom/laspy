from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: User ID of the EVLR that may carry waveform packets inside the LAS/LAZ file
WAVEFORM_PACKET_EVLR_USER_ID = "LASF_Spec"
#: Record ID of the EVLR that may carry waveform packets inside the LAS/LAZ file
WAVEFORM_PACKET_EVLR_RECORD_ID = 65535
#: Size in bytes of an EVLR header; per LAS 1.4 the first internal packet
#: starts at offset 60 relative to the start of the waveform EVLR
EVLR_HEADER_SIZE = 60


def iter_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    last = start
    for idx in indices[1:]:
        idx_int = int(idx)
        if idx_int == last + 1:
            last = idx_int
        else:
            runs.append((start, last))
            start = idx_int
            last = idx_int
    runs.append((start, last))
    return runs


def deduplicate_waveform_indices(
    offsets: NDArray[np.uint64],
    has_waveform_mask: NDArray[np.bool_],
    waveform_size: int,
    offset_origin: int | None = None,
) -> tuple[NDArray[np.uint64], NDArray[np.int64]]:
    if not has_waveform_mask.any():
        return np.array([], dtype=np.uint64), np.array([], dtype=np.int64)

    valid_offsets = offsets[has_waveform_mask]
    if offset_origin is None:
        # Legacy behaviour used for external .wdp files: the point offset
        # corresponding to the first byte of the source is inferred from the
        # common remainder of the point offsets.
        offset_remainder = np.remainder(valid_offsets, waveform_size)
        base_remainder = offset_remainder[0]
        if np.any(offset_remainder != base_remainder):
            raise NotImplementedError(
                "Unsupported waveform byte offset layout; "
                "laspy currently requires a single fixed byte-offset remainder "
                "for all waveform packets"
            )
    else:
        base_remainder = offset_origin

    relative_offsets = valid_offsets.astype(np.int64) - int(base_remainder)
    if np.any(relative_offsets < 0) or np.any(
        np.remainder(relative_offsets, waveform_size) != 0
    ):
        raise NotImplementedError(
            "Unsupported waveform byte offset layout; "
            "laspy currently requires a single fixed byte-offset remainder "
            "for all waveform packets"
        )

    valid_indices = (relative_offsets // waveform_size).astype(np.uint64)
    return np.unique(valid_indices, return_inverse=True)
