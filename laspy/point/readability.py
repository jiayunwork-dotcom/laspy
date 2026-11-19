"""Tracking of which point fields were actually read from a source.

When a LAZ file is read with a selective
:class:`~laspy.DecompressionSelection`, the skipped fields are still present
in the in-memory point record (the point format does not change), but their
content is meaningless.  This module provides the bookkeeping used by
*strict read mode* to distinguish a field that was genuinely read from one
that was simply skipped:

- :func:`readable_fields_for_selection` computes, for a given point format
  and decompression selection, the set of field names that the backend is
  going to decode.
- :class:`FieldReadability` is attached to a point record and answers
  readability queries without triggering any (re)reading of the source.

The readability metadata travels with the data: slicing, boolean masking,
copying and the equivalent :class:`~laspy.LasData` operations all preserve
it, so a view of a record keeps the same guarantees as the record itself.

.. versionadded:: 2.6
"""

from typing import FrozenSet, Iterable, Set

from .._compression.selection import DecompressionReference
from . import dims
from .dims import OLD_LASPY_NAMES

#: Names of the fields decoded by each :class:`DecompressionSelection` flag.
#: Names that do not exist in a given point format are simply filtered out.
#:
#: Note for point formats 6+: the base item only carries X, Y, the return
#: number, the number of returns and the scanner channel. The scan direction
#: and edge-of-flight-line flags live in the separate FLAGS item there, even
#: though they share the ``bit_fields`` packed byte with the return fields in
#: the on-disk point structure (verified against both the lazrs and laszip
#: backends).
SELECTION_FIELDS = {
    DecompressionReference.XY_RETURNS_CHANNEL: (
        "X",
        "Y",
        "bit_fields",
        "return_number",
        "number_of_returns",
        "scanner_channel",
    ),
    DecompressionReference.Z: ("Z",),
    DecompressionReference.CLASSIFICATION: (
        "classification",
        "raw_classification",
        "synthetic",
        "key_point",
        "withheld",
    ),
    DecompressionReference.FLAGS: (
        "classification_flags",
        "synthetic",
        "key_point",
        "withheld",
        "overlap",
        "scanner_channel",
        "scan_direction_flag",
        "edge_of_flight_line",
    ),
    DecompressionReference.INTENSITY: ("intensity",),
    DecompressionReference.SCAN_ANGLE: ("scan_angle", "scan_angle_rank"),
    DecompressionReference.USER_DATA: ("user_data",),
    DecompressionReference.POINT_SOURCE_ID: ("point_source_id",),
    DecompressionReference.GPS_TIME: ("gps_time",),
    DecompressionReference.RGB: ("red", "green", "blue"),
    DecompressionReference.NIR: ("nir",),
    DecompressionReference.WAVEPACKET: (
        "wavepacket_index",
        "wavepacket_offset",
        "wavepacket_size",
        "return_point_wave_location",
        "x_t",
        "y_t",
        "z_t",
    ),
    DecompressionReference.ALL_EXTRA_BYTES: (),  # resolved dynamically (extra dims)
}


def readable_fields_for_selection(
    point_format, selection: DecompressionReference
) -> FrozenSet[str]:
    """Returns the names of the fields that *selection* decodes.

    The result is intersected with the fields actually declared by
    *point_format* (standard sub-fields, composed fields and extra
    dimensions), so asking for the RGB of a point format without colors
    simply does not add anything.

    The base flag always decodes ``X``, ``Y`` and the return/channel packed
    fields. When the extra-bytes flag is set, every extra dimension of the
    point format is considered readable.
    """
    format_fields: Set[str] = set(point_format.dimension_names)
    # The packed (composed) fields are part of the numpy dtype but not of
    # point_format.dimension_names (which lists the un-packed sub-fields).
    format_fields.update(dims.ALL_POINT_FORMATS_DTYPE[point_format.id].names)

    readable: Set[str] = set()
    for flag, names in SELECTION_FIELDS.items():
        if selection.is_set(flag):
            readable.update(names)
    if selection.is_set(DecompressionReference.ALL_EXTRA_BYTES):
        readable.update(point_format.extra_dimension_names)

    return frozenset(readable & format_fields)


class FieldReadability:
    """Immutable description of the fields readable on a point record.

    A field is *readable* when it exists in the point format and was actually
    decoded from the source. Querying the readability never touches the
    underlying file: the information is fully determined at opening time.

    Use :meth:`is_readable` / :meth:`readable_fields` to inspect the state,
    the point record itself raises
    :class:`~laspy.errors.FieldNotReadableError` on forbidden accesses when
    strict mode is enabled.
    """

    __slots__ = ("_point_format_id", "_readable", "_all_fields")

    def __init__(self, point_format, readable: Iterable[str]):
        all_fields: Set[str] = set(point_format.dimension_names)
        all_fields.update(dims.ALL_POINT_FORMATS_DTYPE[point_format.id].names)
        self._point_format_id = point_format.id
        self._all_fields: FrozenSet[str] = frozenset(all_fields)
        self._readable: FrozenSet[str] = frozenset(readable) & self._all_fields

    @classmethod
    def all_readable(cls, point_format) -> "FieldReadability":
        """Builds a readability where every field of the format is readable.

        This is used for uncompressed LAS, old point formats (which do not
        support selective decompression) and fully decompressed reads.
        """
        all_fields: Set[str] = set(point_format.dimension_names)
        all_fields.update(dims.ALL_POINT_FORMATS_DTYPE[point_format.id].names)
        return cls(point_format, all_fields)

    @property
    def point_format_id(self) -> int:
        return self._point_format_id

    def _resolve_name(self, field_name: str) -> str:
        try:
            return OLD_LASPY_NAMES[field_name]
        except KeyError:
            pass
        # The scaled x/y/z views are direct views of the packed X/Y/Z fields.
        if field_name in ("x", "y", "z"):
            return field_name.upper()
        return field_name

    def is_readable(self, field_name: str) -> bool:
        """Returns True if *field_name* was read and can be accessed.

        The query does not perform any I/O and never raises: names that do
        not exist in the point format report ``False``.
        """
        return self._resolve_name(field_name) in self._readable

    def is_known_field(self, field_name: str) -> bool:
        """Returns True if *field_name* exists in the point format."""
        return self._resolve_name(field_name) in self._all_fields

    def readable_fields(self) -> FrozenSet[str]:
        """Returns the frozen set of field names that can be accessed.

        Both packed (composed) names, e.g. ``bit_fields``, and their
        sub-field names, e.g. ``return_number``, are included when readable.
        """
        return self._readable

    def unreadable_fields(self) -> FrozenSet[str]:
        """Returns the fields of the format that were not read."""
        return self._all_fields - self._readable

    def ensure_readable(self, field_name: str) -> None:
        """Raises :class:`FieldNotReadableError` if the field is not readable.

        Fields that do not exist in the point format are left untouched here
        (the regular unknown-field error path handles them).
        """
        from ..errors import FieldNotReadableError

        resolved = self._resolve_name(field_name)
        if resolved in self._all_fields and resolved not in self._readable:
            raise FieldNotReadableError(field_name)

    def __repr__(self) -> str:
        return (
            f"<FieldReadability(point_format={self._point_format_id}, "
            f"{len(self._readable)}/{len(self._all_fields)} fields readable)>"
        )
