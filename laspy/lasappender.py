import io
from typing import BinaryIO, Iterable, Optional, Union

from ._pointappender import IPointAppender
from .compression import LazBackend
from .coordinate import (
    CoordinateGrid,
    CoordinateGuard,
    ensure_grid_matches,
    resolve_coordinate_guard,
    validate_target_grid,
)
from .errors import LaspyException
from .header import LasHeader
from .point.record import PackedPointRecord
from .vlrs.vlrlist import VLRList


class LasAppender:
    """Allows to append points to and existing LAS/LAZ file.

    Appending to LAZ is only supported by the lazrs backend

    Extra bytes statistics
    ----------------------
    When the file advertises the min/max of an extra dimension, those
    values are trusted: the appended batches are merged into them and the
    points already stored in the file are never read again. When a file
    does not advertise them, they stay unknown after the append; they are
    not replaced by the extrema of the newly appended batch, which would
    wrongly present that batch as describing the whole file. Reconstructing
    them would require reading every existing point (a potentially large
    amount of I/O, done silently), so this is left to the caller if the
    statistics are needed.
    """

    def __init__(
        self,
        dest: BinaryIO,
        laz_backend: Optional[Union[LazBackend, Iterable[LazBackend]]] = None,
        closefd: bool = True,
        encoding_errors: str = "strict",
        coordinate_guard=None,
    ) -> None:
        if not dest.seekable():
            raise TypeError("Expected the 'dest' to be a seekable file object")
        header = LasHeader.read_from(dest)
        if laz_backend is None:
            laz_backend = [
                bck for bck in LazBackend.detect_available() if bck.supports_append
            ]

        self.dest = dest
        self.header = header
        self.coordinate_guard: Optional[CoordinateGuard] = None

        # The statistics already stored in the file are trusted and are the
        # starting point of the merge. The old points are not read again.
        try:
            eb_vlr = header.vlrs.get("ExtraBytesVlr")[0]
        except IndexError:
            pass
        else:
            eb_vlr.restart_tracking()

        if coordinate_guard is not None and coordinate_guard is not False:
            # The grid to use is the one already stored in the file: it is
            # never modified by an append session.
            if isinstance(coordinate_guard, CoordinateGrid):
                ensure_grid_matches(
                    coordinate_guard, header.scales, header.offsets, "file"
                )

            self.coordinate_guard = resolve_coordinate_guard(
                coordinate_guard, header.scales, header.offsets
            )
            validate_target_grid(header.scales, header.offsets, "file")

        if not header.are_points_compressed:
            self.points_appender = UncompressedPointAppender(self.dest)
            self.dest.seek(
                (self.header.point_count * self.header.point_format.size)
                + self.header.offset_to_point_data,
                io.SEEK_SET,
            )
        else:
            self.points_appender = self._create_laz_backend(laz_backend)

        if header.version.minor >= 4 and header.number_of_evlrs > 0:
            assert (
                self.dest.tell() <= self.header.start_of_first_evlr
            ), "The position is past the start of evlrs"
            pos = self.dest.tell()
            self.dest.seek(self.header.start_of_first_evlr, io.SEEK_SET)
            self.evlrs: Optional[VLRList] = VLRList.read_from(
                self.dest, self.header.number_of_evlrs, extended=True
            )
            dest.seek(self.header.start_of_first_evlr, io.SEEK_SET)
            self.dest.seek(pos, io.SEEK_SET)
        else:
            self.evlrs: Optional[VLRList] = None

        self.closefd = closefd
        self.encoding_errors = encoding_errors

    def append_points(self, points: PackedPointRecord) -> None:
        """Append the points to the file, the points
        must have the same point format as the points
        already contained within the file.

        When the coordinate guard is enabled, the points must be a
        :class:`laspy.ScaleAwarePointRecord` carrying their source grid.
        Their coordinates are re-quantized onto the grid already stored in
        the file.  A batch is rejected in full if one of its points cannot be
        stored within the error budget; the file and its header statistics are
        then left untouched.

        :param points: The points to append
        """
        points_to_append = points
        if self.coordinate_guard is not None:
            # Validate and convert before touching anything: a rejected batch
            # must not write a single point nor grow the header's statistics.
            points_to_append = self.coordinate_guard.convert_points(
                points,
                self.header.point_format,
                self.header.scales,
                self.header.offsets,
            )
        elif points.point_format != self.header.point_format:
            raise LaspyException("Point formats do not match")

        if self.header.max_point_count() - self.header.point_count < len(points):
            raise LaspyException(
                "Cannot write {} points as it would exceed the maximum number of points the file"
                "can store. Current point count: {}, max point count: {}".format(
                    len(points), self.header.point_count, self.header.max_point_count()
                )
            )

        self.points_appender.append_points(points_to_append)
        self.header.grow(points_to_append)

    def close(self) -> None:
        self.points_appender.done()
        self._write_evlrs()
        # The statistics stored in the file are trusted and the new batches
        # were merged into them, but a component which never had a valid
        # value must not advertise its optional extrema as known.
        self.header.finalize_extra_bytes_statistics()
        self._write_updated_header()

        if self.closefd:
            self.dest.close()

    def _write_evlrs(self) -> None:
        if (
            self.header.version.minor >= 4
            and self.evlrs is not None
            and len(self.evlrs) > 0
        ):
            self.header.number_of_evlr = len(self.evlrs)
            self.header.start_of_first_evlr = self.dest.tell()
            self.evlrs.write_to(self.dest, as_extended=True)

    def _write_updated_header(self) -> None:
        pos = self.dest.tell()
        self.dest.seek(0, io.SEEK_SET)
        self.header.write_to(
            self.dest, ensure_same_size=True, encoding_errors=self.encoding_errors
        )
        self.dest.seek(pos, io.SEEK_SET)

    def _create_laz_backend(
        self,
        laz_backend: Union[LazBackend, Iterable[LazBackend]] = (
            LazBackend.LazrsParallel,
            LazBackend.Lazrs,
        ),
    ) -> IPointAppender:
        try:
            laz_backend = iter(laz_backend)
        except TypeError:
            laz_backend = (laz_backend,)

        last_error: Optional[Exception] = None
        for backend in laz_backend:
            try:
                return backend.create_appender(self.dest, self.header)
            except Exception as e:
                last_error = e
        if last_error is not None:
            raise LaspyException(f"Could not initialize a laz backend: {last_error}")
        else:
            raise LaspyException(f"No valid laz backend selected")

    def __enter__(self) -> "LasAppender":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class UncompressedPointAppender(IPointAppender):
    """
    Appending points in the simple uncompressed case.
    """

    def __init__(self, dest: BinaryIO) -> None:
        self.dest = dest

    def append_points(self, points: PackedPointRecord) -> None:
        self.dest.write(points.memoryview())

    def done(self) -> None:
        pass
