import logging
from copy import deepcopy
from typing import BinaryIO, Iterable, Optional, Union

from ._pointwriter import IPointWriter
from .compression import LazBackend
from .coordinate import (
    CoordinateGrid,
    CoordinateGuard,
    resolve_coordinate_guard,
    validate_target_grid,
)
from .errors import LaspyException
from .header import LasHeader
from .point import dims
from .point.record import PackedPointRecord
from .vlrs.vlrlist import VLRList

logger = logging.getLogger(__name__)


class LasWriter:
    """
    Allows to write a complete LAS/LAZ file to the destination.
    """

    def __init__(
        self,
        dest: BinaryIO,
        header: LasHeader,
        do_compress: Optional[bool] = None,
        laz_backend: Optional[Union[LazBackend, Iterable[LazBackend]]] = None,
        closefd: bool = True,
        encoding_errors: str = "strict",
        coordinate_guard=None,
    ) -> None:
        """
        Parameters
        ----------
        dest: file_object
            file object where the LAS/LAZ will be written

        header: LasHeader
            The header of the file to be written

        do_compress: bool, optional
            Whether the file data should be written as LAS (uncompressed)
            or LAZ (compressed).
            If None, the file won't be compressed, unless a laz_backend is provided

        laz_backend: LazBackend or list of LazBackend, optional
            The LazBackend to use (or if it is a sequence the LazBackend to try)
            for the compression

        closefd: bool, default True
            should the `dest` be closed when the writer is closed

        encoding_errors: str, default 'strict'
            How encoding errors should be treated.
            Possible values and their explanation can be seen here:
            https://docs.python.org/3/library/codecs.html#error-handlers.

        coordinate_guard: bool, number, sequence of 3 numbers or CoordinateGuard, optional
            Enables the coordinate precision guard (default: disabled).

            When enabled, every point record given to :meth:`write_points` must
            be a :class:`laspy.ScaleAwarePointRecord`, i.e. it must carry the
            scales and offsets of its source.  Its coordinates are re-quantized
            onto the grid of this writer's header before being written.

            ``True`` uses the header scales as the per-axis maximum error (one
            grid step).  A number or a sequence of three numbers sets the
            maximum error per axis explicitly.  A :class:`laspy.CoordinateGrid`
            or :class:`laspy.CoordinateGuard` can also be given.  A whole batch
            is rejected if one of its points cannot be stored within the error
            budget, and nothing is written for that batch.
        """
        self.closefd = closefd
        self.encoding_errors = encoding_errors
        self.header = deepcopy(header)
        # The point writer will take take of creating and writing
        # the correct laszip vlr, however we have to make sure
        # no prior laszip vlr exists
        try:
            self.header.vlrs.pop(header.vlrs.index("LasZipVlr"))
        except ValueError:
            pass
        self.header.partial_reset()

        self.dest = dest
        self.done = False
        self.coordinate_guard: Optional[CoordinateGuard] = None

        if isinstance(coordinate_guard, CoordinateGrid):
            # The caller derived a grid with make_coordinate_grid() for a new
            # file: this writer defines the file header, so the grid is
            # applied to it.  The caller's header object is not modified.
            coordinate_guard.apply_to(self.header)
            if coordinate_guard.max_errors is not None:
                coordinate_guard = CoordinateGuard.from_grid(coordinate_guard)
            else:
                coordinate_guard = True

        if coordinate_guard is not None and coordinate_guard is not False:
            # Normalize and validate the header grid once, before writing the
            # file header.
            self.header.scales, self.header.offsets = validate_target_grid(
                self.header.scales, self.header.offsets
            )
            self.coordinate_guard = resolve_coordinate_guard(
                coordinate_guard, self.header.scales, self.header.offsets
            )

        dims.raise_if_version_not_compatible_with_fmt(
            header.point_format.id, str(self.header.version)
        )

        if laz_backend is not None:
            if do_compress is None:
                do_compress = True
            self.laz_backend = laz_backend
        else:
            if do_compress is None:
                do_compress = False
            self.laz_backend = LazBackend.detect_available()
        self.header.are_points_compressed = do_compress

        if do_compress:
            self.point_writer: IPointWriter = self._create_laz_backend(self.laz_backend)
        else:
            self.point_writer: IPointWriter = UncompressedPointWriter(self.dest)

        self.point_writer.write_initial_header_and_vlrs(
            self.header, self.encoding_errors
        )

    def write_points(self, points: PackedPointRecord) -> None:
        """
        .. note ::

            If you are writing points coming from multiple different input files
            into one output file, you have to make sure the point record
            you write all use the same scales and offset of the writer.

            You can use :meth:`.LasData.change_scaling` or :meth:`.ScaleAwarePointRecord.change_scaling`
            to do that.

        Parameters
        ----------
        points: PackedPointRecord or ScaleAwarePointRecord
                The points to be written

        Raises
        ------
        LaspyException
            If the point format of the points does not match
            the point format of the writer.

        CoordinatePrecisionError
            When the coordinate guard is enabled and the batch cannot be
            stored on the writer's coordinate grid (see
            :mod:`laspy.coordinate`).  The whole batch is rejected.
        """
        if not points:
            return

        if self.done:
            raise LaspyException("Cannot write points anymore")

        points_to_write = points
        if self.coordinate_guard is not None:
            # Validation happens before any state is mutated: a rejected batch
            # leaves both the header statistics and the file untouched.
            points_to_write = self.coordinate_guard.convert_points(
                points,
                self.header.point_format,
                self.header.scales,
                self.header.offsets,
            )
        elif points.point_format != self.header.point_format:
            raise LaspyException("Incompatible point formats")

        if self.header.max_point_count() - self.header.point_count < len(points):
            raise LaspyException(
                "Cannot write {} points as it would exceed the maximum number of points the file"
                "can store. Current point count: {}, max point count: {}".format(
                    len(points), self.header.point_count, self.header.max_point_count()
                )
            )

        self.header.grow(points_to_write)
        self.point_writer.write_points(points_to_write)

    def write_evlrs(self, evlrs: VLRList) -> None:
        """Writes the EVLRs to the file

        Parameters
        ----------
        evlrs: VLRList
               The EVLRs to be written

        Raises
        ------
        LaspyException
            If the file's version is not >= 1.4
        """
        if self.header.version.minor < 4:
            raise LaspyException(
                "EVLRs are not supported on files with version less than 1.4"
            )

        if len(evlrs) > 0:
            self.point_writer.done()
            self.done = True
            self.header.number_of_evlrs = len(evlrs)
            self.header.start_of_first_evlr = self.dest.tell()
            evlrs.write_to(self.dest, as_extended=True)

    def close(self) -> None:
        """Closes the writer.

        flushes the points, updates the header, making it impossible
        to write points afterwards.
        """
        if self.point_writer is not None:
            if not self.done:
                self.point_writer.done()

            if self.header.point_count == 0:
                self.header.maxs = [0.0, 0.0, 0.0]
                self.header.mins = [0.0, 0.0, 0.0]

            # Only the statistics of the batches that were actually written
            # are kept. A component that never had a valid value in those
            # batches must not advertise its optional extrema as known.
            self.header.finalize_extra_bytes_statistics()

            self.point_writer.write_updated_header(self.header, self.encoding_errors)
        if self.closefd:
            self.dest.close()
        self.done = True

    def _create_laz_backend(
        self, laz_backends: Union[LazBackend, Iterable[LazBackend]]
    ) -> "IPointWriter":
        try:
            laz_backends = iter(laz_backends)
        except TypeError:
            laz_backends = (laz_backends,)

        last_error: Optional[Exception] = None
        for backend in laz_backends:
            try:
                if not backend.is_available():
                    raise LaspyException(f"The '{backend}' is not available")

                return backend.create_writer(self.dest, self.header)
            except Exception as e:
                logger.error(e)
                last_error = e

        if last_error is not None:
            raise LaspyException(f"No LazBackend could be initialized: {last_error}")
        else:
            raise LaspyException("No LazBackend selected, cannot compress")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class UncompressedPointWriter(IPointWriter):
    """
    Writing points in the simple uncompressed case.
    """

    def __init__(self, dest: BinaryIO) -> None:
        self.dest = dest

    @property
    def destination(self) -> BinaryIO:
        return self.dest

    def write_points(self, points: PackedPointRecord) -> None:
        self.dest.write(points.memoryview())

    def done(self) -> None:
        pass
