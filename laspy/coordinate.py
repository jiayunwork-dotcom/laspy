"""Coordinate grids and the coordinate precision guard.

LAS stores ``x``, ``y`` and ``z`` as signed 32 bit integers together with a
scale and an offset per axis::

    stored_x = round((x - offset_x) / scale_x)

Reading a file decodes the integers with ``X * scale_x + offset_x``.

When points coming from several sources (each with its own scale/offset) are
written into the same file, every point must be re-quantized onto the grid
defined by the *destination* header.  Forgetting that step silently stores
points at the wrong place: the integers of the source are written as-is and
are later decoded with the destination's scale/offset.

This module provides:

* :func:`make_coordinate_grid`, which derives a deterministic
  ``(scales, offsets)`` pair from per-axis coordinate bounds and per-axis
  maximum absolute errors,
* :class:`CoordinateGrid`, the result of that derivation,
* :class:`CoordinateGuard`, which re-quantizes ``ScaleAwarePointRecord`` onto
  a destination grid and rejects a whole batch if any point cannot be
  represented within the allowed error.

The guard is opt-in: ``laspy.open(..., coordinate_guard=...)``.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from .errors import (
    CoordinateIntegerOverflow,
    CoordinateOutOfBounds,
    CoordinatePrecisionExceeded,
    CoordinateRangeTooPrecise,
    CoordinateGridError,
    IncompatiblePointFormat,
    InvalidCoordinateBounds,
    InvalidCoordinateTolerance,
    InvalidSourceGrid,
    InvalidTargetGrid,
    MissingSourceGrid,
    NonFiniteCoordinate,
)
from .point.record import PackedPointRecord, ScaleAwarePointRecord

INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max

# Inclusive number of values representable by a signed 32 bit integer.
_NUM_INT32_VALUES = 1 << 32
# Maximum number of grid steps (intervals) an axis may need.  There are
# 2**32 representable integers, hence 2**32 - 1 intervals between the extreme
# values; we use at most 2**32 - 5 of them so that, even after the offset is
# centred on the interval, at least one value stays unused at each end of the
# range and rounding an endpoint can never land on an out-of-range value.
_MAX_INTERVALS = (1 << 32) - 5

# Unit roundoff of float64 (2**-53).  We use it to bound the error introduced
# by the floating point operations of the encode/decode round trip.
_UNIT_ROUNDOFF = 2.0**-53

_AXIS_NAMES = ("x", "y", "z")

ToleranceLike = Union[float, int, Sequence[float], np.ndarray]


def _as_float_triplet(value, what: str, error_cls) -> np.ndarray:
    """Converts *value* to a finite float64 array of shape (3,)."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise error_cls(f"{what} must be three finite numbers") from None

    if array.ndim == 0 or array.shape != (3,):
        raise error_cls(f"{what} must contain exactly three values (one per axis)")

    if not np.all(np.isfinite(array)):
        raise error_cls(f"{what} must only contain finite values")

    return array


def _as_tolerances(value: ToleranceLike, what: str = "max_errors") -> np.ndarray:
    """Converts *value* to finite, strictly positive float64 array of shape (3,).

    A single scalar is broadcast to all three axes.
    """
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise InvalidCoordinateTolerance(
            f"{what} must be a finite positive number or three finite positive numbers"
        ) from None

    if array.ndim == 0:
        array = np.full(3, float(array), dtype=np.float64)
    elif array.shape != (3,):
        raise InvalidCoordinateTolerance(
            f"{what} must contain exactly three values (one per axis)"
        )

    if not np.all(np.isfinite(array)):
        raise InvalidCoordinateTolerance(f"{what} must only contain finite values")
    if np.any(array <= 0.0):
        raise InvalidCoordinateTolerance(f"{what} must be strictly positive")

    return array.copy()


def _validate_scales_and_offsets(
    scales, offsets, error_cls, what: str
) -> "tuple[np.ndarray, np.ndarray]":
    scales = np.asarray(scales, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)

    if scales.shape != (3,):
        raise error_cls(f"{what} must contain exactly three scales")
    if offsets.shape != (3,):
        raise error_cls(f"{what} must contain exactly three offsets")
    if not np.all(np.isfinite(scales)) or not np.all(np.isfinite(offsets)):
        raise error_cls(f"{what} must only contain finite values")
    if np.any(scales <= 0.0):
        raise error_cls(f"{what} scales must be strictly positive")

    return scales, offsets


def _encode(coordinates: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Returns the (not yet rounded) integer coordinates for *coordinates*."""
    with np.errstate(over="ignore", invalid="ignore"):
        return (coordinates - offset) / scale


def _decode(integers: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Decodes integer coordinates exactly like ``ScaledArrayView`` does."""
    return integers.astype(np.float64) * scale + offset


@dataclass(frozen=True, eq=False)
class CoordinateGrid:
    """The coordinate grid stored in a LAS header.

    Attributes
    ----------
    scales: numpy.ndarray
        The three (strictly positive, finite) scales, one per axis.
    offsets: numpy.ndarray
        The three (finite) offsets, one per axis.
    max_errors: numpy.ndarray or None
        The per-axis maximum absolute error that was used to derive the grid,
        when it was created by :func:`make_coordinate_grid`.
    mins, maxs: numpy.ndarray or None
        The closed interval each axis was derived for, when the grid was
        created by :func:`make_coordinate_grid`.
    """

    scales: np.ndarray
    offsets: np.ndarray
    max_errors: Optional[np.ndarray] = None
    mins: Optional[np.ndarray] = None
    maxs: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        scales, offsets = _validate_scales_and_offsets(
            self.scales, self.offsets, CoordinateGridError, "coordinate grid"
        )
        object.__setattr__(self, "scales", scales.copy())
        object.__setattr__(self, "offsets", offsets.copy())

        if self.max_errors is not None:
            max_errors = _as_tolerances(self.max_errors)
            object.__setattr__(self, "max_errors", max_errors)

        if (self.mins is None) != (self.maxs is None):
            raise CoordinateGridError(
                "a coordinate grid must define both mins and maxs, or neither"
            )
        if self.mins is not None:
            mins = _as_float_triplet(self.mins, "mins", InvalidCoordinateBounds)
            maxs = _as_float_triplet(self.maxs, "maxs", InvalidCoordinateBounds)
            if np.any(mins > maxs):
                raise InvalidCoordinateBounds(
                    "the coordinate grid's intervals are reversed"
                )
            object.__setattr__(self, "mins", mins)
            object.__setattr__(self, "maxs", maxs)

    def apply_to(self, header) -> None:
        """Copies this grid's scales and offsets into a header."""
        header.scales = self.scales.copy()
        header.offsets = self.offsets.copy()

    def __repr__(self) -> str:
        return "<CoordinateGrid(scales={}, offsets={})>".format(
            self.scales.tolist(), self.offsets.tolist()
        )


def make_coordinate_grid(
    mins: Sequence[float],
    maxs: Sequence[float],
    max_errors: ToleranceLike,
) -> CoordinateGrid:
    """Derives a header coordinate grid from coordinate bounds and tolerances.

    Parameters
    ----------
    mins, maxs: sequences of three numbers
        The closed interval ``[mins[i], maxs[i]]`` is the range of coordinates
        that axis ``i`` (x, y, z) must be able to represent.  ``mins[i]`` may
        be greater than zero, cross zero or be equal to ``maxs[i]``.
    max_errors: number or sequence of three numbers
        The maximum absolute error allowed, per axis, between the coordinate
        that is encoded and the coordinate decoded back from the file.  A
        scalar is applied to all three axes.  Values must be finite and
        strictly positive.

    Returns
    -------
    CoordinateGrid
        A deterministic grid.  Calling this function twice with the same
        inputs returns the same scales and offsets.

    Raises
    ------
    InvalidCoordinateBounds
        If an interval is reversed or contains a non-finite value.
    InvalidCoordinateTolerance
        If a maximum error is not a finite, strictly positive number.
    CoordinateRangeTooPrecise
        If the interval cannot be represented within the error budget: either
        because it would need more distinct integers than a signed 32 bit
        integer can hold, or because the float64 round trip at that magnitude
        cannot stay within the requested error.

    Notes
    -----
    The scale of an axis is set to its maximum absolute error.  Rounding to
    the nearest integer then costs at most half of that budget, leaving the
    other half for the floating point operations of the encode/decode round
    trip, as well as for the ``scale * integer + offset`` decoding performed
    by readers.  The offset is the midpoint of the interval, which makes the
    integer coordinates as small as possible (and thereby limits the loss of
    precision when coordinates are far away from zero).  Nearly the full
    signed 32 bit range is used: ``2**32 - 5`` grid steps, so that the
    centred interval keeps a few unused integers at each end of the range and
    rounding never lands on an out-of-range value.  If the width divided by
    the scale needs more integers than that, no grid within the budget exists
    and an error is raised instead of silently storing coordinates on a grid
    that cannot meet the tolerance.

    A single-valued axis (``mins[i] == maxs[i]``) is stored exactly: its only
    coordinate maps to the integer 0 and is decoded back to the offset value.
    """
    mins = _as_float_triplet(mins, "mins", InvalidCoordinateBounds)
    maxs = _as_float_triplet(maxs, "maxs", InvalidCoordinateBounds)
    max_errors = _as_tolerances(max_errors)

    if np.any(mins > maxs):
        bad_axis = int(np.flatnonzero(mins > maxs)[0])
        raise InvalidCoordinateBounds(
            f"mins[{bad_axis}] ({mins[bad_axis]!r}) is greater than "
            f"maxs[{bad_axis}] ({maxs[bad_axis]!r}): the interval is reversed"
        )

    scales = max_errors.copy()
    offsets = np.empty(3, dtype=np.float64)

    for axis, (lo, hi, tolerance) in enumerate(zip(mins, maxs, max_errors)):
        name = _AXIS_NAMES[axis]
        width = hi - lo

        if not math.isfinite(width):
            raise CoordinateRangeTooPrecise(
                f"the {name} interval [{lo!r}, {hi!r}] is too wide to be "
                f"represented with a finite float64 width"
            )

        # Midpoint, computed as lo/2 + hi/2 to avoid overflowing the sum of
        # two large values.  For a degenerate interval this is exactly the
        # single value of the interval.
        offset = lo if lo == hi else lo * 0.5 + hi * 0.5
        if not math.isfinite(offset):
            raise CoordinateRangeTooPrecise(
                f"the midpoint of the {name} interval [{lo!r}, {hi!r}] is not finite"
            )

        if width > 0.0:
            needed_intervals = width / tolerance
            if not math.isfinite(needed_intervals) or math.ceil(
                needed_intervals
            ) > _MAX_INTERVALS:
                raise CoordinateRangeTooPrecise(
                    f"the {name} interval [{lo!r}, {hi!r}] needs "
                    f"{needed_intervals!r} grid steps at the allowed error "
                    f"{tolerance!r}, which does not fit in the signed 32 bit "
                    f"integer range"
                )

            magnitude = max(abs(lo), abs(hi))
            # Bound on the error of the floating point operations of the
            # encode/decode round trip: encoding subtracts the offset (exact
            # for nearby values, as a subtraction of two floats of the same
            # binade is), divides by the scale (relative error ~eps, i.e.
            # ~eps*width once re-multiplied) and the decoding multiplies and
            # adds the offset (relative error ~eps of the result, i.e.
            # ~eps*magnitude).  We use a factor 2 for safety.
            floating_point_error = 2.0 * _UNIT_ROUNDOFF * (width + magnitude)
            # Half of the scale is spent on rounding to the nearest integer,
            # the rest of the budget is left for floating point operations.
            if 0.5 * tolerance + floating_point_error > tolerance:
                raise CoordinateRangeTooPrecise(
                    f"the {name} interval [{lo!r}, {hi!r}] cannot meet the "
                    f"error budget {tolerance!r}: float64 arithmetic at this "
                    f"coordinate magnitude already introduces about "
                    f"{floating_point_error!r} of error"
                )

        # Final, exact check of the grid that will be stored in the header.
        # It verifies the actual operations a reader will do, including the
        # rounding of the interval endpoints.
        for coordinate in (lo, hi):
            index = np.rint(_encode(np.float64(coordinate), tolerance, offset))
            if not np.isfinite(index) or index < INT32_MIN or index > INT32_MAX:
                raise CoordinateRangeTooPrecise(
                    f"the {name} interval [{lo!r}, {hi!r}] does not fit in the "
                    f"signed 32 bit integer range at the allowed error "
                    f"{tolerance!r}"
                )
            decoded = _decode(index, tolerance, offset)
            if abs(decoded - coordinate) > tolerance:
                raise CoordinateRangeTooPrecise(
                    f"the {name} interval [{lo!r}, {hi!r}] cannot meet the "
                    f"error budget {tolerance!r}: endpoint {coordinate!r} "
                    f"would be decoded as {decoded!r}"
                )

        offsets[axis] = offset

    return CoordinateGrid(
        scales=scales, offsets=offsets, max_errors=max_errors, mins=mins, maxs=maxs
    )


class CoordinateGuard:
    """Re-quantizes point records onto a destination grid, atomically.

    A guard is enabled when opening a file for writing or appending:

    >>> import laspy
    >>> grid = laspy.make_coordinate_grid([0, 0, 0], [10, 10, 10], 0.001)
    >>> guard = laspy.CoordinateGuard(grid.max_errors)

    Every point record given to :meth:`laspy.LasWriter.write_points` or
    :meth:`laspy.LasAppender.append_points` must then be a
    :class:`laspy.ScaleAwarePointRecord` (i.e. it must carry its source
    scales and offsets).  The whole batch is validated before anything is
    written; if one point is invalid, the batch is rejected in full and the
    writer's header statistics are left untouched.

    Parameters
    ----------
    max_errors: number or sequence of three numbers
        Per-axis maximum absolute error allowed between the decoded source
        coordinate and the decoded coordinate stored in the destination.
    """

    def __init__(self, max_errors: ToleranceLike) -> None:
        self.max_errors = _as_tolerances(max_errors)
        # Optional closed interval, per axis.  When set (which happens when the
        # guard is derived from a grid built by make_coordinate_grid), points
        # outside of it are rejected.
        self.mins: Optional[np.ndarray] = None
        self.maxs: Optional[np.ndarray] = None

    def with_bounds(self, mins, maxs) -> "CoordinateGuard":
        """Returns a guard that also rejects coordinates outside ``[mins, maxs]``."""
        mins = _as_float_triplet(mins, "mins", InvalidCoordinateBounds)
        maxs = _as_float_triplet(maxs, "maxs", InvalidCoordinateBounds)
        if np.any(mins > maxs):
            raise InvalidCoordinateBounds("the guard's intervals are reversed")
        guard = CoordinateGuard(self.max_errors)
        guard.mins = mins
        guard.maxs = maxs
        return guard

    def __repr__(self) -> str:
        return f"<CoordinateGuard(max_errors={self.max_errors.tolist()})>"

    @classmethod
    def from_grid(cls, grid: CoordinateGrid) -> "CoordinateGuard":
        """Creates a guard using the tolerances recorded in a grid."""
        if grid.max_errors is None:
            raise InvalidCoordinateTolerance(
                "the coordinate grid does not carry max_errors"
            )
        guard = cls(grid.max_errors)
        if grid.mins is not None and grid.maxs is not None:
            guard.mins = grid.mins.copy()
            guard.maxs = grid.maxs.copy()
        return guard

    def convert_points(
        self,
        points: PackedPointRecord,
        point_format,
        target_scales: Sequence[float],
        target_offsets: Sequence[float],
    ) -> PackedPointRecord:
        """Validates and converts a batch to the destination grid.

        The input record is never modified.  Returns a new
        :class:`PackedPointRecord` sharing the point format, point order and
        every non-coordinate field of the input.

        Raises
        ------
        MissingSourceGrid
            If *points* is not a :class:`ScaleAwarePointRecord`.
        InvalidSourceGrid
            If the source scales/offsets are not usable.
        InvalidTargetGrid
            If the destination scales/offsets are not usable.
        IncompatiblePointFormat
            If *points* does not have the destination point format.
        NonFiniteCoordinate
            If a decoded source coordinate is not finite.
        CoordinateIntegerOverflow
            If a coordinate cannot be stored in a signed 32 bit integer.
        CoordinateOutOfBounds
            If the grid was built for a closed interval and a source
            coordinate falls outside of it.
        CoordinatePrecisionExceeded
            If re-quantization would move a coordinate by more than the
            allowed error.
        """
        if not isinstance(points, ScaleAwarePointRecord):
            raise MissingSourceGrid(
                "the coordinate guard needs points that carry their source grid: "
                "expected a ScaleAwarePointRecord, got a bare PackedPointRecord"
            )

        if points.point_format != point_format:
            raise IncompatiblePointFormat(
                "the points' point format does not match the destination point format"
            )

        source_scales, source_offsets = _validate_scales_and_offsets(
            points.scales, points.offsets, InvalidSourceGrid, "source point record"
        )
        target_scales, target_offsets = _validate_scales_and_offsets(
            target_scales, target_offsets, InvalidTargetGrid, "destination header"
        )

        array = points.array
        dimension_names = array.dtype.names or ()
        missing = [name for name in ("X", "Y", "Z") if name not in dimension_names]
        if missing:
            raise IncompatiblePointFormat(
                f"the points are missing coordinate dimensions {missing}"
            )

        converted = array.copy()

        # First pass: decode the source coordinates and check that every
        # integer coordinate fits in int32.  We do this for all axes before
        # applying the (more expensive) precision check so that the error
        # reported to the caller is not dependent on the axis order.
        quantized = []
        coordinates = []
        for axis, name in enumerate(("X", "Y", "Z")):
            with np.errstate(over="ignore", invalid="ignore"):
                source_coordinates = (
                    array[name].astype(np.float64) * source_scales[axis]
                    + source_offsets[axis]
                )
            if not np.all(np.isfinite(source_coordinates)):
                bad = np.flatnonzero(~np.isfinite(source_coordinates))[0]
                raise NonFiniteCoordinate(
                    f"point {int(bad)} has a non-finite "
                    f"{_AXIS_NAMES[axis]} coordinate"
                )

            if self.mins is not None:
                outside = (source_coordinates < self.mins[axis]) | (
                    source_coordinates > self.maxs[axis]
                )
                if np.any(outside):
                    bad = int(np.flatnonzero(outside)[0])
                    raise CoordinateOutOfBounds(
                        f"point {bad}'s {_AXIS_NAMES[axis]} coordinate "
                        f"({float(source_coordinates[bad])!r}) is outside the "
                        f"grid's interval [{float(self.mins[axis])!r}, "
                        f"{float(self.maxs[axis])!r}]"
                    )

            with np.errstate(over="ignore", invalid="ignore"):
                integer_coordinates = _encode(
                    source_coordinates, target_scales[axis], target_offsets[axis]
                )

            if np.any(~np.isfinite(integer_coordinates)):
                bad = np.flatnonzero(~np.isfinite(integer_coordinates))[0]
                raise CoordinateIntegerOverflow(
                    f"point {int(bad)}'s {_AXIS_NAMES[axis]} coordinate cannot "
                    f"be represented as a signed 32 bit integer"
                )

            rounded = np.rint(integer_coordinates)
            out_of_range = (rounded < INT32_MIN) | (rounded > INT32_MAX)
            if np.any(out_of_range):
                bad = int(np.flatnonzero(out_of_range)[0])
                raise CoordinateIntegerOverflow(
                    f"point {bad}'s {_AXIS_NAMES[axis]} coordinate "
                    f"({float(source_coordinates[bad])!r}) would be stored as "
                    f"the integer {float(rounded[bad])!r}, which is outside the "
                    f"signed 32 bit integer range"
                )

            quantized.append(rounded)
            coordinates.append(source_coordinates)

        # Second pass: check the error actually introduced by the round trip.
        for axis, name in enumerate(("X", "Y", "Z")):
            integers = quantized[axis].astype(np.int32)
            decoded = _decode(integers, target_scales[axis], target_offsets[axis])
            with np.errstate(over="ignore", invalid="ignore"):
                error = np.abs(decoded - coordinates[axis])
            exceeded = error > self.max_errors[axis]
            if np.any(exceeded):
                bad = int(np.flatnonzero(exceeded)[0])
                raise CoordinatePrecisionExceeded(
                    f"point {bad}'s {_AXIS_NAMES[axis]} coordinate "
                    f"({float(coordinates[axis][bad])!r}) would be moved by "
                    f"{float(error[bad])!r}, more than the allowed "
                    f"{float(self.max_errors[axis])!r}"
                )
            converted[name] = integers

        return PackedPointRecord(converted, points.point_format, points.readability)


def validate_target_grid(scales, offsets, what: str = "destination header"):
    """Validates a destination grid's scales/offsets and normalizes them to
    float64 arrays.  Raises :class:`InvalidTargetGrid` on unusable values."""
    return _validate_scales_and_offsets(scales, offsets, InvalidTargetGrid, what)


def resolve_coordinate_guard(
    value,
    scales: Sequence[float],
    offsets: Sequence[float],
) -> Optional[CoordinateGuard]:
    """Turns the value given to ``laspy.open(..., coordinate_guard=...)`` into
    a :class:`CoordinateGuard` (or ``None`` when the guard is disabled).

    ``True`` enables the guard with, per axis, a maximum error equal to the
    target grid's scale (one grid step).  A number or a sequence of three
    numbers is used directly as the per-axis maximum error.  Passing ``None``
    or ``False`` disables the guard.
    """
    if value is None or value is False:
        return None

    if isinstance(value, CoordinateGuard):
        return value

    if isinstance(value, CoordinateGrid):
        if value.max_errors is not None:
            return CoordinateGuard.from_grid(value)
        # A grid without recorded tolerances (not produced by
        # make_coordinate_grid): fall back to one grid step of the target.
        value = True

    if value is True:
        scales = np.asarray(scales, dtype=np.float64)
        if scales.shape != (3,) or not np.all(np.isfinite(scales)):
            raise InvalidTargetGrid(
                "cannot derive a default coordinate tolerance: the target "
                "header scales are not three finite values"
            )
        if np.any(scales <= 0.0):
            raise InvalidTargetGrid(
                "cannot derive a default coordinate tolerance: the target "
                "header scales must be strictly positive"
            )
        return CoordinateGuard(scales)

    return CoordinateGuard(value)


def ensure_grid_matches(
    expected: CoordinateGrid,
    scales: Sequence[float],
    offsets: Sequence[float],
    what: str = "file",
) -> None:
    """Raises if a stored grid differs from an expected one."""
    _, _ = _validate_scales_and_offsets(scales, offsets, InvalidTargetGrid, what)
    actual = CoordinateGrid(scales=scales, offsets=offsets)
    if not np.array_equal(actual.scales, expected.scales) or not np.array_equal(
        actual.offsets, expected.offsets
    ):
        raise InvalidTargetGrid(
            f"the {what}'s coordinate grid does not match the expected grid: "
            f"{what} has scales={actual.scales.tolist()}, "
            f"offsets={actual.offsets.tolist()} while the expected grid has "
            f"scales={expected.scales.tolist()}, offsets={expected.offsets.tolist()}"
        )
