""" All the custom exceptions types
"""


class LaspyException(Exception):
    pass


class UnknownExtraType(LaspyException):
    pass


class PointFormatNotSupported(LaspyException):
    pass


class FileVersionNotSupported(LaspyException):
    pass


class LazError(LaspyException):
    pass


class FieldNotReadableError(LaspyException):
    """Raised in strict read mode when accessing a point field that exists
    in the point format but was not decompressed/read from the source.

    Fields that do not exist in the point format keep raising their usual
    error instead of this one.

    .. versionadded:: 2.6
    """

    def __init__(self, field_name: str):
        self.field_name = field_name
        super().__init__(
            f"Point field '{field_name}' exists in the point format but was not "
            f"read (it was skipped by the selective decompression selection)"
        )


class IncompatibleDataFormat(LaspyException):
    pass


class CoordinateGridError(LaspyException, ValueError):
    """Base class for errors raised while deriving a coordinate grid."""


class InvalidCoordinateBounds(CoordinateGridError):
    """The given coordinate bounds are reversed, non-finite or unusable."""


class InvalidCoordinateTolerance(CoordinateGridError):
    """The given maximum absolute error is not a finite positive number."""


class CoordinateRangeTooPrecise(CoordinateGridError):
    """The range cannot be represented within the error budget."""


class CoordinatePrecisionError(LaspyException):
    """Base class for errors raised by the coordinate precision guard."""


class InvalidTargetGrid(CoordinatePrecisionError):
    """The destination header does not define a usable coordinate grid."""


class InvalidSourceGrid(CoordinatePrecisionError):
    """The point record's source scales/offsets are not usable."""


class MissingSourceGrid(CoordinatePrecisionError):
    """Points were given as a PackedPointRecord, without scales/offsets."""


class NonFiniteCoordinate(CoordinatePrecisionError):
    """A source coordinate is not finite."""


class IncompatiblePointFormat(CoordinatePrecisionError):
    """The point record's format does not match the destination format."""


class CoordinateIntegerOverflow(CoordinatePrecisionError):
    """A coordinate does not fit into the signed 32 bit integers."""


class CoordinatePrecisionExceeded(CoordinatePrecisionError):
    """Re-quantizing a coordinate would move it by more than allowed."""


class CoordinateOutOfBounds(CoordinatePrecisionError):
    """A coordinate is outside the closed interval the grid was built for."""
