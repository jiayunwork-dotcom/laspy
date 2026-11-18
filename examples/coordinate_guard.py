"""Writing several batches of points with different source grids into one file.

Run with::

    python examples/coordinate_guard.py

The laser scans of a flight campaign are delivered as several batches, each
with its own scale and offset.  All of them must end up in a single file whose
header defines one coordinate grid.  ``coordinate_guard`` makes
``write_points`` / ``append_points`` re-quantize every batch onto that grid and
refuse a whole batch if one of its points cannot be stored within the error
budget.
"""

import io

import laspy

# ---------------------------------------------------------------------------
# 1. Derive the file header's grid from the campaign's coordinate bounds.
#    X, Y and Z are each a closed interval [min, max] and each axis accepts a
#    maximum absolute error of 1 mm.
# ---------------------------------------------------------------------------
grid = laspy.make_coordinate_grid(
    mins=[0.0, 0.0, 0.0],
    maxs=[1000.0, 1000.0, 200.0],
    max_errors=0.001,
)
print("grid scales: ", grid.scales)
print("grid offsets:", grid.offsets)

header = laspy.LasHeader(point_format=3, version="1.2")

# A batch as delivered by one flight line: its own scale/offset.
def batch(coords, scale, offset):
    points = laspy.ScaleAwarePointRecord.zeros(
        len(coords),
        point_format=laspy.PointFormat(3),
        scales=[scale] * 3,
        offsets=[offset] * 3,
    )
    points.x = [c[0] for c in coords]
    points.y = [c[1] for c in coords]
    points.z = [c[2] for c in coords]
    points.return_number = [1] * len(coords)
    return points


destination = io.BytesIO()

# The first batch is written by the session that creates the file.  The grid
# above defines the header; the batch is re-quantized onto it.
with laspy.open(
    destination, mode="w", header=header, closefd=False, coordinate_guard=grid
) as writer:
    writer.write_points(
        batch([(10.0, 10.0, 10.0), (20.0, 25.0, 12.5)], scale=0.5, offset=0.0)
    )

# Subsequent batches come from other source grids and are appended.  The grid
# already stored in the file is used; the header cannot be changed anymore.
destination.seek(0)
with laspy.open(destination, mode="a", closefd=False, coordinate_guard=True) as appender:
    appender.append_points(
        batch([(30.0, 30.0, 30.0)], scale=0.125, offset=2048.0)
    )

destination.seek(0)
result = laspy.read(destination)
print("points:      ", len(result.points))
print("coordinates: ", result.xyz)
print("header grid: ", result.header.scales, result.header.offsets)
