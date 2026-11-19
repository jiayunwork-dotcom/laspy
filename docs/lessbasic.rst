=================
Less Basic Things
=================


Extra Dimensions
================

The LAS Specification version 1.4 defines a standard way to add extra dimensions to
a LAS file.

In laspy you can add extra dimensions using the :meth:`.LasData.add_extra_dim` function


The Allowed base types for an extra dimensions are:

+-------------------------+-------------+-------------+
|       laspy name        | size (bits) |     type    |
+=========================+=============+=============+
|     u1 or uint8         |     8       |  unsigned   |
+-------------------------+-------------+-------------+
|     i1 or int8          |     8       |   signed    |
+-------------------------+-------------+-------------+
|     u2 or uint16        |     16      |   unsigned  |
+-------------------------+-------------+-------------+
|     i2 or int16         |     16      |    signed   |
+-------------------------+-------------+-------------+
|     u4 or uint32        |     32      |   unsigned  |
+-------------------------+-------------+-------------+
|     i4 or int32         |     32      |    signed   |
+-------------------------+-------------+-------------+
|     u8 or uint64        |     64      |   unsigned  |
+-------------------------+-------------+-------------+
|     i8 or int64         |     64      |    signed   |
+-------------------------+-------------+-------------+
|     f4 or float32       |     32      |   floating  |
+-------------------------+-------------+-------------+
|     f8 or float64       |     64      |   floating  |
+-------------------------+-------------+-------------+

You can prepend the number '2' or '3' to one of the above base type to define an extra dimension
that is array of 2 or 3 elements per points.
Example: 3u2 -> each points will have an extra dimension that is an array of 3 * 16 bits


Here we are adding a new dimension called "codification" where each value is stored on a 64 bit unsigned integer
and an array field of 3 doubles for each points.


.. code-block:: python

    import laspy
    import numpy as np

    las = laspy.read("somefile.las")

    las.add_extra_dim(laspy.ExtraBytesParams(
        name="codification",
        type=np.uint64,
        description="More classes available"
    ))

    las.add_extra_dim(laspy.ExtraBytesParams(name="mysterious", type="3f8"))



.. note::

    Although the specification of the ExtraBytesVlr appeared in the 1.4 LAS Spec, laspy allows to
    add new dimensions to file with version < 1.4

.. note::

   If you are adding multiple extra dimensions use :meth:`.LasData.add_extra_dims`
   as it is more efficient (it allows to allocate all the dimensions at once instead
   of re-allocating each time a new dimension is added.


Selective decompression and strict read mode
============================================

LAZ 1.4 files using point formats 6 to 10 can be decompressed *partially*:
the :class:`.DecompressionSelection` passed to :func:`laspy.open`,
:func:`laspy.read` or :meth:`.CopcReader.open` selects which groups of fields
the backend decodes. Skipping fields you do not need (for example the colors
when you only compute an intensity histogram) makes the decompression cheaper.

However, a partially decompressed point record still exposes every field of
its point format: the skipped fields simply contain placeholder bytes, which
can be zero or even look like plausible values. To tell "this field was not
read" apart from "this field was read and its value is zero", open the reader
with ``strict=True``:

.. code-block:: python

    import laspy
    from laspy.errors import FieldNotReadableError

    selection = laspy.DecompressionSelection.base().decompress_intensity()

    with laspy.open("points.laz", decompression_selection=selection, strict=True) as reader:
        # Inspect what is available without reading a single point.
        print(reader.readable_fields())
        print(reader.is_field_readable("Z"))  # False

        points = reader.read_points(10_000)
        points.intensity  # works, it was decoded
        try:
            points.Z
        except FieldNotReadableError as error:
            print(error)  # the message contains the field name: 'Z'

In strict mode:

* accessing a field that exists in the point format but was skipped raises
  :class:`.FieldNotReadableError` (the error object carries the field name);
* fields absent from the point format keep raising their usual error;
* a value that was genuinely decoded as zero reads perfectly fine;
* :meth:`.LasData.readable_fields`, :meth:`.LasData.is_field_readable`,
  :meth:`.LasReader.readable_fields`, :meth:`.LasReader.is_field_readable`
  and the matching methods on :class:`.CopcReader` answer readability
  queries without triggering any (re)reading;
* the information follows the data through slicing, boolean masking,
  copying, chunked reads and the equivalent :class:`.LasData` operations,
  and stays valid after the reader is closed -- including on empty results;
* the header and the point format are unchanged, and the underlying numpy
  structured array (``points.array``) remains directly accessible;
* uncompressed LAS and point formats that do not support selective
  decompression simply report every field as readable.

Omitting ``strict`` (or passing ``strict=False``) keeps the historical
behaviour: skipped fields stay accessible and return their placeholder data.

COPC window queries
-------------------

:meth:`.CopcReader.query` accepts 2D (XY only) and 3D bounds. With a 3D
window the reader must filter on the actual elevation, even when the caller
did not select ``Z``. In that case laspy decodes ``Z`` internally for the
duration of the filtering step, but the field stays unreadable through the
public API unless it was part of the selection:

.. code-block:: python

    selection = laspy.DecompressionSelection.base()  # Z not selected
    with laspy.CopcReader.open("cloud.copc.laz",
                               decompression_selection=selection,
                               strict=True) as reader:
        result = reader.query(bounds_3d)   # filtered on the real Z
        result.X                            # readable
        result.Z                            # raises FieldNotReadableError

The returned points and their order are identical to a fully decompressed
query with the same window. An XY-only window never decodes ``Z`` at all.


Custom VLRs
===========

Provided you have a valid user_id and record_id (meaning that they are not taken by a VLR described in the LAS specification)
You can add you own VLRs to a file

Fast & Easy way
---------------

The fastest and easiest way to add your custom VLR to a file is to create a :class:`.VLR`,
set its record_data (which must be bytes) and add it to the VLR list.


>>> import laspy
>>> vlr = laspy.vlrs.VLR(user_id='A UserId', record_id=1, description='Example VLR')
>>> vlr
<VLR(user_id: 'A UserId', record_id: '1', data len: 0)>
>>> vlr.record_data = b'somebytes'
>>> vlr
<VLR(user_id: 'A UserId', record_id: '1', data len: 9)>
>>> las = laspy.create()
>>> las.vlrs
[]
>>> las.vlrs.append(vlr)
>>> las.vlrs
[<VLR(user_id: 'A UserId', record_id: '1', data len: 9)>]


Complete & Harder way
---------------------

While the way shown above is quick & easy it might not be perfect for complex VLRs.
Also when reading a file that has custom VLR, laspy won't be able to automatically parse its data
into a better structure, so you will have to find the VLR in the vlrs list and parse it yourself
one laspy is done.

One way to automate this task is to create your own Custom VLR Class that extends
:class:`.BaseKnownVLR` by implementing the missing methods laspy
will be able to automatically parse the VLR when reading the file & write it when saving the file.

>>> class CustomVLR(laspy.vlrs.BaseKnownVLR):
...     def __init__(self):
...         super().__init__()
...         self.numbers = []
...
...     @staticmethod
...     def official_user_id():
...         return "CustomId"
...
...     @staticmethod
...     def official_record_ids():
...         return 1,
...
...     def record_data_bytes(self):
...         return bytes(self.numbers)
...
...     def parse_record_data(self, record_data):
...         self.numbers = [b for b in record_data]
...
...     def __repr__(self):
...         return "<MyCustomVLR>"

>>> import numpy as np
>>> cvlr = CustomVLR()
>>> cvlr.numbers
[]
>>> cvlr.numbers = [1,2, 3]
>>> las = laspy.create()
>>> las.vlrs.append(cvlr)
>>> las.vlrs
[<MyCustomVLR>]
>>> las.x = np.array([1.0, 2.0])
>>> las = laspy.lib.write_then_read_again(las)
>>> las.vlrs
[<MyCustomVLR>]
>>> las.vlrs[0].numbers
[1, 2, 3]

