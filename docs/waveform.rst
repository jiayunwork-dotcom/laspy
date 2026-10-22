Waveform Support
================

laspy can read and write LAS waveform point formats, but the current
implementation intentionally supports only a subset of the LAS 1.4 waveform
specification.

API
---

Use ``fullwave`` when opening a file for reading:

The default is ``fullwave="never"`` (no waveform samples are loaded unless you
opt in).

.. code-block:: python

   import laspy

   with laspy.open("example.laz", fullwave="never") as reader:
       las = reader.read()

   with laspy.open("example.laz", fullwave="lazy") as reader:
       points = reader.read_points(1024)
       waveforms = points["waveform"]

   with laspy.open("example.laz", fullwave="eager") as reader:
       las = reader.read()
       waveforms = las.points["waveform"]

``fullwave="never"``
   Do not load waveform samples.

``fullwave="lazy"``
   Keep waveform samples in their storage (external ``.wdp`` file or the
   internal EVLR) until ``points["waveform"]`` is accessed.

``fullwave="eager"``
   Load waveform samples while reading points.

Writing
-------

Waveform packet location is selected with the global encoding flags:

- ``header.global_encoding.waveform_data_packets_internal = True`` and
  ``waveform_data_packets_external = False`` stores the packets inside the
  LAS/LAZ file, in an EVLR with User ID ``LASF_Spec`` and Record ID 65535.
  This requires LAS 1.4. No sibling ``.wdp`` file is created.
- Any other flag combination with points carrying waveforms writes the
  packets to a sibling ``.wdp`` file next to the ``.las`` or ``.laz`` file.
  If both flags are set, the external ``.wdp`` file is used.
- When none of the points carry a waveform, both flags are cleared and no
  waveform data is written.

For internal storage, per-point ``wavepacket_offset`` values are relative to
the start of the waveform EVLR header, so the first packet starts at offset
60. The header's ``start_of_waveform_data_packet_record`` points at the
waveform EVLR header, which may sit anywhere among the other EVLRs; existing
EVLRs are preserved.

When waveform data is present, writing to a filesystem path emits either the
embedded EVLR or the sibling ``.wdp`` file, depending on the flags above.

Writing waveform-enabled data to a file-like object is not supported.

Current Scope
-------------

The current implementation supports:

- External waveform storage in a sibling ``.wdp`` file.
- Internal waveform storage in the ``LASF_Spec``/65535 EVLR (LAS 1.4).
- Uncompressed waveform packets only.
- Byte-aligned sample widths only.
- A single waveform layout across all waveform descriptors in the file.
- Waveform packets addressed as fixed-size records.

Unsupported LAS 1.4 Features
----------------------------

The LAS 1.4 specification allows more than the current implementation supports.
laspy currently raises ``NotImplementedError`` or ``ValueError`` for these
cases.

- Non-byte-aligned waveform sample widths are not supported.
- Multiple waveform descriptor layouts in one file are not supported.
- General byte-offset-based packet layouts are not supported when they do not
  behave like fixed-size records.
