Checking another decoder
========================

RDEFpy ships small synthetic product files together with their expected
decoding as plain text. A decoder that reproduces the text files reads the
format the same way. RDEFpy itself is tested against them
(``tests/test_reference.py``).

The files are in `tests/reference/
<https://github.com/atogni/RDEFpy/tree/main/tests/reference>`_ of the
repository and in the source distribution, not in the wheel. They are
written by ``tests/make_reference.py`` from ``tests/rdef_spec.py``, a plain
transcription of CCSDS 506.1-B-1 and of the DSN 0222 internal-agency layout
[DSN0222]_ that does not import RDEFpy. To regenerate them from the
repository root:

.. code-block:: text

   python tests/make_reference.py

Cases
-----

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Case
     - Contents
   * - ``sample_1bit`` ... ``sample_16bit``
     - One file per sample size (1, 2, 4, 8, 16 bits): spacecraft 1,
       station DS24, 2 kHz, two one-second records of a full-scale tone.
       Copies of the files in ``tests/data/``.
   * - ``steered_ps0``
     - 16 bits, 1 kHz, three records with a steered downconversion
       (:math:`c_1` to :math:`c_3` nonzero, phase and phase rate continuous
       from record to record), picoseconds 0, ``agency_flag`` 0.
   * - ``dsn_agency_block``
     - The same steering with picoseconds 50,000, ``agency_flag`` 3 and the
       40-byte internal-agency block filled with the DSN 0222 layout.

``tests/reference/README.md`` gives the details of each case.

Files and how to compare
------------------------

Every case has the product file (``.prd``) and these text files:

.. list-table::
   :header-rows: 1
   :widths: 12 52 36

   * - File
     - Contents
     - Compare
   * - ``.hdr``
     - Every header field of every record (Table 5-1, little-endian, §5.1).
     - Exactly.
   * - ``.tab``
     - The decoded samples as the §5.3.6 values :math:`2k+1` (Table 5-2
       packing). One line per 10 samples: record number (from 1), index in
       the record of the line's first sample (from 0), then 10 I/Q pairs.
     - Exactly. A difference usually means the bit order inside the 32-bit
       words, the I/Q order or the :math:`2k+1` mapping.
   * - ``.hex``
     - The whole file as bytes: offset, then 16 bytes per line.
     - Exactly.
   * - ``.dc``
     - ``steered_ps0`` and ``dsn_agency_block`` only. Per record, with
       :math:`t_0` = TIME TAG SECOND OF DAY (§5.2.6): the total
       downconversion frequency at the first sample (Hz), its mean over the
       record's second (Hz), and the variable-part phase at the first sample
       (turns).
     - Within floating-point rounding; RDEFpy's own test allows 4e-6 Hz and
       1e-9 turns.
   * - ``.agency``
     - ``dsn_agency_block`` only. The internal-agency block (header bytes
       132-171) of every record, field by field with the DSN 0222 layout
       (Table 3-1, Figure 3-4).
     - Integers exactly.

A ``.hdr`` file starts:

.. code-block:: text

   sample_8bit.prd: record headers (CCSDS 506.1-B-1 Table 5-1)

   Record 1 of 2, file offset 0
     record_label                             = RDEF
     record_length                            = 4176
     record_version_id                        = 1
     station_id                               = 24
     ...

From the same header fields and samples, a conforming writer reproduces the
``.prd`` byte for byte. The files do not cover error handling (truncated
files, bad labels, values out of range): that is in RDEFpy's own tests
(``tests/test_spec_*.py``).
