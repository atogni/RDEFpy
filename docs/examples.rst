Examples
========

Runnable scripts from the ``examples/`` directory of the repository. Each runs
in seconds from the repository root, reads the small files in ``tests/data``,
and writes only to a temporary directory.

Read an observation file
------------------------

Reads an observation file (§4) and the product files it lists.

.. literalinclude:: ../examples/read_observation.py
   :language: python

Validate observation files
--------------------------

Checks observation files against §4 and exits with status 1 if any fails.

.. literalinclude:: ../examples/validate_observation.py
   :language: python

.. code-block:: text

   $ python examples/validate_observation.py tests/data/EXMPn000tIsDS24r01c00-26039120000.obs
   OK    tests/data/EXMPn000tIsDS24r01c00-26039120000.obs: 1 scans, 5 product files

Summarise a product file
------------------------

File summary with ``info()``, the header of the first record, and the power
of each record.

.. literalinclude:: ../examples/read_product_summary.py
   :language: python

Samples and time tags
---------------------

``pf.samples`` and ``pf.timetags`` take any sample range across records;
``read_samples(out=buf)`` decodes straight to complex into one reused array.

.. literalinclude:: ../examples/samples_and_timetags.py
   :language: python

.. code-block:: text

   2026-02-08T12:00:00.998500000 (38521-53019j)
   2026-02-08T12:00:00.999000000 (53019-38521j)
   2026-02-08T12:00:00.999500000 (62327-20251j)
   2026-02-08T12:00:01.000000000 (65535+1j)
   2026-02-08T12:00:01.000500000 (62327+20251j)
   2026-02-08T12:00:01.001000000 (53019+38521j)
   mean power 96.33 dB
   loop matches pf.samples: True

Select and write
----------------

Selects the records of a time window, writes them with ``write_product`` and
reads them back; selecting every record reproduces the file byte for byte.

.. literalinclude:: ../examples/select_and_write.py
   :language: python

.. code-block:: text

   2 records -> 1 selected, first at 2026-02-08T12:00:01.000000000
   written 8176 bytes, read back 1 record
   samples equal: True
   bytes equal: True

Downconversion frequency and phase
----------------------------------

Downconversion frequency and phase (§5.2.6) per record and per sample. The
script gives a copy of the fixture one frequency ramp in the headers,
continuous in phase and frequency across the record boundary.

.. literalinclude:: ../examples/downconversion.py
   :language: python

.. code-block:: text

   per-record [Hz]: [8405000000.000 8405000000.000]
   first sample [Hz]: [8405000100.000 8405000094.000]
   record mean  [Hz]: [8405000097.000 8405000091.000]
   phase at first sample [turns]: [0.000 97.000]
   per-sample [Hz]:    [8405000094.003 8405000094.000]
   per-sample [turns]: [96.953 97.000]
   2026-02-08T12:00:00.500000000  8405000097.000 Hz
   2026-02-08T12:00:01.500000000  8405000091.000 Hz

Carrier of a synthetic tone
---------------------------

Builds a file of a complex tone of known frequency, quantized to the ``2k+1``
codes of 16-bit samples, and compares ``extract_carrier`` with the truth.

.. literalinclude:: ../examples/carrier_synthetic.py
   :language: python

.. code-block:: text

   truth frequency before downconversion 8405000123.456700 Hz (baseband 123.4567 Hz)
   2026-02-08T12:00:00.500000000  before DC 8405000123.456700 Hz  error +0.000 mHz  baseband 123.456701 Hz
   2026-02-08T12:00:01.500000000  before DC 8405000123.456700 Hz  error +0.000 mHz  baseband 123.456701 Hz
   2026-02-08T12:00:02.500000000  before DC 8405000123.456699 Hz  error -0.001 mHz  baseband 123.456700 Hz

Large files
-----------

Counts records and reads headers without touching the samples, then iterates
the file one record at a time, with ``set_threads`` for the decoding.

.. literalinclude:: ../examples/stream_large_file.py
   :language: python
