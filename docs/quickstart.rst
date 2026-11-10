Quick start
===========

The examples use the small product file shipped with the tests,
``tests/data/EXMPn001tSsDS24r01c01-26039120000.prd`` (two one-second
records at 2 kHz, 16-bit samples); call it ``scan.prd``.

Read a product file
-------------------

.. code-block:: python

   from rdefpy import read_product

   pf = read_product("scan.prd")  # records above 1 kHz stay on disk until accessed
   pf.info()

.. code-block:: text

   ProductFile EXMPn001tSsDS24r01c01-26039120000
                 path: .../EXMPn001tSsDS24r01c01-26039120000.prd
              records: 2
        first_timetag: 2026-02-08T12:00:00.000000000Z
         last_timetag: 2026-02-08T12:00:01.000000000Z
          consecutive: True
          sample_rate: 2,000 Hz
          sample_size: 16 bit
        spacecraft_id: 1
           station_id: 24
                 band: X
   frequency_steering: False

A :class:`~rdefpy.ProductFile` is a sequence of one-second
:class:`~rdefpy.Record` objects. ``print(pf[0])`` summarises the first
record and ``print(pf[0].header)`` shows its header fields; ``rec.header``
holds all of them (:class:`~rdefpy.RecordHeader`) and ``rec.data`` the samples
(:class:`~rdefpy.RecordData`):

.. code-block:: python

   for rec in pf:                 # one record at a time
       x = rec.data.samples       # complex I + jQ, NumPy array
       print(rec.header.timetag, rec.data.power_db)

.. code-block:: text

   2026-02-08T12:00:00.000000000 96.32945284169084
   2026-02-08T12:00:01.000000000 96.32945284169084

At high sample rates, decode straight to complex and refill one array
(copy anything you keep):

.. code-block:: python

   buf = None
   for rec in pf:
       buf = rec.read_samples(out=buf)

Samples, times and downconversion across records
------------------------------------------------

The file-wide views index the whole file like one array, across record
boundaries:

.. code-block:: python

   x = pf.samples[:1024]                      # complex samples
   t = pf.timetags[:1024]                     # datetime64[ns] of each sample
   f = pf.downconversion_frequency[:1024]     # Hz, at each sample (§5.2.6)
   phi = pf.downconversion_phase[:1024]       # turns, variable part (§5.2.6.2)
   f0 = pf.record_downconversion_frequency    # Hz, at each record's first sample
   f_mean = pf.mean_downconversion_frequency  # Hz, mean over each record

``pf.samples`` raises :class:`ValueError` if a range crosses a time gap;
see :doc:`guide/samples_and_time` and :doc:`guide/downconversion`.

Carrier frequency
-----------------

.. code-block:: python

   from rdefpy import extract_carrier

   c = extract_carrier(pf, integration_time=1.0, max_rate_hz_s=10.0)
   c["time"]           # block midpoints: 12:00:00.5, 12:00:01.5
   c["frequency_before_downconversion"]  # Hz, one per block

See :doc:`guide/carrier`.

Browse
------

With matplotlib installed, an interactive window shows the spectrum, the
carrier and the headers around an epoch:

.. code-block:: python

   from rdefpy import browse

   browse(pf, epoch="2026-02-08T12:00:00.5", integration_time=0.5)

See :doc:`guide/browse`.

Select and write
----------------

.. code-block:: python

   from rdefpy import write_product

   # start <= timetag < stop: here the first record only
   part = pf.select(start_epoch="2026-02-08T12:00:00", stop_epoch="2026-02-08T12:00:01")
   write_product("part.prd", part)

See :doc:`guide/writing`.

Observation files
-----------------

An observation file (``.obs``) lists the product files of each scan:

.. code-block:: python

   from rdefpy import read_observation, read_product

   for scan in read_observation("session.obs"):
       for entry in scan.products:
           pf = read_product(entry.datafile)

See :doc:`guide/observation_files`, and :doc:`examples` for complete
scripts.
