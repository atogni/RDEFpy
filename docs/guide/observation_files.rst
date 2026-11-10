Observation files
=================

An observation file (``.obs``) is a short ASCII index of a tracking session
(§4). It names the receiving station, lists the scans, and for each scan
lists the product files recorded, one per frequency channel.
:func:`rdefpy.read_observation` parses and validates it and returns an
:class:`rdefpy.ObservationFile`. RDEFpy does not write observation files.

The fixture ``tests/data/EXMPn000tIsDS24r01c00-26039120000.obs``, a
simulated example:

.. code-block:: text

   # Observation File
   V VERSION = 1
   # Simulated example, not a real observation.
   R STATION = DS24
   Z
   #
   # SCAN_NUM  SRC_ID     START_TIME         STOP_TIME          RA         DEC         TFREQ
   # ------------------------------------------------------------------------------------------
   S 001       EXMP       2026-039T12:00:00  2026-039T12:00:02  999        999         8400000000.0000
   #
   # DATAFILE                                COH_FLAG    DOR_MULT     FSUB           HARMONIC
   # ------------------------------------------------------------------------------------------
   D EXMPn001tSsDS24r01c01-26039120000.prd   T           0            0.0            0
   D EXMPn001tSsDS24r01c02-26039120000.prd   T           1/440        0.0            -1
   D EXMPn001tSsDS24r01c03-26039120000.prd   T           1/440        0.0            1
   D EXMPn001tSsDS24r01c04-26039120000.prd   T           1/440        0.0            -2
   D EXMPn001tSsDS24r01c05-26039120000.prd   T           1/440        0.0            2
   Z
   E *=END=*

The first character of a line is its type: ``#`` comment, ``V`` version,
``R`` receiving station, ``T`` transmitting station (absent for one-way
data), ``Z`` end of a section, ``S`` scan, ``D`` product file, ``F`` log
line, ``E`` end line.

.. code-block:: python

   from rdefpy import read_observation, read_product

   obs = read_observation("EXMPn000tIsDS24r01c00-26039120000.obs")
   obs.info()
   for scan in obs:
       for entry in scan.products:
           pf = read_product(entry.datafile)
           print(scan.scan_num, entry.harmonic, entry.dor_mult,
                 len(pf), pf[0].header.sample_size)

.. code-block:: text

   ObservationFile EXMPn000tIsDS24r01c00-26039120000
                 path: ...\tests\data\EXMPn000tIsDS24r01c00-26039120000.obs
              version: 1
      receive_station: DS24
     transmit_station: None (one-way)
                scans: 1
             products: 5
             scan 001: EXMP (spacecraft) 2026-02-08T12:00:00Z to 2026-02-08T12:00:02Z, 5 products
   1 0 0 2 16
   1 -1 1/440 2 8
   1 1 1/440 2 4
   1 -2 1/440 2 2
   1 2 1/440 2 1

The objects
-----------

:class:`rdefpy.ObservationFile`
   ``path``, ``version``, ``receive_station``, ``transmit_station``
   (``None`` for one-way data), ``scans`` and ``logs`` (the ``F`` lines). It
   is iterable, sized and indexable over its scans.

:class:`rdefpy.Scan`
   One ``S`` line (Table 4-1): ``scan_num``, ``src_id``, ``start_time`` and
   ``stop_time`` (``numpy.datetime64``, 1 s precision), ``ra`` and ``dec``
   in degrees (``None`` when the file gives 999), ``tfreq`` in Hz,
   ``products``, and ``kind``: ``"quasar"`` when ``tfreq`` is 0, else
   ``"spacecraft"``.

:class:`rdefpy.ProductEntry`
   One ``D`` line (Table 4-2), one frequency channel: ``datafile`` (a
   :class:`pathlib.Path`, the name joined to the observation file's
   directory), ``coherent``, ``dor_mult`` (an exact
   :class:`fractions.Fraction`), ``fsub`` and ``harmonic``. The fields are
   stored as written; the tone frequency is not computed.

Validation
----------

The reader is strict: a file that breaks a "shall" of §4 raises
:class:`ValueError` naming the file, the line number and the rule.

.. code-block:: text

   session.obs:9: RA '400' outside 0 to 360: 'S 001       EXMP       2026-039T12:00:00  2026-039T12:00:02  400        999         8400000000.0000'

* **Structure**: header section, then scan sections, each closed by ``Z``,
  then optional log lines and the end line; every scan has at least one
  product line.
* **Lines** (§4.1.3): at most 180 characters, printable ASCII, no blank
  lines, every line terminated (LF, CR, CR/LF or LF/CR).
* **Scan line**: scan number ``001`` to ``999`` and increasing; ``TFREQ``
  not negative, at most 16 significant digits; a four-character source ID
  for a spacecraft, up to 16 for a quasar; times as ``YYYY-DDDThh:mm:ss``;
  RA within 0 to 360 and DEC within -90 to +90, or 999.
* **Product line**: a bare file name, coherence flag ``T`` or ``F``,
  integer harmonic, ``DOR_MULT`` a fraction of two integers when coherent,
  ``FSUB`` in decimal notation when not.
* **Stations**: exactly four characters.

A scan that stops before it starts only gives a :class:`UserWarning`. A
leap second (``hh:mm:60``) is rejected, because :class:`numpy.datetime64`
cannot hold it.

Not checked: the version against the product files' (§4.2.4), the file
names of §6.2, that the listed product files exist, and that scan times
agree with the product files' time tags.
