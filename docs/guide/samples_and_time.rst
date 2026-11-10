Samples and time
================

Samples
-------

Every I and Q component is the value :math:`2k + 1` of §5.3.6, with
:math:`k` the two's-complement code read from the file: odd integers
symmetric about zero, at most :math:`2^{b} - 1` in magnitude for :math:`b`
bits. The complex sample is :math:`x = I + jQ`.

**The samples are returned as recorded.** RDEFpy never removes a mean,
applies a window, filters, scales or interpolates: on an open-loop recording
the carrier can sit in the DC bin.

``scan.prd`` below is the fixture of the :doc:`../quickstart`.

.. code-block:: python

   import numpy as np
   from rdefpy import read_product

   pf = read_product("scan.prd")        # 2 records at 2 kHz
   x = pf[0].data.samples               # one record, complex128
   y = pf.samples[1990:2010]            # any range, across records
   t = pf.timetags[1998:2002]           # datetime64[ns] of each sample
   print(y.shape, y.dtype)
   print(t)

   buf = None
   for rec in pf:                       # decode straight to complex,
       buf = rec.read_samples(out=buf)  # refilling one array

.. code-block:: text

   (20,) complex128
   ['2026-02-08T12:00:00.999000000' '2026-02-08T12:00:00.999500000'
    '2026-02-08T12:00:01.000000000' '2026-02-08T12:00:01.000500000']

* :attr:`rdefpy.RecordData.samples` builds a new array on each access.
  ``i_samples`` and ``q_samples`` are the integers (``int8`` for 1, 2 and 4
  bits, ``int16`` for 8, ``int32`` for 16).
* ``read_product(..., dtype=np.complex64)`` halves the memory, losslessly:
  the values are small integers.
* :meth:`rdefpy.Record.read_samples` gives the same values, bit for bit,
  without the integer arrays. ``out`` must be a 1-D, C-contiguous, aligned,
  writeable complex64 or complex128 array of exactly ``len(rec)`` samples;
  it is overwritten by the next call.
* :attr:`rdefpy.ProductFile.samples` is a view of the whole file: nothing
  is read until it is indexed, and only the records a range touches are
  read. Slicing follows NumPy. ``np.asarray(pf.samples)`` reads the whole
  file.

Gaps
----

A ``pf.samples`` range raises :class:`ValueError` when it crosses records
that are not 1 s apart, a change of sample rate, or a record shorter than
one second: read each side separately. ``pf.timetags`` and the
downconversion views never raise on a gap. To find the gaps:

.. code-block:: python

   dt = np.diff(pf.timetag_array)
   gaps = np.flatnonzero(dt != np.timedelta64(1, "s"))   # between records i and i + 1

Time tags
---------

Each header carries the time of the record's first sample as year, day of
year, second of day and picoseconds of the second (Table 5-1), in station
time, an approximation to UTC (§5.2.5). It is not converted.

* :attr:`rdefpy.RecordHeader.timetag` combines the fields into one
  ``numpy.datetime64[ns]``, picoseconds rounded to the nearest nanosecond.
  It raises :class:`ValueError` for a year outside 1678 to 2261, a day of
  year outside 1 to 366 or a second of day outside 0 to 86400; such a record
  has no time tag and time-window selection never includes it.
* :attr:`rdefpy.ProductFile.timetag_array` has one entry per record;
  :attr:`rdefpy.ProductFile.timetags` one per sample, sample *k* of a record
  at ``header.timetag + k / sample_rate``. Both read no samples.
* A leap-second record (second of day 86400) gets the same ``timetag`` as
  second 0 of the next day, because NumPy time has no leap second. The
  header fields keep them apart; ``consecutive`` is then False and a
  ``pf.samples`` range across the two raises.

Tagging an estimate over a block
--------------------------------

An estimate over a block that starts at :math:`t_0` and lasts
:math:`T = N / f_s` is tagged at the **midpoint of the block**, not at the
mean of its sample times:

.. math::

   t_\mathrm{tag} = t_0 + \frac{T}{2}, \qquad
   \text{not} \quad t_0 + \frac{N - 1}{2 f_s} .

For one-second records the tag is at the half second (x.500). The same
holds for averages of consecutive records. :func:`rdefpy.extract_carrier`
tags its blocks this way (:doc:`carrier`); the downconversion frequency
that belongs to such a block is its mean over the block
(:doc:`downconversion`).
