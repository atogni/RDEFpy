Carrier extraction
==================

:func:`rdefpy.extract_carrier` estimates the carrier frequency before
downconversion of a :class:`~rdefpy.ProductFile`, one value per block of
``integration_time`` seconds. It is the one estimator in RDEFpy; everything
else in the package is a format layer.

.. code-block:: python

   from rdefpy import extract_carrier, read_product

   pf = read_product("tests/data/EXMPn001tSsDS24r01c01-26039120000.prd")
   c = extract_carrier(pf, integration_time=1.0, max_rate_hz_s=0)   # a steady tone
   for r in c:
       print(r["time"], r["frequency_before_downconversion"], r["frequency"],
             r["rate"], r["downconversion_frequency"])

.. code-block:: text

   2026-02-08T12:00:00.500000000 8405000100.0 99.99999998657594 0.0 8405000000.0
   2026-02-08T12:00:01.500000000 8405000100.0 100.00000003356524 0.0 8405000000.0

Method
------

Each block is processed independently, on the samples as recorded
(:doc:`samples_and_time`); zero-padding (``zero_padding``, default 3) alters
no sample.

1. A bank of trial chirp rates is searched: for each rate the block is
   dechirped, zero-padded and Fourier transformed, and the maximum over
   (rate, frequency bin) gives a coarse rate and frequency. This is a
   fractional Fourier transform (FrFT) evaluated in the frequency/rate
   plane.
2. The rate is refined by Successive Parabolic Interpolation (SPI) on the
   logarithm of the power against rate.
3. The frequency is refined by SPI on the DTFT magnitude of the block
   dechirped at the final rate.

The estimator is that of Togni, Zannoni and Tortora, "Accurate frequency
estimation through iterative parabolic interpolations", *Signal Processing*
237 (2025) 110094, https://doi.org/10.1016/j.sigpro.2025.110094. Cite it
when you use the method (:doc:`/citing`). The per-block estimators are
:func:`rdefpy.carrier.frft_frequency` and
:func:`rdefpy.carrier.spi_frequency`.

Output
------

A NumPy structured array, one row per block, of dtype
``rdefpy.carrier.CARRIER_DTYPE``:

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - Field
     - Type
     - Meaning
   * - ``time``
     - ``datetime64[ns]``
     - Block midpoint, t0 + T/2 (:doc:`samples_and_time`).
   * - ``frequency_before_downconversion``
     - float, Hz
     - Mean frequency before downconversion over the block,
       ``downconversion_frequency + frequency`` (§5.2.6.1).
   * - ``frequency``
     - float, Hz
     - Mean baseband carrier frequency over the block.
   * - ``rate``
     - float, Hz/s
     - Baseband carrier rate.
   * - ``downconversion_frequency``
     - float, Hz
     - Mean downconversion frequency over the block, integrated exactly
       from the header polynomials of every record the block covers
       (:doc:`downconversion`).

``frequency_before_downconversion`` is exact for any downconverter steering
while the baseband frequency is linear in time over the block.

Blocks and time tags
--------------------

* The first block starts at the first sample of the first record, or at
  ``start_epoch`` (the forms ``read_product`` accepts), rounded to the
  nearest sample.
* ``integration_time * sample_rate`` must be a whole number of samples. A
  trailing partial block is not used.
* ``whole_second=True`` reports the blocks on the whole-second grid when
  the record time tags have a fractional second: the first block starts at
  the sample nearest the next whole second, and the estimates are carried
  to the grid tag with ``frequency + rate * dt``. The samples are neither
  shifted nor interpolated.

Rate search
-----------

Each block searches its own baseband chirp rate, starting from 0 Hz/s:

* The search starts at ``±rate_range_hz_s`` (default 2 Hz/s), in steps of
  ``rate_step_cells`` (default 2) cells of 1/T² Hz/s.
* While the winning rate is in the outer quarter of the range, the range
  doubles, up to ``max_rate_hz_s``. A winner still on the edge at the cap
  gives a :class:`RuntimeWarning`.
* ``max_rate_hz_s=None`` (default) caps the search at 0.75 fs/T;
  ``max_rate_hz_s=0`` skips it: plain FFT + SPI, for a stable tone.
* A rate far outside the starting range can be missed at low
  signal-to-noise ratio: choose ``rate_range_hz_s`` to cover the rates you
  expect.

Cost grows with the range and with T²: set ``max_rate_hz_s`` to a bound on
the rate you expect.

NaN results and errors
----------------------

* A block that spans a time gap (records more than 1 µs off 1 s apart, or
  an unusable time tag) or covers a record with a nonzero ``validity_flag``
  is NaN in every field but ``time``.
* A block that covers a Millisecond Predict record (:doc:`downconversion`)
  has NaN ``downconversion_frequency`` and
  ``frequency_before_downconversion``; ``frequency`` and ``rate`` are still
  estimated.
* A block whose samples are all zero or not finite has NaN
  ``frequency_before_downconversion``, ``frequency`` and ``rate``.
* Mixed sample rates, an empty file and a bad ``integration_time`` raise
  :class:`ValueError`.

Threads and memory
------------------

Blocks run in parallel on the thread pool set by
:func:`rdefpy.set_threads`; the file is read in one pass, each record
decoded once. With the Cython extension the kernels run compiled and
without the GIL; otherwise an equivalent NumPy path runs
(:doc:`/installation`).

``memory_limit`` bounds the blocks in flight, counting about 400 bytes per
block sample:

.. code-block:: text

   max(1, min(2 * threads, memory_limit // (400 * samples per block)))

The default, ``None``, is half of the memory the process can still take (2
GiB if it cannot be read). A single block over the budget still runs,
alone, with a :class:`RuntimeWarning`; ``memory_limit=math.inf`` ignores
the budget. The estimates are the same whatever the limit.
