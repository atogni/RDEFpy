Downconversion
==============

The samples are at baseband: the receiver has mixed the signal down by a
known, possibly time-varying local oscillator. The header carries what is
needed to undo that (§5.2.6), a fixed frequency plus a variable one:

.. math::

   f_\mathrm{DC}(t) = f_\mathrm{RF-IF} + f_\mathrm{IF-CHAN}
       + c_1 + 2 c_2 (t - t_0) + 3 c_3 (t - t_0)^2 .

* **Fixed part** (§5.2.6.1): the header fields ``rf_to_if_downconv`` and
  ``if_to_channel_downconv``. Its phase depends on the experiment epoch,
  "generally unknown", so it is not in the file.
* **Variable part** (§5.2.6.2, §5.2.6.3): the phase polynomial, in turns,
  updated for every record,

  .. math::

     \varphi_\mathrm{DC,var}(t) = \Phi + c_0 + c_1 (t - t_0)
         + c_2 (t - t_0)^2 + c_3 (t - t_0)^3 ,

  with :math:`\Phi` = ``channel_accumulated_phase`` and :math:`c_0` to
  :math:`c_3` = ``channel_phase_polynomial_coefficient_0`` to ``_3``.
* **Origin** :math:`t_0`: the second boundary, TIME TAG SECOND OF DAY
  (§5.2.6.2, Table 5-1; DSN 0222 Table 3-1 and §3.4 say the same
  [DSN0222]_). The first sample is picoseconds after it.

The phase after downconversion is the phase before it minus the
downconverter phase (§5.2.6.1), so a line at baseband frequency
:math:`f_\mathrm{bb}` has the frequency before downconversion (the sky
frequency of DSN radio science)

.. math::

   f_\mathrm{before} = f_\mathrm{DC} + f_\mathrm{bb} .

API
---

All are header-only: no samples are read. ``scan.prd`` below is the fixture
of the :doc:`../quickstart`.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Property
     - Value
   * - :attr:`rdefpy.Record.downconversion_frequency`,
       :attr:`rdefpy.ProductFile.downconversion_frequency`
     - Total :math:`f_\mathrm{DC}` at every sample, Hz. The file view is
       indexed like ``pf.samples``.
   * - :attr:`rdefpy.Record.downconversion_phase`,
       :attr:`rdefpy.ProductFile.downconversion_phase`
     - :math:`\varphi_\mathrm{DC,var}` at every sample, turns: the variable
       part only, so not the integral of the total frequency.
   * - :attr:`rdefpy.ProductFile.record_downconversion_frequency`,
       :attr:`rdefpy.ProductFile.record_downconversion_phase`
     - One value per record, at its first sample. The frequency is what
       ``band``, ``min_downconv_hz`` and ``max_downconv_hz`` compare.
   * - :attr:`rdefpy.ProductFile.mean_downconversion_frequency`
     - One value per record: the mean of :math:`f_\mathrm{DC}` over the
       record's second, to be tagged at the record midpoint
       (:doc:`samples_and_time`). It is not the frequency at the midpoint
       when :math:`c_3 \neq 0`.

.. code-block:: python

   from rdefpy import read_product

   pf = read_product("scan.prd")                 # fixed downconversion
   f = pf.downconversion_frequency[1998:2002]    # Hz, across the record boundary
   phi = pf.downconversion_phase[1998:2002]      # turns
   print(pf.record_downconversion_frequency)
   print(pf.mean_downconversion_frequency)

.. code-block:: text

   [8.405e+09 8.405e+09]
   [8.405e+09 8.405e+09]

Added to a baseband estimate over the same second, the mean gives the mean
frequency before downconversion over the record;
:func:`rdefpy.extract_carrier` does this for any block length
(:doc:`carrier`).

Both fixed header fields must be set, or evaluating a frequency raises
:class:`ValueError`. Phase is computed in float64: when :math:`\Phi` is
large, take the fraction from the polynomial terms alone.

Another origin: t0_offset
-------------------------

The polynomial starts at the record's whole second, TIME TAG SECOND OF DAY
(§5.2.6.2). A producer may refer it to another instant instead, such as the
first sample. ``t0_offset`` (seconds; not given: origin at the whole second
of the time tag) moves the origin to TIME TAG SECOND OF DAY +
``t0_offset`` for every record:

.. code-block:: python

   pf = read_product("scan.prd", t0_offset=0.25)  # polynomial origin 0.25 s after the whole second

Use it only when you know the producer's origin. As a hint,
:func:`rdefpy.read_product` and :func:`rdefpy.read_headers` warn once
(:class:`RuntimeWarning`) when a record's picoseconds exceed DSN 0222's
limit of 100,000 (Table 3-1). Passing any ``t0_offset``, even 0.0, silences
the warning.

Millisecond Predict records
---------------------------

In DSN 0222 "Millisecond Predict" mode [DSN0222]_ (§2.2, Table 3-1),
:math:`c_1`, :math:`c_2` and :math:`c_3` are NaN: the downconverter follows
predicts that are not in the file. RDEFpy reads such records without a
warning and writes them, each coefficient as 0222's ``7fffffffffffffff``.
Their downconversion frequency and phase are NaN; the ``band`` /
``min_downconv_hz`` / ``max_downconv_hz`` filters compare their fixed part.
Any other NaN, and any infinity, is outside the standard (§5.2.4): the
reader warns, the writer refuses.

.. [DSN0222] DSN 820-013, module 0222-Science, "Open Loop Data Interface",
   Rev. A, 5 June 2019, JPL D-76531, Jet Propulsion Laboratory.
   https://pds-geosciences.wustl.edu/radiosciencedocs/urn-nasa-pds-radiosci_documentation/dsn_0222-science/dsn_0222-science.2019-06-05.pdf
