API reference
=============

Everything below is importable from the top-level package,
``from rdefpy import ...``, except the ``rdefpy.carrier`` estimators, which
are imported from that module.

.. currentmodule:: rdefpy

Product files (§5)
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   read_product
   read_headers
   scan_product
   write_product
   pack_record_header
   pack_record_data
   ProductFile
   Record
   RecordHeader
   RecordData

.. data:: DSN_AGENCY_DTYPE

   NumPy structured dtype (40 bytes) of the DSN OLR internal-agency block,
   DSN 820-013 0222-Science Rev. A Table 3-1: used by :meth:`RecordHeader.agency_fields` and
   :meth:`ProductFile.agency_fields`
   with ``agency_layout="dsn"`` (or ``"auto"`` on ``agency_flag`` 3). See
   :doc:`guide/product_files`.

Observation files (§4)
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   read_observation
   ObservationFile
   Scan
   ProductEntry

Carrier extraction
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   extract_carrier

The per-block estimators behind it, for use on your own sample arrays
(``from rdefpy.carrier import ...``):

.. autosummary::
   :toctree: generated
   :nosignatures:

   carrier.frft_frequency
   carrier.spi_frequency

.. data:: rdefpy.carrier.CARRIER_DTYPE

   NumPy structured dtype of the array returned by :func:`extract_carrier`:
   ``time``, ``frequency_before_downconversion``, ``frequency``, ``rate``,
   ``downconversion_frequency``. See :doc:`guide/carrier`.

Utilities
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   browse
   set_threads

.. data:: __version__

   The installed RDEFpy version, as a string (from the package metadata).
