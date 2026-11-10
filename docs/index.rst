RDEFpy
======

Open-source Python reader and writer for the CCSDS Delta-DOR Raw Data
Exchange Format (RDEF), CCSDS 506.1-B-1 (June 2013), also published as
ISO 20208:2015.

RDEFpy reads and writes the standard's product files (``.prd``, §5):
binary one-second records of complex I/Q samples with their full
downconversion metadata, as recorded by open-loop receivers for radio
science and Delta-DOR. It reads and validates the ASCII observation files
(``.obs``, §4) that index them. It depends on NumPy alone; an optional
Cython extension speeds up decoding.

.. code-block:: python

   from rdefpy import read_product

   pf = read_product("scan.prd")
   pf.info()                      # time span, sample rate, bits, band
   x = pf.samples[:1024]          # complex I + jQ, across records
   t = pf.timetags[:1024]         # datetime64[ns] of each sample

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   guide/product_files
   guide/samples_and_time
   guide/downconversion
   guide/carrier
   guide/browse
   guide/writing
   guide/observation_files
   guide/performance

.. toctree::
   :maxdepth: 2
   :caption: Reference

   examples
   api
   spec_mapping
   reference_files
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing
   citing
