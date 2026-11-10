Contributing
============

Issues and pull requests are welcome at
https://github.com/atogni/RDEFpy.

The low-level decoding (the ``2k+1`` transform of §5.3.6, the bit offsets,
the header struct) is the format itself and is meant to stay as it is;
changes there need a reference to the standard. Performance improvements in
decoding, reading or processing data are welcome.

Development setup
-----------------

The project uses `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: console

   git clone https://github.com/atogni/RDEFpy
   cd RDEFpy
   uv sync                       # builds the Cython extension if a compiler is found
   uv run python -m pytest

The tests pin the low-level decoding for every sample size against the
standard's bit layout, and run both with and without the Cython extension
in continuous integration.

``tests/reference/`` holds small product files with their expected
decoding as text, written without RDEFpy; see :doc:`reference_files`.

Building this documentation
---------------------------

.. code-block:: console

   uv run --group docs python -m sphinx -W --keep-going -b html docs docs/_build/html

Then open ``docs/_build/html/index.html``.
