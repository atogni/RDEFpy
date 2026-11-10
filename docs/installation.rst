Installation
============

RDEFpy needs Python 3.11 or later and NumPy (1.26 or later). It has no
other runtime dependency. The record browser, :func:`rdefpy.browse`, also
needs matplotlib.

.. code-block:: console

   pip install rdefpy                                  # from PyPI
   pip install "rdefpy[browse]"                        # with matplotlib
   pip install git+https://github.com/atogni/RDEFpy    # latest, from GitHub

.. code-block:: python

   import rdefpy
   print(rdefpy.__version__)

The Cython extension
--------------------

The prebuilt wheels include an optional Cython extension that speeds up I/Q
unpacking and carrier extraction. Where no wheel matches your platform, pip
builds from source, which needs a C compiler: Microsoft C++ Build Tools on
Windows, ``gcc`` on Linux, the Xcode command-line tools on macOS. Without
one the build prints a warning and installs the NumPy path only. Both paths
give the same results; the tests run on each. To check that the extension
was built:

.. code-block:: console

   python -c "import rdefpy.prd.unpack_cyth"

No error means it is in use; ``ImportError`` means the NumPy path.
