Installation
============

Requirements
------------

* Python **3.9** or newer.
* Runtime dependencies, installed automatically:

  .. list-table::
     :header-rows: 1
     :widths: 22 78

     * - Package
       - Used for
     * - ``numpy``
       - All array handling.
     * - ``scipy``
       - Pairwise distances (``scipy.spatial.distance.cdist``).
     * - ``scikit-learn``
       - k-means grouping in :mod:`aeroopt.analysis`.
     * - ``numexpr``
       - Evaluating constraint strings such as ``"x1 ** 2 + x2 ** 2 - 0.64"``.
     * - ``pydoe``
       - Latin hypercube sampling. The ``seed=`` argument requires the 0.9 series.
     * - ``openpyxl``
       - Exporting a database to ``.xlsx``.

From PyPI
---------

.. code-block:: bash

   pip install aeroopt

Optional extras
---------------

.. code-block:: bash

   pip install "aeroopt[surrogate]"   # smt, for aeroopt.utils.surrogate.Kriging
   pip install "aeroopt[examples]"    # matplotlib, for the example scripts
   pip install "aeroopt[docs]"        # sphinx + furo, to build this documentation
   pip install "aeroopt[tests]"       # pytest

The surrogate extra is separate because ``smt`` is a heavy dependency and is
only imported when :class:`~aeroopt.utils.surrogate.Kriging` is instantiated.
The abstract :class:`~aeroopt.utils.surrogate.SurrogateModel` interface can be
implemented against any other backend without installing it.

From source
-----------

.. code-block:: bash

   git clone https://github.com/swayli94/AeroOpt.git
   cd AeroOpt
   pip install -e ".[surrogate,examples,tests]"

Running the tests
-----------------

.. code-block:: bash

   pytest

Building the documentation
--------------------------

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs/source docs/build/html

Checking the import
-------------------

.. code-block:: python

   import aeroopt
   from aeroopt.core import Problem, Database, MultiProcessEvaluation
   from aeroopt.optimization import OptNSGAII, SettingsNSGAII
   from aeroopt.utils import benchmark

   print(aeroopt.__version__)
