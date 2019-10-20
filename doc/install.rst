============
Installation
============

Check dependencies
------------------
We currently support ``Python 3.5+``.

For the package: ``numpy>=1.11``, ``scipy>=0.17``, ``scikit-learn>=0.18``

Additionally, for examples: ``pandas>=0.20``

Both `Canopy <https://www.enthought.com/products/canopy/>`__
and `Anaconda <https://www.continuum.io/downloads>`__
ship with a recent version of all these packages.

Additionally, for development, tests and coverage: ``pytest``, ``pytest-cov``, ``coverage``, ``flake8``

Additionally, for building documentation: ``sphinx``, ``sphinx-gallery``, ``pillow``, ``numpydoc``, ``matplotlib``

In case you have other distributions of Python, you can install
the dependencies using ``pip``.

.. code-block:: bash

   pip install numpy scipy
   pip install -U scikit-learn

Get pyglmnet
------------
If you would like to install the latest stable release:

.. code-block:: bash

    pip install pyglmnet

To install the development version of ``pyglmnet``, first clone the repository

.. code-block:: bash

	pip install https://api.github.com/repos/glm-tools/pyglmnet/zipball/master
