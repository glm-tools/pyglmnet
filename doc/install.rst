============
Installation
============

Check dependencies
------------------
You only need ``NumPy``, ``SciPy`` for the package and additionally ``scikit-learn``
if you would like to run the examples. We currently support
``Python 2.7``, ``Python 3.5``
``scikit-learn >= 0.18``,
``NumPy >= 1.11`` and ``SciPy >= 0.17``.

Both `Canopy <https://www.enthought.com/products/canopy/>`__
and `Anaconda <https://www.continuum.io/downloads>`__
ship with a recent version of all these packages.

In case you have other distributions of Python, you can install
``NumPy`` and ``SciPy`` using ``pip``.

.. code-block:: bash

   pip install numpy
   pip install scipy

To install ``scikit-learn`` follow these
`installation instructions <http://scikit-learn.org/stable/install.html>`__.

Get pyglmnet
------------
If you would like to install the latest stable release:

.. code-block:: bash

    pip install pyglmnet

To install the development version of ``pyglmnet``, first clone the repository

.. code-block:: bash

    git clone http://github.com/glm-tools/pyglmnet

Then run ``setup.py`` to install the library

.. code-block:: bash

    cd pyglmnet
    python setup.py develop install
