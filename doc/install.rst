============
Installation
============

Check dependencies
------------------
You only need ``NumPy``, ``SciPy`` and (optional) ``scikit-learn`` as
requirements. We recommend ``Python >= 2.6``, ``scikit-learn >= 0.17``, ``NumPy >= 1.6.1`` and
``SciPy >= 0.14``.

Both `Canopy <https://www.enthought.com/products/canopy/>`__
and `Anaconda <https://www.continuum.io/downloads>`__
ship with a recent version of ``scikit-learn``.

In case you have other distributions of Python, you can install ``NumPy`` and ``SciPy``
using ``pip``.

.. code-block:: bash

   pip install numpy
   pip install scipy

To install ``scikit-learn`` follow these
`installation instructions <http://scikit-learn.org/stable/install.html>`__.



Get pyglmnet
------------
To install ``pyglmnet``, first clone the repository

.. code-block:: bash

    git clone http://github.com/pavanramkumar/pyglmnet

Then run ``setup.py`` to install the library

.. code-block:: bash

    cd pyglmnet
    python setup.py develop install
