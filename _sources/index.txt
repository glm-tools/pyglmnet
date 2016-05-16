.. pyglmnet documentation master file, created by
   sphinx-quickstart on Mon May  9 19:01:11 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================
pyglmnet |release|
==================


``pyglmnet`` is a Python library implementing elastic-net
regularized generalized linear models (GLM). Notations and approach
are followed as in `Friedman, J., Hastie, T., & Tibshirani, R. (2010) <https://core.ac.uk/download/files/153/6287975.pdf>`_.
The implementation is aim to suite users the same way along side with popular GLM
`R package <https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html>`_.
The only different is that we still use batch gradient descent rather than
coordinate gradient descent which os very fast for ``n_samples x n_features``
size up to 10000 by 1000.


Links / Examples and Resources
==============================

* Repository: https://github.com/pavanramkumar/pyglmnet
* Documentation: http://pavanramkumar.github.io/pyglmnet
* :ref:`general_examples`


Contents
========

.. toctree::
   :maxdepth: 1

   api
   resources
   developers


Installation
============

You only need ``numpy``, ``scipy`` and (optional) ``scikit-learn`` as
requiments. You can use ``pip`` to install all dependencies,

.. code-block:: bash

   pip install numpy
   pip install scipy
   pip install scikit-learn

To install ``pyglmnet``, first clone the repository

.. code-block:: bash

    git clone http://github.com/pavanramkumar/pyglmnet

Then run ``setup.py`` to install the library

.. code-block:: bash

    python setup.py develop install


Questions / Errors / Bugs
=========================

If you have a question about the code or find errors or bugs,
please `report it here <https://github.com/pavanramkumar/pyglmnet/issues>`__.
For more specific question, feel free to email directly to us.
