=======================
Developer Documentation
=======================

Contributing
------------

We welcome any pull requests.

* Fork the `repository <https://github.com/pavanramkumar/pyglmnet>`_
* Develop and push to your branch
* Create new pull requests

You can run ``nosetests tests`` before for making pull requests
to ensure that the changes work. We are continuously adding tests
with more coverage.


Building Documentation
----------------------
The following is needed in order to build the documentation.

*    ``pip install sphinx``
*    ``pip install sphinx-gallery``
*    ``pip install Image``
*    ``pip install matplotlib``

We use ``sphinx`` to generate documentation page. To build documentation page, run::

    make html

All static files will be built in ``_build/html/`` where you can open them using the web browser.

To remove the built files from your local repository, run::

    make clean

To push built documentation page to ``gh-pages``, simply run::

    make install
