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
The following should be installed in order to build the documentation.

*    `sphinx <https://github.com/sphinx-doc/sphinx/>`_
*    `sphinx-gallery <https://github.com/sphinx-gallery/sphinx-gallery/>`_
*    `pillow <https://github.com/python-pillow/Pillow/>`_
*    `matplotlib <https://github.com/matplotlib/matplotlib/>`_

Shortcut: ``pip install sphinx sphinx-gallery pillow matplotlib``

Some dependencies are necessary to build the documentation (i.e. ``sphinx``, ``sphinx-gallery`` and ``matplotlib``). At the root project folder, you can simply run::

    make doc-dependencies

We use ``sphinx`` to generate documentation page. To build documentation page, run::

    make doc

All static files will be built in ``_build/html/`` where you can open them using the web browser. To do so, run::

    make doc-run

To remove the built files from your local repository, run::

    make clean

To push built documentation page to ``gh-pages``, simply run::

    make doc-publish
