=======================
Developer Documentation
=======================

We could use help with a number of ongoing issues. If you'd like to help,
check the `issue tree <https://github.com/glm-tools/pyglmnet/issues>`_
as well as our
`requests for pull requests <https://glm-tools.github.io/pyglmnet/requests.html>`_.

If you've decided to make a pull request, here is an outline of the
development workflow that we have successfully adopted.

Setting up
----------

Fork the `repository <https://github.com/glm-tools/pyglmnet>`_
to your own account (i.e. github user).

Clone or pull your forked repository locally. If you are doing it for the
first time:

.. code:: bash

  git clone https://github.com/<your-user-name>/pyglmnet

Or if you have already cloned the repository:

.. code:: bash

  git pull origin master

Setup a remote to point to the main upstream repository.

.. code:: bash

  git remote add upstream https://github.com/glm-tools/pyglmnet
  git remote -v

You should see this:

.. code:: console

  upstream	  https://github.com/glm-tools/pyglmnet.git (fetch)
  upstream	  https://github.com/glm-tools/pyglmnet.git (push)
  origin	http://github.com/<your-user-name>/pyglmnet.git (fetch)
  origin	http://github.com/<your-user-name>/pyglmnet.git (push)

Before you start developing a feature, make sure that your local master is
up to date. This can save you from a lot of merge conflicts later.

.. code:: bash

  git checkout master
  git pull upstream master

In case your own fork is behind the main repo, update your fork:

.. code:: bash

  git push origin master

Once you are done with these housekeeping steps, you are ready to start
developing.

Develop
-------
Make sure you develop each feature on a new branch:

.. code:: bash

  git checkout -b feat

Develop your changes, and once you are satisfied we need to do
a couple of things before adding and committing them.

First if it is a major feature, consider writing a test. You can do this by
editing ``tests/test_pyglmnet.py``.

Second, once you have written your tests, run them locally.
Install ``pytest`` and ``pytest-cov``:

.. code:: bash

  pip install pytest pytest-cov


.. code:: bash

  py.test --cov=pyglmnet tests/

If you don't see error messages, go ahead and test with a pep8 style checker.
We use flake8.  Install ``flake8``:

.. code:: bash

  pip install flake8
  

.. code:: bash

  flake8 --count pyglmnet

If you don't see any errors, you are good to add and commit.

Add all the files changed and commit them with a short and meaningful message.
We recommend one commit per conceptual change since it helps us keep track
of what happened more easily.

Note: If you are making changes to the documentation, you will see a number
of new files built when you locally build the documentation including the
folders: `_build`, `auto_examples`, `generated`, `modules`, and `tutorials`.
DO NOT add or commit any of these! Only add and commit the files you manually
changed (typically `.rst` or `.py` files). Once a pull request is made and
merged, we will build the documentation to be hosted separately (see below).

Once committed, push your local branch to a branch in your fork.

.. code:: bash

  git push origin feat:feat

Make pull request
-----------------
From the ``feat`` branch of your fork: https://github.com/<your-user-name>/pyglmnet
you can create a pull request on to the main repo. Give the PR a meaningful
name. We recommend prefixing it with a ``[WIP]`` if the feature is being built.
If you think it is ready to merge, prefix with ``[MRG]``.

If it's a complicated feature that can evolve better with feedback, we highly
recommend making the PR when it's a work in progress (WIP). In the PR message box,
it's typically good to associate it with an issue (.e.g. "address #253")
in addition to concisely describing the most salient changes made.

Once your PR is made, the tests will run. If there are errors, they will
be reported on the PR's page.

Major PRs are followed by a process of peer review by one of the maintainers
commenting on the code and suggesting changes.

For making changes to the PR, make changes to your local ``feat`` branch
and push to your fork's ``feat`` branch just as you did before making the PR.
Your new commits will be automatically associated with the PR and tested.

Sometimes you may make tiny formatting changes that are not worth retesting
with our continuous integration systems. For these changes, include a ``[ci skip]``
prefix in your commit message. However, use this trick sparingly!

After all suggested changes are resolved, add your name to the `whats_new`
page in the appropriate section. This should be the last commit in the PR
before it can be merged.

Once the PR is merged, you can optionally delete the ``feat`` branch both
locally and on your fork.

Build documentation
------------------
The following should be installed in order to build the documentation.

*    `sphinx <https://github.com/sphinx-doc/sphinx/>`_
*    `sphinx-gallery <https://github.com/sphinx-gallery/sphinx-gallery/>`_
*    `pillow <https://github.com/python-pillow/Pillow/>`_
*    `numpydoc <https://github.com/numpy/numpydoc/>`_
*    `matplotlib <https://github.com/matplotlib/matplotlib/>`_

Shortcut:

.. code:: bash

  pip install sphinx sphinx-gallery pillow numpydoc matplotlib

We use ``sphinx`` to generate documentation page.
To build the documentation pages locally,
run::

    make html

All static files will be built in ``_build/html/``
where you can open them using the web browser.

To remove the built files from your local repository, run::

    make clean

To push built documentation page to ``gh-pages``, simply run::

    make install
