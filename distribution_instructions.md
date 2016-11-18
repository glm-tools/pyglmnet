# Instructions for distributing `pyglmnet`

### Prerequisites
Make sure you have the following packages installed locally:
  `twine`

Create a file called `.pypirc` **in the project directory root.** This file is
included in the `.gitignore` file, as it contains sensitive info. The file
generally is stuctured like this:

```
[distutils]
index-servers =
  pypi
  pypitest

[pypi]
repository: https://pypi.python.org/pypi
username: organization-username
password: organization-password

[pypitest]
repository: https://testpypi.python.org/pypi
username: organization-username
password: organization-password
```

Obviously, make sure you're using the proper passwords and usernames.

### Setup

Update the `setup.py` file as needed with new versions, authors, packages etc.
Then in the root file of the directory, run:
```bash
$ python setup.py sdist bdist_wheel
```
Which will create the standard distribution files.

### Registration and uploading
This is where we will now use `twine` and the custom `.pypirc` file in the root
of the project directory. First, register the package to the `pypitest` server.

```bash
$ twine register -r pypitest --config-file=.pypirc dist/pyglmnet-<verion number>.tar.gz
```

Next, upload.

```bash
$ twine upload -r pypitest --config-file=.pypirc dist/pyglmnet-<version number>.tar.gz
```

To make sure the test worked, head over to the [PyPI test page](https://testpypi.python.org/pypi).
You should see `pyglmnet <version number>` near the top of the list (if you head directly over).

Great, looks like it worked. Let's register, upload and release for real!

```bash
$ twine register -r pypi --config-file=.pypirc dist/pyglmnet-<version number>.tar.gz
```
Finally, upload!

```bash
$ twine upload -r pypi --config-file=.pypirc dist/pyglmnet-<version number>.tar.gz
```
