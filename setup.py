#! /usr/bin/env python
import setuptools  # noqa; we are using a setuptools namespace
import os


def get_version():
    """
    Get the version without importing, so as not to invoke dependency
    requirements.
    """
    base, _ = os.path.split(os.path.realpath(__file__))
    file = os.path.join(base, "pyglmnet", "__init__.py")

    for line in open(file, "r"):
        if "__version__" in line:
            return line.split("=")[1].strip().strip("'").strip('"')


descr = """Elastic-net regularized generalized linear models."""

DISTNAME = "pyglmnet"
DESCRIPTION = descr
MAINTAINER = "Pavan Ramkumar"
MAINTAINER_EMAIL = "pavan.ramkumar@gmail.com"
LICENSE = "MIT"
URL = 'http://glm-tools.github.io/pyglmnet/'
DOWNLOAD_URL = "https://github.com/glm-tools/pyglmnet.git"
VERSION = get_version()  # Get version without importing

if __name__ == "__main__":
    setuptools.setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        long_description=open("README.rst").read(),
        long_description_content_type='text/x-rst',
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
        platforms="any",
        packages=["pyglmnet"],
        project_urls={
            'Documentation': 'http://glm-tools.github.io/pyglmnet/',
            'Bug Reports': 'https://github.com/glm-tools/pyglmnet/issues',
            'Source': 'https://github.com/glm-tools/pyglmnet',
        },
    )
