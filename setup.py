#! /usr/bin/env python
import setuptools  # noqa; we are using a setuptools namespace
from numpy.distutils.core import setup

descr = """Elastic-net regularized generalized linear models."""

DISTNAME = 'pyglmnet'
DESCRIPTION = descr
MAINTAINER = 'Pavan Ramkumar'
MAINTAINER_EMAIL = 'pavan.ramkumar@gmail.com'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/glm-tools/pyglmnet.git'
VERSION = '1.0.0'

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
          ],
          platforms='any',
          packages=[
              'pyglmnet'
          ],
          )
