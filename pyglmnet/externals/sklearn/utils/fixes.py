"""Compatibility fixes for older version of python, numpy and scipy
If you add content to this file, please give the version of the package
at which the fixe is no longer needed.
"""
# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Fabian Pedregosa <fpedregosa@acm.org>
#          Lars Buitinck
#
# License: BSD 3 clause

import numpy as np


def _parse_version(version_string):
    version = []
    for x in version_string.split('.'):
        try:
            version.append(int(x))
        except ValueError:
            # x may be of the form dev-1ea1592
            version.append(x)
    return tuple(version)


np_version = _parse_version(np.__version__)

if np_version < (1, 13):
    def _object_dtype_isnan(X):
        return np.frompyfunc(lambda x: x != x, 1, 1)(X).astype(bool)
else:
    def _object_dtype_isnan(X):
        return X != X
