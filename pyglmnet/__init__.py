"""Pyglmnet: A python implementation of elastic-net regularized generalized linear models"""

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.devN' where N is an integer.
#

__version__ = '1.2.dev0'


from .pyglmnet import GLM, GLMCV, _grad_L2loss, _L2loss, simulate_glm, _gradhess_logloss_1d, _loss, ALLOWED_DISTRS
from .utils import softmax, label_binarizer, set_log_level
from .datasets import fetch_tikhonov_data, fetch_rgc_data, fetch_community_crime_data, fetch_group_lasso_data
from . import externals
