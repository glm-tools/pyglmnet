"""
The :mod:`sklearn.exceptions` module includes all custom warnings and error
classes used across scikit-learn.
"""

__all__ = ['NotFittedError',
           'NonBLASDotWarning']


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    Examples
    --------
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.exceptions import NotFittedError
    >>> try:
    ...     LinearSVC().predict([[1, 2], [2, 3], [3, 4]])
    ... except NotFittedError as e:
    ...     print(repr(e))
    NotFittedError("This LinearSVC instance is not fitted yet. Call 'fit' with
    appropriate arguments before using this estimator."...)
    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.
    """


class EfficiencyWarning(UserWarning):
    """Warning used to notify the user of inefficient computation.
    This warning notifies the user that the efficiency may not be optimal due
    to some reason which may be included as a part of the warning message.
    This may be subclassed into a more specific Warning class.
    .. versionadded:: 0.18
    """


class NonBLASDotWarning(EfficiencyWarning):
    """Warning used when the dot operation does not use BLAS.
    This warning is used to notify the user that BLAS was not used for dot
    operation and hence the efficiency may be affected.
    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation, extends EfficiencyWarning.
    """
