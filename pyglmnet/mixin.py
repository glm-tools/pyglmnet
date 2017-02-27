class EstimatorMixin(object):
    """Mixin class for estimators."""

    def get_params(self):  # noqa: D102
        pass

    def set_params(self, **params):
        """Set parameters (mimics sklearn API)."""
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.iteritems():
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self
