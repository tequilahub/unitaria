from types import GenericAlias

import numpy as np


_NOT_FOUND = object()


class cached_property:
    """
    Clone of `functools.cached_property`. Modified to hide the traceback
    """

    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                f"Cannot assign the same cached_property to two different names ({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance, owner=None):
        # This local variable tells many tracers/debuggers to skip this frame
        __tracebackhide__ = True
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError("Cannot use cached_property instance without calling __set_name__ on it.")
        try:
            cache = instance.__dict__
        except AttributeError:  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            val = self.func(instance)
            try:
                cache[self.attrname] = val
            except TypeError:
                msg = (
                    f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                    f"does not support item assignment for caching {self.attrname!r} property."
                )
                raise TypeError(msg) from None
        return val

    __class_getitem__ = classmethod(GenericAlias)


def poly_sup_norm(poly, range=(-1.0, 1.0)):
    """
    Computes the sup norm of a polynomial in the range [-1, 1]
    """

    extrema = poly.deriv().roots()
    extrema = extrema[np.logical_and(extrema > range[0], extrema < range[1])]
    extrema = extrema[np.abs(np.imag(extrema)) < 1e-6]
    extrema = np.concatenate((extrema, [range[0], range[1]]))

    return np.max(np.abs(poly(extrema)))
