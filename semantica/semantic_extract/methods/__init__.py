"""Extraction method implementations, split by kind.

Public names are re-exported from the section modules; patching a name on this package
reaches the module that defines it.
"""

import sys
import types

from . import _common, scoring, entities, relations, triplets, dispatch  # noqa: F401

_SUBMODULES = (_common, scoring, entities, relations, triplets, dispatch,)
_LAST_OWNER = {}


def _owner_of(name):
    for module in _SUBMODULES:
        if name in vars(module):
            return module
    return _LAST_OWNER.get(name)


class _ExtractionMethodsModule(types.ModuleType):
    def __getattr__(self, name):
        for module in _SUBMODULES:
            if name in vars(module):
                return vars(module)[name]
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        owner = _owner_of(name)
        if owner is None:
            super().__setattr__(name, value)
        else:
            setattr(owner, name, value)

    def __delattr__(self, name):
        # mock.patch removes a patched name and then restores the original itself.
        for module in _SUBMODULES:
            if name in vars(module):
                _LAST_OWNER[name] = module
                delattr(module, name)
                return
        super().__delattr__(name)


sys.modules[__name__].__class__ = _ExtractionMethodsModule
