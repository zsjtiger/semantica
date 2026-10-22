"""Semantica command-line interface.

The root ``main`` group lives in ``_common``; each command group is a module
that registers its commands on it when imported. Helpers are looked up on
the module that defines them, so reads and writes through this package
reach the live definition.
"""

import sys
import types

from . import (
    _common,
    core_cmd,
    kg_cmd,
    pipeline_cmd,
    services_cmd,
    embed_cmd,
    reason_cmd,
    decision_cmd,
    temporal_cmd,
    provenance_cmd,
    validate_cmd,
    ontology_cmd,
    visualize_cmd,
    store_cmd,
    backup_cmd,
)  # noqa: F401  (importing registers the commands)
from ._common import main  # noqa: F401

_SUBMODULES = (
    _common,
    core_cmd,
    kg_cmd,
    pipeline_cmd,
    services_cmd,
    embed_cmd,
    reason_cmd,
    decision_cmd,
    temporal_cmd,
    provenance_cmd,
    validate_cmd,
    ontology_cmd,
    visualize_cmd,
    store_cmd,
    backup_cmd,
)


class _CLIModule(types.ModuleType):
    def __getattr__(self, name):
        for module in _SUBMODULES:
            if name in vars(module):
                return vars(module)[name]
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        for module in _SUBMODULES:
            if name in vars(module):
                setattr(module, name, value)
                return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _CLIModule
