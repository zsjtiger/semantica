"""Ontology API routes, mounted under /api/ontology.

Each endpoint group lives in its own module. They are included here in the
original registration order, which FastAPI relies on for route matching.
"""


import os
import asyncio
import ipaddress
import logging
import socket
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from typing_extensions import Literal

import sys
import types

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ...session import GraphSession
from ...dependencies import get_session
from ...utils.rdf_parser import _safe_parse_rdf


# Create router
router = APIRouter(prefix="/api/ontology", tags=["ontology"])

from . import registry  # noqa: F401
from . import wildcard  # noqa: F401
from . import _common  # noqa: F401
from . import drafts  # noqa: F401
from . import proposals  # noqa: F401
from . import versions  # noqa: F401
from . import alignment  # noqa: F401

router.include_router(registry.router)
router.include_router(wildcard.router)
router.include_router(drafts.router)
router.include_router(proposals.router)
router.include_router(versions.router)
router.include_router(alignment.router)

_SUBMODULES = (_common, registry, wildcard, drafts, proposals, versions, alignment,)


_LAST_OWNER = {}


def _owner_of(name):
    for module in _SUBMODULES:
        if name in vars(module):
            return module
    # A name removed by delattr (e.g. at the end of mock.patch) is restored to
    # the module that held it, not onto this package object.
    return _LAST_OWNER.get(name)


class _OntologyRoutesModule(types.ModuleType):
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
        for module in _SUBMODULES:
            if name in vars(module):
                _LAST_OWNER[name] = module
                delattr(module, name)
                return
        super().__delattr__(name)


sys.modules[__name__].__class__ = _OntologyRoutesModule
