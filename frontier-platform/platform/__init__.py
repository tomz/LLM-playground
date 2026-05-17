"""Frontier model training platform — architecture-only skeleton.

This package name collides with the stdlib ``platform`` module. Tooling
(pytest's terminal reporter, pip, etc.) calls e.g. ``platform.python_version()``
during startup, so we transparently proxy any missing attribute to the real
stdlib module by loading it from its file location.
"""
from __future__ import annotations

__version__ = "0.0.1-blueprint"


def _load_stdlib_platform():
    import importlib.util, sys, sysconfig, os
    stdlib_dir = sysconfig.get_paths()["stdlib"]
    path = os.path.join(stdlib_dir, "platform.py")
    spec = importlib.util.spec_from_file_location("_stdlib_platform", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_stdlib_platform"] = mod
    spec.loader.exec_module(mod)
    return mod


_stdlib = _load_stdlib_platform()


def __getattr__(name):
    # Proxy any missing attribute to stdlib platform.
    return getattr(_stdlib, name)
