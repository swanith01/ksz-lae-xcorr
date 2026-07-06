"""
utils/config.py
================
Single entry point for loading configs/*.yaml. Every script and module
should get its parameters through `load_config()` rather than hardcoding
box/cosmology/path values a second time.

Usage:
    from ksz_lae_xcorr.utils.config import load_config
    cfg = load_config("configs/fiducial.yaml")
    cfg.box.box_len_mpc      # attribute access
    cfg["box"]["box_len_mpc"]  # or dict-style
"""

from __future__ import annotations

import os
import re
from typing import Any

import yaml

_PATH_REF_RE = re.compile(r"\$\{([a-zA-Z0-9_.]+)\}")


class Config(dict):
    """Dict subclass with attribute access and nested Config wrapping."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _resolve_refs(obj: Any, root: dict) -> Any:
    """Resolve '${a.b.c}' style references against the root config dict."""
    if isinstance(obj, dict):
        return {k: _resolve_refs(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_refs(v, root) for v in obj]
    if isinstance(obj, str):
        def _sub(match: "re.Match[str]") -> str:
            key_path = match.group(1).split(".")
            node: Any = root
            for key in key_path:
                node = node[key]
            return str(node)

        # Resolve repeatedly in case a reference points to another reference.
        prev = None
        cur = obj
        while prev != cur:
            prev = cur
            cur = _PATH_REF_RE.sub(_sub, cur)
        return cur
    return obj


def _expand_paths(obj: Any) -> Any:
    """Expand '~' in any string that looks like a filesystem path."""
    if isinstance(obj, dict):
        return {k: _expand_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_paths(v) for v in obj]
    if isinstance(obj, str) and (obj.startswith("~") or obj.startswith("/")):
        return os.path.expanduser(obj)
    return obj


def load_config(path: str, overrides: dict | None = None) -> Config:
    """
    Load a YAML config, resolve ${...} cross-references, expand '~' paths,
    and apply optional dict overrides (e.g. from a configs/variants/ file
    or CLI args), then return it as an attribute-accessible Config.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if overrides:
        raw = _deep_update(raw, overrides)

    resolved = _resolve_refs(raw, raw)
    resolved = _expand_paths(resolved)
    return _wrap(resolved)


def _deep_update(base: dict, update: dict) -> dict:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base
