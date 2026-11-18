"""Serializable provenance for exported runs."""
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
import importlib.metadata
import platform
import subprocess


def json_value(obj):
    if is_dataclass(obj):
        return json_value(asdict(obj))
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_value(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def runtime_metadata():
    root = Path(__file__).resolve().parents[1]
    git = {}
    for name, args in (("commit", ["rev-parse", "HEAD"]), ("working_tree", ["status", "--porcelain"])):
        try:
            git[name] = subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            git[name] = "unavailable"
    versions = {}
    for name in ("optical-networking-gym", "numpy", "gymnasium", "streamlit", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed as distribution"
    return {"python": platform.python_version(), "platform": platform.platform(), "git": git, "versions": versions}
