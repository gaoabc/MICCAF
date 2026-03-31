from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigNode(dict):
    def __getattr__(self, name: str) -> Any:
        if name not in self:
            raise AttributeError(name)
        value = self[name]
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            value = ConfigNode(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def clone(self) -> "ConfigNode":
        return ConfigNode(deepcopy(dict(self)))


DEFAULT_OVERRIDES = {
    "aublations": {
        "disable_iic": False,
        "disable_mmi": False,
        "disable_iaf": False,
    }
}


def _recursive_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _recursive_update(base[key], value)
        else:
            base[key] = value
    return base



def load_config(path: str | Path) -> ConfigNode:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _recursive_update(deepcopy(DEFAULT_OVERRIDES), cfg)
    return ConfigNode(cfg)
