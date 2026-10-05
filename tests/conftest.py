import copy
from pathlib import Path

import pytest
import yaml

from bot.config import config_from_dict

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw_cfg() -> dict:
    with open(ROOT / "config" / "config.example.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def cfg(raw_cfg):
    return config_from_dict(copy.deepcopy(raw_cfg))
