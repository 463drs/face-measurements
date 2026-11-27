"""Experiment configs: YAML files in `config/` loaded into pydantic models."""
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict

# src/facemeasure/config.py -> repo root. Needs an editable install (`pip install -e .`), not a copy in site-packages.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

# Relative paths in configs are resolved against the project root.
ProjectPath = Annotated[Path, AfterValidator(lambda p: p if p.is_absolute() else PROJECT_ROOT / p)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)  # typos in config keys are errors


class DataConfig(_Strict):
    paths: list[ProjectPath]


class ComponentConfig(_Strict):
    """A registry entry: `name` selects the factory, `params` are passed to it as kwargs."""
    name: str
    params: Annotated[dict[str, Any], BeforeValidator(lambda v: v or {})] = {}


class RunConfig(_Strict):
    name: str
    out_root: ProjectPath = Path("runs")
    save_overlays: int = 0
    seed: int = 0
    notes: str = ""


class Config(_Strict):
    data: DataConfig
    detector: ComponentConfig
    landmark_model: ComponentConfig
    target_selector: ComponentConfig
    run: RunConfig


def load_config(name: str | Path) -> Config:
    """Load a config by name (`"300w_fan"` -> `config/300w_fan.yaml`) or by path to a .yaml file."""
    path = Path(name)
    if path.suffix not in (".yaml", ".yml"):
        path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"config {str(name)!r} not found at {path}; available: {available}")
    return Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
