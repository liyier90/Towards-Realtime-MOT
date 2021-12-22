from pathlib import Path
from typing import Any, Dict, List


def parse_model_cfg(config_path: Path) -> List[Dict[str, Any]]:
    """Parse model configuration with context manager"""
    with open(config_path) as infile:
        lines = [
            line
            for line in map(str.strip, infile.readlines())
            if line and not line.startswith("#")
        ]
    module_defs: List[Dict[str, Any]] = []
    for line in lines:
        if line.startswith("["):
            module_defs.append({})
            module_defs[-1]["type"] = line[1:-1].rstrip()
            if module_defs[-1]["type"] == "convolutional":
                module_defs[-1]["batch_normalize"] = 0
        else:
            key, value = tuple(map(str.strip, line.split("=")))
            if value.startswith("$"):
                value = module_defs[0].get(value.strip("$"), None)
            module_defs[-1][key] = value
    return module_defs


def parse_data_cfg(path):
    """Parses the data configuration file"""
    options = dict()
    options["gpus"] = "0"
    options["num_workers"] = "10"
    with open(path, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue
        key, value = line.split("=")
        options[key.strip()] = value.strip()
    return options
