import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Union

def py2dict(config_name: str, config_root: str = "configs") -> dict:
    """Convert python file to dictionary.
    The main use - config parser.
    file:
    ```
    a = 1
    b = 3
    c = range(10)
    ```
    will be converted to
    {'a':1,
     'b':3,
     'c': range(10)
    }
    """
    if not config_name or "." in config_name:
        raise ValueError("Invalid config_name. It must not be empty or contain dots.")
    
    config_path = Path(config_root) / f"{config_name}.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"No such config file: {config_path}")

    try:
        module_path = f"{config_root.replace('/', '.')}.{config_name}"
        mod = import_module(module_path)
        cfg_dict = {
            name: value for name, value in mod.__dict__.items()
            if not name.startswith("__")
        }
        return cfg_dict
    except FileNotFoundError as e:
        raise e
    except ImportError as e:
        raise ImportError(f"Cannot import module {module_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading config: {e}")
