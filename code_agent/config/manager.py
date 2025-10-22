"""
Configuration Manager

Manages loading and validation of configuration from various sources.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import yaml


class ConfigManager:
    """
    Comprehensive configuration manager.

    Supports loading from:
    - YAML files
    - JSON files
    - Environment variables
    - Python dictionaries
    """

    def __init__(self) -> None:
        """Initialize configuration manager."""
        self.config: dict[str, Any] = {}

    def load_from_file(self, config_path: str | Path) -> dict[str, Any]:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file (YAML or JSON)

        Returns:
            Loaded configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        suffix = config_path.suffix.lower()

        try:
            if suffix in {".yaml", ".yml"}:
                with open(config_path, encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
            elif suffix == ".json":
                with open(config_path, encoding="utf-8") as f:
                    loaded = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

            if not isinstance(loaded, dict):
                raise ValueError("Configuration file must contain a dictionary")

            self.config = cast(dict[str, Any], loaded)
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}") from e

        # Resolve environment variables
        self._resolve_env_vars()

        return self.config

    def load_from_dict(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Load configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Configuration dictionary
        """
        self.config = config_dict.copy()
        self._resolve_env_vars()
        return self.config

    def load_from_env(self, prefix: str = "CODEAGENT_") -> dict[str, Any]:
        """
        Load configuration from environment variables.

        Args:
            prefix: Environment variable prefix

        Returns:
            Configuration dictionary
        """
        config: dict[str, str] = {}

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix) :].lower()
                config[config_key] = value

        self.config.update(config)
        return self.config

    def _resolve_env_vars(self) -> None:
        """Resolve environment variable references in config."""

        def resolve_value(value: Any) -> Any:
            if isinstance(value, str):
                # Replace ${VAR_NAME} with environment variable
                import re

                pattern = r"\$\{([^}]+)\}"

                def replacer(match: Any) -> str:
                    var_name = match.group(1)
                    return os.environ.get(var_name, match.group(0))

                return re.sub(pattern, replacer, value)
            if isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [resolve_value(v) for v in value]
            return value

        resolved = resolve_value(self.config)
        if not isinstance(resolved, dict):
            raise ValueError("Configuration must resolve to a dictionary")
        self.config = cast(dict[str, Any], resolved)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value: Any = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    manager = ConfigManager()
    return manager.load_from_file(config_path)
