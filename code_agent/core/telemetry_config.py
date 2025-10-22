"""
Telemetry Configuration

Configuration classes for telemetry collection, export, and privacy.

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .telemetry_types import ExportFormat

# ============================================================================
# Validation Utilities
# ============================================================================


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""

    pass


def validate_port(port: int, name: str = "port") -> None:
    """
    Validate port number.

    Args:
        port: Port number to validate
        name: Name of the port parameter for error messages

    Raises:
        ConfigValidationError: If port is invalid
    """
    if not isinstance(port, int):
        raise ConfigValidationError(f"{name} must be an integer, got {type(port).__name__}")
    if not 1 <= port <= 65535:
        raise ConfigValidationError(f"{name} must be between 1 and 65535, got {port}")


def validate_positive_number(value: int | float, name: str = "value", allow_zero: bool = False) -> None:
    """
    Validate positive number.

    Args:
        value: Number to validate
        name: Name of the parameter for error messages
        allow_zero: Whether to allow zero

    Raises:
        ConfigValidationError: If value is invalid
    """
    if not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{name} must be a number, got {type(value).__name__}")

    if allow_zero:
        if value < 0:
            raise ConfigValidationError(f"{name} must be >= 0, got {value}")
    else:
        if value <= 0:
            raise ConfigValidationError(f"{name} must be > 0, got {value}")


def validate_sample_rate(rate: float) -> None:
    """
    Validate sample rate.

    Args:
        rate: Sample rate to validate (0.0 to 1.0)

    Raises:
        ConfigValidationError: If rate is invalid
    """
    if not isinstance(rate, (int, float)):
        raise ConfigValidationError(f"sample_rate must be a number, got {type(rate).__name__}")
    if not 0.0 <= rate <= 1.0:
        raise ConfigValidationError(f"sample_rate must be between 0.0 and 1.0, got {rate}")


def validate_path(path: Path | str, name: str = "path", must_exist: bool = False, create: bool = False) -> Path:
    """
    Validate and normalize path.

    Args:
        path: Path to validate
        name: Name of the parameter for error messages
        must_exist: Whether path must already exist
        create: Whether to create path if it doesn't exist

    Returns:
        Normalized Path object

    Raises:
        ConfigValidationError: If path is invalid
    """
    if not isinstance(path, (Path, str)):
        raise ConfigValidationError(f"{name} must be a Path or string, got {type(path).__name__}")

    path_obj = Path(path) if isinstance(path, str) else path

    # Check for path traversal attempts
    try:
        path_obj.resolve()
    except (OSError, RuntimeError) as e:
        raise ConfigValidationError(f"{name} is invalid: {e}") from e

    if must_exist and not path_obj.exists():
        raise ConfigValidationError(f"{name} does not exist: {path_obj}")

    if create and not path_obj.exists():
        try:
            path_obj.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ConfigValidationError(f"Cannot create {name}: {e}") from e

    return path_obj


def validate_url(url: str, name: str = "url") -> None:
    """
    Validate URL.

    Args:
        url: URL to validate
        name: Name of the parameter for error messages

    Raises:
        ConfigValidationError: If URL is invalid
    """
    if not isinstance(url, str):
        raise ConfigValidationError(f"{name} must be a string, got {type(url).__name__}")

    if not url:
        raise ConfigValidationError(f"{name} cannot be empty")

    # Basic URL validation
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ConfigValidationError(f"{name} must start with http:// or https://, got: {url}")


# ============================================================================
# Telemetry Configuration
# ============================================================================


@dataclass
class TelemetryPrivacyConfig:
    """
    Privacy configuration for telemetry.

    Attributes:
        enabled: Enable privacy features
        sanitize_prompts: Remove sensitive data from prompts
        sanitize_errors: Remove sensitive data from error messages
        anonymize_user_data: Anonymize user identifiers
        max_prompt_length: Maximum prompt length to log (truncate longer)
    """

    enabled: bool = True
    sanitize_prompts: bool = True
    sanitize_errors: bool = True
    anonymize_user_data: bool = True
    max_prompt_length: int = 500

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_positive_number(self.max_prompt_length, "max_prompt_length", allow_zero=False)
        if self.max_prompt_length > 100000:
            raise ConfigValidationError("max_prompt_length cannot exceed 100000 characters")


@dataclass
class TelemetrySamplingConfig:
    """
    Sampling configuration for telemetry.

    Attributes:
        enabled: Enable sampling
        sample_rate: Sample rate (0.0 to 1.0)
        always_sample_errors: Always sample error events
        min_duration_ms: Minimum duration to sample (for performance metrics)
    """

    enabled: bool = False
    sample_rate: float = 1.0
    always_sample_errors: bool = True
    min_duration_ms: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_sample_rate(self.sample_rate)
        validate_positive_number(self.min_duration_ms, "min_duration_ms", allow_zero=True)


@dataclass
class FileExportConfig:
    """
    File-based export configuration.

    Attributes:
        enabled: Enable file export
        output_dir: Output directory for telemetry files
        format: Export format (JSON or CSV)
        rotation_size_mb: File size for rotation (MB)
        max_files: Maximum number of files to keep
    """

    enabled: bool = False
    output_dir: Path = field(default_factory=lambda: Path("telemetry_data"))
    format: ExportFormat = ExportFormat.JSON
    rotation_size_mb: int = 10
    max_files: int = 10

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate and normalize path
        self.output_dir = validate_path(self.output_dir, "output_dir", must_exist=False, create=False)

        # Validate rotation size
        validate_positive_number(self.rotation_size_mb, "rotation_size_mb", allow_zero=False)
        if self.rotation_size_mb > 10000:  # 10GB max
            raise ConfigValidationError("rotation_size_mb cannot exceed 10000 MB")

        # Validate max files
        validate_positive_number(self.max_files, "max_files", allow_zero=False)
        if self.max_files > 1000:
            raise ConfigValidationError("max_files cannot exceed 1000")

        # Check write permissions if enabled
        if self.enabled:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                # Test write permission
                test_file = self.output_dir / ".telemetry_write_test"
                test_file.touch()
                test_file.unlink()
            except OSError as e:
                raise ConfigValidationError(f"Cannot write to output_dir {self.output_dir}: {e}") from e


@dataclass
class PrometheusExportConfig:
    """
    Prometheus export configuration.

    Attributes:
        enabled: Enable Prometheus export
        port: Prometheus metrics port
        host: Prometheus metrics host
        path: Metrics endpoint path
    """

    enabled: bool = False
    port: int = 9090
    host: str = "localhost"
    path: str = "/metrics"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_port(self.port, "port")

        if not isinstance(self.host, str) or not self.host:
            raise ConfigValidationError("host must be a non-empty string")

        if not isinstance(self.path, str) or not self.path.startswith("/"):
            raise ConfigValidationError("path must be a string starting with /")


@dataclass
class OpenTelemetryExportConfig:
    """
    OpenTelemetry export configuration.

    Attributes:
        enabled: Enable OpenTelemetry export
        endpoint: OTLP endpoint URL
        protocol: Protocol (grpc or http/protobuf)
        headers: Additional headers for export
        insecure: Use insecure connection
    """

    enabled: bool = False
    endpoint: str = "http://localhost:4317"
    protocol: str = "grpc"
    headers: dict[str, str] = field(default_factory=dict)
    insecure: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_url(self.endpoint, "endpoint")

        if self.protocol not in ("grpc", "http/protobuf"):
            raise ConfigValidationError(f"protocol must be 'grpc' or 'http/protobuf', got '{self.protocol}'")


@dataclass
class HTTPExportConfig:
    """
    HTTP endpoint export configuration.

    Attributes:
        enabled: Enable HTTP export
        endpoint: HTTP endpoint URL
        method: HTTP method (POST or PUT)
        headers: HTTP headers
        batch_size: Number of events to batch before sending
        timeout_seconds: Request timeout
    """

    enabled: bool = False
    endpoint: str = ""
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    batch_size: int = 100
    timeout_seconds: int = 30

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.enabled:
            validate_url(self.endpoint, "endpoint")

        if self.method not in ("POST", "PUT", "PATCH"):
            raise ConfigValidationError(f"method must be POST, PUT, or PATCH, got '{self.method}'")

        validate_positive_number(self.batch_size, "batch_size", allow_zero=False)
        if self.batch_size > 10000:
            raise ConfigValidationError("batch_size cannot exceed 10000")

        validate_positive_number(self.timeout_seconds, "timeout_seconds", allow_zero=False)
        if self.timeout_seconds > 300:  # 5 minutes max
            raise ConfigValidationError("timeout_seconds cannot exceed 300")


@dataclass
class TelemetryConfig:
    """
    Complete telemetry configuration.

    Attributes:
        enabled: Enable telemetry collection
        privacy: Privacy configuration
        sampling: Sampling configuration
        file_export: File export configuration
        prometheus_export: Prometheus export configuration
        opentelemetry_export: OpenTelemetry export configuration
        http_export: HTTP export configuration
        buffer_size: Event buffer size
        flush_interval_seconds: Interval to flush events
        enable_routing_metrics: Enable routing decision metrics
        enable_enhancement_metrics: Enable prompt enhancement metrics
        enable_classification_metrics: Enable classification metrics
        enable_performance_metrics: Enable performance metrics
        enable_cost_tracking: Enable cost tracking
    """

    enabled: bool = False
    privacy: TelemetryPrivacyConfig = field(default_factory=TelemetryPrivacyConfig)
    sampling: TelemetrySamplingConfig = field(default_factory=TelemetrySamplingConfig)
    file_export: FileExportConfig = field(default_factory=FileExportConfig)
    prometheus_export: PrometheusExportConfig = field(default_factory=PrometheusExportConfig)
    opentelemetry_export: OpenTelemetryExportConfig = field(default_factory=OpenTelemetryExportConfig)
    http_export: HTTPExportConfig = field(default_factory=HTTPExportConfig)
    buffer_size: int = 1000
    flush_interval_seconds: int = 60
    enable_routing_metrics: bool = True
    enable_enhancement_metrics: bool = True
    enable_classification_metrics: bool = True
    enable_performance_metrics: bool = True
    enable_cost_tracking: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_positive_number(self.buffer_size, "buffer_size", allow_zero=False)
        if self.buffer_size > 1000000:  # 1M max
            raise ConfigValidationError("buffer_size cannot exceed 1000000")

        validate_positive_number(self.flush_interval_seconds, "flush_interval_seconds", allow_zero=False)
        if self.flush_interval_seconds > 3600:  # 1 hour max
            raise ConfigValidationError("flush_interval_seconds cannot exceed 3600")

        # Validate at least one export backend is enabled if telemetry is enabled
        if self.enabled and not any(
            [
                self.file_export.enabled,
                self.prometheus_export.enabled,
                self.opentelemetry_export.enabled,
                self.http_export.enabled,
            ]
        ):
            raise ConfigValidationError("At least one export backend must be enabled when telemetry is enabled")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "privacy": {
                "enabled": self.privacy.enabled,
                "sanitize_prompts": self.privacy.sanitize_prompts,
                "sanitize_errors": self.privacy.sanitize_errors,
                "anonymize_user_data": self.privacy.anonymize_user_data,
                "max_prompt_length": self.privacy.max_prompt_length,
            },
            "sampling": {
                "enabled": self.sampling.enabled,
                "sample_rate": self.sampling.sample_rate,
                "always_sample_errors": self.sampling.always_sample_errors,
                "min_duration_ms": self.sampling.min_duration_ms,
            },
            "file_export": {
                "enabled": self.file_export.enabled,
                "output_dir": str(self.file_export.output_dir),
                "format": self.file_export.format.value,
                "rotation_size_mb": self.file_export.rotation_size_mb,
                "max_files": self.file_export.max_files,
            },
            "prometheus_export": {
                "enabled": self.prometheus_export.enabled,
                "port": self.prometheus_export.port,
                "host": self.prometheus_export.host,
                "path": self.prometheus_export.path,
            },
            "opentelemetry_export": {
                "enabled": self.opentelemetry_export.enabled,
                "endpoint": self.opentelemetry_export.endpoint,
                "protocol": self.opentelemetry_export.protocol,
                "insecure": self.opentelemetry_export.insecure,
            },
            "http_export": {
                "enabled": self.http_export.enabled,
                "endpoint": self.http_export.endpoint,
                "method": self.http_export.method,
                "batch_size": self.http_export.batch_size,
                "timeout_seconds": self.http_export.timeout_seconds,
            },
            "buffer_size": self.buffer_size,
            "flush_interval_seconds": self.flush_interval_seconds,
            "enable_routing_metrics": self.enable_routing_metrics,
            "enable_enhancement_metrics": self.enable_enhancement_metrics,
            "enable_classification_metrics": self.enable_classification_metrics,
            "enable_performance_metrics": self.enable_performance_metrics,
            "enable_cost_tracking": self.enable_cost_tracking,
        }


# ============================================================================
# Helper Functions
# ============================================================================


def create_default_telemetry_config() -> TelemetryConfig:
    """
    Create default telemetry configuration.

    Returns:
        Default telemetry config (disabled by default)
    """
    return TelemetryConfig(
        enabled=False,
        privacy=TelemetryPrivacyConfig(enabled=True),
        sampling=TelemetrySamplingConfig(enabled=False, sample_rate=1.0),
    )


def create_file_telemetry_config(output_dir: Path | str = "telemetry_data") -> TelemetryConfig:
    """
    Create telemetry config with file export enabled.

    Args:
        output_dir: Output directory for telemetry files

    Returns:
        Telemetry config with file export
    """
    return TelemetryConfig(
        enabled=True,
        file_export=FileExportConfig(
            enabled=True,
            output_dir=Path(output_dir) if isinstance(output_dir, str) else output_dir,
        ),
    )


def create_prometheus_telemetry_config(port: int = 9090) -> TelemetryConfig:
    """
    Create telemetry config with Prometheus export enabled.

    Args:
        port: Prometheus metrics port

    Returns:
        Telemetry config with Prometheus export
    """
    return TelemetryConfig(
        enabled=True,
        prometheus_export=PrometheusExportConfig(
            enabled=True,
            port=port,
        ),
    )


def create_opentelemetry_config(endpoint: str = "http://localhost:4317") -> TelemetryConfig:
    """
    Create telemetry config with OpenTelemetry export enabled.

    Args:
        endpoint: OTLP endpoint URL

    Returns:
        Telemetry config with OpenTelemetry export
    """
    return TelemetryConfig(
        enabled=True,
        opentelemetry_export=OpenTelemetryExportConfig(
            enabled=True,
            endpoint=endpoint,
        ),
    )


__all__ = [
    "TelemetryPrivacyConfig",
    "TelemetrySamplingConfig",
    "FileExportConfig",
    "PrometheusExportConfig",
    "OpenTelemetryExportConfig",
    "HTTPExportConfig",
    "TelemetryConfig",
    "create_default_telemetry_config",
    "create_file_telemetry_config",
    "create_prometheus_telemetry_config",
    "create_opentelemetry_config",
]
