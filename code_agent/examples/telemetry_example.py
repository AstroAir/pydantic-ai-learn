"""
Telemetry System Examples

Demonstrates how to use the telemetry system with the routing system.

Author: The Augster
Python Version: 3.12+
"""

from pathlib import Path

from code_agent.core import (
    TelemetryConfig,
    create_code_agent,
    create_default_routing_config,
    create_default_telemetry_config,
    create_file_telemetry_config,
    create_opentelemetry_config,
    create_prometheus_telemetry_config,
)


def example_1_basic_telemetry() -> None:
    """Example 1: Basic telemetry with file export."""
    print("=" * 80)
    print("Example 1: Basic Telemetry with File Export")
    print("=" * 80)

    # Create telemetry config with file export
    telemetry_config = create_file_telemetry_config(output_dir="telemetry_data")

    # Create routing config
    routing_config = create_default_routing_config()
    routing_config.enabled = True
    routing_config.telemetry_config = telemetry_config

    # Create agent with telemetry
    create_code_agent(
        model="openai:gpt-4o-mini",
        routing_config=routing_config,
    )

    print("✓ Agent created with telemetry enabled")
    print(f"  Telemetry output: {telemetry_config.file_export.output_dir}")
    print(f"  Format: {telemetry_config.file_export.format.value}")
    print()


def example_2_prometheus_telemetry() -> None:
    """Example 2: Prometheus metrics export."""
    print("=" * 80)
    print("Example 2: Prometheus Metrics Export")
    print("=" * 80)

    # Create telemetry config with Prometheus export
    telemetry_config = create_prometheus_telemetry_config(port=9090)

    # Create routing config
    routing_config = create_default_routing_config()
    routing_config.enabled = True
    routing_config.telemetry_config = telemetry_config

    # Create agent
    create_code_agent(
        model="openai:gpt-4o-mini",
        routing_config=routing_config,
    )

    print("✓ Agent created with Prometheus export")
    print(
        f"  Metrics endpoint: http://{telemetry_config.prometheus_export.host}:{telemetry_config.prometheus_export.port}{telemetry_config.prometheus_export.path}"
    )
    print()


def example_3_opentelemetry() -> None:
    """Example 3: OpenTelemetry export."""
    print("=" * 80)
    print("Example 3: OpenTelemetry Export")
    print("=" * 80)

    # Create telemetry config with OpenTelemetry export
    telemetry_config = create_opentelemetry_config(endpoint="http://localhost:4317")

    # Create routing config
    routing_config = create_default_routing_config()
    routing_config.enabled = True
    routing_config.telemetry_config = telemetry_config

    # Create agent
    create_code_agent(
        model="openai:gpt-4o-mini",
        routing_config=routing_config,
    )

    print("✓ Agent created with OpenTelemetry export")
    print(f"  OTLP endpoint: {telemetry_config.opentelemetry_export.endpoint}")
    print(f"  Protocol: {telemetry_config.opentelemetry_export.protocol}")
    print()


def example_4_custom_telemetry() -> None:
    """Example 4: Custom telemetry configuration."""
    print("=" * 80)
    print("Example 4: Custom Telemetry Configuration")
    print("=" * 80)

    from code_agent.core import (
        FileExportConfig,
        TelemetryPrivacyConfig,
        TelemetrySamplingConfig,
    )

    # Create custom telemetry config
    telemetry_config = TelemetryConfig(
        enabled=True,
        # Privacy settings
        privacy=TelemetryPrivacyConfig(
            enabled=True,
            sanitize_prompts=True,
            sanitize_errors=True,
            anonymize_user_data=True,
            max_prompt_length=500,
        ),
        # Sampling settings
        sampling=TelemetrySamplingConfig(
            enabled=True,
            sample_rate=0.5,  # Sample 50% of events
            always_sample_errors=True,
            min_duration_ms=10.0,  # Only sample operations > 10ms
        ),
        # File export
        file_export=FileExportConfig(
            enabled=True,
            output_dir=Path("custom_telemetry"),
            rotation_size_mb=5,
            max_files=20,
        ),
        # Feature flags
        enable_routing_metrics=True,
        enable_enhancement_metrics=True,
        enable_classification_metrics=True,
        enable_performance_metrics=True,
        enable_cost_tracking=True,
    )

    # Create routing config
    routing_config = create_default_routing_config()
    routing_config.enabled = True
    routing_config.telemetry_config = telemetry_config

    # Create agent
    create_code_agent(
        model="openai:gpt-4o-mini",
        routing_config=routing_config,
    )

    print("✓ Agent created with custom telemetry")
    print(f"  Privacy: {telemetry_config.privacy.enabled}")
    print(f"  Sampling: {telemetry_config.sampling.sample_rate * 100}%")
    print(f"  Routing metrics: {telemetry_config.enable_routing_metrics}")
    print(f"  Cost tracking: {telemetry_config.enable_cost_tracking}")
    print()


def example_5_disabled_telemetry() -> None:
    """Example 5: Telemetry disabled (default)."""
    print("=" * 80)
    print("Example 5: Telemetry Disabled (Default)")
    print("=" * 80)

    # Create default telemetry config (disabled)
    telemetry_config = create_default_telemetry_config()

    # Create routing config
    routing_config = create_default_routing_config()
    routing_config.enabled = True
    routing_config.telemetry_config = telemetry_config

    # Create agent
    create_code_agent(
        model="openai:gpt-4o-mini",
        routing_config=routing_config,
    )

    print("✓ Agent created with telemetry disabled")
    print(f"  Telemetry enabled: {telemetry_config.enabled}")
    print(f"  Privacy protection: {telemetry_config.privacy.enabled}")
    print()


def example_6_multiple_exporters() -> None:
    """Example 6: Multiple telemetry exporters."""
    print("=" * 80)
    print("Example 6: Multiple Telemetry Exporters")
    print("=" * 80)

    from code_agent.core import (
        FileExportConfig,
        HTTPExportConfig,
        PrometheusExportConfig,
    )

    # Create telemetry config with multiple exporters
    telemetry_config = TelemetryConfig(
        enabled=True,
        # File export
        file_export=FileExportConfig(
            enabled=True,
            output_dir=Path("telemetry_data"),
        ),
        # Prometheus export
        prometheus_export=PrometheusExportConfig(
            enabled=True,
            port=9090,
        ),
        # HTTP export
        http_export=HTTPExportConfig(
            enabled=True,
            endpoint="https://telemetry.example.com/api/events",
            method="POST",
            headers={"Authorization": "Bearer YOUR_TOKEN"},
            batch_size=100,
        ),
    )

    # Create routing config
    routing_config = create_default_routing_config()
    routing_config.enabled = True
    routing_config.telemetry_config = telemetry_config

    # Create agent
    create_code_agent(
        model="openai:gpt-4o-mini",
        routing_config=routing_config,
    )

    print("✓ Agent created with multiple exporters")
    print(f"  File export: {telemetry_config.file_export.enabled}")
    print(f"  Prometheus export: {telemetry_config.prometheus_export.enabled}")
    print(f"  HTTP export: {telemetry_config.http_export.enabled}")
    print()


def example_7_telemetry_config_dict() -> None:
    """Example 7: Telemetry configuration as dictionary."""
    print("=" * 80)
    print("Example 7: Telemetry Configuration as Dictionary")
    print("=" * 80)

    # Create telemetry config
    telemetry_config = create_file_telemetry_config()

    # Convert to dictionary
    config_dict = telemetry_config.to_dict()

    print("Telemetry configuration:")
    print(f"  Enabled: {config_dict['enabled']}")
    print(f"  Privacy enabled: {config_dict['privacy']['enabled']}")
    print(f"  Sanitize prompts: {config_dict['privacy']['sanitize_prompts']}")
    print(f"  File export: {config_dict['file_export']['enabled']}")
    print(f"  Output dir: {config_dict['file_export']['output_dir']}")
    print(f"  Routing metrics: {config_dict['enable_routing_metrics']}")
    print(f"  Cost tracking: {config_dict['enable_cost_tracking']}")
    print()


def main() -> None:
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "TELEMETRY SYSTEM EXAMPLES" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    examples = [
        example_1_basic_telemetry,
        example_2_prometheus_telemetry,
        example_3_opentelemetry,
        example_4_custom_telemetry,
        example_5_disabled_telemetry,
        example_6_multiple_exporters,
        example_7_telemetry_config_dict,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"✗ Example failed: {e}")
            print()

    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Enable telemetry in your agent configuration")
    print("2. Choose appropriate export backend(s)")
    print("3. Configure privacy and sampling settings")
    print("4. Monitor metrics and optimize routing decisions")
    print()


if __name__ == "__main__":
    main()
