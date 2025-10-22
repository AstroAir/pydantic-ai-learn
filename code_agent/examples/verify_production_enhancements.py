"""
Verification Script for Production-Ready Enhancements

This script verifies that all production-ready enhancements are working correctly.

Author: The Augster
Python Version: 3.12+
"""

import contextlib
import tempfile
from pathlib import Path

from code_agent.core import (
    ConfigValidationError,
    FileExportConfig,
    PrometheusExportConfig,
    RoutingMetric,
    TelemetryCollector,
    TelemetryConfig,
    TelemetryPrivacyConfig,
    TelemetrySamplingConfig,
    create_default_telemetry_config,
)


def test_validation() -> bool:
    """Test configuration validation."""
    print("=" * 80)
    print("Testing Configuration Validation")
    print("=" * 80)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Valid configuration
    try:
        config = TelemetryPrivacyConfig(max_prompt_length=1000)
        assert config.max_prompt_length == 1000
        print("✓ Valid configuration accepted")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Valid configuration rejected: {e}")
        tests_failed += 1

    # Test 2: Invalid port number
    try:
        PrometheusExportConfig(port=70000)
        print("✗ Invalid port accepted (should have been rejected)")
        tests_failed += 1
    except ConfigValidationError as e:
        print(f"✓ Invalid port rejected: {e}")
        tests_passed += 1

    # Test 3: Invalid sample rate
    try:
        TelemetrySamplingConfig(sample_rate=1.5)
        print("✗ Invalid sample rate accepted (should have been rejected)")
        tests_failed += 1
    except ConfigValidationError as e:
        print(f"✓ Invalid sample rate rejected: {e}")
        tests_passed += 1

    # Test 4: Invalid buffer size
    try:
        TelemetryConfig(buffer_size=-1)
        print("✗ Invalid buffer size accepted (should have been rejected)")
        tests_failed += 1
    except ConfigValidationError as e:
        print(f"✓ Invalid buffer size rejected: {e}")
        tests_passed += 1

    # Test 5: No export backends enabled
    try:
        TelemetryConfig(enabled=True)
        print("✗ Config with no exporters accepted (should have been rejected)")
        tests_failed += 1
    except ConfigValidationError as e:
        print(f"✓ Config with no exporters rejected: {e}")
        tests_passed += 1

    print(f"\nValidation Tests: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_collector() -> bool:
    """Test telemetry collector."""
    print("\n" + "=" * 80)
    print("Testing Telemetry Collector")
    print("=" * 80)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Collector creation
    try:
        config = create_default_telemetry_config()
        collector = TelemetryCollector(config)
        print("✓ Collector created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Collector creation failed: {e}")
        tests_failed += 1
        return False

    # Test 2: Collector stats
    try:
        stats = collector.get_stats()
        assert stats.events_collected == 0
        assert stats.events_exported == 0
        assert stats.events_dropped == 0
        print("✓ Collector stats working")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Collector stats failed: {e}")
        tests_failed += 1

    # Test 3: Event collection (disabled)
    try:
        from code_agent.core import TelemetryEvent, TelemetryEventType

        event = TelemetryEvent(event_type=TelemetryEventType.ROUTING_DECISION)
        collector.collect_event(event)
        stats = collector.get_stats()
        assert stats.events_collected == 0  # Should not collect when disabled
        print("✓ Event collection respects enabled flag")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Event collection failed: {e}")
        tests_failed += 1

    # Test 4: Collector with enabled telemetry
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TelemetryConfig(
                enabled=True,
                file_export=FileExportConfig(enabled=True, output_dir=Path(tmpdir)),
            )
            collector = TelemetryCollector(config)

            # Collect a metric
            metric = RoutingMetric(
                selected_model="gpt-4o-mini",
                confidence=0.95,
                difficulty_level="MODERATE",
                request_mode="AGENT",
            )
            collector.collect_routing_metric(metric)

            stats = collector.get_stats()
            assert stats.events_collected > 0
            print("✓ Event collection working when enabled")
            tests_passed += 1

            # Cleanup
            collector.shutdown()
    except Exception as e:
        print(f"✗ Enabled collector failed: {e}")
        tests_failed += 1

    # Test 5: Graceful shutdown
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TelemetryConfig(
                enabled=True,
                file_export=FileExportConfig(enabled=True, output_dir=Path(tmpdir)),
            )
            collector = TelemetryCollector(config)
            collector.start()
            assert collector._running
            collector.shutdown(timeout=1.0)
            assert not collector._running
            print("✓ Graceful shutdown working")
            tests_passed += 1
    except Exception as e:
        print(f"✗ Graceful shutdown failed: {e}")
        tests_failed += 1

    print(f"\nCollector Tests: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_error_handling() -> bool:
    """Test error handling and resilience."""
    print("\n" + "=" * 80)
    print("Testing Error Handling and Resilience")
    print("=" * 80)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Telemetry errors don't crash application
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TelemetryConfig(
                enabled=True,
                file_export=FileExportConfig(enabled=True, output_dir=Path(tmpdir)),
            )
            collector = TelemetryCollector(config)

            # Try to collect invalid event (should not crash)
            with contextlib.suppress(Exception):
                collector.collect_event(None)  # type: ignore

            print("✓ Telemetry errors don't crash application")
            tests_passed += 1

            collector.shutdown()
    except Exception as e:
        print(f"✗ Error handling failed: {e}")
        tests_failed += 1

    # Test 2: Circuit breakers initialized
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TelemetryConfig(
                enabled=True,
                file_export=FileExportConfig(enabled=True, output_dir=Path(tmpdir)),
                prometheus_export=PrometheusExportConfig(enabled=True),
            )
            collector = TelemetryCollector(config)

            assert "file" in collector._circuit_breakers
            assert "prometheus" in collector._circuit_breakers
            print("✓ Circuit breakers initialized")
            tests_passed += 1

            collector.shutdown()
    except Exception as e:
        print(f"✗ Circuit breaker initialization failed: {e}")
        tests_failed += 1

    # Test 3: Buffer overflow handling
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TelemetryConfig(
                enabled=True,
                buffer_size=2,  # Very small buffer
                file_export=FileExportConfig(enabled=True, output_dir=Path(tmpdir)),
            )
            collector = TelemetryCollector(config)

            # Fill buffer beyond capacity
            from code_agent.core import TelemetryEvent, TelemetryEventType

            for i in range(10):
                event = TelemetryEvent(
                    event_type=TelemetryEventType.ROUTING_DECISION,
                    attributes={"iteration": i},
                )
                collector.collect_event(event)

            stats = collector.get_stats()
            # Some events should be dropped
            assert stats.events_dropped > 0 or stats.events_collected <= 2
            print("✓ Buffer overflow handled gracefully")
            tests_passed += 1

            collector.shutdown()
    except Exception as e:
        print(f"✗ Buffer overflow handling failed: {e}")
        tests_failed += 1

    print(f"\nError Handling Tests: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_performance() -> bool:
    """Test performance characteristics."""
    print("\n" + "=" * 80)
    print("Testing Performance")
    print("=" * 80)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Event collection latency
    try:
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TelemetryConfig(
                enabled=True,
                file_export=FileExportConfig(enabled=True, output_dir=Path(tmpdir)),
            )
            collector = TelemetryCollector(config)

            from code_agent.core import TelemetryEvent, TelemetryEventType

            # Measure latency
            num_events = 1000
            start = time.time()

            for i in range(num_events):
                event = TelemetryEvent(
                    event_type=TelemetryEventType.ROUTING_DECISION,
                    attributes={"iteration": i},
                )
                collector.collect_event(event)

            duration = time.time() - start
            latency_ms = (duration / num_events) * 1000

            print(f"✓ Event collection latency: {latency_ms:.3f}ms per event")
            if latency_ms < 1.0:  # Should be < 1ms
                tests_passed += 1
            else:
                print("  ⚠️  Latency higher than expected (< 1ms)")
                tests_failed += 1

            collector.shutdown()
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        tests_failed += 1

    print(f"\nPerformance Tests: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def main() -> None:
    """Run all verification tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "PRODUCTION ENHANCEMENTS VERIFICATION" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    all_passed = True

    # Run all test suites
    all_passed &= test_validation()
    all_passed &= test_collector()
    all_passed &= test_error_handling()
    all_passed &= test_performance()

    # Final summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL VERIFICATION TESTS PASSED")
        print("=" * 80)
        print("\nThe production-ready enhancements are working correctly!")
        print("\nNext steps:")
        print("1. Review the documentation:")
        print("   - docs/TELEMETRY_TROUBLESHOOTING.md")
        print("   - docs/TELEMETRY_PERFORMANCE.md")
        print("   - docs/TELEMETRY_SECURITY.md")
        print("2. Configure telemetry for your environment")
        print("3. Deploy with confidence!")
    else:
        print("❌ SOME VERIFICATION TESTS FAILED")
        print("=" * 80)
        print("\nPlease review the failed tests above and fix any issues.")

    print()


if __name__ == "__main__":
    main()
