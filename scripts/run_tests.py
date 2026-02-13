"""
Comprehensive Test Runner for Real-Time Object Tracking System

Provides easy execution of:
- Unit tests for all components
- Integration tests
- Performance benchmarks
- Live visualization
- Statistics collection
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# LOGGER SETUP
# ============================================================================

def setup_logging(level=logging.INFO):
    """Configure logging for test runner."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - TestRunner - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("TestRunner")


# ============================================================================
# TEST RUNNER COMMANDS
# ============================================================================

class TestRunner:
    """Manages test execution and reporting."""
    
    def __init__(self, project_root: Path, logger: logging.Logger):
        """
        Initialize test runner.
        
        Args:
            project_root: Root directory of project
            logger: Logger instance
        """
        self.project_root = project_root
        self.logger = logger
        self.results = {}
        self.start_time = None
    
    def run_command(self, command: str, description: str) -> bool:
        """
        Run a shell command and report results.
        
        Args:
            command: Command to run
            description: Human-readable description
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Running: {description}")
        self.logger.debug(f"Command: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.project_root),
                capture_output=False,
                timeout=300
            )
            
            success = result.returncode == 0
            self.results[description] = {
                "status": "PASSED" if success else "FAILED",
                "return_code": result.returncode,
                "timestamp": datetime.now().isoformat()
            }
            
            if success:
                self.logger.info(f"✓ {description} passed")
            else:
                self.logger.error(f"✗ {description} failed (exit code: {result.returncode})")
            
            return success
        
        except subprocess.TimeoutExpired:
            self.logger.error(f"✗ {description} timed out")
            self.results[description] = {
                "status": "TIMEOUT",
                "timestamp": datetime.now().isoformat()
            }
            return False
        
        except Exception as e:
            self.logger.error(f"✗ {description} error: {e}")
            self.results[description] = {
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            return False
    
    def run_unit_tests(self) -> bool:
        """Run unit tests."""
        self.logger.info("\n" + "="*70)
        self.logger.info("RUNNING UNIT TESTS")
        self.logger.info("="*70)
        
        return self.run_command(
            f'"{sys.executable}" -m pytest tests/test_tracking_pipeline.py -v',
            "Unit Tests (pytest)"
        )
    
    def run_unittest_suite(self) -> bool:
        """Run unittest suite."""
        self.logger.info("\n" + "="*70)
        self.logger.info("RUNNING UNITTEST SUITE")
        self.logger.info("="*70)
        
        return self.run_command(
            f'"{sys.executable}" tests/test_tracking_pipeline.py',
            "Unittest Suite"
        )
    
    def run_live_detection(self) -> bool:
        """Run live detection test."""
        self.logger.info("\n" + "="*70)
        self.logger.info("RUNNING LIVE DETECTION TEST")
        self.logger.info("="*70)
        
        return self.run_command(
            f'"{sys.executable}" scripts/test_detection_live.py',
            "Live Detection Test"
        )
    
    def run_visualization(self) -> bool:
        """Run visualization."""
        self.logger.info("\n" + "="*70)
        self.logger.info("RUNNING VISUALIZATION")
        self.logger.info("="*70)
        
        return self.run_command(
            f'"{sys.executable}" scripts/visualize_tracking.py',
            "Tracking Visualization"
        )
    
    def run_detection_test(self) -> bool:
        """Run detection test."""
        self.logger.info("\n" + "="*70)
        self.logger.info("RUNNING DETECTION TEST")
        self.logger.info("="*70)
        
        return self.run_command(
            f'"{sys.executable}" scripts/test_detection.py',
            "Detection Test"
        )
    
    def run_all_tests(self) -> bool:
        """Run all tests."""
        self.logger.info("\n" + "="*70)
        self.logger.info("RUNNING ALL TESTS")
        self.logger.info("="*70)
        
        all_passed = True
        
        # Run unit tests
        all_passed &= self.run_unit_tests()
        
        # Run detection test
        all_passed &= self.run_detection_test()
        
        # Run live detection test
        all_passed &= self.run_live_detection()
        
        return all_passed
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        for test_name, result in self.results.items():
            status = result.get("status", "UNKNOWN")
            symbol = "✓" if status == "PASSED" else "✗" if status != "UNKNOWN" else "?"
            print(f"  {symbol} {test_name}: {status}")
        
        # Count results
        passed = sum(1 for r in self.results.values() if r.get("status") == "PASSED")
        total = len(self.results)
        
        print(f"\nTotal: {passed}/{total} passed")
        print("="*70 + "\n")
        
        return passed == total
    
    def save_results(self, output_path: Path):
        """Save test results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results saved to: {output_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Test Runner for Object Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python scripts/run_tests.py --all
  
  # Run specific test
  python scripts/run_tests.py --unit
  python scripts/run_tests.py --live
  python scripts/run_tests.py --visualize
  
  # Run with debug logging
  python scripts/run_tests.py --all --debug
  
  # Save results
  python scripts/run_tests.py --all --save results.json
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests"
    )
    parser.add_argument(
        "--detection",
        action="store_true",
        help="Run detection test"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live detection test"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Run visualization"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save test results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(log_level)
    
    # Initialize runner
    runner = TestRunner(PROJECT_ROOT, logger)
    
    logger.info(f"Starting Test Runner")
    logger.info(f"Project Root: {PROJECT_ROOT}")
    
    # Default: if no tests specified, show help
    if not any([args.all, args.unit, args.detection, args.live, args.visualize]):
        parser.print_help()
        return 1
    
    # Run requested tests
    all_passed = True
    
    if args.all:
        all_passed &= runner.run_all_tests()
    else:
        if args.unit:
            all_passed &= runner.run_unit_tests()
        
        if args.detection:
            all_passed &= runner.run_detection_test()
        
        if args.live:
            all_passed &= runner.run_live_detection()
        
        if args.visualize:
            all_passed &= runner.run_visualization()
    
    # Print summary
    runner.print_summary()
    
    # Save results if requested
    if args.save:
        runner.save_results(Path(args.save))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
