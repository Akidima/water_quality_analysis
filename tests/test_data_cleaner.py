"""
Test script for the production-ready data cleaner with Pydantic validation and Dask.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

import importlib.util
import sys

# Import module with hyphen in name
spec = importlib.util.spec_from_file_location("data_cleaner", Path(__file__).parent / "data-cleaner.py")
data_cleaner = importlib.util.module_from_spec(spec)
sys.modules["data_cleaner"] = data_cleaner
spec.loader.exec_module(data_cleaner)

DataCleanerConfig = data_cleaner.DataCleanerConfig
WaterDataCleaner = data_cleaner.WaterDataCleaner
CleaningReport = data_cleaner.CleaningReport
validate_config = data_cleaner.validate_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_config_validation():
    """Test Pydantic configuration validation."""
    print("\n" + "="*80)
    print("TEST 1: Configuration Validation")
    print("="*80)
    
    try:
        # Test valid configuration
        config = DataCleanerConfig(
            lat_min=-90.0,
            lat_max=90.0,
            lon_min=-180.0,
            lon_max=180.0,
            strict_mode=True
        )
        validate_config(config)
        print("✓ Valid configuration passed")
        
        # Test invalid latitude range
        try:
            invalid_config = DataCleanerConfig(
                lat_min=100.0,  # Invalid
                lat_max=90.0
            )
            print("✗ Invalid latitude should have failed")
            return False
        except ValueError as e:
            print(f"✓ Invalid latitude correctly rejected: {e}")
        
        # Test invalid min > max
        try:
            invalid_config = DataCleanerConfig(
                lat_min=50.0,
                lat_max=40.0  # max < min
            )
            print("✗ Invalid min/max range should have failed")
            return False
        except ValueError as e:
            print(f"✓ Invalid min/max correctly rejected: {e}")
        
        print("\n✓ All configuration validation tests passed")
        return True
        
    except Exception as e:
        print(f"\n✗ Configuration validation test failed: {e}")
        return False


def test_config_defaults():
    """Test default configuration values."""
    print("\n" + "="*80)
    print("TEST 2: Default Configuration Values")
    print("="*80)
    
    try:
        config = DataCleanerConfig()
        
        assert config.lat_min == -90.0, "Default lat_min should be -90.0"
        assert config.lat_max == 90.0, "Default lat_max should be 90.0"
        assert config.lon_min == -180.0, "Default lon_min should be -180.0"
        assert config.lon_max == 180.0, "Default lon_max should be 180.0"
        assert config.strict_mode == True, "Default strict_mode should be True"
        assert config.remove_duplicates == True, "Default remove_duplicates should be True"
        assert config.outlier_std_threshold == 3.0, "Default outlier_std_threshold should be 3.0"
        
        print("✓ All default values are correct")
        print(f"  - lat_min: {config.lat_min}")
        print(f"  - lat_max: {config.lat_max}")
        print(f"  - lon_min: {config.lon_min}")
        print(f"  - lon_max: {config.lon_max}")
        print(f"  - strict_mode: {config.strict_mode}")
        print(f"  - remove_duplicates: {config.remove_duplicates}")
        print(f"  - outlier_std_threshold: {config.outlier_std_threshold}")
        
        return True
        
    except AssertionError as e:
        print(f"✗ Default value test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_cleaner_initialization():
    """Test WaterDataCleaner initialization."""
    print("\n" + "="*80)
    print("TEST 3: Cleaner Initialization")
    print("="*80)
    
    try:
        # Test with default config
        cleaner1 = WaterDataCleaner()
        assert cleaner1.config is not None, "Config should be initialized"
        assert cleaner1.report is None, "Report should be None initially"
        print("✓ Cleaner initialized with default config")
        
        # Test with custom config
        custom_config = DataCleanerConfig(
            strict_mode=False,
            outlier_std_threshold=2.5
        )
        cleaner2 = WaterDataCleaner(custom_config)
        assert cleaner2.config.strict_mode == False, "Custom strict_mode not set"
        assert cleaner2.config.outlier_std_threshold == 2.5, "Custom threshold not set"
        print("✓ Cleaner initialized with custom config")
        
        print("\n✓ All initialization tests passed")
        return True
        
    except AssertionError as e:
        print(f"✗ Initialization test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_pydantic_serialization():
    """Test Pydantic model serialization."""
    print("\n" + "="*80)
    print("TEST 4: Pydantic Serialization")
    print("="*80)
    
    try:
        config = DataCleanerConfig(
            strict_mode=True,
            remove_duplicates=True,
            outlier_std_threshold=3.0
        )
        
        # Test model_dump
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict), "model_dump should return dict"
        assert 'strict_mode' in config_dict, "strict_mode should be in dict"
        assert config_dict['strict_mode'] == True, "strict_mode value incorrect"
        print("✓ Configuration serialized to dict successfully")
        
        # Test JSON serialization
        import json
        config_json = json.dumps(config_dict, indent=2)
        assert isinstance(config_json, str), "JSON serialization should return string"
        print("✓ Configuration serialized to JSON successfully")
        
        # Test deserialization
        config_from_dict = DataCleanerConfig(**config_dict)
        assert config_from_dict.strict_mode == config.strict_mode, "Deserialization failed"
        print("✓ Configuration deserialized from dict successfully")
        
        print("\n✓ All serialization tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Serialization test failed: {e}")
        return False


def test_field_validators():
    """Test Pydantic field validators."""
    print("\n" + "="*80)
    print("TEST 5: Field Validators")
    print("="*80)
    
    try:
        # Test valid latitude
        config = DataCleanerConfig(lat_min=-45.0, lat_max=45.0)
        print("✓ Valid latitude range accepted")
        
        # Test invalid latitude (out of range)
        try:
            config = DataCleanerConfig(lat_min=-100.0)
            print("✗ Should have rejected invalid latitude")
            return False
        except ValueError:
            print("✓ Invalid latitude rejected")
        
        # Test valid longitude
        config = DataCleanerConfig(lon_min=-90.0, lon_max=90.0)
        print("✓ Valid longitude range accepted")
        
        # Test invalid longitude (out of range)
        try:
            config = DataCleanerConfig(lon_min=-200.0)
            print("✗ Should have rejected invalid longitude")
            return False
        except ValueError:
            print("✓ Invalid longitude rejected")
        
        print("\n✓ All field validator tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Field validator test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING ALL TESTS FOR DATA CLEANER")
    print("="*80)
    
    tests = [
        ("Configuration Validation", test_config_validation),
        ("Default Configuration", test_config_defaults),
        ("Cleaner Initialization", test_cleaner_initialization),
        ("Pydantic Serialization", test_pydantic_serialization),
        ("Field Validators", test_field_validators)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}", exc_info=True)
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
