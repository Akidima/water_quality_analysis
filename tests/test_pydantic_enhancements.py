"""
Test suite for Pydantic enhancements module.
"""

import sys
from pathlib import Path
import logging

# Import the enhancements module
from pydantic_enhancements import (
    ColumnNameValidator,
    NumericStatistics,
    CategoricalStatistics,
    DatasetStatistics,
    ErrorHandler,
    ColumnValidationError,
    RangeValidationError,
    DataQualityError,
    ApplicationSettings,
    load_settings,
    validate_dataset_structure
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_column_validation():
    """Test column name validation (Enhancement 1)."""
    print("\n" + "="*80)
    print("TEST 1: Column Name Validation")
    print("="*80)
    
    # Test with valid columns
    valid_cols = ['Latitude', 'Longitude', 'Water Company', 'Spill Events 2023']
    results = ColumnNameValidator.validate_columns(valid_cols)
    assert len(results['valid']) == 4, "All columns should be valid"
    print("✓ Valid columns recognized correctly")
    
    # Test with typos
    typo_cols = ['Receuving Environment', 'Water Company']
    results = ColumnNameValidator.validate_columns(typo_cols)
    assert len(results['suggestions']) > 0, "Should suggest corrections for typos"
    print(f"✓ Typo detection working: {results['suggestions']}")
    
    # Test dataset structure validation
    test_columns = ['Latitude', 'Longitude', 'Site Name', 'Random Column']
    structure_results = validate_dataset_structure(
        test_columns,
        required_patterns=[r'^Latitude$', r'^Longitude$']
    )
    print(f"✓ Structure validation working: {len(structure_results['valid'])} valid columns")
    
    return True


def test_statistical_models():
    """Test statistical summary models (Enhancement 2)."""
    print("\n" + "="*80)
    print("TEST 2: Statistical Summary Models")
    print("="*80)
    
    # Test NumericStatistics
    numeric_stats = NumericStatistics(
        column_name='Temperature',
        count=1000,
        mean=20.5,
        std=5.2,
        min=10.0,
        median=20.0,
        q25=17.5,
        q75=23.5,
        max=35.0,
        missing_count=50,
        missing_percent=5.0,
        outlier_count=10,
        outlier_percent=1.0
    )
    
    quality_score = numeric_stats.quality_score()
    assert 0 <= quality_score <= 100, "Quality score should be 0-100"
    print(f"✓ NumericStatistics quality score: {quality_score:.2f}/100")
    
    # Test CategoricalStatistics
    cat_stats = CategoricalStatistics(
        column_name='Water Company',
        count=1000,
        unique_count=10,
        unique_percent=1.0,
        top_value='Thames Water',
        top_frequency=300,
        missing_count=20,
        missing_percent=2.0
    )
    
    cat_quality = cat_stats.quality_score()
    assert 0 <= cat_quality <= 100, "Quality score should be 0-100"
    print(f"✓ CategoricalStatistics quality score: {cat_quality:.2f}/100")
    
    # Test DatasetStatistics
    dataset_stats = DatasetStatistics(
        total_rows=1000,
        total_columns=10,
        numeric_columns={'Temperature': numeric_stats},
        categorical_columns={'Water Company': cat_stats},
        memory_usage_mb=5.2,
        overall_missing_percent=3.5,
        duplicate_rows=15
    )
    
    overall_quality = dataset_stats.overall_quality_score()
    assert 0 <= overall_quality <= 100, "Overall quality should be 0-100"
    print(f"✓ DatasetStatistics overall quality: {overall_quality:.2f}/100")
    
    # Test summary report
    summary = dataset_stats.summary_report()
    assert 'dataset_overview' in summary, "Should have dataset overview"
    assert 'quality_score' in summary, "Should have quality score"
    print(f"✓ Summary report generated: {summary['quality_score']:.2f} quality score")
    
    return True


def test_error_handlers():
    """Test custom error handlers (Enhancement 3)."""
    print("\n" + "="*80)
    print("TEST 3: Custom Error Handlers")
    print("="*80)
    
    error_handler = ErrorHandler()
    
    # Test ColumnValidationError
    col_error = ColumnValidationError(
        'Invalid_Col',
        'Does not match pattern',
        'Use standard naming'
    )
    error_handler.add_error(col_error)
    assert error_handler.has_errors(), "Should have errors"
    print(f"✓ ColumnValidationError: {col_error.message}")
    
    # Test RangeValidationError
    range_error = RangeValidationError(
        'Latitude',
        value=95.0,
        min_value=-90.0,
        max_value=90.0
    )
    error_handler.add_error(range_error)
    print(f"✓ RangeValidationError: {range_error.message}")
    
    # Test DataQualityError
    quality_error = DataQualityError(
        'High missing data',
        affected_rows=100,
        total_rows=1000,
        threshold=0.05
    )
    error_handler.add_error(quality_error)
    print(f"✓ DataQualityError: {quality_error.message}")
    
    # Test warnings
    error_handler.add_warning('This is a warning message')
    assert error_handler.has_warnings(), "Should have warnings"
    print("✓ Warning system working")
    
    # Test error report
    report = error_handler.get_error_report()
    assert report['error_count'] == 3, "Should have 3 errors"
    assert report['warning_count'] == 1, "Should have 1 warning"
    print(f"✓ Error report: {report['error_count']} errors, {report['warning_count']} warnings")
    
    # Test error to dict
    error_dict = col_error.to_dict()
    assert 'error_type' in error_dict, "Should have error_type"
    assert 'message' in error_dict, "Should have message"
    print("✓ Error serialization working")
    
    return True


def test_application_settings():
    """Test application settings management (Enhancement 4)."""
    print("\n" + "="*80)
    print("TEST 4: Application Settings Management")
    print("="*80)
    
    # Test default settings
    settings = ApplicationSettings()
    assert settings.app_name == "Water Quality Analysis", "Default app name incorrect"
    assert settings.max_workers == 4, "Default max_workers incorrect"
    assert settings.lat_min == 49.0, "Default lat_min incorrect"
    print(f"✓ Default settings loaded: {settings.app_name} v{settings.app_version}")
    
    # Test custom settings
    custom_settings = ApplicationSettings(
        app_name="Custom App",
        max_workers=8,
        lat_min=50.0,
        lat_max=60.0,
        debug=True
    )
    assert custom_settings.max_workers == 8, "Custom max_workers not set"
    print(f"✓ Custom settings: {custom_settings.max_workers} workers")
    
    # Test validation
    try:
        invalid_settings = ApplicationSettings(max_workers=100)
        print("✗ Should have rejected invalid max_workers")
        return False
    except Exception:
        print("✓ Settings validation working (rejected invalid value)")
    
    # Test log level validation
    try:
        invalid_log = ApplicationSettings(log_level="INVALID")
        print("✗ Should have rejected invalid log level")
        return False
    except Exception:
        print("✓ Log level validation working")
    
    # Test directory creation
    assert settings.data_directory.exists(), "Data directory should exist"
    assert settings.export_directory.exists(), "Export directory should exist"
    print("✓ Directory creation working")
    
    # Test to_cleaner_config conversion
    cleaner_config = settings.to_cleaner_config()
    assert 'lat_min' in cleaner_config, "Should have lat_min"
    assert 'n_partitions' in cleaner_config, "Should have n_partitions"
    print(f"✓ Cleaner config conversion: {len(cleaner_config)} parameters")
    
    # Test to_loader_config conversion
    loader_config = settings.to_loader_config()
    assert 'chunk_size' in loader_config, "Should have chunk_size"
    assert 'max_memory_mb' in loader_config, "Should have max_memory_mb"
    print(f"✓ Loader config conversion: {len(loader_config)} parameters")
    
    return True


def test_pydantic_validation():
    """Test Pydantic validation features."""
    print("\n" + "="*80)
    print("TEST 5: Pydantic Validation Features")
    print("="*80)
    
    # Test field validators
    try:
        invalid_numeric = NumericStatistics(
            column_name='Test',
            count=-1,  # Invalid: negative count
            mean=10.0
        )
        print("✗ Should have rejected negative count")
        return False
    except Exception:
        print("✓ Field validators working (rejected negative count)")
    
    # Test range validators
    try:
        invalid_cat = CategoricalStatistics(
            column_name='Test',
            count=100,
            unique_count=150,  # Invalid: more unique than count
            unique_percent=150.0
        )
        print("✗ Should have rejected invalid unique_count")
        return False
    except Exception:
        print("✓ Model validators working (rejected invalid unique_count)")
    
    # Test type coercion
    numeric_stats = NumericStatistics(
        column_name='Test',
        count='100',  # String should be coerced to int
        mean='20.5',  # String should be coerced to float
        min=10.0,
        max=30.0
    )
    assert isinstance(numeric_stats.count, int), "Should coerce to int"
    assert isinstance(numeric_stats.mean, float), "Should coerce to float"
    print("✓ Type coercion working")
    
    # Test JSON serialization
    stats_dict = numeric_stats.model_dump()
    assert isinstance(stats_dict, dict), "Should serialize to dict"
    print("✓ JSON serialization working")
    
    return True


def run_all_tests():
    """Run all enhancement tests."""
    print("\n" + "="*80)
    print("TESTING PYDANTIC ENHANCEMENTS")
    print("="*80)
    
    tests = [
        ("Column Validation", test_column_validation),
        ("Statistical Models", test_statistical_models),
        ("Error Handlers", test_error_handlers),
        ("Application Settings", test_application_settings),
        ("Pydantic Validation", test_pydantic_validation)
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
        print("\n🎉 All Pydantic enhancements working correctly!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
