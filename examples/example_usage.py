"""
Example: Using All Pydantic Enhancements with Water Quality Data
"""

from pathlib import Path
from scripts.pydantic_enhancements import (
    ApplicationSettings,
    ColumnNameValidator,
    ErrorHandler
)
from scripts.data_cleaner import WaterDataCleaner, DataCleanerConfig

def example_1_column_validation():
    """Example 1: Validate your dataset columns"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Column Validation")
    print("="*80)
    
    # Simulate your dataset columns
    your_columns = [
        'Latitude', 'Longitude', 'Water Company',
        'Spill Events 2023', 'Spill Events 2024',
        'Receuving Environment',  # Typo!
        'Site Name', 'Permit Number'
    ]
    
    print(f"\nValidating {len(your_columns)} columns...")
    results = ColumnNameValidator.validate_columns(your_columns)
    
    print(f"\n✅ Valid columns: {len(results['valid'])}")
    for col in results['valid'][:5]:  # Show first 5
        print(f"   • {col}")
    
    if results['suggestions']:
        print(f"\n⚠️  Typos detected with suggestions:")
        for suggestion in results['suggestions']:
            print(f"   • '{suggestion['column']}'")
            print(f"     Suggested: {suggestion['suggestions']}")


def example_2_settings_based_cleaning():
    """Example 2: Clean data using settings"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Settings-Based Data Cleaning")
    print("="*80)
    
    # Create custom settings
    settings = ApplicationSettings(
        app_name="My Water Analysis",
        max_workers=4,
        lat_min=49.0,  # UK bounds
        lat_max=61.0,
        lon_min=-11.0,
        lon_max=2.0,
        missing_threshold=0.15,
        outlier_threshold=2.5,
        strict_validation=True
    )
    
    print(f"\n📋 Configuration:")
    print(f"   App: {settings.app_name}")
    print(f"   Workers: {settings.max_workers}")
    print(f"   Geographic bounds:")
    print(f"     Latitude: [{settings.lat_min}, {settings.lat_max}]")
    print(f"     Longitude: [{settings.lon_min}, {settings.lon_max}]")
    print(f"   Thresholds:")
    print(f"     Missing data: {settings.missing_threshold * 100}%")
    print(f"     Outliers: {settings.outlier_threshold} std devs")
    
    # Convert to cleaner config
    cleaner_params = settings.to_cleaner_config()
    config = DataCleanerConfig(**cleaner_params)
    
    print(f"\n✅ Config ready for cleaning with {len(cleaner_params)} parameters")


def example_3_error_handling():
    """Example 3: Structured error handling"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Structured Error Handling")
    print("="*80)
    
    from scripts.pydantic_enhancements import (
        ErrorHandler,
        ColumnValidationError,
        RangeValidationError,
        DataQualityError
    )
    
    # Create error handler
    handler = ErrorHandler()
    
    # Simulate various errors
    handler.add_error(
        ColumnValidationError('Invalid_Column', 'Does not match pattern')
    )
    
    handler.add_error(
        RangeValidationError('Latitude', 95.0, -90.0, 90.0)
    )
    
    handler.add_error(
        DataQualityError('High missing data', 150, 1000, 0.10)
    )
    
    handler.add_warning('Column naming convention not followed')
    handler.add_warning('Detected potential duplicate rows')
    
    # Generate report
    report = handler.get_error_report()
    
    print(f"\n📊 Error Report:")
    print(f"   Errors: {report['error_count']}")
    print(f"   Warnings: {report['warning_count']}")
    
    print(f"\n❌ Errors:")
    for error in report['errors']:
        print(f"   • {error['error_type']}")
        print(f"     {error['message']}")
    
    print(f"\n⚠️  Warnings:")
    for warning in report['warnings']:
        print(f"   • {warning}")


def example_4_quality_scoring():
    """Example 4: Data quality scoring"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Data Quality Scoring")
    print("="*80)
    
    from scripts.pydantic_enhancements import (
        NumericStatistics,
        CategoricalStatistics,
        DatasetStatistics
    )
    
    # Example numeric column
    lat_stats = NumericStatistics(
        column_name='Latitude',
        count=1000,
        mean=52.5,
        std=2.3,
        min=49.0,
        median=52.7,
        max=61.0,
        q25=51.2,
        q75=54.1,
        missing_count=10,
        missing_percent=1.0,
        outlier_count=5,
        outlier_percent=0.5
    )
    
    # Example categorical column
    company_stats = CategoricalStatistics(
        column_name='Water Company',
        count=1000,
        unique_count=10,
        unique_percent=1.0,
        top_value='Thames Water',
        top_frequency=300,
        missing_count=20,
        missing_percent=2.0
    )
    
    # Overall dataset
    dataset_stats = DatasetStatistics(
        total_rows=1000,
        total_columns=15,
        numeric_columns={'Latitude': lat_stats},
        categorical_columns={'Water Company': company_stats},
        memory_usage_mb=5.2,
        overall_missing_percent=3.5,
        duplicate_rows=15
    )
    
    print(f"\n📊 Quality Scores:")
    print(f"\n   Numeric Columns:")
    print(f"   • Latitude: {lat_stats.quality_score():.2f}/100")
    
    print(f"\n   Categorical Columns:")
    print(f"   • Water Company: {company_stats.quality_score():.2f}/100")
    
    print(f"\n   Overall Dataset: {dataset_stats.overall_quality_score():.2f}/100")
    
    # Generate summary
    summary = dataset_stats.summary_report()
    print(f"\n📋 Dataset Overview:")
    print(f"   Rows: {summary['dataset_overview']['rows']:,}")
    print(f"   Columns: {summary['dataset_overview']['columns']}")
    print(f"   Memory: {summary['dataset_overview']['memory_mb']:.2f} MB")
    print(f"   Missing: {summary['dataset_overview']['missing_percent']:.2f}%")


def example_5_complete_pipeline():
    """Example 5: Complete pipeline with all features"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Complete Data Cleaning Pipeline")
    print("="*80)
    
    # This is what you would do with your actual data file:
    data_file = Path(__file__).parent / 'data' / 'national_water_plan.csv'
    
    if data_file.exists():
        print(f"\n✅ Data file found: {data_file.name}")
        print(f"   Size: {data_file.stat().st_size / 1e6:.2f} MB")
        
        print("\n📝 To clean this data with all enhancements:")
        print("   1. Create .env file with your settings (optional)")
        print("   2. Run:")
        print()
        print("   from pydantic_enhancements import ApplicationSettings")
        print("   from scripts.data_cleaner import clean_water_data_with_settings")
        print()
        print("   settings = ApplicationSettings()")
        print(f"   df, report = clean_water_data_with_settings('{data_file}')")
        print()
        print("   # View quality score")
        print("   print(report.quality_metrics['overall_quality_score'])")
    else:
        print(f"\n⚠️  Data file not found: {data_file}")
        print("   Place your CSV file in the 'data' directory")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PYDANTIC ENHANCEMENTS - PRACTICAL EXAMPLES")
    print("="*80)
    
    # Run all examples
    example_1_column_validation()
    example_2_settings_based_cleaning()
    example_3_error_handling()
    example_4_quality_scoring()
    example_5_complete_pipeline()
    
    print("\n" + "="*80)
    print("✅ All examples completed successfully!")
    print("="*80)
    print("\n💡 Next steps:")
    print("   1. Review PYDANTIC_ENHANCEMENTS_GUIDE.md for detailed docs")
    print("   2. Create your .env file for custom settings")
    print("   3. Run: python scripts/data-cleaner.py")
    print("   4. Or use: clean_water_data_with_settings() in your code")
    print("="*80 + "\n")
