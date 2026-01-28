# Data Cleaner Usage Guide

This guide shows you the best ways to use the `data-cleaner.py` script to clean and preprocess water quality data.

## Overview

The `data-cleaner.py` module provides comprehensive data cleaning capabilities with:
- ✅ Pydantic validation for configuration
- ✅ Dask parallel processing for large datasets
- ✅ Automatic backup creation
- ✅ Comprehensive cleaning reports
- ✅ Configurable cleaning strategies

---

## Option 1: Direct Script Execution (Recommended for Quick Use) ⭐

The simplest way to use the data cleaner is to run it directly. It uses default settings and processes the default data file.

### Basic Usage

```bash
# Run with default settings (cleans data/national_water_plan.csv)
python scripts/data-cleaner.py
```

This will:
- Load data from `data/national_water_plan.csv`
- Clean the data using default configuration
- Save cleaned data to `export/cleaned_data/cleaned_water_data.csv`
- Generate a cleaning report at `export/cleaned_data/cleaning_report.json`
- Create a backup of the original data

---

## Option 2: Using the Convenience Function (Recommended for Python Scripts) ⭐

For programmatic use, import and use the convenience function:

### Basic Example

```python
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, 'scripts')

# Import using importlib (because of hyphenated filename)
import importlib.util
spec = importlib.util.spec_from_file_location("data_cleaner", "scripts/data-cleaner.py")
data_cleaner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_cleaner)

# Use the convenience function
cleaned_df, report = data_cleaner.clean_water_data(
    filepath='data/national_water_plan.csv',
    output_dir='export/cleaned_data'
)

print(f"Cleaned {report.cleaned_shape[0]:,} rows")
print(f"Removed {report.quality_metrics['rows_removed']:,} rows")
```

### Advanced Example with Custom Configuration

```python
from pathlib import Path
import sys
import importlib.util

# Import the module
spec = importlib.util.spec_from_file_location("data_cleaner", "scripts/data-cleaner.py")
data_cleaner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_cleaner)

# Create custom configuration
config = data_cleaner.DataCleanerConfig(
    strict_mode=True,
    remove_duplicates=True,
    remove_outliers=True,
    outlier_std_threshold=3.0,
    min_valid_spill_years=3,
    save_cleaning_report=True,
    output_directory='export/cleaned_data',
    output_filename='cleaned_water_data.csv',
    report_output_path='cleaning_report.json',
    n_partitions=4  # Adjust based on your system
)

# Clean the data
cleaned_df, report = data_cleaner.clean_water_data(
    filepath='data/national_water_plan.csv',
    config=config,
    output_dir='export/cleaned_data',
    use_data_loader=True  # Use DataLoader for optimized loading
)

# Access report details
print(f"Original: {report.original_shape[0]:,} rows")
print(f"Cleaned:  {report.cleaned_shape[0]:,} rows")
print(f"Retained: {report.quality_metrics['rows_retained_percent']:.2f}%")
```

---

## Option 3: Using the WaterDataCleaner Class Directly

For maximum control, use the `WaterDataCleaner` class directly:

### Example

```python
from pathlib import Path
import sys
import importlib.util
import dask.dataframe as dd

# Import modules
spec = importlib.util.spec_from_file_location("data_cleaner", "scripts/data-cleaner.py")
data_cleaner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_cleaner)

# Import data loader if needed
spec_loader = importlib.util.spec_from_file_location("data_loader", "scripts/data-loader.py")
data_loader = importlib.util.module_from_spec(spec_loader)
spec_loader.loader.exec_module(data_loader)

# Load data first
loader_config = data_loader.DataConfig(filepath='data/national_water_plan.csv')
loader = data_loader.DataLoader(loader_config)
df, load_report = loader.load_and_explore_data()

# Create cleaner configuration
cleaner_config = data_cleaner.DataCleanerConfig(
    strict_mode=True,
    remove_duplicates=True,
    remove_outliers=True,
    outlier_std_threshold=3.0,
    min_valid_spill_years=3,
    n_partitions=4
)

# Create cleaner instance
cleaner = data_cleaner.WaterDataCleaner(cleaner_config)

# Clean the data
cleaned_df, cleaning_report = cleaner.clean_data(
    df=df,
    output_dir='export/cleaned_data'
)

# Access results
print(f"Processing time: {cleaning_report.processing_time_seconds:.2f} seconds")
print(f"Quality score: {cleaning_report.quality_metrics.get('overall_quality_score', 'N/A')}")
```

---

## Option 4: Using with ApplicationSettings (Advanced)

For production environments, use `ApplicationSettings` for centralized configuration:

```python
from pathlib import Path
import sys
import importlib.util

# Import modules
spec = importlib.util.spec_from_file_location("data_cleaner", "scripts/data-cleaner.py")
data_cleaner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_cleaner)

# Clean with settings (loads from environment or .env file)
cleaned_df, report = data_cleaner.clean_water_data_with_settings(
    filepath='data/national_water_plan.csv',
    env_file=Path('.env')  # Optional: specify .env file path
)
```

---

## Configuration Options

### DataCleanerConfig Parameters

The `DataCleanerConfig` class provides extensive configuration options:

#### Column Configuration

```python
config = DataCleanerConfig(
    # Required columns that must be present
    required_columns=['Latitude', 'Longitude', 'Water company', 'River Basin District'],
    
    # Optional columns to use if present
    optional_columns=['Site Name', 'Receiving Environment', 'Permit Number'],
    
    # Coordinate columns for geographic validation
    coordinate_columns=['Latitude', 'Longitude'],
    
    # Spill event columns by year
    spill_year_columns=[
        'Spill Events 2020', 'Spill Events 2021', 'Spill Events 2022',
        'Spill Events 2023', 'Spill Events 2024', 'Spill Events 2025'
    ],
    
    # Text columns to validate
    text_columns=['Water company', 'River Basin District', 'Site Name'],
    
    # Numeric columns
    numeric_columns=['Latitude', 'Longitude']
)
```

#### Validation Configuration

```python
config = DataCleanerConfig(
    # Latitude range validation
    lat_min=-90.0,
    lat_max=90.0,
    
    # Longitude range validation
    lon_min=-180.0,
    lon_max=180.0,
    
    # Spill year range
    spill_year_min=2020,
    spill_year_max=2025,
    
    # Validation strictness
    strict_validation=True,
    strict_mode=True  # Fail if required columns missing
)
```

#### Cleaning Parameters

```python
config = DataCleanerConfig(
    # Duplicate handling
    remove_duplicates=True,
    
    # Missing value handling
    fill_missing_values=False,  # Set to True to fill instead of remove
    fill_value=None,  # Value to use for filling (if fill_missing_values=True)
    missing_value_threshold=0.1,  # Remove rows with >10% missing values
    
    # Outlier handling
    remove_outliers=True,
    outlier_std_threshold=3.0,  # Remove values beyond 3 standard deviations
    
    # Spill year validation
    min_valid_spill_years=3,  # Minimum valid spill years required
    
    # Behavior flags
    remove_invalid_coordinates=True,
    remove_invalid_spill_years=True,
    remove_invalid_text_values=True,
    create_backup=True  # Create backup before cleaning
)
```

#### Output Configuration

```python
config = DataCleanerConfig(
    # Output settings
    save_cleaning_report=True,
    output_directory='export/cleaned_data',
    output_filename='cleaned_water_data.csv',
    report_output_path='cleaning_report.json',
    
    # Logging
    log_level='INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    
    # Dask configuration
    n_partitions=4  # Number of partitions for parallel processing (None for auto)
)
```

---

## Cleaning Report Structure

The cleaning report (`CleaningReport`) contains comprehensive information about the cleaning process:

```python
report.original_shape          # Tuple: (rows, columns) before cleaning
report.cleaned_shape           # Tuple: (rows, columns) after cleaning
report.original_columns        # List of original column names
report.cleaned_columns         # List of cleaned column names
report.original_missing_values # Dict: column -> missing count
report.cleaned_missing_values  # Dict: column -> missing count
report.removal_breakdown       # Dict: reason -> count of removed rows
report.quality_metrics        # Dict: various quality metrics
report.processing_time_seconds # Float: processing time in seconds
report.errors                  # List: any errors encountered
report.warnings               # List: any warnings generated
```

### Example: Accessing Report Data

```python
cleaned_df, report = clean_water_data('data/national_water_plan.csv')

# Print summary
print(f"Original: {report.original_shape[0]:,} rows × {report.original_shape[1]} columns")
print(f"Cleaned:  {report.cleaned_shape[0]:,} rows × {report.cleaned_shape[1]} columns")
print(f"Rows Removed: {report.quality_metrics['rows_removed']:,}")
print(f"Rows Retained: {report.quality_metrics['rows_retained_percent']:.2f}%")
print(f"Processing Time: {report.processing_time_seconds:.2f} seconds")

# Check removal breakdown
if report.removal_breakdown:
    print("\nRemoval Breakdown:")
    for reason, count in report.removal_breakdown.items():
        print(f"  • {reason}: {count:,}")

# Check quality metrics
print(f"\nQuality Metrics:")
print(f"  • Initial Missing: {report.quality_metrics['initial_missing_percent']:.2f}%")
print(f"  • Final Missing:   {report.quality_metrics['final_missing_percent']:.2f}%")
print(f"  • Duplicates Removed: {report.quality_metrics['duplicate_reduction']}")

# Check for warnings/errors
if report.warnings:
    print(f"\n⚠️  Warnings ({len(report.warnings)}):")
    for warning in report.warnings:
        print(f"    {warning}")

if report.errors:
    print(f"\n❌ Errors ({len(report.errors)}):")
    for error in report.errors:
        print(f"    {error}")
```

---

## Common Use Cases

### Use Case 1: Quick Clean with Defaults

```python
from pathlib import Path
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("data_cleaner", "scripts/data-cleaner.py")
data_cleaner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_cleaner)

# Simple one-liner
cleaned_df, report = data_cleaner.clean_water_data('data/national_water_plan.csv')
print(f"Cleaned {report.cleaned_shape[0]:,} rows")
```

### Use Case 2: Custom Cleaning Strategy

```python
# More lenient cleaning (keep more data)
config = DataCleanerConfig(
    strict_mode=False,  # Don't fail on missing columns
    remove_outliers=False,  # Keep outliers
    min_valid_spill_years=1,  # Require only 1 valid year
    missing_value_threshold=0.5,  # Allow up to 50% missing values
    remove_duplicates=True  # Still remove duplicates
)

cleaned_df, report = clean_water_data(
    filepath='data/national_water_plan.csv',
    config=config
)
```

### Use Case 3: Strict Cleaning (High Quality)

```python
# Very strict cleaning (high quality, fewer rows)
config = DataCleanerConfig(
    strict_mode=True,
    remove_duplicates=True,
    remove_outliers=True,
    outlier_std_threshold=2.5,  # Stricter outlier detection
    min_valid_spill_years=4,  # Require 4 valid years
    missing_value_threshold=0.05,  # Only allow 5% missing values
    remove_invalid_coordinates=True,
    remove_invalid_spill_years=True,
    remove_invalid_text_values=True
)

cleaned_df, report = clean_water_data(
    filepath='data/national_water_plan.csv',
    config=config
)
```

### Use Case 4: Fill Missing Values Instead of Removing

```python
# Fill missing values instead of removing rows
config = DataCleanerConfig(
    fill_missing_values=True,
    fill_value=0,  # Fill numeric columns with 0
    remove_duplicates=True,
    remove_outliers=True
)

cleaned_df, report = clean_water_data(
    filepath='data/national_water_plan.csv',
    config=config
)
```

### Use Case 5: Process Large Dataset with Custom Partitions

```python
# Optimize for large datasets
config = DataCleanerConfig(
    n_partitions=8,  # More partitions for parallel processing
    create_backup=True,
    save_cleaning_report=True
)

cleaned_df, report = clean_water_data(
    filepath='data/large_dataset.csv',
    config=config,
    output_dir='export/cleaned_data'
)
```

---

## Integration with Data Loader

The data cleaner integrates seamlessly with the data loader:

```python
from pathlib import Path
import sys
import importlib.util

# Import both modules
spec_loader = importlib.util.spec_from_file_location("data_loader", "scripts/data-loader.py")
data_loader = importlib.util.module_from_spec(spec_loader)
spec_loader.loader.exec_module(data_loader)

spec_cleaner = importlib.util.spec_from_file_location("data_cleaner", "scripts/data-cleaner.py")
data_cleaner = importlib.util.module_from_spec(spec_cleaner)
spec_cleaner.loader.exec_module(data_cleaner)

# Load data
loader_config = data_loader.DataConfig(filepath='data/national_water_plan.csv')
loader = data_loader.DataLoader(loader_config)
df, load_report = loader.load_and_explore_data()

# Clean data
cleaner_config = data_cleaner.DataCleanerConfig()
cleaner = data_cleaner.WaterDataCleaner(cleaner_config)
cleaned_df, cleaning_report = cleaner.clean_data(df, output_dir='export/cleaned_data')

print(f"Loaded: {load_report.metadata.rows:,} rows")
print(f"Cleaned: {cleaning_report.cleaned_shape[0]:,} rows")
```

---

## Running in Conda Environment

Always run in your conda environment:

```bash
# Activate environment first
conda activate /Users/akidimageorge/Desktop/AI-ML/water_quality_analysis/env

# Then run the script
python scripts/data-cleaner.py

# Or use conda run
conda run -p /Users/akidimageorge/Desktop/AI-ML/water_quality_analysis/env python scripts/data-cleaner.py
```

---

## Troubleshooting

### Issue: Module not found
**Solution:** Make sure you're running from the project root or use absolute paths. Use `importlib.util` for hyphenated filenames.

### Issue: File not found
**Solution:** Ensure the input CSV file exists at the specified path. Check the default path: `data/national_water_plan.csv`.

### Issue: Validation errors
**Solution:** Check that required columns are present in your dataset. Set `strict_mode=False` to allow missing optional columns.

### Issue: Memory errors
**Solution:** Adjust `n_partitions` in the configuration. For very large datasets, use fewer partitions or process in chunks.

### Issue: No output files created
**Solution:** Check that `save_cleaning_report=True` and `output_directory` is writable. Verify file permissions.

### Issue: Configuration validation fails
**Solution:** Use `validate_config()` to check your configuration:
```python
from data_cleaner import DataCleanerConfig, validate_config

config = DataCleanerConfig(...)
validate_config(config)  # Will raise error if invalid
```

---

## Examples

### Example 1: Basic Cleaning
```python
from pathlib import Path
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("data_cleaner", "scripts/data-cleaner.py")
data_cleaner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_cleaner)

cleaned_df, report = data_cleaner.clean_water_data('data/national_water_plan.csv')
print(f"✓ Cleaned {report.cleaned_shape[0]:,} rows")
```

### Example 2: Custom Configuration
```python
config = DataCleanerConfig(
    strict_mode=True,
    remove_duplicates=True,
    remove_outliers=True,
    outlier_std_threshold=3.0,
    min_valid_spill_years=3,
    output_directory='export/cleaned_data',
    output_filename='my_cleaned_data.csv',
    n_partitions=4
)

cleaned_df, report = clean_water_data(
    filepath='data/national_water_plan.csv',
    config=config
)
```

### Example 3: Access Cleaned Data
```python
cleaned_df, report = clean_water_data('data/national_water_plan.csv')

# Convert to pandas if needed (for small datasets)
import pandas as pd
cleaned_pd = cleaned_df.compute()  # Convert Dask DataFrame to Pandas

# Or work with Dask DataFrame directly
print(cleaned_df.head())
print(cleaned_df.describe())
```

---

## Next Steps

After cleaning data, you can:
1. **Use cleaned data for analysis** - Pass to `data_analysis.py` for EDA
2. **Load in Jupyter notebooks** - Import cleaned CSV for interactive analysis
3. **Use for modeling** - Clean data is ready for predictive modeling
4. **Review cleaning report** - Check `cleaning_report.json` for detailed metrics
5. **Restore from backup** - Original data is saved in `export/cleaned_data/backup_*.csv`

---

## Configuration Reference

### Quick Configuration Cheat Sheet

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strict_mode` | bool | `True` | Fail if required columns missing |
| `remove_duplicates` | bool | `True` | Remove duplicate rows |
| `remove_outliers` | bool | `True` | Remove statistical outliers |
| `outlier_std_threshold` | float | `3.0` | Standard deviations for outlier detection |
| `min_valid_spill_years` | int | `3` | Minimum valid spill years required |
| `missing_value_threshold` | float | `0.1` | Max missing value percentage (0-1) |
| `fill_missing_values` | bool | `False` | Fill instead of remove missing values |
| `create_backup` | bool | `True` | Create backup before cleaning |
| `n_partitions` | int | `None` | Dask partitions (None = auto) |
| `output_directory` | str | `'cleaned_data'` | Output directory path |
| `output_filename` | str | `'cleaned_data.csv'` | Output filename |
| `log_level` | str | `'INFO'` | Logging level |

---

## Best Practices

1. **Always create backups** - Set `create_backup=True` (default)
2. **Review cleaning reports** - Check `cleaning_report.json` after cleaning
3. **Start with defaults** - Use default configuration first, then customize
4. **Validate configuration** - Use `validate_config()` before cleaning
5. **Use DataLoader** - Set `use_data_loader=True` for optimized loading
6. **Check warnings** - Review `report.warnings` for potential issues
7. **Monitor processing time** - Use `report.processing_time_seconds` to optimize
8. **Adjust partitions** - Tune `n_partitions` based on dataset size and system resources

---

For more details, see the main documentation in `docs/DATA_CLEANER_DOCUMENTATION.md`.

