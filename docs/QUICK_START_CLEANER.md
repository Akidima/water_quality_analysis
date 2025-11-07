# Water Data Cleaner - Quick Start Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install 'dask[complete]' pydantic
```

### Step 2: Run with Defaults
```python
from scripts.data_cleaner import clean_water_data

# Clean your data (one line!)
cleaned_df, report = clean_water_data('data/your_file.csv')

# Check results
print(f"Cleaned: {report.original_shape[0]} → {report.cleaned_shape[0]} rows")
```

### Step 3: View Results
```python
# View quality metrics
print(report.quality_metrics)

# View what was removed
print(report.removal_breakdown)
```

---

## 📋 Common Usage Patterns

### Pattern 1: Basic Cleaning
```python
from scripts.data_cleaner import clean_water_data

df, report = clean_water_data(
    filepath='data/water_data.csv',
    output_dir='export/cleaned'
)
```

### Pattern 2: Custom Configuration
```python
from scripts.data_cleaner import DataCleanerConfig, WaterDataCleaner

config = DataCleanerConfig(
    lat_min=49.0,  # UK latitude range
    lat_max=61.0,
    strict_mode=True,
    remove_outliers=True,
    n_partitions=4  # Use 4 CPU cores
)

cleaner = WaterDataCleaner(config)
df, report = cleaner.clean_data(your_dataframe)
```

### Pattern 3: With Data Loader (Recommended)
```python
from scripts.data_loader import load_and_explore_data
from scripts.data_cleaner import WaterDataCleaner, DataCleanerConfig

# Load data efficiently
df, load_report = load_and_explore_data('data/water_data.csv')

# Clean data
config = DataCleanerConfig(strict_mode=True)
cleaner = WaterDataCleaner(config)
cleaned_df, clean_report = cleaner.clean_data(df)
```

---

## 🎯 Key Configuration Options

### Essential Settings
```python
config = DataCleanerConfig(
    # Geographic validation
    lat_min=-90.0,  # Min latitude
    lat_max=90.0,   # Max latitude
    lon_min=-180.0, # Min longitude
    lon_max=180.0,  # Max longitude
    
    # Cleaning behavior
    strict_mode=True,           # Fail on missing required columns
    remove_duplicates=True,     # Remove duplicate rows
    remove_outliers=True,       # Remove statistical outliers
    outlier_std_threshold=3.0,  # Standard deviations for outliers
    
    # Missing data
    missing_value_threshold=0.1, # Max 10% missing per row
    fill_missing_values=False,   # Don't fill, remove instead
    
    # Performance
    n_partitions=4,  # Number of parallel partitions
    create_backup=True,  # Backup before cleaning
)
```

---

## 📊 Understanding the Report

### Report Structure
```python
report.original_shape          # (10000, 25) - rows, columns before
report.cleaned_shape           # (8500, 25) - rows, columns after
report.quality_metrics         # Dict with quality scores
report.removal_breakdown       # What was removed and why
report.errors                  # List of errors (empty if successful)
report.warnings                # List of warnings
report.processing_time_seconds # How long it took
```

### Quality Metrics
```python
{
    'rows_removed': 1500,
    'rows_retained_percent': 85.0,
    'initial_missing_percent': 5.2,
    'final_missing_percent': 2.1,
    'duplicate_reduction': 100
}
```

### Removal Breakdown
```python
{
    'missing_coordinates': 50,
    'invalid_latitude': 30,
    'invalid_longitude': 20,
    'insufficient_spill_years': 800,
    'empty_text_fields': 500,
    'duplicates': 100
}
```

---

## 🔧 Command Line Usage

### Run directly
```bash
cd scripts
python data-cleaner.py
```

### Run tests
```bash
cd scripts
python test_data_cleaner.py
```

---

## ⚡ Performance Tips

### Small Files (<1GB)
```python
config = DataCleanerConfig(n_partitions=2)
```

### Medium Files (1-10GB)
```python
config = DataCleanerConfig(n_partitions=4)
```

### Large Files (>10GB)
```python
config = DataCleanerConfig(
    n_partitions=8,
    create_backup=False  # Save disk space
)
```

---

## ❌ Common Errors & Fixes

### Error: "No module named 'dask'"
```bash
pip install 'dask[complete]'
```

### Error: "Missing required columns"
```python
# Set strict_mode=False to continue anyway
config = DataCleanerConfig(strict_mode=False)
```

### Error: "Memory error"
```python
# Increase partitions
config = DataCleanerConfig(n_partitions=8)
```

---

## 📝 What Gets Cleaned?

1. ✅ **Geographic Coordinates**
   - Invalid latitude/longitude values
   - Out-of-range coordinates
   - Missing coordinates

2. ✅ **Spill Events**
   - Statistical outliers (>3 standard deviations)
   - Invalid numeric values
   - Insufficient data points

3. ✅ **Text Fields**
   - Extra whitespace
   - Empty strings
   - Rows with all empty text fields

4. ✅ **Duplicates**
   - Exact duplicate rows

5. ✅ **Missing Data**
   - Rows exceeding missing value threshold

---

## 🎨 Example Output

```
================================================================================
CLEANING SUMMARY
================================================================================
Original Shape: 10,000 rows × 25 columns
Cleaned Shape:  8,500 rows × 25 columns
Rows Removed:   1,500
Rows Retained:  85.00%
Processing Time: 45.32 seconds

Removal Breakdown:
  • missing_coordinates: 50
  • invalid_latitude: 30
  • invalid_longitude: 20
  • insufficient_spill_years: 800
  • empty_text_fields: 500
  • duplicates: 100

Quality Metrics:
  • Initial Missing: 5.20%
  • Final Missing:   2.10%
  • Duplicates Removed: 100

✓ Data cleaning completed successfully!
✓ Cleaned data saved to: export/cleaned_data/cleaned_water_data.csv
✓ Report saved to: export/cleaned_data/cleaning_report.json
================================================================================
```

---

## 🔗 Integration with Other Scripts

### With data-loader.py
```python
from scripts.data_loader import load_and_explore_data
from scripts.data_cleaner import clean_water_data

# Load efficiently
df, load_report = load_and_explore_data('data.csv')

# Clean
cleaned_df, clean_report = clean_water_data(
    filepath=None,  # Already loaded
    use_data_loader=False  # Skip loading
)
```

### Pipeline Example
```python
# Complete pipeline
from scripts.data_loader import DataLoader, DataConfig
from scripts.data_cleaner import WaterDataCleaner, DataCleanerConfig

# 1. Load
data_config = DataConfig(filepath='data.csv')
loader = DataLoader(data_config)
df, load_report = loader.load_and_explore_data()

# 2. Clean
clean_config = DataCleanerConfig(strict_mode=True)
cleaner = WaterDataCleaner(clean_config)
cleaned_df, clean_report = cleaner.clean_data(df, 'export/cleaned')

# 3. Use cleaned data
# ... your analysis code ...
```

---

## 📚 More Information

- Full documentation: `DATA_CLEANER_DOCUMENTATION.md`
- Source code: `scripts/data-cleaner.py`
- Tests: `scripts/test_data_cleaner.py`
- Data loader: `scripts/data-loader.py`

---

## ✅ Verification

Run tests to verify everything works:
```bash
cd scripts
python test_data_cleaner.py
```

Expected output:
```
🎉 All tests passed successfully!
Total: 5/5 tests passed
```

---

## 🆘 Need Help?

1. Check the error message
2. Review `DATA_CLEANER_DOCUMENTATION.md`
3. Run `python test_data_cleaner.py` to verify installation
4. Check that all dependencies are installed

---

**Ready to clean your data? Start with Step 1 above! 🚀**
