# Data Loader Usage Guide

This guide shows you the best ways to run the `data-loader.py` script.

## Option 1: Using Command-Line Arguments (Recommended) ⭐

The script now supports a full CLI interface with argparse. This is the **best option** for flexibility and production use.

### Basic Usage

```bash
# Run with default settings (loads data/national_water_plan.csv)
python scripts/data-loader.py

# Load a specific file
python scripts/data-loader.py --file data/my_data.csv

# Short form
python scripts/data-loader.py -f data/my_data.csv
```

### Advanced Options

```bash
# Load with custom chunk size for large files
python scripts/data-loader.py --file data/large_data.csv --chunk-size 10000

# Specify custom output path
python scripts/data-loader.py --file data/my_data.csv --output export/my_report.json

# Change log level for debugging
python scripts/data-loader.py --file data/my_data.csv --log-level DEBUG

# Disable data type optimization
python scripts/data-loader.py --file data/my_data.csv --no-dtype-optimization

# Don't show the first 5 rows
python scripts/data-loader.py --file data/my_data.csv --no-show-head

# Set maximum memory threshold
python scripts/data-loader.py --file data/my_data.csv --max-memory-mb 1000
```

### All Available Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--file` | `-f` | Path to CSV file | `data/national_water_plan.csv` |
| `--output` | `-o` | Output report path | `export/data_exploration_report.json` |
| `--chunk-size` | - | Rows per chunk | Auto-detect |
| `--max-memory-mb` | - | Max memory before chunking | 500 |
| `--no-dtype-optimization` | - | Disable dtype optimization | False |
| `--log-level` | - | Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL) | INFO |
| `--show-head` | - | Show first 5 rows | True |
| `--no-show-head` | - | Don't show first 5 rows | False |
| `--help` | `-h` | Show help message | - |

### Help Command

```bash
python scripts/data-loader.py --help
```

---

## Option 2: Using the Wrapper Script

A convenience wrapper script is available in the project root:

```bash
# From project root
python load_data.py

# With arguments
python load_data.py --file data/my_data.csv --output export/my_report.json
```

---

## Option 3: Import as a Module (For Python Scripts)

You can import and use the functions in your own Python scripts:

```python
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, 'scripts')

# Import using importlib (because of hyphenated filename)
import importlib.util
spec = importlib.util.spec_from_file_location("data_loader", "scripts/data-loader.py")
data_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_loader)

# Use the convenience function
df, report = data_loader.load_and_explore_data(
    filepath='data/national_water_plan.csv',
    chunk_size=10000,
    dtype_optimization=True
)

print(f"Loaded {report.metadata.rows:,} rows")
```

Or use the classes directly:

```python
from data_loader import DataLoader, DataConfig

# Create custom configuration
config = DataConfig(
    filepath='data/my_data.csv',
    chunk_size=10000,
    max_memory_mb=1000,
    dtype_optimization=True
)

# Create loader and load data
loader = DataLoader(config)
df, report = loader.load_and_explore_data()

# Save report
loader.save_exploration_report(report, 'export/my_report.json')
```

---

## Option 4: Running in Conda Environment

Always run in your conda environment:

```bash
# Activate environment first
conda activate /Users/akidimageorge/Desktop/AI-ML/water_quality_analysis/env

# Then run the script
python scripts/data-loader.py --file data/my_data.csv

# Or use conda run
conda run -p /Users/akidimageorge/Desktop/AI-ML/water_quality_analysis/env python scripts/data-loader.py --file data/my_data.csv
```

---

## Option 5: Using Python's `-m` Flag (After Renaming)

**Note:** This option requires renaming `data-loader.py` to `data_loader.py` to follow Python naming conventions.

If you rename the file, you can run it as a module:

```bash
python -m scripts.data_loader --file data/my_data.csv
```

---

## Comparison of Options

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **CLI Arguments** ⭐ | Flexible, production-ready, easy to use | None | **Recommended for all use cases** |
| Wrapper Script | Convenient from project root | Extra file | Quick access from root |
| Import as Module | Programmatic access | Requires importlib workaround | Integration with other scripts |
| Direct Execution | Simple | No flexibility | Quick testing only |
| Python `-m` flag | Standard Python way | Requires file rename | If you rename the file |

---

## Recommended Approach

**For most users: Use Option 1 (CLI Arguments)** ⭐

This gives you:
- ✅ Full control over all parameters
- ✅ Easy to use from command line
- ✅ Production-ready
- ✅ Help documentation built-in
- ✅ Works with any file path
- ✅ Configurable logging

Example:
```bash
python scripts/data-loader.py --file data/my_data.csv --output export/report.json --log-level INFO
```

---

## Troubleshooting

### Issue: Module not found
**Solution:** Make sure you're running from the project root or use absolute paths.

### Issue: File not found
**Solution:** Use `--file` with the correct path, or ensure default file exists at `data/national_water_plan.csv`.

### Issue: Permission errors
**Solution:** Ensure output directory is writable, or specify a different `--output` path.

### Issue: Memory errors
**Solution:** Use `--chunk-size` to process in smaller chunks, or increase `--max-memory-mb`.

---

## Examples

### Example 1: Quick Load
```bash
python scripts/data-loader.py
```

### Example 2: Load Large File
```bash
python scripts/data-loader.py --file data/large_dataset.csv --chunk-size 5000 --max-memory-mb 1000
```

### Example 3: Debug Mode
```bash
python scripts/data-loader.py --file data/my_data.csv --log-level DEBUG
```

### Example 4: Custom Output
```bash
python scripts/data-loader.py --file data/my_data.csv --output reports/exploration.json --no-show-head
```

---

## Next Steps

After loading data, you can:
1. Use the generated report JSON for analysis
2. Pass the DataFrame to `data-cleaner.py` for cleaning
3. Use it in `data_analysis.py` for exploratory analysis
4. Import it in Jupyter notebooks for interactive analysis


