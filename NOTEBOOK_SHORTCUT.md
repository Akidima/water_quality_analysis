# Notebook Execution Shortcuts

Quick ways to run the `comprehensive_modeling.ipynb` notebook.

## Available Shortcuts

### Option 1: Python Script (Recommended)
```bash
python3 run_notebook.py
```

### Option 2: Shell Script
```bash
./run_notebook.sh
```

### Option 3: Direct Python Execution
```bash
python3 -c "
import json, sys, subprocess
from pathlib import Path
nb = json.load(open('notebooks/predictive_modeling/comprehensive_modeling.ipynb'))
# ... (execution code)
"
```

## What the Shortcut Does

1. ✅ Reads the notebook file
2. ✅ Extracts all code cells
3. ✅ Executes them sequentially
4. ✅ Handles errors gracefully
5. ✅ Provides execution status

## Output

The shortcut will:
- Execute all notebook cells
- Display progress and results
- Show model performance metrics
- Generate predictions
- Save the trained model

## Execution Time

Typically takes **15-20 seconds** to complete.

## Files Created

After execution, you'll find:
- `artifacts/spills_pipeline.joblib` - Trained model
- `notebooks/predictive_modeling/execution_report.json` - Detailed execution report
- `notebooks/predictive_modeling/EXECUTION_REPORT.md` - Human-readable report

## Troubleshooting

If you encounter issues:
1. Ensure you're in the project root directory
2. Check that all dependencies are installed
3. Verify data files exist in `export/cleaned_data/` or `data/`
4. Check Python version (requires Python 3.7+)

## Example Usage

```bash
# Navigate to project root
cd /path/to/water_quality_analysis

# Run the notebook
python3 run_notebook.py

# Or use the shell script
./run_notebook.sh
```

## Notes

- The shortcut automatically handles path resolution
- All imports are configured automatically
- Errors in individual cells won't stop the entire execution
- The script cleans up temporary files after execution

