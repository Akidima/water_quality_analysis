#!/usr/bin/env python3
"""
Shortcut script to run the comprehensive_modeling.ipynb notebook.
Usage: 
    python run_notebook.py          # Normal mode
    python run_notebook.py --fast    # Fast mode (skips/speeds up hyperparameter tuning)
"""

import json
import sys
import argparse
from pathlib import Path
import subprocess
from datetime import datetime

def run_notebook(fast_mode: bool = False):
    """Execute the comprehensive modeling notebook and generate a timestamped report.
    
    Args:
        fast_mode: If True, uses faster settings for hyperparameter tuning
    """
    
    # Get project root
    project_root = Path(__file__).parent
    notebook_path = project_root / "notebooks" / "predictive_modeling" / "comprehensive_modeling.ipynb"
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return 1
    
    # Generate timestamp for report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = project_root / "notebooks" / "predictive_modeling"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("RUNNING COMPREHENSIVE MODELING NOTEBOOK")
    print("=" * 80)
    print(f"Notebook: {notebook_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'🚀 FAST MODE' if fast_mode else 'Normal'}")
    print(f"Report will be saved with timestamp: {timestamp}\n")
    
    # Read notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Create execution script with report generation
    script_content = []
    script_content.append("# Generated from comprehensive_modeling.ipynb\n")
    script_content.append("import sys\n")
    script_content.append("from pathlib import Path\n")
    script_content.append("import os\n")
    script_content.append("import json\n")
    script_content.append("import pandas as pd\n")
    script_content.append("import numpy as np\n")
    script_content.append("from datetime import datetime\n")
    script_content.append("PROJECT_ROOT = Path.cwd()\n")
    script_content.append("os.chdir(str(PROJECT_ROOT))\n")
    script_content.append("sys.path.insert(0, str(PROJECT_ROOT))\n\n")
    script_content.append("# Initialize execution report\n")
    script_content.append(f"execution_report = {{\n")
    script_content.append(f"    'timestamp': '{timestamp}',\n")
    script_content.append(f"    'start_time': datetime.now().isoformat(),\n")
    script_content.append(f"    'notebook_path': str(Path('notebooks/predictive_modeling/comprehensive_modeling.ipynb')),\n")
    script_content.append(f"    'cells_executed': [],\n")
    script_content.append(f"    'errors': [],\n")
    script_content.append(f"    'warnings': [],\n")
    script_content.append(f"    'data_info': {{}},\n")
    script_content.append(f"    'model_metrics': {{}},\n")
    script_content.append(f"    'predictions': []\n")
    script_content.append(f"}}\n\n")
    
    code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
    
    for cell_idx, cell in enumerate(code_cells, 1):
        source = ''.join(cell['source'])
        if source.strip():
            # Fix path resolution
            source = source.replace(
                'NOTEBOOK_DIR = Path(__file__).resolve().parent',
                'NOTEBOOK_DIR = Path.cwd() / "notebooks" / "predictive_modeling"'
            )
            source = source.replace(
                'PROJECT_ROOT = NOTEBOOK_DIR.parent.parent',
                'PROJECT_ROOT = Path.cwd()'
            )
            
            # Fast mode optimizations
            if fast_mode:
                # Replace grid search with random search
                if 'grid_search' in source and 'HyperparameterTuner' in source:
                    source = source.replace('grid_search', 'random_search')
                    source = source.replace('cv=5', 'cv=3, n_iter=10')
                    source = source.replace('cv=3', 'cv=3, n_iter=10')
                # Reduce CV folds in model comparison
                if 'compare_models' in source and 'cv=5' in source:
                    source = source.replace('cv=5', 'cv=3')
                # Skip some visualizations if needed
                if 'plot_learning_curves' in source:
                    source = '# Skipped in fast mode: ' + source.split('\n')[0] + '\n# Learning curves take time\npass\n'
            
            script_content.append(f"\n# Cell {cell_idx}\n")
            script_content.append("try:\n")
            for line in source.split('\n'):
                if line.strip():
                    script_content.append(f"    {line}\n")
                else:
                    script_content.append("\n")
            script_content.append(f"    execution_report['cells_executed'].append({{'cell': {cell_idx}, 'status': 'completed'}})\n")
            script_content.append("except Exception as e:\n")
            script_content.append(f"    execution_report['cells_executed'].append({{'cell': {cell_idx}, 'status': 'failed'}})\n")
            script_content.append(f"    execution_report['errors'].append({{'cell': {cell_idx}, 'error': str(e)}})\n")
            script_content.append(f"    print(f'❌ Error in cell {cell_idx}: {{e}}')\n")
            script_content.append("    import traceback\n")
            script_content.append("    traceback.print_exc()\n")
        
        # Capture data info
        if 'df = pd.read_csv' in source:
            script_content.append(f"    if 'df' in locals():\n")
            script_content.append(f"        execution_report['data_info']['shape'] = list(df.shape)\n")
            script_content.append(f"        execution_report['data_info']['columns'] = list(df.columns)\n")
            script_content.append(f"        execution_report['data_info']['memory_mb'] = float(df.memory_usage(deep=True).sum() / 1024**2)\n")
            script_content.append(f"        execution_report['data_info']['missing_pct'] = float(df.isnull().sum().sum() / df.size * 100)\n")
        
        # Capture model metrics
        if 'metrics = trainer.train' in source:
            script_content.append(f"    if 'metrics' in locals():\n")
            script_content.append(f"        execution_report['model_metrics'] = {{k: float(v) for k, v in metrics.items()}}\n")
        
        # Capture predictions
        if 'predictions = trainer.predict' in source or 'predictions = predictor.predict' in source:
            script_content.append(f"    if 'predictions' in locals():\n")
            script_content.append(f"        if isinstance(predictions, (list, np.ndarray)):\n")
            script_content.append(f"            execution_report['predictions'] = [float(p) for p in list(predictions)[:10]]\n")
            script_content.append(f"        else:\n")
            script_content.append(f"            execution_report['predictions'] = [float(predictions)]\n")
    
    # Add report saving at the end
    script_content.append("\n# Finalize and save report\n")
    script_content.append(f"execution_report['end_time'] = datetime.now().isoformat()\n")
    script_content.append(f"execution_report['duration'] = (datetime.fromisoformat(execution_report['end_time']) - datetime.fromisoformat(execution_report['start_time'])).total_seconds()\n")
    script_content.append(f"\n# Save timestamped reports\n")
    script_content.append(f"report_json = Path('notebooks/predictive_modeling/execution_report_{timestamp}.json')\n")
    script_content.append(f"report_md = Path('notebooks/predictive_modeling/EXECUTION_REPORT_{timestamp}.md')\n")
    script_content.append(f"report_json.parent.mkdir(parents=True, exist_ok=True)\n\n")
    script_content.append(f"# Save JSON report\n")
    script_content.append(f"with open(report_json, 'w') as f:\n")
    script_content.append(f"    json.dump(execution_report, f, indent=2, default=str)\n")
    script_content.append(f"print(f'\\n✅ JSON Report saved: {{report_json}}')\n\n")
    script_content.append(f"# Save Markdown report\n")
    script_content.append(f"with open(report_md, 'w') as f:\n")
    script_content.append(f"    f.write('# Comprehensive Modeling Notebook - Execution Report\\n\\n')\n")
    script_content.append(f"    f.write(f'**Execution Timestamp:** {{execution_report[\"timestamp\"]}}\\n')\n")
    script_content.append(f"    f.write(f'**Start Time:** {{execution_report[\"start_time\"]}}\\n')\n")
    script_content.append(f"    f.write(f'**End Time:** {{execution_report[\"end_time\"]}}\\n')\n")
    script_content.append(f"    f.write(f'**Duration:** {{execution_report[\"duration\"]:.2f}} seconds\\n\\n')\n")
    script_content.append(f"    f.write('## Summary\\n\\n')\n")
    script_content.append(f"    completed = [c for c in execution_report['cells_executed'] if c['status'] == 'completed']\n")
    script_content.append(f"    failed = [c for c in execution_report['cells_executed'] if c['status'] == 'failed']\n")
    script_content.append(f"    f.write(f'- ✅ Completed Cells: {{len(completed)}}/{{len(execution_report[\"cells_executed\"])}}\\n')\n")
    script_content.append(f"    f.write(f'- ❌ Failed Cells: {{len(failed)}}\\n')\n")
    script_content.append(f"    if execution_report.get('model_metrics'):\n")
    script_content.append(f"        f.write(f'- 🎯 R² Score: {{execution_report[\"model_metrics\"].get(\"r2\", \"N/A\"):.4f}}\\n')\n")
    script_content.append(f"    f.write('\\n## Details\\n\\n')\n")
    script_content.append(f"    if execution_report.get('data_info'):\n")
    script_content.append(f"        f.write('### Data Information\\n\\n')\n")
    script_content.append(f"        f.write(f'- Shape: {{execution_report[\"data_info\"].get(\"shape\", \"N/A\")}}\\n')\n")
    script_content.append(f"        f.write(f'- Memory: {{execution_report[\"data_info\"].get(\"memory_mb\", 0):.2f}} MB\\n')\n")
    script_content.append(f"    if execution_report.get('model_metrics'):\n")
    script_content.append(f"        f.write('\\n### Model Metrics\\n\\n')\n")
    script_content.append(f"        for metric, value in execution_report['model_metrics'].items():\n")
    script_content.append(f"            f.write(f'- {{metric}}: {{value:.6f}}\\n')\n")
    script_content.append(f"print(f'✅ Markdown Report saved: {{report_md}}')\n")
    
    # Write temporary script
    temp_script = project_root / "temp_run_notebook.py"
    with open(temp_script, 'w') as f:
        f.write(''.join(script_content))
    
    # Execute
    try:
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            cwd=str(project_root),
            text=True
        )
        
        if result.returncode == 0:
            print("\n" + "=" * 80)
            print("✅ NOTEBOOK EXECUTION COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\n📊 Reports generated with timestamp: {timestamp}")
            print(f"   - JSON: notebooks/predictive_modeling/execution_report_{timestamp}.json")
            print(f"   - Markdown: notebooks/predictive_modeling/EXECUTION_REPORT_{timestamp}.md")
        else:
            print("\n" + "=" * 80)
            print(f"⚠️  NOTEBOOK EXECUTION COMPLETED WITH EXIT CODE: {result.returncode}")
            print("=" * 80)
        
        return result.returncode
        
    finally:
        # Cleanup
        if temp_script.exists():
            temp_script.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run comprehensive modeling notebook')
    parser.add_argument('--fast', action='store_true', 
                       help='Use fast mode (faster hyperparameter tuning)')
    args = parser.parse_args()
    
    exit_code = run_notebook(fast_mode=args.fast)
    sys.exit(exit_code)

