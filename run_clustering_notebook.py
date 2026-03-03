#!/usr/bin/env python3
"""
Script to execute the comprehensive_clustering.ipynb notebook.
"""

import json
import sys
from pathlib import Path

def run_clustering_notebook():
    """Execute the clustering notebook."""
    
    project_root = Path(__file__).parent
    notebook_path = project_root / "notebooks" / "site_clustering" / "comprehensive_clustering.ipynb"
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return 1
    
    print("=" * 80)
    print("RUNNING COMPREHENSIVE CLUSTERING NOTEBOOK")
    print("=" * 80)
    print(f"Notebook: {notebook_path}\n")
    
    # Read notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Extract code cells and combine source lines
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = source
            code_cells.append(code)
    
    # Combine all code
    full_code = '\n\n'.join(code_cells)
    
    # Write to temp file
    temp_script = project_root / 'temp_clustering_script.py'
    temp_script.write_text(full_code)
    
    print("Executing notebook code...\n")
    
    # Execute - try to use env Python if available, otherwise use system Python
    try:
        import subprocess
        python_exec = project_root / "env" / "bin" / "python"
        if python_exec.exists():
            python_cmd = str(python_exec)
            print(f"Using Python from: {python_cmd}\n")
        else:
            python_cmd = sys.executable
            print(f"Using system Python: {python_cmd}\n")
        
        result = subprocess.run(
            [python_cmd, str(temp_script)],
            cwd=str(project_root),
            text=True
        )
        
        if result.returncode == 0:
            print("\n" + "=" * 80)
            print("✅ NOTEBOOK EXECUTION COMPLETED SUCCESSFULLY!")
            print("=" * 80)
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
    exit_code = run_clustering_notebook()
    sys.exit(exit_code)
