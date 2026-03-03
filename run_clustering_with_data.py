#!/usr/bin/env python3
"""
Script to run clustering analysis with actual data.
Loads data, calculates Avg_Annual_Spills, and performs clustering.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues

# Import clustering functions from notebook
# We'll execute the notebook code first, then use it
import json

def load_and_prepare_data():
    """Load data and calculate Avg_Annual_Spills."""
    data_file = project_root / "export" / "cleaned_data" / "cleaned_water_data.csv"
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return None
    
    print(f"📊 Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Calculate Avg_Annual_Spills from spill event columns
    spill_cols = [col for col in df.columns if 'Spill Events' in col]
    if spill_cols:
        print(f"   Found spill columns: {spill_cols}")
        df['Avg_Annual_Spills'] = df[spill_cols].mean(axis=1)
        print(f"   ✅ Calculated Avg_Annual_Spills (mean: {df['Avg_Annual_Spills'].mean():.2f})")
    else:
        print("   ⚠️  No spill event columns found!")
        return None
    
    # Calculate Spill_Trend if we have multiple years
    if len(spill_cols) >= 2:
        df['Spill_Trend'] = df[spill_cols[-1]] - df[spill_cols[0]]
        print(f"   ✅ Calculated Spill_Trend")
    
    # Clean flag columns - convert 'Y'/'N' to 1/0, handle other values
    flag_columns = [col for col in df.columns if 'Flag' in col]
    for col in flag_columns:
        if col in df.columns:
            # Convert Y/N to 1/0, keep numeric values, convert others to 0
            df[col] = df[col].replace({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0})
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            print(f"   ✅ Cleaned flag column: {col}")
    
    return df

def run_clustering():
    """Run the clustering analysis."""
    # Load notebook code
    notebook_path = project_root / "notebooks" / "site_clustering" / "comprehensive_clustering.ipynb"
    
    print("\n" + "=" * 80)
    print("RUNNING CLUSTERING ANALYSIS")
    print("=" * 80)
    
    # Load data
    df = load_and_prepare_data()
    if df is None:
        return 1
    
    # Execute notebook code to get clustering functions
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Extract and execute code cells
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = source
            code_cells.append(code)
    
    # Combine code
    full_code = '\n\n'.join(code_cells)
    
    # Execute in namespace
    namespace = {}
    exec(full_code, namespace)
    
    # Get the clustering function
    perform_site_clustering = namespace.get('perform_site_clustering')
    if not perform_site_clustering:
        print("❌ Could not find perform_site_clustering function")
        return 1
    
    # Run clustering
    print("\n🔍 Starting clustering analysis...")
    try:
        cluster_analysis, high_risk_sites = perform_site_clustering(df)
        
        if cluster_analysis:
            print("\n" + "=" * 80)
            print("✅ CLUSTERING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\n📊 Results:")
            print(f"   Optimal clusters: {cluster_analysis.get('optimal_k', 'N/A')}")
            print(f"   Quality: {cluster_analysis.get('quality_interpretation', 'N/A')}")
            
            if high_risk_sites is not None:
                print(f"   High-risk sites identified: {len(high_risk_sites)}")
            
            print(f"\n📁 Output files:")
            print(f"   - plots/clustering_analysis.png")
            print(f"   - models/clustering_model.pkl")
            print(f"   - models/clustering_scaler.pkl")
            print(f"   - models/clustering_model_metadata.pkl")
            
            return 0
        else:
            print("❌ Clustering failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during clustering: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = run_clustering()
    sys.exit(exit_code)
