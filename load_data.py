#!/usr/bin/env python
"""
Convenience wrapper script to run the data loader from the project root.

This script allows you to run the data loader easily:
    python load_data.py
    python load_data.py --file data/my_file.csv
    python load_data.py --help
"""

import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / 'scripts'
sys.path.insert(0, str(scripts_dir))

# Import and run the main function
if __name__ == "__main__":
    # Import the module (note: hyphenated filename requires special handling)
    import importlib.util
    loader_module_path = scripts_dir / 'data-loader.py'
    spec = importlib.util.spec_from_file_location("data_loader", loader_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {loader_module_path}")
    data_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_loader)
    
    # Run the main function
    sys.exit(data_loader.main())

