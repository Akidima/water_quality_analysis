# Water Quality Analysis

Production-grade utilities for exploring, validating, and cleaning large-scale water quality datasets.  
The toolkit combines Pydantic-powered configuration, Dask-based scalability, and automated reporting to streamline data readiness for downstream analytics or modelling.

---

## Highlights
- **Robust ingestion** via `scripts/data-loader.py`, including configurable NA handling, chunked reads, dtype optimisation, and validation reports.
- **Cleaning workflows** in `scripts/data-cleaner.py` with reusable transformations and test coverage.
- **Pydantic enhancements** for schema management and consistent configuration across components.
- **Extensive documentation** under `docs/` plus example notebooks/scripts for quick experimentation.

---

## Project Layout
```
water_quality_analysis/
├── data/                      # Source datasets (sample CSV included)
├── docs/                      # Detailed implementation & enhancement guides
├── examples/                  # How-to scripts illustrating API usage
├── export/                    # Generated reports (e.g., data_exploration_report.json)
├── scripts/                   # Core modules (loader, cleaner, pydantic helpers)
├── tests/                     # Pytest suite covering validation & cleaning logic
├── environment.yml            # Reproducible Conda environment (Python 3.13)
└── README.md
```

---

## Getting Started

### 1. Clone
```bash
git clone https://github.com/<your-user>/water_quality_analysis.git
cd water_quality_analysis
```

### 2. Create the Conda Environment
> **Note:** `environment.yml` was authored with a prefix pointing to the original dev machine.  
> Update the first line (`name:`) to something like `water_quality_analysis`, or override the prefix when creating the env.

```bash
conda env create --name water_quality_analysis --file environment.yml
# or, to reuse the on-disk env layout:
# conda env create --prefix ./env --file environment.yml

conda activate water_quality_analysis
```

### 3. Verify key dependencies
```bash
python -c "import dask, pydantic; print(dask.__version__, pydantic.__version__)"
```

---

## Usage

### Load & Explore Data
```bash
python scripts/data-loader.py
```
- Produces an `ExplorationReport` with validation stats, missing-value analysis, and memory metrics.
- Saves a JSON report to `export/data_exploration_report.json`.

### Clean Data
```bash
python scripts/data-cleaner.py --help
```
- CLI options let you target specific cleaning pipelines, output formats, and logging levels.

### Programmatic API Example
```python
from scripts.data_loader import DataLoader, DataConfig

config = DataConfig(
    filepath="data/national_water_plan.csv",
    required_columns=["Location", "Sample Date"],
)
df, report = DataLoader(config).load_and_explore_data()
print(report.metadata.model_dump())
```

---

## Testing
```bash
pytest
```

Continuous integration-style checks validate:
- Data loader validation boundaries (`tests/test_pydantic_validation.py`)
- Cleaner transformations (`tests/test_data_cleaner.py`)
- Pydantic schema extensions (`tests/test_pydantic_enhancements.py`)

---

## Documentation & Resources
- Quick-start and enhancement guides live in `docs/`.
- Generated reports (EDA summaries, cleaner outputs) drop into `export/`.
- Example workflows: `examples/example_usage.py`.

---

## Contributing
1. Fork the repo and create a feature branch.
2. Keep code formatted and linted; ensure tests pass.
3. Submit a PR describing motivation, changes, and validation steps.

---

## License
Specify your project’s license (e.g., MIT, Apache-2.0) here to clarify usage rights.


