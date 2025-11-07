# Pydantic Validation Quick Reference

## Import Models

```python
from scripts.data_loader import (
    DataConfig,
    ValidationReport,
    ValidationMetadata,
    ValidationStats,
    ExplorationReport,
    ExplorationMetadata
)
```

## Common Usage Patterns

### 1. Create Configuration with Validation

```python
from scripts.data_loader import DataConfig

# Basic configuration
config = DataConfig(filepath='data/my_data.csv')

# Advanced configuration with validation
config = DataConfig(
    filepath='data/water_quality.csv',
    chunk_size=10000,              # Must be > 0
    max_memory_mb=500,              # Must be 0 < x ≤ 10000
    required_columns=['Location', 'pH', 'Temperature'],
    usecols=['Location', 'pH', 'Temperature', 'Date'],
    dtype_optimization=True
)

# Access configuration values
print(config.filepath)
print(config.chunk_size)
print(config.na_values)  # Default NA values automatically set
```

### 2. Load Data with Validated Configuration

```python
from scripts.data_loader import DataLoader, DataConfig

# Create validated config
config = DataConfig(
    filepath='data/national_water_plan.csv',
    max_memory_mb=1000
)

# Load data
loader = DataLoader(config)
df, exploration_report = loader.load_and_explore_data()

# Access report attributes (all type-safe!)
print(f"Rows: {exploration_report.metadata.rows}")
print(f"Columns: {exploration_report.metadata.columns}")
print(f"Duration: {exploration_report.duration}s")
print(f"Valid: {exploration_report.validation.is_valid}")
```

### 3. Export Report to JSON

```python
# Method 1: Using the loader's save method
loader.save_exploration_report(
    report=exploration_report,
    output_path='reports/data_exploration.json'
)

# Method 2: Direct JSON serialization
json_dict = exploration_report.model_dump()
json_string = exploration_report.model_dump_json(indent=2)

# Save manually
import json
with open('report.json', 'w') as f:
    json.dump(exploration_report.model_dump(), f, indent=2, default=str)
```

### 4. Validation Error Handling

```python
from pydantic import ValidationError

try:
    # This will fail - negative chunk size
    config = DataConfig(
        filepath='data.csv',
        chunk_size=-100
    )
except ValidationError as e:
    print("Validation errors:")
    for error in e.errors():
        print(f"  Field: {error['loc']}")
        print(f"  Error: {error['msg']}")
        print(f"  Type: {error['type']}")
```

### 5. Working with Validation Reports

```python
# After loading data
df, exploration_report = loader.load_and_explore_data()

# Check validation status
if exploration_report.validation.is_valid:
    print("✓ Data is valid!")
else:
    print("✗ Validation failed:")
    for error in exploration_report.validation.errors:
        print(f"  - {error}")

# Check warnings
if exploration_report.validation.warnings:
    print("⚠ Warnings:")
    for warning in exploration_report.validation.warnings:
        print(f"  - {warning}")

# Access statistics (type-safe!)
stats = exploration_report.validation.stats
if stats:
    print(f"Total rows: {stats.total_rows}")
    print(f"Missing data: {stats.missing_values_percent:.2f}%")
    print(f"Duplicates: {stats.duplicate_rows}")
```

## Field Constraints Reference

### DataConfig Fields

| Field | Type | Constraints | Default |
|-------|------|-------------|---------|
| `filepath` | `str` | Non-empty, must end with `.csv` | `'national_water_plan.csv'` |
| `na_values` | `List[str]` | Non-empty list | `['', 'TBC', 'N/A', ...]` |
| `chunk_size` | `Optional[int]` | Must be > 0 (if set) | `None` |
| `max_memory_mb` | `int` | Must be 0 < x ≤ 10000 | `500` |
| `required_columns` | `Optional[List[str]]` | Must be subset of `usecols` | `None` |
| `usecols` | `Optional[List[str]]` | No constraints | `None` |
| `dtype_optimization` | `bool` | No constraints | `True` |

### ValidationMetadata Fields

| Field | Type | Constraints |
|-------|------|-------------|
| `rows` | `int` | ≥ 0 |
| `columns` | `int` | ≥ 0 |
| `memory_usage` | `float` | ≥ 0.0 |

### ValidationStats Fields

| Field | Type | Constraints |
|-------|------|-------------|
| `total_rows` | `int` | ≥ 0 |
| `total_columns` | `int` | ≥ 0 |
| `memory_usage_mb` | `float` | ≥ 0.0 |
| `missing_values_percent` | `float` | 0.0 ≤ x ≤ 100.0 |
| `duplicate_rows` | `int` | ≥ 0 |

### ExplorationMetadata Fields

| Field | Type | Constraints |
|-------|------|-------------|
| `rows` | `int` | ≥ 0 |
| `columns` | `int` | ≥ 0 |
| `memory_usage` | `float` | ≥ 0.0 |
| `missing_values_percent` | `float` | 0.0 ≤ x ≤ 100.0 |
| `duplicate_rows` | `int` | ≥ 0 |

## Common Validation Errors

### 1. Empty Filepath
```python
config = DataConfig(filepath='')
# ❌ ValidationError: Filepath cannot be empty
```

### 2. Wrong File Extension
```python
config = DataConfig(filepath='data.txt')
# ❌ ValidationError: Filepath must point to a CSV file (.csv extension)
```

### 3. Invalid Numeric Constraints
```python
config = DataConfig(chunk_size=-100)
# ❌ ValidationError: Input should be greater than 0

config = DataConfig(max_memory_mb=20000)
# ❌ ValidationError: Input should be less than or equal to 10000
```

### 4. Column Consistency
```python
config = DataConfig(
    required_columns=['col1', 'col2'],
    usecols=['col1']  # col2 is missing!
)
# ❌ ValidationError: Required columns {'col2'} must be included in usecols
```

## Model Serialization

### To Dictionary
```python
config_dict = config.model_dump()
report_dict = exploration_report.model_dump()
```

### To JSON String
```python
config_json = config.model_dump_json(indent=2)
report_json = exploration_report.model_dump_json(indent=2)
```

### From Dictionary (Validation Applied)
```python
config_dict = {
    'filepath': 'data.csv',
    'chunk_size': 5000
}
config = DataConfig(**config_dict)  # Validates on creation
```

## Best Practices

### ✅ DO

- Always wrap configuration creation in try-except for ValidationError
- Use type hints when working with Pydantic models
- Access model fields as attributes (e.g., `config.filepath`)
- Use `model_dump()` for serialization
- Take advantage of IDE autocomplete with Pydantic models

### ❌ DON'T

- Don't access `__dict__` directly (use `model_dump()` instead)
- Don't skip validation error handling
- Don't modify model fields after creation without re-validation
- Don't use dictionary access (e.g., `config['filepath']`)

## Testing Your Code

```python
from pydantic import ValidationError

def test_config_creation():
    """Test that valid config is created successfully."""
    config = DataConfig(
        filepath='test.csv',
        chunk_size=1000
    )
    assert config.filepath == 'test.csv'
    assert config.chunk_size == 1000

def test_invalid_config():
    """Test that invalid config raises ValidationError."""
    try:
        config = DataConfig(filepath='')
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert 'Filepath cannot be empty' in str(e)

# Run tests
test_config_creation()
test_invalid_config()
print("✓ All tests passed!")
```

## Resources

- **Pydantic Documentation**: https://docs.pydantic.dev/
- **Full Implementation Details**: See `PYDANTIC_IMPLEMENTATION.md`
- **Test Suite**: Run `python tests/test_pydantic_validation.py`

---

**Quick Tip**: Use your IDE's autocomplete (Ctrl+Space or Cmd+Space) when working with Pydantic models to discover available fields and their types!
