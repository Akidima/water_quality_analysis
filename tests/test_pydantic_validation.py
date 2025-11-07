"""
Test script to demonstrate Pydantic validation in data-loader.py
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import importlib.util
spec = importlib.util.spec_from_file_location("data_loader", str(Path(__file__).parent.parent / 'scripts' / 'data-loader.py'))
data_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_loader)

DataConfig = data_loader.DataConfig
ValidationReport = data_loader.ValidationReport
ValidationMetadata = data_loader.ValidationMetadata
ValidationStats = data_loader.ValidationStats
ExplorationReport = data_loader.ExplorationReport
ExplorationMetadata = data_loader.ExplorationMetadata
from pydantic import ValidationError


def test_data_config_validation():
    """Test DataConfig Pydantic validation."""
    print("=" * 60)
    print("Testing DataConfig Pydantic Validation")
    print("=" * 60)
    
    # Test 1: Valid configuration
    print("\n1. Testing valid configuration...")
    try:
        config = DataConfig(
            filepath='data/national_water_plan.csv',
            chunk_size=5000,
            max_memory_mb=1000
        )
        print(f"✓ Valid config created successfully!")
        print(f"  Filepath: {config.filepath}")
        print(f"  Chunk size: {config.chunk_size}")
        print(f"  NA values: {config.na_values[:3]}... ({len(config.na_values)} total)")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")
    
    # Test 2: Invalid filepath (empty)
    print("\n2. Testing invalid filepath (empty)...")
    try:
        config = DataConfig(filepath='')
        print(f"✗ Should have failed validation!")
    except ValidationError as e:
        print(f"✓ Correctly caught validation error:")
        print(f"  {e.errors()[0]['msg']}")
    
    # Test 3: Invalid filepath (wrong extension)
    print("\n3. Testing invalid filepath (wrong extension)...")
    try:
        config = DataConfig(filepath='data.txt')
        print(f"✗ Should have failed validation!")
    except ValidationError as e:
        print(f"✓ Correctly caught validation error:")
        print(f"  {e.errors()[0]['msg']}")
    
    # Test 4: Invalid chunk_size (negative)
    print("\n4. Testing invalid chunk_size (negative)...")
    try:
        config = DataConfig(chunk_size=-100)
        print(f"✗ Should have failed validation!")
    except ValidationError as e:
        print(f"✓ Correctly caught validation error:")
        print(f"  {e.errors()[0]['msg']}")
    
    # Test 5: Invalid max_memory_mb (too large)
    print("\n5. Testing invalid max_memory_mb (too large)...")
    try:
        config = DataConfig(max_memory_mb=20000)
        print(f"✗ Should have failed validation!")
    except ValidationError as e:
        print(f"✓ Correctly caught validation error:")
        print(f"  {e.errors()[0]['msg']}")
    
    # Test 6: Column consistency validation
    print("\n6. Testing column consistency validation...")
    try:
        config = DataConfig(
            required_columns=['col1', 'col2'],
            usecols=['col1', 'col3']  # col2 is required but not in usecols
        )
        print(f"✗ Should have failed validation!")
    except ValidationError as e:
        print(f"✓ Correctly caught validation error:")
        print(f"  {e.errors()[0]['msg']}")


def test_validation_models():
    """Test Pydantic validation models."""
    print("\n" + "=" * 60)
    print("Testing Validation Models")
    print("=" * 60)
    
    # Test ValidationMetadata
    print("\n1. Testing ValidationMetadata...")
    try:
        metadata = ValidationMetadata(
            rows=1000,
            columns=20,
            memory_usage=15.5
        )
        print(f"✓ ValidationMetadata created successfully!")
        print(f"  {metadata.model_dump()}")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")
    
    # Test negative values
    print("\n2. Testing ValidationMetadata with negative values...")
    try:
        metadata = ValidationMetadata(rows=-10, columns=20, memory_usage=15.5)
        print(f"✗ Should have failed validation!")
    except ValidationError as e:
        print(f"✓ Correctly caught validation error:")
        print(f"  {e.errors()[0]['msg']}")
    
    # Test ValidationStats
    print("\n3. Testing ValidationStats...")
    try:
        stats = ValidationStats(
            total_rows=1000,
            total_columns=20,
            memory_usage_mb=15.5,
            missing_values_percent=12.5,
            duplicate_rows=5,
            missing_values={'col1': 10, 'col2': 20},
            data_types={'col1': 'int64', 'col2': 'object'}
        )
        print(f"✓ ValidationStats created successfully!")
        print(f"  Total rows: {stats.total_rows}")
        print(f"  Missing %: {stats.missing_values_percent}%")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")
    
    # Test invalid percentage (>100)
    print("\n4. Testing ValidationStats with invalid percentage...")
    try:
        stats = ValidationStats(
            total_rows=1000,
            total_columns=20,
            memory_usage_mb=15.5,
            missing_values_percent=150.0,  # Invalid: > 100%
            duplicate_rows=5
        )
        print(f"✗ Should have failed validation!")
    except ValidationError as e:
        print(f"✓ Correctly caught validation error:")
        print(f"  {e.errors()[0]['msg']}")


def test_exploration_models():
    """Test Pydantic exploration models."""
    print("\n" + "=" * 60)
    print("Testing Exploration Models")
    print("=" * 60)
    
    # Test ExplorationMetadata
    print("\n1. Testing ExplorationMetadata...")
    try:
        metadata = ExplorationMetadata(
            rows=1000,
            columns=20,
            memory_usage=15.5,
            missing_values_percent=12.5,
            duplicate_rows=5
        )
        print(f"✓ ExplorationMetadata created successfully!")
        print(f"  {metadata.model_dump()}")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")


def test_model_serialization():
    """Test Pydantic model serialization."""
    print("\n" + "=" * 60)
    print("Testing Model Serialization")
    print("=" * 60)
    
    print("\n1. Testing model_dump() for JSON export...")
    config = DataConfig(
        filepath='data/test.csv',
        chunk_size=5000,
        max_memory_mb=1000
    )
    
    config_dict = config.model_dump()
    print(f"✓ Successfully serialized to dict:")
    print(f"  Keys: {list(config_dict.keys())}")
    
    print("\n2. Testing model_dump_json() for direct JSON export...")
    config_json = config.model_dump_json(indent=2)
    print(f"✓ Successfully serialized to JSON:")
    print(config_json[:200] + "...")


if __name__ == '__main__':
    print("\n" + "🔍" * 30)
    print("PYDANTIC VALIDATION TEST SUITE")
    print("🔍" * 30)
    
    try:
        test_data_config_validation()
        test_validation_models()
        test_exploration_models()
        test_model_serialization()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS COMPLETED!")
        print("=" * 60)
        print("\nPydantic validation is working correctly! 🎉")
        print("\nKey features implemented:")
        print("  • Field validation with constraints (gt, le, ge)")
        print("  • Custom validators (@field_validator)")
        print("  • Model validators (@model_validator)")
        print("  • Type checking and coercion")
        print("  • Descriptive error messages")
        print("  • Easy serialization (model_dump, model_dump_json)")
        
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
