"""
Pydantic Enhancements Module
Additional validators, models, and error handlers for water quality data processing.
"""

import re
from typing import Dict, Any, List, Optional, Union, ClassVar
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Enhancement 1: Column Name/Pattern Validators
# ============================================================================

class ColumnNameValidator(BaseModel):
    """Validator for column naming conventions."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Allowed patterns for column names
    ALLOWED_PATTERNS: ClassVar[Dict[str, str]] = {
        'coordinate': r'^(Latitude|Longitude|Easting|Northing)$',
        'spill_event': r'^Spill Events \d{4}$',
        'water_company': r'^Water Company$',
        'river_basin': r'^River Basin District$',
        'site_name': r'^Site Name$',
        'permit': r'^Permit Number$',
        'receiving_env': r'^Receiving Environment$',
        'duration': r'^Duration \(Hours\)$',
        'date': r'^(Date|Start Date|End Date)$',
        'count': r'.*Count$',
        'total': r'.*Total$',
    }
    
    column_name: str = Field(..., description="Column name to validate")
    
    @field_validator('column_name')
    @classmethod
    def validate_column_name(cls, v: str) -> str:
        """Validate column name against patterns."""
        # Check if it matches any allowed pattern
        for pattern_name, pattern in cls.ALLOWED_PATTERNS.items():
            if re.match(pattern, v, re.IGNORECASE):
                logger.debug(f"Column '{v}' matches pattern '{pattern_name}'")
                return v
        
        # Warn about non-standard column names
        logger.warning(f"Column '{v}' does not match any standard pattern")
        return v
    
    @classmethod
    def validate_columns(cls, columns: List[str]) -> Dict[str, List[str]]:
        """
        Validate a list of column names and categorize them.
        
        Returns:
            Dict with 'valid', 'warnings', and 'suggestions' keys
        """
        results = {
            'valid': [],
            'warnings': [],
            'suggestions': []
        }
        
        for col in columns:
            # Always try to get suggestions
            suggestions = cls._suggest_corrections(col)
            
            try:
                validated = cls(column_name=col)
                results['valid'].append(col)
                # Add suggestions even for valid columns if typos detected
                if suggestions:
                    results['suggestions'].append({
                        'column': col,
                        'suggestions': suggestions
                    })
            except ValidationError as e:
                results['warnings'].append(col)
                # Suggest corrections for common typos
                if suggestions:
                    results['suggestions'].append({
                        'column': col,
                        'suggestions': suggestions
                    })
        
        return results
    
    @staticmethod
    def _suggest_corrections(column_name: str) -> List[str]:
        """Suggest corrections for misspelled column names."""
        common_corrections = {
            'receuving': 'Receiving',
            'enviroment': 'Environment',
            'permitt': 'Permit',
            'compnay': 'Company',
            'latitute': 'Latitude',
            'longitute': 'Longitude',
        }
        
        suggestions = []
        col_lower = column_name.lower()
        
        for typo, correction in common_corrections.items():
            if typo in col_lower:
                # Case-insensitive replacement
                import re
                pattern = re.compile(re.escape(typo), re.IGNORECASE)
                suggested = pattern.sub(correction, column_name)
                suggestions.append(suggested)
        
        return suggestions


# ============================================================================
# Enhancement 2: Statistical Summary Models
# ============================================================================

class NumericStatistics(BaseModel):
    """Pydantic model for numeric column statistics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    column_name: str
    count: int = Field(ge=0)
    mean: Optional[float] = None
    std: Optional[float] = Field(default=None, ge=0.0)
    min: Optional[float] = None
    q25: Optional[float] = None  # 25th percentile
    median: Optional[float] = None
    q75: Optional[float] = None  # 75th percentile
    max: Optional[float] = None
    missing_count: int = Field(default=0, ge=0)
    missing_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    outlier_count: int = Field(default=0, ge=0)
    outlier_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    
    @field_validator('mean', 'median')
    @classmethod
    def validate_central_tendency(cls, v: Optional[float], info) -> Optional[float]:
        """Validate that central tendency measures are reasonable."""
        if v is not None:
            values = info.data
            if 'min' in values and 'max' in values:
                if values['min'] is not None and values['max'] is not None:
                    if not (values['min'] <= v <= values['max']):
                        raise ValueError(
                            f"Mean/Median {v} outside range [{values['min']}, {values['max']}]"
                        )
        return v
    
    def quality_score(self) -> float:
        """
        Calculate data quality score (0-100).
        
        Based on:
        - Missing data percentage (lower is better)
        - Outlier percentage (lower is better)
        - Data availability (higher is better)
        """
        score = 100.0
        
        # Penalize for missing data
        score -= self.missing_percent * 0.5
        
        # Penalize for outliers
        score -= self.outlier_percent * 0.3
        
        # Penalize if no statistics available
        if self.mean is None or self.std is None:
            score -= 20.0
        
        return max(0.0, min(100.0, score))


class CategoricalStatistics(BaseModel):
    """Pydantic model for categorical column statistics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    column_name: str
    count: int = Field(ge=0)
    unique_count: int = Field(ge=0)
    unique_percent: float = Field(ge=0.0, le=100.0)
    top_value: Optional[str] = None
    top_frequency: int = Field(default=0, ge=0)
    missing_count: int = Field(default=0, ge=0)
    missing_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    
    @field_validator('unique_count')
    @classmethod
    def validate_unique_count(cls, v: int, info) -> int:
        """Validate that unique count doesn't exceed total count."""
        if 'count' in info.data and v > info.data['count']:
            raise ValueError(
                f"Unique count {v} cannot exceed total count {info.data['count']}"
            )
        return v
    
    def quality_score(self) -> float:
        """Calculate data quality score (0-100)."""
        score = 100.0
        
        # Penalize for missing data
        score -= self.missing_percent * 0.5
        
        # Penalize if too many unique values (might indicate data quality issues)
        if self.unique_percent > 90:
            score -= 20.0
        
        # Penalize if no top value identified
        if self.top_value is None:
            score -= 10.0
        
        return max(0.0, min(100.0, score))


class DatasetStatistics(BaseModel):
    """Comprehensive statistics for entire dataset."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    total_rows: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    numeric_columns: Dict[str, NumericStatistics] = Field(default_factory=dict)
    categorical_columns: Dict[str, CategoricalStatistics] = Field(default_factory=dict)
    memory_usage_mb: float = Field(ge=0.0)
    overall_missing_percent: float = Field(ge=0.0, le=100.0)
    duplicate_rows: int = Field(ge=0)
    
    def overall_quality_score(self) -> float:
        """Calculate overall dataset quality score."""
        scores = []
        
        # Numeric column scores
        for stats in self.numeric_columns.values():
            scores.append(stats.quality_score())
        
        # Categorical column scores
        for stats in self.categorical_columns.values():
            scores.append(stats.quality_score())
        
        if not scores:
            return 0.0
        
        avg_score = sum(scores) / len(scores)
        
        # Penalize for overall missing data
        avg_score -= self.overall_missing_percent * 0.3
        
        return max(0.0, min(100.0, avg_score))
    
    def summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        return {
            'dataset_overview': {
                'rows': self.total_rows,
                'columns': self.total_columns,
                'memory_mb': round(self.memory_usage_mb, 2),
                'missing_percent': round(self.overall_missing_percent, 2),
                'duplicates': self.duplicate_rows
            },
            'quality_score': round(self.overall_quality_score(), 2),
            'numeric_columns': len(self.numeric_columns),
            'categorical_columns': len(self.categorical_columns),
            'column_quality_scores': {
                **{name: round(stats.quality_score(), 2) 
                   for name, stats in self.numeric_columns.items()},
                **{name: round(stats.quality_score(), 2) 
                   for name, stats in self.categorical_columns.items()}
            }
        }


# ============================================================================
# Enhancement 3: Custom Error Handlers
# ============================================================================

class DataValidationError(Exception):
    """Base exception for data validation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'details': self.details
        }


class ColumnValidationError(DataValidationError):
    """Error for column-related validation failures."""
    
    def __init__(self, column_name: str, issue: str, suggestion: Optional[str] = None):
        details = {
            'column_name': column_name,
            'issue': issue
        }
        if suggestion:
            details['suggestion'] = suggestion
        
        message = f"Column '{column_name}' validation failed: {issue}"
        if suggestion:
            message += f" (Suggestion: {suggestion})"
        
        super().__init__(message, details)


class RangeValidationError(DataValidationError):
    """Error for value range validation failures."""
    
    def __init__(
        self, 
        field_name: str, 
        value: Any, 
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ):
        details = {
            'field_name': field_name,
            'value': value,
            'min_value': min_value,
            'max_value': max_value
        }
        
        range_str = ""
        if min_value is not None and max_value is not None:
            range_str = f"[{min_value}, {max_value}]"
        elif min_value is not None:
            range_str = f">= {min_value}"
        elif max_value is not None:
            range_str = f"<= {max_value}"
        
        message = f"Value {value} for '{field_name}' is outside valid range {range_str}"
        super().__init__(message, details)


class DataQualityError(DataValidationError):
    """Error for data quality issues."""
    
    def __init__(
        self,
        issue: str,
        affected_rows: int,
        total_rows: int,
        threshold: Optional[float] = None
    ):
        percent = (affected_rows / total_rows * 100) if total_rows > 0 else 0
        
        details = {
            'issue': issue,
            'affected_rows': affected_rows,
            'total_rows': total_rows,
            'affected_percent': round(percent, 2)
        }
        
        if threshold is not None:
            details['threshold'] = threshold
        
        message = f"Data quality issue: {issue} affects {affected_rows}/{total_rows} rows ({percent:.2f}%)"
        if threshold is not None:
            message += f" (threshold: {threshold * 100:.1f}%)"
        
        super().__init__(message, details)


class ErrorHandler:
    """Centralized error handler for validation errors."""
    
    def __init__(self):
        self.errors: List[DataValidationError] = []
        self.warnings: List[str] = []
    
    def add_error(self, error: DataValidationError) -> None:
        """Add an error to the handler."""
        self.errors.append(error)
        logger.error(error.message)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the handler."""
        self.warnings.append(warning)
        logger.warning(warning)
    
    def has_errors(self) -> bool:
        """Check if any errors were recorded."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if any warnings were recorded."""
        return len(self.warnings) > 0
    
    def get_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report."""
        return {
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': [error.to_dict() for error in self.errors],
            'warnings': self.warnings
        }
    
    def clear(self) -> None:
        """Clear all errors and warnings."""
        self.errors.clear()
        self.warnings.clear()


# ============================================================================
# Enhancement 4: Pydantic Settings Management
# ============================================================================

class ApplicationSettings(BaseSettings):
    """
    Application-wide settings using Pydantic BaseSettings.
    
    Loads settings from environment variables or .env file.
    """
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )
    
    # Application metadata
    app_name: str = Field(default="Water Quality Analysis", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    # Data paths
    data_directory: Path = Field(default=Path("data"), description="Data directory")
    export_directory: Path = Field(default=Path("export"), description="Export directory")
    backup_directory: Path = Field(default=Path("backups"), description="Backup directory")
    
    # Processing configuration
    max_workers: int = Field(default=4, gt=0, le=32, description="Maximum parallel workers")
    chunk_size: int = Field(default=10000, gt=0, description="Data chunk size")
    memory_limit_mb: int = Field(default=1000, gt=0, description="Memory limit in MB")
    
    # Data quality thresholds
    missing_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    duplicate_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    outlier_threshold: float = Field(default=3.0, gt=0.0)
    
    # Geographic bounds (UK defaults)
    lat_min: float = Field(default=49.0, ge=-90.0, le=90.0)
    lat_max: float = Field(default=61.0, ge=-90.0, le=90.0)
    lon_min: float = Field(default=-11.0, ge=-180.0, le=180.0)
    lon_max: float = Field(default=2.0, ge=-180.0, le=180.0)
    
    # Validation settings
    strict_validation: bool = Field(default=True, description="Enable strict validation")
    create_backups: bool = Field(default=True, description="Create data backups")
    save_reports: bool = Field(default=True, description="Save processing reports")
    
    @field_validator('data_directory', 'export_directory', 'backup_directory')
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Create directories if they don't exist."""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper
    
    def configure_logging(self) -> None:
        """Configure logging based on settings."""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=self.log_format,
            filename=str(self.log_file) if self.log_file else None
        )
    
    def to_cleaner_config(self) -> Dict[str, Any]:
        """Convert to DataCleanerConfig parameters."""
        return {
            'lat_min': self.lat_min,
            'lat_max': self.lat_max,
            'lon_min': self.lon_min,
            'lon_max': self.lon_max,
            'missing_value_threshold': self.missing_threshold,
            'duplicate_value_threshold': self.duplicate_threshold,
            'outlier_std_threshold': self.outlier_threshold,
            'strict_mode': self.strict_validation,
            'create_backup': self.create_backups,
            'save_cleaning_report': self.save_reports,
            'output_directory': str(self.export_directory),
            'n_partitions': self.max_workers,
            'log_level': self.log_level,
            'log_format': self.log_format
        }
    
    def to_loader_config(self) -> Dict[str, Any]:
        """Convert to DataConfig parameters."""
        return {
            'chunk_size': self.chunk_size,
            'max_memory_mb': self.memory_limit_mb,
            'dtype_optimization': True
        }


# ============================================================================
# Utility Functions
# ============================================================================

def load_settings(env_file: Optional[Path] = None) -> ApplicationSettings:
    """
    Load application settings from environment or .env file.
    
    Args:
        env_file: Optional path to .env file
        
    Returns:
        ApplicationSettings instance
    """
    if env_file:
        return ApplicationSettings(_env_file=str(env_file))
    return ApplicationSettings()


def validate_dataset_structure(
    columns: List[str],
    required_patterns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate dataset structure against expected patterns.
    
    Args:
        columns: List of column names
        required_patterns: Optional list of regex patterns that must be present
        
    Returns:
        Validation results dictionary
    """
    validator = ColumnNameValidator
    results = validator.validate_columns(columns)
    
    if required_patterns:
        missing_patterns = []
        for pattern in required_patterns:
            if not any(re.match(pattern, col, re.IGNORECASE) for col in columns):
                missing_patterns.append(pattern)
        
        if missing_patterns:
            results['missing_required_patterns'] = missing_patterns
    
    return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("PYDANTIC ENHANCEMENTS DEMO")
    print("="*80)
    
    # Enhancement 1: Column validation
    print("\n1. Column Name Validation:")
    test_columns = [
        'Latitude', 'Longitude', 'Water Company',
        'Spill Events 2023', 'Receuving Environment',  # Typo!
        'Site Name', 'Permit Number'
    ]
    
    results = ColumnNameValidator.validate_columns(test_columns)
    print(f"   Valid columns: {len(results['valid'])}")
    print(f"   Warnings: {len(results['warnings'])}")
    if results['suggestions']:
        print(f"   Suggestions: {results['suggestions']}")
    
    # Enhancement 2: Statistical models
    print("\n2. Statistical Summary Models:")
    numeric_stats = NumericStatistics(
        column_name='Latitude',
        count=1000,
        mean=52.5,
        std=2.3,
        min=49.0,
        median=52.7,
        max=61.0,
        q25=51.2,
        q75=54.1,
        missing_count=10,
        missing_percent=1.0
    )
    print(f"   Quality Score: {numeric_stats.quality_score():.2f}/100")
    
    # Enhancement 3: Custom error handling
    print("\n3. Custom Error Handlers:")
    error_handler = ErrorHandler()
    error_handler.add_error(
        ColumnValidationError('Invalid_Column', 'Does not match pattern', 'Use standard naming')
    )
    error_handler.add_warning('High missing data percentage detected')
    print(f"   Errors: {error_handler.get_error_report()['error_count']}")
    print(f"   Warnings: {error_handler.get_error_report()['warning_count']}")
    
    # Enhancement 4: Settings management
    print("\n4. Application Settings:")
    settings = ApplicationSettings()
    print(f"   App: {settings.app_name} v{settings.app_version}")
    print(f"   Workers: {settings.max_workers}")
    print(f"   Data dir: {settings.data_directory}")
    print(f"   Geographic bounds: Lat [{settings.lat_min}, {settings.lat_max}], "
          f"Lon [{settings.lon_min}, {settings.lon_max}]")
    
    print("\n" + "="*80)
    print("✅ All enhancements working correctly!")
    print("="*80)
