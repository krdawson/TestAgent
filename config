# config.py - Configuration Management
"""
Configuration management for the Operations Analyst Agent
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = "sqlite"  # sqlite, postgresql, mysql
    host: str = "localhost"
    port: int = 5432
    name: str = "ops_agent"
    username: str = ""
    password: str = ""
    sqlite_path: str = "ops_agent.db"

@dataclass
class DataConfig:
    """Data processing configuration"""
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    backup_path: str = "data/backups"
    max_file_size_mb: int = 100
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["csv", "xlsx", "json"]

@dataclass
class MetricsConfig:
    """Metrics calculation configuration"""
    # Support metrics thresholds
    max_resolution_time_days: int = 5
    min_satisfaction_score: float = 4.0
    
    # Implementation metrics thresholds
    max_implementation_days: int = 60
    min_completion_rate: float = 80.0
    
    # Usage metrics thresholds
    min_feature_usage_score: float = 0.6
    min_active_users: int = 10
    
    # Forecasting parameters
    forecast_periods: int = 6
    confidence_interval: float = 0.95
    min_historical_points: int = 12

@dataclass
class APIConfig:
    """API configuration"""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    max_tokens: int = 1000
    temperature: float = 0.3
    
class AppConfig:
    """Main application configuration"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.data = DataConfig()
        self.metrics = MetricsConfig()
        self.api = APIConfig()
        self.load_from_env()
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        # Database
        self.database.type = os.getenv("DB_TYPE", self.database.type)
        self.database.host = os.getenv("DB_HOST", self.database.host)
        self.database.port = int(os.getenv("DB_PORT", self.database.port))
        self.database.name = os.getenv("DB_NAME", self.database.name)
        self.database.username = os.getenv("DB_USERNAME", self.database.username)
        self.database.password = os.getenv("DB_PASSWORD", self.database.password)
        
        # API keys
        self.api.openai_api_key = os.getenv("OPENAI_API_KEY", self.api.openai_api_key)
        self.api.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", self.api.anthropic_api_key)
        
        # Create directories
        Path(self.data.raw_data_path).mkdir(parents=True, exist_ok=True)
        Path(self.data.processed_data_path).mkdir(parents=True, exist_ok=True)
        Path(self.data.backup_path).mkdir(parents=True, exist_ok=True)
