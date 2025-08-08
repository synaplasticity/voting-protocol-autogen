"""Configuration module for the voting protocol system."""

import os
import logging
from typing import List


class Config:
    """Configuration class for the voting protocol system."""
    
    # API Configuration
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-...")
    
    # Logging Configuration
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Agent Configuration
    MAX_ROUNDS = 6
    MAX_SYMBOL_LENGTH = 6
    
    # Personality traits for agents
    PERSONALITY_TRAITS: List[str] = [
        "creative", "precise", "minimalist", "verbose", "technical", "funny"
    ]
    
    # Default tasks for testing
    DEFAULT_TASKS: List[str] = [
        "Translate 'I love AI' into Japanese",
        "Get current weather in Paris",
        "Summarize this article: 'AI is transforming banking'",
        "Convert 100 USD to EUR",
        "Generate a haiku about neural networks"
    ]
    
    # Output file configuration
    SYMBOL_TABLE_FILE = "symbol_table_log.json"
    
    @classmethod
    def setup_logging(cls) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(level=cls.LOG_LEVEL, format=cls.LOG_FORMAT)
        return logging.getLogger(__name__)
    
    @classmethod
    def validate_api_key(cls) -> bool:
        """Validate that API key is properly configured."""
        return cls.OPENAI_API_KEY and not cls.OPENAI_API_KEY.startswith("sk-...")
