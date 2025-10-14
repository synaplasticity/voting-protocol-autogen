"""Configuration module for the voting protocol system."""

import os
import logging
from typing import List


class Config:
    """Configuration class for the voting protocol system."""
    
    # API Configuration
    # OpenAI API
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "77FMH69-1D84SSQ-KGCGKZT-GA553PH")
    
    # AnyLLM Configuration
    ANYLLM_API_KEY = os.environ.get("ANYLLM_API_KEY", "77FMH69-1D84SSQ-KGCGKZT-GA553PH")
    ANYLLM_API_BASE = "http://localhost:3001/api/v1/workspace/test-llama3/chat"
    ANYLLM_MODEL = "Llama3.2"
    
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
        "Translate 'I love AI' into Japanese"
    ]
    # DEFAULT_TASKS: List[str] = [
    #     "Translate 'I love AI' into Japanese",
    #     "Get current weather in Paris",
    #     "Summarize this article: 'AI is transforming banking'",
    #     "Convert 100 USD to EUR",
    #     "Generate a haiku about neural networks"
    # ]
    
    # Output file configuration
    SYMBOL_TABLE_FILE = "symbol_table_log.json"
    
    @classmethod
    def setup_logging(cls) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(level=cls.LOG_LEVEL, format=cls.LOG_FORMAT)
        return logging.getLogger(__name__)
    
    @classmethod
    def get_llm_config(cls, use_anyllm: bool = True) -> dict:
        """Get the LLM configuration.
        
        Args:
            use_anyllm: If True, use the local anyllm configuration. If False, use OpenAI.
            
        Returns:
            dict: Configuration dictionary for the LLM
        """
        import os
        
        # Ensure proxy environment variables are not set
        os.environ["NO_PROXY"] = "*"
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""
        
        if use_anyllm:
            return {
                "config_list": [{
                    "model": cls.ANYLLM_MODEL,
                    "base_url": cls.ANYLLM_API_BASE,
                    "api_key": cls.ANYLLM_API_KEY,
                    "api_type": "open_ai",
                    "model_client_cls": "CustomAnyLLMClient",
                }]
            }
        return {
            "config_list": [{
                "model": "gpt-4",  # Default to GPT-4 if not using anyllm
                "api_key": cls.OPENAI_API_KEY,
            }]
        }
    
    @classmethod
    def validate_api_key(cls) -> bool:
        """Validate that API key is properly configured."""
        return cls.OPENAI_API_KEY #and not cls.OPENAI_API_KEY.startswith("sk-...")
