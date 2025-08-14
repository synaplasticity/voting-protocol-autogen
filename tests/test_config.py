"""Unit tests for the config module."""

import unittest
import os
import logging
from unittest.mock import patch
from voting_protocol.config import Config


class TestConfig(unittest.TestCase):
    """Test cases for the Config class."""
    
    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        self.assertEqual(Config.LOG_LEVEL, logging.INFO)
        self.assertEqual(Config.LOG_FORMAT, '%(asctime)s - %(levelname)s - %(message)s')
        self.assertEqual(Config.MAX_ROUNDS, 6)
        self.assertEqual(Config.MAX_SYMBOL_LENGTH, 6)
        self.assertEqual(Config.SYMBOL_TABLE_FILE, "symbol_table_log.json")
    
    def test_personality_traits(self):
        """Test that personality traits are defined correctly."""
        expected_traits = ["creative", "precise", "minimalist", "verbose", "technical", "funny"]
        self.assertEqual(Config.PERSONALITY_TRAITS, expected_traits)
        self.assertIsInstance(Config.PERSONALITY_TRAITS, list)
        self.assertTrue(len(Config.PERSONALITY_TRAITS) > 0)
    
    def test_default_tasks(self):
        """Test that default tasks are defined correctly."""
        self.assertIsInstance(Config.DEFAULT_TASKS, list)
        self.assertTrue(len(Config.DEFAULT_TASKS) > 0)
        # Check that all tasks are strings
        for task in Config.DEFAULT_TASKS:
            self.assertIsInstance(task, str)
            self.assertTrue(len(task) > 0)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
    def test_openai_api_key_from_env(self):
        """Test that OpenAI API key is read from environment."""
        # Need to reload the module to pick up the new environment variable
        import importlib
        from voting_protocol import config
        importlib.reload(config)
        self.assertEqual(config.Config.OPENAI_API_KEY, 'sk-test123')
    
    def test_setup_logging(self):
        """Test that logging setup works correctly."""
        logger = Config.setup_logging()
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'voting_protocol.config')
    
    def test_validate_api_key_valid(self):
        """Test API key validation with valid key."""
        with patch.object(Config, 'OPENAI_API_KEY', 'sk-valid123'):
            self.assertTrue(Config.validate_api_key())
    
    def test_validate_api_key_invalid(self):
        """Test API key validation with invalid key."""
        with patch.object(Config, 'OPENAI_API_KEY', 'sk-...'):
            self.assertFalse(Config.validate_api_key())
        
        with patch.object(Config, 'OPENAI_API_KEY', ''):
            self.assertFalse(Config.validate_api_key())
        
        with patch.object(Config, 'OPENAI_API_KEY', None):
            self.assertFalse(Config.validate_api_key())


if __name__ == '__main__':
    unittest.main()
