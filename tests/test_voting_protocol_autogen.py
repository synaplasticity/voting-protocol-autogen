"""Unit tests for the voting_protocol_autogen module."""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import the autogen module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import voting_protocol_autogen as autogen_module


class TestVotingProtocolAutogen(unittest.TestCase):
    """Test cases for the voting_protocol_autogen module."""
    
    def test_symbol_table_initialization(self):
        """Test that symbol_table is initialized as empty list."""
        # Reset symbol_table for testing
        autogen_module.symbol_table = []
        self.assertEqual(autogen_module.symbol_table, [])
        self.assertIsInstance(autogen_module.symbol_table, list)
    
    def test_get_symbol_history_empty(self):
        """Test get_symbol_history with empty symbol table."""
        autogen_module.symbol_table = []
        history = autogen_module.get_symbol_history()
        self.assertEqual(history, {})
    
    def test_get_symbol_history_with_data(self):
        """Test get_symbol_history with symbol data."""
        autogen_module.symbol_table = [
            {"symbol": "TEST1", "task": "task1"},
            {"symbol": "TEST2", "task": "task2"},
            {"symbol": "TEST1", "task": "task3"},
        ]
        
        history = autogen_module.get_symbol_history()
        expected = {"TEST1": 2, "TEST2": 1}
        self.assertEqual(history, expected)
    
    def test_personality_traits_defined(self):
        """Test that personality traits are properly defined."""
        self.assertIsInstance(autogen_module.PERSONALITY_TRAITS, list)
        self.assertTrue(len(autogen_module.PERSONALITY_TRAITS) > 0)
        
        expected_traits = ["creative", "precise", "minimalist", "verbose", "technical", "funny"]
        self.assertEqual(autogen_module.PERSONALITY_TRAITS, expected_traits)
    
    def test_random_personality(self):
        """Test random_personality function."""
        personality = autogen_module.random_personality()
        self.assertIn(personality, autogen_module.PERSONALITY_TRAITS)
        self.assertIsInstance(personality, str)
    
    def test_random_personality_multiple_calls(self):
        """Test that random_personality can return different values."""
        personalities = set()
        for _ in range(20):  # Multiple calls to increase chance of different values
            personalities.add(autogen_module.random_personality())
        
        # All returned personalities should be valid
        for personality in personalities:
            self.assertIn(personality, autogen_module.PERSONALITY_TRAITS)
    
    @patch('voting_protocol_autogen.AssistantAgent')
    @patch('voting_protocol_autogen.random_personality')
    def test_make_speaker_empty_memory(self, mock_random_personality, mock_assistant_agent):
        """Test make_speaker with empty memory."""
        autogen_module.symbol_table = []
        mock_random_personality.return_value = "creative"
        mock_agent = MagicMock()
        mock_assistant_agent.return_value = mock_agent
        
        speaker = autogen_module.make_speaker()
        
        # Verify AssistantAgent was called
        mock_assistant_agent.assert_called_once()
        call_args = mock_assistant_agent.call_args
        
        # Check agent name
        self.assertEqual(call_args[1]['name'], 'Speaker')
        
        # Check system message contains expected elements
        system_message = call_args[1]['system_message']
        self.assertIn('propose two short symbolic names', system_message)
        self.assertIn('creative', system_message)
        self.assertIn('Option A', system_message)
        self.assertIn('Option B', system_message)
        
        self.assertEqual(speaker, mock_agent)
    
    @patch('voting_protocol_autogen.AssistantAgent')
    @patch('voting_protocol_autogen.random_personality')
    def test_make_speaker_with_memory(self, mock_random_personality, mock_assistant_agent):
        """Test make_speaker with memory history."""
        autogen_module.symbol_table = [
            {"symbol": "SYM1", "task": "task1"},
            {"symbol": "SYM1", "task": "task2"},
            {"symbol": "SYM2", "task": "task3"},
        ]
        mock_random_personality.return_value = "technical"
        mock_agent = MagicMock()
        mock_assistant_agent.return_value = mock_agent
        
        speaker = autogen_module.make_speaker()
        
        # Verify AssistantAgent was called
        mock_assistant_agent.assert_called_once()
        call_args = mock_assistant_agent.call_args
        
        # Check system message contains memory information
        system_message = call_args[1]['system_message']
        self.assertIn('Past popular symbols', system_message)
        self.assertIn('SYM1 (2)', system_message)
        self.assertIn('SYM2 (1)', system_message)
        self.assertIn('technical', system_message)
        
        self.assertEqual(speaker, mock_agent)
    
    def tearDown(self):
        """Clean up after each test."""
        # Reset symbol_table to avoid test interference
        autogen_module.symbol_table = []


if __name__ == '__main__':
    unittest.main()
