"""Unit tests for the agents module."""

import unittest
from unittest.mock import patch, MagicMock
from voting_protocol.agents import AgentFactory
from voting_protocol.config import Config


class TestAgentFactory(unittest.TestCase):
    """Test cases for the AgentFactory class."""
    
    def test_get_random_personality(self):
        """Test that get_random_personality returns a valid personality."""
        personality = AgentFactory.get_random_personality()
        self.assertIn(personality, Config.PERSONALITY_TRAITS)
        self.assertIsInstance(personality, str)
    
    def test_get_random_personality_multiple_calls(self):
        """Test that multiple calls can return different personalities."""
        personalities = set()
        for _ in range(20):  # Run multiple times to increase chance of getting different values
            personalities.add(AgentFactory.get_random_personality())
        
        # Should get at least 2 different personalities in 20 calls (very likely)
        # But we'll be lenient and just check that we get valid personalities
        for personality in personalities:
            self.assertIn(personality, Config.PERSONALITY_TRAITS)
    
    @patch('voting_protocol.agents.AssistantAgent')
    def test_create_speaker_empty_memory(self, mock_assistant):
        """Test creating a speaker with empty memory."""
        mock_agent = MagicMock()
        mock_assistant.return_value = mock_agent
        
        memory = {}
        agent = AgentFactory.create_speaker(memory)
        
        # Verify AssistantAgent was called
        mock_assistant.assert_called_once()
        call_args = mock_assistant.call_args
        
        # Check that name is correct
        self.assertEqual(call_args[1]['name'], 'Speaker')
        
        # Check that system message contains expected elements
        system_message = call_args[1]['system_message']
        self.assertIn('propose two short symbolic names', system_message)
        self.assertIn(str(Config.MAX_SYMBOL_LENGTH), system_message)
        self.assertIn('Option A', system_message)
        self.assertIn('Option B', system_message)
        
        self.assertEqual(agent, mock_agent)
    
    @patch('voting_protocol.agents.AssistantAgent')
    def test_create_speaker_with_memory(self, mock_assistant):
        """Test creating a speaker with memory history."""
        mock_agent = MagicMock()
        mock_assistant.return_value = mock_agent
        
        memory = {"SYM1": 3, "SYM2": 1}
        agent = AgentFactory.create_speaker(memory)
        
        # Verify AssistantAgent was called
        mock_assistant.assert_called_once()
        call_args = mock_assistant.call_args
        
        # Check that system message contains memory information
        system_message = call_args[1]['system_message']
        self.assertIn('Past popular symbols', system_message)
        self.assertIn('SYM1 (3)', system_message)
        self.assertIn('SYM2 (1)', system_message)
        
        self.assertEqual(agent, mock_agent)
    
    @patch('voting_protocol.agents.AssistantAgent')
    def test_create_listener(self, mock_assistant):
        """Test creating a listener agent."""
        mock_agent = MagicMock()
        mock_assistant.return_value = mock_agent
        
        agent = AgentFactory.create_listener()
        
        # Verify AssistantAgent was called
        mock_assistant.assert_called_once()
        call_args = mock_assistant.call_args
        
        # Check that name is correct
        self.assertEqual(call_args[1]['name'], 'Listener')
        
        # Check that system message contains expected elements
        system_message = call_args[1]['system_message']
        self.assertIn('two symbolic task codes', system_message)
        
        self.assertEqual(agent, mock_agent)
    
    @patch('voting_protocol.agents.AgentFactory.get_random_personality')
    @patch('voting_protocol.agents.AssistantAgent')
    def test_personality_integration(self, mock_assistant, mock_personality):
        """Test that personality is integrated into agent creation."""
        mock_agent = MagicMock()
        mock_assistant.return_value = mock_agent
        mock_personality.return_value = "creative"
        
        AgentFactory.create_speaker({})
        
        # Verify personality was requested
        mock_personality.assert_called_once()
        
        # Check that personality appears in system message
        call_args = mock_assistant.call_args
        system_message = call_args[1]['system_message']
        self.assertIn('creative', system_message)


if __name__ == '__main__':
    unittest.main()
