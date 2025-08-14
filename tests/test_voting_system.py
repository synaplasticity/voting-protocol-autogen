"""Unit tests for the voting_system module."""

import unittest
import os
from unittest.mock import patch, MagicMock, call
from voting_protocol.voting_system import VotingSystem
from voting_protocol.config import Config


class TestVotingSystem(unittest.TestCase):
    """Test cases for the VotingSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('voting_protocol.voting_system.Config.setup_logging'):
            with patch('voting_protocol.voting_system.Config.validate_api_key', return_value=True):
                self.voting_system = VotingSystem()
    
    def test_init(self):
        """Test VotingSystem initialization."""
        with patch('voting_protocol.voting_system.Config.setup_logging') as mock_logging:
            with patch('voting_protocol.voting_system.Config.validate_api_key', return_value=True):
                vs = VotingSystem()
                
                self.assertIsNotNone(vs.config)
                self.assertIsNotNone(vs.memory)
                mock_logging.assert_called_once()
    
    def test_init_invalid_api_key(self):
        """Test VotingSystem initialization with invalid API key."""
        with patch('voting_protocol.voting_system.Config.setup_logging'):
            with patch('voting_protocol.voting_system.Config.validate_api_key', return_value=False):
                with patch('voting_protocol.voting_system.Config.setup_logging') as mock_setup:
                    mock_logger = MagicMock()
                    mock_setup.return_value = mock_logger
                    
                    vs = VotingSystem()
                    mock_logger.warning.assert_called_once_with("OpenAI API key not properly configured!")
    
    @patch('voting_protocol.voting_system.GroupChatManager')
    @patch('voting_protocol.voting_system.GroupChat')
    @patch('voting_protocol.voting_system.AgentFactory')
    def test_process_task(self, mock_agent_factory, mock_group_chat, mock_manager):
        """Test processing a single task."""
        # Mock agents
        mock_speaker = MagicMock()
        mock_listener = MagicMock()
        mock_negotiator = MagicMock()
        mock_user = MagicMock()
        
        mock_agent_factory.create_speaker.return_value = mock_speaker
        mock_agent_factory.create_listener.return_value = mock_listener
        mock_agent_factory.create_negotiator.return_value = mock_negotiator
        mock_agent_factory.create_user_proxy.return_value = mock_user
        
        # Mock group chat and manager
        mock_groupchat_instance = MagicMock()
        mock_group_chat.return_value = mock_groupchat_instance
        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance
        
        # Mock the chat result
        mock_manager_instance.initiate_chat.return_value = MagicMock()
        mock_groupchat_instance.messages = [
            {"name": "Speaker", "content": "Option A: TEST1 - Test\nOption B: TEST2 - Test"},
            {"name": "Listener", "content": "I vote for Option A"},
            {"name": "Negotiator", "content": "Final selection: Option A"}
        ]
        
        # Test the method
        result = self.voting_system.process_task("Test task")
        
        # Verify agents were created
        mock_agent_factory.create_speaker.assert_called_once()
        mock_agent_factory.create_listener.assert_called_once()
        mock_agent_factory.create_negotiator.assert_called_once()
        mock_agent_factory.create_user_proxy.assert_called_once()
        
        # Verify group chat was set up
        mock_group_chat.assert_called_once()
        mock_manager.assert_called_once()
        
        # Verify chat was initiated
        mock_manager_instance.initiate_chat.assert_called_once()
        
        self.assertIsNotNone(result)
    
    def test_process_tasks_default(self):
        """Test processing default tasks."""
        with patch.object(self.voting_system, 'process_task', return_value="TEST") as mock_process:
            results = self.voting_system.process_tasks()
            
            # Should process all default tasks
            self.assertEqual(mock_process.call_count, len(Config.DEFAULT_TASKS))
            self.assertEqual(len(results), len(Config.DEFAULT_TASKS))
            
            # Verify each default task was processed
            expected_calls = [call(task) for task in Config.DEFAULT_TASKS]
            mock_process.assert_has_calls(expected_calls)
    
    def test_process_tasks_custom(self):
        """Test processing custom tasks."""
        custom_tasks = ["Task 1", "Task 2"]
        
        with patch.object(self.voting_system, 'process_task', return_value="TEST") as mock_process:
            results = self.voting_system.process_tasks(custom_tasks)
            
            # Should process custom tasks
            self.assertEqual(mock_process.call_count, len(custom_tasks))
            self.assertEqual(len(results), len(custom_tasks))
            
            # Verify each custom task was processed
            expected_calls = [call(task) for task in custom_tasks]
            mock_process.assert_has_calls(expected_calls)
    
    def test_get_symbol_history(self):
        """Test getting symbol history."""
        expected_history = {"SYM1": 2, "SYM2": 1}
        
        with patch.object(self.voting_system.memory, 'get_symbol_history', return_value=expected_history):
            history = self.voting_system.get_symbol_history()
            self.assertEqual(history, expected_history)
    
    def test_save_memory(self):
        """Test saving memory."""
        with patch.object(self.voting_system.memory, 'save_to_file') as mock_save:
            self.voting_system.save_memory()
            mock_save.assert_called_once()
    
    def test_save_memory_custom_filename(self):
        """Test saving memory with custom filename."""
        filename = "custom_file.json"
        
        with patch.object(self.voting_system.memory, 'save_to_file') as mock_save:
            self.voting_system.save_memory(filename)
            mock_save.assert_called_once_with(filename)


if __name__ == '__main__':
    unittest.main()
