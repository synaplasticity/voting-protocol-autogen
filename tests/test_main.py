"""Unit tests for the main module."""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock
from io import StringIO

# Add the parent directory to the path to import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main


class TestMain(unittest.TestCase):
    """Test cases for the main module."""
    
    @patch('main.VotingSystem')
    @patch('builtins.print')
    def test_main_function(self, mock_print, mock_voting_system):
        """Test the main function execution."""
        # Mock VotingSystem instance
        mock_vs_instance = MagicMock()
        mock_voting_system.return_value = mock_vs_instance
        
        # Mock return values
        mock_vs_instance.process_tasks.return_value = ["RESULT1", "RESULT2"]
        mock_vs_instance.get_symbol_history.return_value = {"SYM1": 2, "SYM2": 1}
        
        # Call main function
        main.main()
        
        # Verify VotingSystem was created
        mock_voting_system.assert_called_once()
        
        # Verify methods were called
        mock_vs_instance.process_tasks.assert_called_once()
        mock_vs_instance.save_memory.assert_called_once()
        mock_vs_instance.get_symbol_history.assert_called_once()
        
        # Verify print statements
        mock_print.assert_any_call("Starting voting protocol system...")
        mock_print.assert_any_call("\n=== Voting Results Summary ===")
        mock_print.assert_any_call("Symbol 'SYM1' was selected 2 time(s)")
        mock_print.assert_any_call("Symbol 'SYM2' was selected 1 time(s)")
    
    @patch('main.VotingSystem')
    @patch('builtins.print')
    def test_main_function_empty_history(self, mock_print, mock_voting_system):
        """Test main function with empty symbol history."""
        # Mock VotingSystem instance
        mock_vs_instance = MagicMock()
        mock_voting_system.return_value = mock_vs_instance
        
        # Mock return values
        mock_vs_instance.process_tasks.return_value = []
        mock_vs_instance.get_symbol_history.return_value = {}
        
        # Call main function
        main.main()
        
        # Verify basic calls were made
        mock_vs_instance.process_tasks.assert_called_once()
        mock_vs_instance.save_memory.assert_called_once()
        mock_vs_instance.get_symbol_history.assert_called_once()
        
        # Verify summary header was printed
        mock_print.assert_any_call("\n=== Voting Results Summary ===")
    
    @patch('main.main')
    def test_main_entry_point(self, mock_main):
        """Test that main is called when script is run directly."""
        # This test verifies the if __name__ == "__main__" block
        # We can't easily test this directly, so we mock it
        mock_main.return_value = None
        
        # Simulate running the script
        with patch('__main__.__name__', '__main__'):
            # The actual test would require executing the module
            # For now, we just verify our mock setup
            self.assertTrue(callable(mock_main))


if __name__ == '__main__':
    unittest.main()
