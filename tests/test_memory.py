"""Unit tests for the memory module."""

import unittest
import json
import os
import tempfile
from datetime import datetime
from unittest.mock import patch, mock_open
from voting_protocol.memory import SymbolMemory


class TestSymbolMemory(unittest.TestCase):
    """Test cases for the SymbolMemory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory = SymbolMemory()
    
    def test_init(self):
        """Test that SymbolMemory initializes correctly."""
        self.assertIsInstance(self.memory.symbol_table, list)
        self.assertEqual(len(self.memory.symbol_table), 0)
    
    def test_get_symbol_history_empty(self):
        """Test get_symbol_history with empty symbol table."""
        history = self.memory.get_symbol_history()
        self.assertEqual(history, {})
    
    def test_get_symbol_history_with_data(self):
        """Test get_symbol_history with symbol data."""
        # Add some test data
        self.memory.symbol_table = [
            {"symbol": "TEST1", "task": "task1", "timestamp": "2023-01-01"},
            {"symbol": "TEST2", "task": "task2", "timestamp": "2023-01-02"},
            {"symbol": "TEST1", "task": "task3", "timestamp": "2023-01-03"},
        ]
        
        history = self.memory.get_symbol_history()
        expected = {"TEST1": 2, "TEST2": 1}
        self.assertEqual(history, expected)
    
    @patch('voting_protocol.memory.datetime')
    def test_add_symbol(self, mock_datetime):
        """Test adding a symbol to memory."""
        mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"
        
        self.memory.add_symbol("Test task", "TESTSYM")
        
        self.assertEqual(len(self.memory.symbol_table), 1)
        entry = self.memory.symbol_table[0]
        self.assertEqual(entry["task"], "Test task")
        self.assertEqual(entry["symbol"], "TESTSYM")
        self.assertEqual(entry["timestamp"], "2023-01-01T12:00:00")
    
    def test_save_to_file(self):
        """Test saving symbol table to file."""
        # Add test data
        self.memory.add_symbol("Test task", "TESTSYM")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            self.memory.save_to_file(tmp_filename)
            
            # Verify file was created and contains correct data
            with open(tmp_filename, 'r') as f:
                data = json.load(f)
            
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["task"], "Test task")
            self.assertEqual(data[0]["symbol"], "TESTSYM")
        finally:
            os.unlink(tmp_filename)
    
    def test_save_to_file_default_filename(self):
        """Test saving to file with default filename."""
        self.memory.add_symbol("Test task", "TESTSYM")
        
        with patch('builtins.open', mock_open()) as mock_file:
            self.memory.save_to_file()
            mock_file.assert_called_once_with("symbol_table_log.json", "w")
    
    def test_load_from_file(self):
        """Test loading symbol table from file."""
        test_data = [
            {"timestamp": "2023-01-01", "task": "task1", "symbol": "SYM1"},
            {"timestamp": "2023-01-02", "task": "task2", "symbol": "SYM2"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            json.dump(test_data, tmp_file)
            tmp_filename = tmp_file.name
        
        try:
            self.memory.load_from_file(tmp_filename)
            
            self.assertEqual(len(self.memory.symbol_table), 2)
            self.assertEqual(self.memory.symbol_table[0]["symbol"], "SYM1")
            self.assertEqual(self.memory.symbol_table[1]["symbol"], "SYM2")
        finally:
            os.unlink(tmp_filename)
    
    def test_load_from_file_not_found(self):
        """Test loading from non-existent file."""
        self.memory.load_from_file("non_existent_file.json")
        self.assertEqual(self.memory.symbol_table, [])
    
    def test_load_from_file_default_filename(self):
        """Test loading from file with default filename."""
        with patch('builtins.open', mock_open(read_data='[]')):
            self.memory.load_from_file()
            self.assertEqual(self.memory.symbol_table, [])


if __name__ == '__main__':
    unittest.main()
