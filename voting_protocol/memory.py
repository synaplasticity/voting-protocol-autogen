"""Memory management for the voting protocol system."""

import json
from datetime import datetime
from typing import List, Dict, Any
from .config import Config


class SymbolMemory:
    """Manages the symbol table memory for the voting system."""
    
    def __init__(self):
        """Initialize the symbol memory."""
        self.symbol_table: List[Dict[str, Any]] = []
    
    def get_symbol_history(self) -> Dict[str, int]:
        """Get the history of symbol usage counts."""
        symbol_counts = {}
        for entry in self.symbol_table:
            symbol = entry["symbol"]
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        return symbol_counts
    
    def add_symbol(self, task: str, symbol: str) -> None:
        """Add a new symbol to the memory table."""
        timestamp = datetime.now().isoformat()
        self.symbol_table.append({
            "timestamp": timestamp,
            "task": task,
            "symbol": symbol
        })
    
    def save_to_file(self, filename: str = None) -> None:
        """Save the symbol table to a JSON file."""
        if filename is None:
            filename = Config.SYMBOL_TABLE_FILE
        
        with open(filename, "w") as f:
            json.dump(self.symbol_table, f, indent=2)
    
    def load_from_file(self, filename: str = None) -> None:
        """Load the symbol table from a JSON file."""
        if filename is None:
            filename = Config.SYMBOL_TABLE_FILE
        
        try:
            with open(filename, "r") as f:
                self.symbol_table = json.load(f)
        except FileNotFoundError:
            self.symbol_table = []
    
    def clear(self) -> None:
        """Clear the symbol table."""
        self.symbol_table = []
    
    def get_all_symbols(self) -> List[Dict[str, Any]]:
        """Get all symbols in the table."""
        return self.symbol_table.copy()
