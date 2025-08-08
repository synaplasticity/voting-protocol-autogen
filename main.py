#!/usr/bin/env python3
"""
Main entry point for the voting protocol system.

This script demonstrates how to use the refactored voting protocol package.
"""

from voting_protocol import VotingSystem


def main():
    """Main function to run the voting protocol system."""
    # Create the voting system
    voting_system = VotingSystem()
    
    # Process the default tasks
    print("Starting voting protocol system...")
    results = voting_system.process_tasks()
    
    # Save the results
    voting_system.save_memory()
    
    # Display summary
    print("\n=== Voting Results Summary ===")
    symbol_history = voting_system.get_symbol_history()
    for symbol, count in symbol_history.items():
        print(f"Symbol '{symbol}' was selected {count} time(s)")


if __name__ == "__main__":
    main()
