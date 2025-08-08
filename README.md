# Voting Protocol System

A multi-agent voting protocol system for collaborative decision making on symbolic task codes using AutoGen agents.

## Overview

This system uses three types of agents to collaboratively decide on symbolic representations of tasks:
- **Speaker**: Proposes multiple symbolic codes for a given task
- **Listener**: Votes on the clarity and efficiency of proposals
- **Negotiator**: Makes the final decision on which symbolic code to use

## Project Structure

```
voting_protocol/
├── __init__.py          # Package initialization
├── agents.py            # Agent factory and definitions
├── config.py            # Configuration settings
├── memory.py            # Symbol memory management
├── message_parser.py    # Message parsing utilities
└── voting_system.py     # Main voting system implementation
main.py                  # Entry point script
requirements.txt         # Python dependencies
setup.py                # Package setup configuration
README.md               # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```python
from voting_protocol import VotingSystem

# Create the voting system
voting_system = VotingSystem()

# Process a single task
result = voting_system.process_task("Translate 'Hello' to French")
print(f"Selected symbol: {result}")

# Process multiple tasks
tasks = ["Task 1", "Task 2", "Task 3"]
results = voting_system.process_tasks(tasks)

# Save the memory
voting_system.save_memory()
```

### Running the Demo

```bash
python main.py
```

## Configuration

The system can be configured through the `Config` class in `voting_protocol/config.py`:

- `MAX_ROUNDS`: Maximum rounds for group chat (default: 6)
- `MAX_SYMBOL_LENGTH`: Maximum length for symbolic codes (default: 6)
- `PERSONALITY_TRAITS`: Available personality traits for agents
- `DEFAULT_TASKS`: Default tasks for testing

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules
- **Memory System**: Tracks symbol usage history for context
- **Configurable**: Easy to customize agent behavior and system parameters
- **Logging**: Comprehensive logging for debugging and monitoring
- **PEP 8 Compliant**: Follows Python coding standards

## License

MIT License
