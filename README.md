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

## AnyLLM Integration (CustomAnyLLMClient)

This project supports routing all LLM calls through an AnyLLM workspace using a custom AutoGen model client.

- **Client class**: `voting_protocol/anyllm_client.py` defines `CustomAnyLLMClient`, implementing AutoGen’s `ModelClient` interface and posting to a workspace-scoped `/chat` endpoint.
- **Why**: The AnyLLM server exposes a simplified chat API (not OpenAI’s `/v1/chat/completions`). `CustomAnyLLMClient` adapts AutoGen’s calls to that API and parses its JSON (`textResponse`/`message`/`response`/`text`).

### Configuration

Set AnyLLM values in `voting_protocol/config.py`:

- `ANYLLM_API_BASE`: e.g. `http://localhost:3001/api/v1/workspace/test-llama3/chat` (note the `/chat` path)
- `ANYLLM_API_KEY`: Your workspace API key
- `ANYLLM_MODEL`: e.g. `Llama3.2`

Enable the custom client in `Config.get_llm_config(use_anyllm=True)` by including:

```python
{
  "model": cls.ANYLLM_MODEL,
  "base_url": cls.ANYLLM_API_BASE,
  "api_key": cls.ANYLLM_API_KEY,
  "api_type": "open_ai",
  "model_client_cls": "CustomAnyLLMClient",
  # Optional network hardening
  "timeout": 180.0,   # or httpx.Timeout(...)
  "retries": 3,
}
```

### Runtime registration

AutoGen requires custom clients to be activated at runtime.

- In `voting_protocol/voting_system.py`:
  - Pass the client class and config to `GroupChat` so the internal auto speaker-selection agents can register it:

```python
groupchat = GroupChat(
    agents=[user, speaker, listener, negotiator],
    messages=[],
    select_speaker_auto_model_client_cls=CustomAnyLLMClient,
    select_speaker_auto_llm_config=Config.get_llm_config(use_anyllm=True),
)
```

  - After constructing `GroupChatManager`, register on its wrapper instance to ensure activation:

```python
manager = GroupChatManager(groupchat=groupchat, llm_config=Config.get_llm_config(use_anyllm=True))
if hasattr(manager, "client") and manager.client is not None:
    manager.client.register_model_client(CustomAnyLLMClient)
```

- In `voting_protocol/agents.py`, agents also register the client after creation (redundant but safe):

```python
if hasattr(agent, "client") and agent.client is not None:
    agent.client.register_model_client(CustomAnyLLMClient)
```

### Endpoint expectations

- The client sends `{ model, mode: "chat", message }` to `ANYLLM_API_BASE`.
- It reads `textResponse` (or `message`/`response`/`text`) from the JSON response.

### Timeouts and retries

- `CustomAnyLLMClient` supports `timeout` and `retries` via config or registration kwargs. Increase `read` timeout if long generations are expected.

### Troubleshooting

- **RuntimeError: Model client(s) ... are not activated**
  - Ensure `model_client_cls: "CustomAnyLLMClient"` is present in the config list, and the registrations above are executed before the first call.

- **TypeError: '_Msg' object is not iterable**
  - Fixed by returning plain strings from `CustomAnyLLMClient.message_retrieval()`.

- **httpx.ReadTimeout**
  - Increase `timeout`/`retries` in the config. Consider reducing prompt size or increasing server-side timeouts.

## License

MIT License
