"""Main voting system implementation."""

import os
from typing import List
from autogen import GroupChat, GroupChatManager
from autogen.oai.client import OpenAIWrapper
from .anyllm_client import CustomAnyLLMClient

from .config import Config
from .agents import AgentFactory
from .memory import SymbolMemory
from .message_parser import MessageParser


class VotingSystem:
    """Main voting system that orchestrates the multi-agent decision making."""
    
    def __init__(self):
        """Initialize the voting system."""
        self.config = Config()
        self.memory = SymbolMemory()
        self.logger = Config.setup_logging()

        try:
            OpenAIWrapper.register_model_client(CustomAnyLLMClient)
        except Exception:
            pass
        
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = self.config.OPENAI_API_KEY
        
        # Validate configuration
        if not self.config.validate_api_key():
            self.logger.warning("OpenAI API key not properly configured!")
    
    def process_task(self, task: str) -> str:
        """Process a single task through the voting protocol."""
        self.logger.info(f"Running task: {task}")
        
        # Get symbol history for context
        symbol_history = self.memory.get_symbol_history()
        
        # Create agents
        speaker = AgentFactory.create_speaker(symbol_history)
        listener = AgentFactory.create_listener()
        negotiator = AgentFactory.create_negotiator()
        user = AgentFactory.create_user_proxy()
        
        # Set up group chat
        groupchat = GroupChat(
            agents=[user, speaker, listener, negotiator],
            messages=[],
            select_speaker_auto_model_client_cls=CustomAnyLLMClient,
            select_speaker_auto_llm_config=Config.get_llm_config(use_anyllm=True)
        )
        #     max_rounds=self.config.MAX_ROUNDS,
        # )
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=Config.get_llm_config(use_anyllm=True)
        )
        # Ensure the manager's wrapper activates the custom client
        try:
            if hasattr(manager, "client") and manager.client is not None:
                manager.client.register_model_client(CustomAnyLLMClient)
        except Exception:
            pass
        
        # Initiate the voting process
        user.initiate_chat(manager, message=f"Please compress this task: {task}")
        
        # Parse the results
        parsed_results = MessageParser.parse_group_chat_messages(groupchat.messages)
        
        # Log the process
        self._log_voting_process(parsed_results)
        
        # Store the result if valid
        final_symbol = parsed_results.get("final_symbol")
        if final_symbol:
            self.memory.add_symbol(task, final_symbol)
            self.logger.info(f"Symbol selected: {final_symbol}")
            return final_symbol
        else:
            self.logger.warning("No valid final selection found.")
            return ""
    
    def process_tasks(self, tasks: List[str] = None) -> List[str]:
        """Process multiple tasks through the voting protocol."""
        if tasks is None:
            tasks = self.config.DEFAULT_TASKS
        
        results = []
        for task in tasks:
            result = self.process_task(task)
            results.append(result)
        
        return results
    
    def save_memory(self, filename: str = None) -> None:
        """Save the symbol memory to file."""
        self.memory.save_to_file(filename)
        filename = filename or self.config.SYMBOL_TABLE_FILE
        self.logger.info(f"Symbol table memory saved to '{filename}'")
    
    def load_memory(self, filename: str = None) -> None:
        """Load the symbol memory from file."""
        self.memory.load_from_file(filename)
        filename = filename or self.config.SYMBOL_TABLE_FILE
        self.logger.info(f"Symbol table memory loaded from '{filename}'")
    
    def get_symbol_history(self) -> dict:
        """Get the current symbol usage history."""
        return self.memory.get_symbol_history()
    
    def _log_voting_process(self, parsed_results: dict) -> None:
        """Log the details of the voting process."""
        if parsed_results.get("option_a") and parsed_results.get("option_b"):
            self.logger.info(
                f"Speaker proposals: Option A: {parsed_results['option_a']}, "
                f"Option B: {parsed_results['option_b']}"
            )
        
        if parsed_results.get("listener_vote"):
            self.logger.info(f"Listener voted for Option {parsed_results['listener_vote']}")
        
        if parsed_results.get("final_selection"):
            self.logger.info(f"Negotiator selected Option {parsed_results['final_selection']}")
