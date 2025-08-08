"""Agent definitions for the voting protocol system."""

import random
from typing import Dict, Any
from autogen import AssistantAgent, UserProxyAgent
from .config import Config


class AgentFactory:
    """Factory class for creating different types of agents."""
    
    @staticmethod
    def get_random_personality() -> str:
        """Get a random personality trait."""
        return random.choice(Config.PERSONALITY_TRAITS)
    
    @staticmethod
    def create_speaker(memory: Dict[str, int]) -> AssistantAgent:
        """Create a Speaker agent that proposes symbolic codes."""
        memory_prompt = (
            "Past popular symbols: " + 
            ", ".join(f"{k} ({v})" for k, v in memory.items()) 
            if memory else ""
        )
        personality = AgentFactory.get_random_personality()
        
        return AssistantAgent(
            name="Speaker",
            system_message=f"""
You receive a task and propose two short symbolic names for it (each max {Config.MAX_SYMBOL_LENGTH} characters).
Use inspiration from previous popular symbols if helpful.
{memory_prompt}
Your personality is: {personality} — adapt your naming style accordingly.
Return your proposals labeled Option A and Option B, with a one-line description.
Example:
Option A: WXPAR - Get weather in Paris
Option B: WTHPR - Weather Paris
Only output these two lines.
"""
        )
    
    @staticmethod
    def create_listener() -> AssistantAgent:
        """Create a Listener agent that votes on proposals."""
        personality = AgentFactory.get_random_personality()
        
        return AssistantAgent(
            name="Listener",
            system_message=f"""
You receive two symbolic task codes.
Vote for the one that you think best represents the task clearly and efficiently.
Just say "I vote for Option A" or "I vote for Option B" with a short reason.
Your personality is: {personality} — base your reasoning style on it.
"""
        )
    
    @staticmethod
    def create_negotiator() -> AssistantAgent:
        """Create a Negotiator agent that makes final decisions."""
        personality = AgentFactory.get_random_personality()
        
        return AssistantAgent(
            name="Negotiator",
            system_message=f"""
You receive the Speaker's proposals and Listener's vote.
Make the final decision on which symbolic code to use.
Say "Final selection: Option A" or "Final selection: Option B" with a justification.
Your personality is: {personality} — use it to guide your logic and tone.
"""
        )
    
    @staticmethod
    def create_user_proxy() -> UserProxyAgent:
        """Create a User proxy agent."""
        return UserProxyAgent(
            name="User",
            code_execution_config=False,
        )
