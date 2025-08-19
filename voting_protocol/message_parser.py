"""Message parsing utilities for the voting protocol system."""

import re
from typing import Optional, Tuple, List, Dict, Any


class MessageParser:
    """Parses messages from group chat to extract voting information."""
    
    @staticmethod
    def extract_speaker_proposals(content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract Option A and Option B from Speaker's message."""
        option_a = None
        option_b = None
        
        match_a = re.search(r"Option A:\s*(\b\w{1,6}\b)", content)
        if match_a:
            option_a = match_a.group(1)
            
        match_b = re.search(r"Option B:\s*(\b\w{1,6}\b)", content)
        if match_b:
            option_b = match_b.group(1)
            
        return option_a, option_b
    
    @staticmethod
    def extract_listener_vote(content: str) -> Optional[str]:
        """Extract the vote from Listener's message."""
        if "I vote for Option A" in content:
            return "A"
        elif "I vote for Option B" in content:
            return "B"
        return None
    
    @staticmethod
    def extract_final_selection(content: str) -> Optional[str]:
        """Extract the final selection from Negotiator's message."""
        match_final = re.search(r"Final selection:\s*Option (A|B)", content)
        if match_final:
            return match_final.group(1)
        return None
    
    @staticmethod
    def parse_group_chat_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse all messages from a group chat and extract voting information.
        
        Args:
            messages: List of message dictionaries with 'name' and 'content' keys
            
        Returns:
            Dictionary containing extracted voting information
            
        Raises:
            ValueError: If Speaker's message doesn't contain both Option A and Option B
        """
        result = {
            "option_a": None,
            "option_b": None,
            "listener_vote": None,
            "final_selection": None,
            "final_symbol": None
        }
        
        for msg in messages:
            sender = msg.get("name", "")
            content = msg.get("content", "")
            
            if sender == "Speaker":
                has_option_a = "Option A:" in content
                has_option_b = "Option B:" in content
                
                if has_option_a and has_option_b:
                    option_a, option_b = MessageParser.extract_speaker_proposals(content)
                    result["option_a"] = option_a
                    result["option_b"] = option_b
                elif has_option_a or has_option_b:
                    raise ValueError("Speaker's message must contain both Option A and Option B")
            
            elif sender == "Listener":
                vote = MessageParser.extract_listener_vote(content)
                if vote:
                    result["listener_vote"] = vote
            
            elif sender == "Negotiator":
                final_option = MessageParser.extract_final_selection(content)
                if final_option and result["option_a"] and result["option_b"]:
                    result["final_selection"] = final_option
                    result["final_symbol"] = (
                        result["option_a"] if final_option == "A" 
                        else result["option_b"]
                    )
        
        return result
