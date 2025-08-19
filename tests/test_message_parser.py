"""Unit tests for the message_parser module."""

import unittest
from voting_protocol.message_parser import MessageParser


class TestMessageParser(unittest.TestCase):
    """Test cases for the MessageParser class."""
    
    def test_extract_speaker_proposals_valid(self):
        """Test extracting valid speaker proposals."""
        content = """
        Option A: WXPAR - Get weather in Paris
        Option B: WTHPR - Weather Paris
        """
        
        option_a, option_b = MessageParser.extract_speaker_proposals(content)
        self.assertEqual(option_a, "WXPAR")
        self.assertEqual(option_b, "WTHPR")
    
    def test_extract_speaker_proposals_partial(self):
        """Test extracting proposals when only one option is present."""
        content_a_only = "Option A: TSTSYM - Test symbol"
        option_a, option_b = MessageParser.extract_speaker_proposals(content_a_only)
        self.assertEqual(option_a, "TSTSYM")
        self.assertIsNone(option_b)
        
        content_b_only = "Option B: ANTSYM - Another symbol"
        option_a, option_b = MessageParser.extract_speaker_proposals(content_b_only)
        self.assertIsNone(option_a)
        self.assertEqual(option_b, "ANTSYM")
    
    def test_extract_speaker_proposals_none(self):
        """Test extracting proposals when no valid options are present."""
        content = "This is just some random text without options"
        option_a, option_b = MessageParser.extract_speaker_proposals(content)
        self.assertIsNone(option_a)
        self.assertIsNone(option_b)
    
    def test_extract_speaker_proposals_max_length(self):
        """Test that symbols longer than 6 characters are handled correctly."""
        content = """
        Option A: TOOLONG - This symbol is too long
        Option B: SHORT - This is fine
        """
        option_a, option_b = MessageParser.extract_speaker_proposals(content)
        self.assertIsNone(option_a)  # Should be None because TOOLONG is > 6 chars
        self.assertEqual(option_b, "SHORT")
    
    def test_extract_listener_vote_option_a(self):
        """Test extracting listener vote for Option A."""
        content = "After careful consideration, I vote for Option A because it's clearer."
        vote = MessageParser.extract_listener_vote(content)
        self.assertEqual(vote, "A")
    
    def test_extract_listener_vote_option_b(self):
        """Test extracting listener vote for Option B."""
        content = "I think Option B is better. I vote for Option B."
        vote = MessageParser.extract_listener_vote(content)
        self.assertEqual(vote, "B")
    
    def test_extract_listener_vote_none(self):
        """Test extracting listener vote when no clear vote is present."""
        content = "I'm not sure which option to choose. Both seem good."
        vote = MessageParser.extract_listener_vote(content)
        self.assertIsNone(vote)
    
    def test_extract_final_selection_option_a(self):
        """Test extracting final selection for Option A."""
        content = "After negotiation, Final selection: Option A"
        selection = MessageParser.extract_final_selection(content)
        self.assertEqual(selection, "A")
    
    def test_extract_final_selection_option_b(self):
        """Test extracting final selection for Option B."""
        content = "The final decision is made. Final selection: Option B"
        selection = MessageParser.extract_final_selection(content)
        self.assertEqual(selection, "B")
    
    def test_extract_final_selection_none(self):
        """Test extracting final selection when no selection is present."""
        content = "We are still discussing the options."
        selection = MessageParser.extract_final_selection(content)
        self.assertIsNone(selection)
    
    def test_parse_group_chat_messages_complete(self):
        """Test parsing complete group chat messages."""
        messages = [
            {
                "name": "Speaker",
                "content": "Option A: WXPAR - Weather Paris\nOption B: WTHPR - Weather in Paris"
            },
            {
                "name": "Listener", 
                "content": "I vote for Option A because it's more concise."
            },
            {
                "name": "Negotiator",
                "content": "Based on the vote, Final selection: Option A"
            }
        ]
        
        result = MessageParser.parse_group_chat_messages(messages)
        
        self.assertEqual(result["option_a"], "WXPAR")
        self.assertEqual(result["option_b"], "WTHPR")
        self.assertEqual(result["listener_vote"], "A")
        self.assertEqual(result["final_selection"], "A")
        self.assertEqual(result["final_symbol"], "WXPAR")
    
    def test_parse_group_chat_messages_incomplete(self):
        """Test parsing incomplete group chat messages raises ValueError."""
        messages = [
            {
                "name": "Speaker",
                "content": "Option A: TSTSYM - Test symbol"  # Missing Option B
            }
        ]
        
        with self.assertRaises(ValueError) as context:
            MessageParser.parse_group_chat_messages(messages)
        self.assertEqual(
            str(context.exception),
            "Speaker's message must contain both Option A and Option B"
        )
    
    def test_parse_group_chat_messages_empty(self):
        """Test parsing empty group chat messages."""
        messages = []
        
        result = MessageParser.parse_group_chat_messages(messages)
        
        self.assertIsNone(result["option_a"])
        self.assertIsNone(result["option_b"])
        self.assertIsNone(result["listener_vote"])
        self.assertIsNone(result["final_selection"])
        self.assertIsNone(result["final_symbol"])


if __name__ == '__main__':
    unittest.main()
