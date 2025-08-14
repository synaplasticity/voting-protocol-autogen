from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
import json
from datetime import datetime
import logging
import re
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "sk-..."  # Replace with your OpenAI key

# --- SYMBOL TABLE MEMORY ---
symbol_table = []

# --- HELPER FUNCTION TO GET SYMBOL HISTORY ---
def get_symbol_history():
    symbol_counts = {}
    for entry in symbol_table:
        symbol = entry["symbol"]
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    return symbol_counts

# --- PERSONALITY TYPES ---
PERSONALITY_TRAITS = [
    "creative", "precise", "minimalist", "verbose", "technical", "funny"
]

def random_personality():
    return random.choice(PERSONALITY_TRAITS)

# --- SPEAKER: Proposes multiple symbolic codes ---
def make_speaker():
    memory = get_symbol_history()
    memory_prompt = "Past popular symbols: " + ", ".join(f"{k} ({v})" for k, v in memory.items()) if memory else ""
    personality = random_personality()
    return AssistantAgent(
        name="Speaker",
        system_message=f"""
You receive a task and propose two short symbolic names for it (each max 6 characters).
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

# --- LISTENER: Votes on clarity of proposals ---
def make_listener():
    personality = random_personality()
    return AssistantAgent(
        name="Listener",
        system_message=f"""
You receive two symbolic task codes.
Vote for the one that you think best represents the task clearly and efficiently.
Just say "I vote for Option A" or "I vote for Option B" with a short reason.
Your personality is: {personality} — base your reasoning style on it.
"""
    )

# --- NEGOTIATOR: Final vote or tie breaker ---
def make_negotiator():
    personality = random_personality()
    return AssistantAgent(
        name="Negotiator",
        system_message=f"""
You receive the Speaker's proposals and Listener's vote.
Make the final decision on which symbolic code to use.
Say "Final selection: Option A" or "Final selection: Option B" with a justification.
Your personality is: {personality} — use it to guide your logic and tone.
"""
    )

# --- USER (Task Giver) ---
user = UserProxyAgent(
    name="User",
    code_execution_config=False,
)

# --- TASK LIST ---
tasks = [
    "Translate 'I love AI' into Japanese",
    "Get current weather in Paris",
    "Summarize this article: 'AI is transforming banking'",
    "Convert 100 USD to EUR",
    "Generate a haiku about neural networks"
]

# --- LOOP THROUGH TASKS ---
for task in tasks:
    speaker = make_speaker()
    listener = make_listener()
    negotiator = make_negotiator()

    groupchat = GroupChat(  
        agents=[user, speaker, listener, negotiator],
        messages=[],
        max_round=6,
    )
    manager = GroupChatManager(groupchat=groupchat)

    logger.info(f"Running task: {task}")
    user.initiate_chat(manager, message=f"Please compress this task: {task}")

    # Extract the final selection and logs from groupchat messages
    final_selection = None
    option_a = None
    option_b = None
    for msg in groupchat.messages:
        sender = msg.get("name", "")
        content = msg.get("content", "")

        if sender == "Speaker" and "Option A:" in content and "Option B:" in content:
            logger.info(f"Speaker proposals:{content}")
            match_a = re.search(r"Option A:\s*(\w{1,6})", content)
            if match_a:
                option_a = match_a.group(1)
            match_b = re.search(r"Option B:\s*(\w{1,6})", content)
            if match_b:
                option_b = match_b.group(1)

        if sender == "Listener" and ("I vote for Option A" in content or "I vote for Option B" in content):
            logger.info(f"Listener vote: {content}")

        if sender == "Negotiator" and "Final selection" in content:
            logger.info(f"Negotiator decision: {content}")
            match_final = re.search(r"Final selection:\s*Option (A|B)", content)
            if match_final:
                final_option = match_final.group(1)
                final_selection = option_a if final_option == "A" else option_b

    if final_selection:
        timestamp = datetime.now().isoformat()
        symbol_table.append({
            "timestamp": timestamp,
            "task": task,
            "symbol": final_selection
        })
        logger.info(f"Symbol selected: {final_selection}")
    else:
        logger.warning("No valid final selection found.")

# --- SAVE SYMBOL TABLE ---
with open("symbol_table_log.json", "w") as f:
    json.dump(symbol_table, f, indent=2)

logger.info("Symbol table memory saved to 'symbol_table_log.json'")
