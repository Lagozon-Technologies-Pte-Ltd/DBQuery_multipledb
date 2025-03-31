import json
import os
from threading import Lock

CHAT_HISTORY_FILE = "chat_history.json"
session_lock = Lock()

# Load chat history from file if available
if os.path.exists(CHAT_HISTORY_FILE):
    with open(CHAT_HISTORY_FILE, "r") as file:
        session_state = json.load(file)
else:
    session_state = {"messages": []}

def save_session():
    """Save the session state to a file for persistence."""
    with session_lock:
        with open(CHAT_HISTORY_FILE, "w") as file:
            json.dump(session_state, file, indent=4)
