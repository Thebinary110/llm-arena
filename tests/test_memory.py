"""Unit tests for ConversationMemory."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory.conversation_memory import ConversationMemory


def test_add_messages():
    """Messages added via add_user_message / add_assistant_message appear in get_history."""
    mem = ConversationMemory(max_turns=5)
    mem.add_user_message("Hello")
    mem.add_assistant_message("Hi there!")

    history = mem.get_history()
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1] == {"role": "assistant", "content": "Hi there!"}


def test_max_turns_sliding_window():
    """When more than max_turns are added, the oldest messages are dropped."""
    mem = ConversationMemory(max_turns=2)
    # Add 3 complete turns (6 messages); maxlen = 4 → oldest 2 messages dropped
    mem.add_user_message("turn1-user")
    mem.add_assistant_message("turn1-assistant")
    mem.add_user_message("turn2-user")
    mem.add_assistant_message("turn2-assistant")
    mem.add_user_message("turn3-user")
    mem.add_assistant_message("turn3-assistant")

    history = mem.get_history()
    assert len(history) == 4  # only last 2 turns kept
    assert history[0]["content"] == "turn2-user"
    assert history[1]["content"] == "turn2-assistant"
    assert history[2]["content"] == "turn3-user"
    assert history[3]["content"] == "turn3-assistant"


def test_clear():
    """After clear(), get_history() returns an empty list."""
    mem = ConversationMemory(max_turns=5)
    mem.add_user_message("Hello")
    mem.add_assistant_message("Hi!")
    mem.clear()

    assert mem.get_history() == []
    assert len(mem) == 0


def test_turn_count():
    """turn_count property returns number of complete user-assistant pairs."""
    mem = ConversationMemory(max_turns=10)
    assert mem.turn_count == 0

    mem.add_user_message("Q1")
    assert mem.turn_count == 0  # incomplete turn

    mem.add_assistant_message("A1")
    assert mem.turn_count == 1

    mem.add_user_message("Q2")
    mem.add_assistant_message("A2")
    assert mem.turn_count == 2


def test_len():
    """__len__ returns total number of individual messages."""
    mem = ConversationMemory(max_turns=10)
    assert len(mem) == 0
    mem.add_user_message("Hello")
    assert len(mem) == 1
    mem.add_assistant_message("Hi")
    assert len(mem) == 2
