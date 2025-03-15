# Copyright (c) Microsoft. All rights reserved.

import json
import os
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.contents.chat_history import ChatHistory

class TestCase:
    def __init__(self, question, expected_answer=None, assertion=None, chat_history=None):
        self.question = question
        self.expected_answer = expected_answer
        self.assertion = assertion
        self.chat_history = chat_history or []


def read_testcases_from_jsonl(file_path):
    testcases = []

    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            testcases.append(TestCase(**data))
    return testcases

def convert_chat_history_to_sk_chat_history(chat_history):
    sk_chat_history = ChatHistory(messages=[])
    
    for entry in chat_history:
        user_message = ChatMessageContent(
            role=AuthorRole.USER,
            content=entry['inputs']['question']
        )
        assistant_message = ChatMessageContent(
            role=AuthorRole.ASSISTANT,
            content=entry['outputs']['answer']
        )
        
        # Add messages to the SK ChatHistory
        sk_chat_history.add_message(user_message)
        sk_chat_history.add_message(assistant_message)
    
    return sk_chat_history

