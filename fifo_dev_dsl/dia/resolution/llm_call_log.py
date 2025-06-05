from dataclasses import dataclass

@dataclass
class LLMCallLog:
    # each llm call is composed exactly of a system prompt a assistant prompt and a user(answer)
    # prompt the system prompt gives the instructions, the assistant prompt gives the 
    # resolution/runtime specific context and the answer/user is the actual output of the llm.
    # a Call Log can be used to fine tune the model behavior by correcting the answer of a model
    # and retraining on the new expected output.
    description: str
    system_prompt: str
    assistant: str
    answer: str