from dataclasses import dataclass

@dataclass
class LLMCallLog:
    """
    Represents a single LLM interaction during DSL resolution.

    Each call log captures the full context of a model invocation, including:
      - the system prompt (high-level role or instruction),
      - the assistant prompt (runtime-specific or resolution context),
      - and the user field (the model's actual output, treated as an 'answer').

    These logs are useful for debugging, tracing resolution steps, or fine-tuning
    model behavior by replacing the answer with an improved output.

    Attributes:
        description (str):
            A short label or tag describing the purpose of this LLM call
            (e.g., "intent sequencing").

        system_prompt (str):
            The system-level prompt used to set the behavior or role of the model.

        assistant (str):
            The assistant prompt containing the DSL resolution context or generated input.

        answer (str):
            The model's response, treated as the 'user' answer in the conversational schema.
    """
    description: str
    system_prompt: str
    assistant: str
    answer: str
