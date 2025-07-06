from typing import Iterator, cast
import difflib

from fifo_tool_datasets.sdk.hf_dataset_adapters.dsl import DSLAdapter
from fifo_tool_airlock_model_env.sdk.client_sdk import call_airlock_model_server
from fifo_tool_airlock_model_env.common.models import GenerationParameters, Message
from fifo_tool_airlock_model_env.common.models import Model


def dsl_similarity_indicator(str1: str, str2: str) -> str:
    """
    Compare two DSL strings and return a similarity indicator with emoji and score.

    The function uses difflib to compute a similarity ratio between two strings
    and returns:
        🟢 for high similarity (>= 90)
        🟡 for medium similarity (>= 70)
        🔴 for low similarity (< 70)

    The result is returned as: "<emoji> <score>" with the score right-aligned (e.g., "🟡  78")
    
    Args:
        str1 (str):
            First DSL string

        str2 (str):
            Second DSL string

    Returns:
        str:
            An emoji indicator and a 3-digit similarity score
    """
    similarity = difflib.SequenceMatcher(None, str1, str2).ratio()
    score = int(similarity * 100)

    if score >= 95:
        emoji = "🟢"
    elif score >= 75:
        emoji = "🟡"
    else:
        emoji = "🔴"

    return f"{emoji} {score:>3}"


adapter_obj = DSLAdapter()
dataset_dict = adapter_obj.from_hub_to_dataset_wide_dict(
    "a6188466/dia-intent-sequencer-robot-arm-dataset"
)
dataset_test = list(cast(Iterator[dict[str, str]], dataset_dict["test"]))

for entry in dataset_test:

    system_prompt = entry["system"]
    input_text = entry["in"]
    expected_dsl_text = entry["out"]

    try:
        model_dsl_text = call_airlock_model_server(
            model=Model.Phi4MiniInstruct,
            adapter="dia-intent-sequencer-robot-arm-adapter",
            messages=[
                Message.system(system_prompt),
                Message.user(input_text)
            ],
            parameters=GenerationParameters(
                max_new_tokens=1024,
                do_sample=False
            ),
            container_name="phi"
        )

    except RuntimeError as e:
        model_dsl_text = ""

    sim = dsl_similarity_indicator(model_dsl_text, expected_dsl_text)
    print(f"{sim} {model_dsl_text} {expected_dsl_text}")
