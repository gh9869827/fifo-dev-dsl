
from typing import List, Optional, Tuple
from common.llm.airlock_model_env.common.models import GenerationParameters, Message, Model, Role
from common.llm.airlock_model_env.sdk.client_sdk import call_airlock_model_server
from common.llm.dia.dsl.elements.base import DslBase
from common.llm.dia.dsl.parser.parser import parse_dsl
from common.llm.dia.resolution.context import ResolutionContext
from common.llm.dia.resolution.enums import ResolutionKind, ResolutionResult
from common.llm.dia.resolution.interaction import Interaction, InteractionAnswer, InteractionRequest
from common.llm.dia.resolution.outcome import ResolutionOutcome
from common.llm.dia.runtime.context import LLMRuntimeContext


def process_user_prompt(runtime_context: LLMRuntimeContext, prompt: str) -> Tuple[Optional[InteractionRequest], List[DslBase]]:

    answer = call_airlock_model_server(
        model=Model.Phi4MiniInstruct,
        adapter="intent-sequencer",
        messages=[
                Message(role=Role.system, content=runtime_context.system_prompt_intent_sequencer),
                Message(role=Role.user, content=prompt)
        ],
        parameters=GenerationParameters(
            max_new_tokens=1024,
            do_sample=False
        ),
        container_name="dev-phi"
    )

    print("main processing")
    print("---")
    print("$")
    print(runtime_context.system_prompt_intent_sequencer)
    print(">")
    print(prompt)
    print("<")
    print(answer)
    print("---")

    dsl_elements = parse_dsl(answer)

    return resume_process_user_prompt(runtime_context, dsl_elements, None)

def resume_process_user_prompt(runtime_context: LLMRuntimeContext, dsl_elements: List[DslBase], interaction: Optional[Interaction]) -> Tuple[Optional[InteractionRequest], List[DslBase]]:

    def fct_resolve(dsl_elements: List[DslBase], kind: set[ResolutionKind], interaction: Optional[Interaction]) -> Tuple[Optional[InteractionRequest], List[DslBase]]:
        while True:
            resolutions_result = ResolutionResult.NOT_APPLICABLE
            new_dsl_elements = []
            skip = False
            outcome = None
            for dsl_element in dsl_elements:
                if skip:
                    new_dsl_elements.append(dsl_element)
                    continue

                outcome = dsl_element.resolve(runtime_context, kind, ResolutionContext(), interaction)
                
                if outcome.result is ResolutionResult.ABORT:
                    # we drop the current intent as it was aborted
                    continue

                # we skip the abort
                resolutions_result = resolutions_result.combine(outcome.result)

                if outcome.result is ResolutionResult.INTERACTION_REQUESTED:
                    skip = True
                
                if outcome.resolved is not None:
                    new_dsl_elements.append(outcome.resolved)

            dsl_elements = new_dsl_elements
            print("fct_resolve after processing all elements", dsl_elements)
            if skip and outcome is not None:
                print("skip", outcome)
                return outcome.interaction, dsl_elements
            
            if resolutions_result is ResolutionResult.NOT_APPLICABLE:
                return None, dsl_elements
        

    requested_interaction, dsl_elements = fct_resolve(dsl_elements, {ResolutionKind.QUERY_USER, ResolutionKind.ASK}, interaction)

    if requested_interaction is not None:
        return requested_interaction, dsl_elements
    
    requested_interaction, dsl_elements = fct_resolve(dsl_elements, ResolutionKind.QUERY_FILL, interaction)

    if requested_interaction is not None:
        return requested_interaction, dsl_elements

    return None, dsl_elements

def process_user_prompt_text_mode(runtime_context: LLMRuntimeContext, prompt: str) -> List[DslBase]:
    requested_interaction, dsl_elements = process_user_prompt(runtime_context, prompt)

    print(requested_interaction, dsl_elements)

    while requested_interaction:

        print(f"< {requested_interaction.message}")
        answer = InteractionAnswer(content=input("> "))

        requested_interaction, dsl_elements = resume_process_user_prompt(
            runtime_context=runtime_context,
            dsl_elements=dsl_elements,
            interaction=Interaction(requested_interaction, answer)
        )

    return dsl_elements
