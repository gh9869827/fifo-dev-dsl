
from common.llm.dia.dsl.elements.abort import Abort
from common.llm.dia.dsl.elements.abort_with_new_dsl import AbortWithNewDsl
from common.llm.dia.dsl.elements.base import DslBase
from common.llm.dia.dsl.elements.element_list import ListElement
from common.llm.dia.dsl.elements.propagate_slots import PropagateSlots
from common.llm.dia.resolution.context import LLMCallLog, ResolutionContext, ResolutionContextStackElement
from common.llm.dia.resolution.enums import AbortBehavior, ResolutionResult
from common.llm.dia.resolution.interaction import Interaction, InteractionAnswer
from common.llm.dia.resolution.outcome import ResolutionOutcome
from common.llm.dia.runtime.context import LLMRuntimeContext
from common.llm.airlock_model_env.common.models import GenerationParameters, Message, Model, Role
from common.llm.airlock_model_env.sdk.client_sdk import call_airlock_model_server
from common.llm.dia.dsl.parser.parser import parse_dsl


def resolve(runtime_context: LLMRuntimeContext,
            resolution_context: ResolutionContext,
            abort_behavior: AbortBehavior,
            interaction: Interaction | None) -> ResolutionOutcome:
    """
    Resolve the current DSL structure using a resumable stack-based traversal.

    Supports mutation, user interaction, and reentry propagation hooks across a tree of DSL
    elements.

    Args:
        runtime_context (LLMRuntimeContext):
            The runtime context with environment and tool references.

        resolution_context (ResolutionContext):
            The resolution context holding the call stack and current processing state.

        abort_behavior (AbortBehavior):
            Specifies how to handle abort signals and intent overrides.

        interaction (Interaction | None):
            User interaction data passed in from outside resolution.

    Returns:
        ResolutionOutcome:
            The final result of the DSL tree resolution.
    """

    args = (runtime_context, resolution_context, abort_behavior, interaction)

    def _replace_current_node_with(
        new_node: DslBase,
        outcome: ResolutionOutcome
    ) -> ResolutionOutcome:
        """
        Replace the current node in the stack with a new DSL node.

        Args:
            new_node (DslBase):
                The new DSL node to install.

            outcome (ResolutionOutcome):
                The originating outcome containing propagated slots or supporting data.

        Returns:
            ResolutionOutcome:
                CHANGED if the replacement occurred successfully, or ABORT.
        """
        assert outcome.node is not None

        core_dsl_elements = []

        for element in outcome.node:
            if isinstance(element, Abort):
                return ResolutionOutcome(
                    result=ResolutionResult.ABORT,
                    node=None
                )
            if isinstance(element, AbortWithNewDsl):
                return ResolutionOutcome(
                    result=ResolutionResult.ABORT,
                    node=element.new_dsl
                )
            if isinstance(element, PropagateSlots):
                resolution_context.add_propagated_slot(element)
            else:
                core_dsl_elements.append(element)

        assert len(resolution_context.call_stack) >= 2

        current = resolution_context.call_stack[-1]
        parent = resolution_context.call_stack[-2]
        parent.obj.update_child(parent.idx - 1, new_node)

        print(f"--> in {parent} replacing {current.obj} by {new_node}")

        resolution_context.call_stack[-1] = ResolutionContextStackElement(new_node, 0)

        return ResolutionOutcome(
            result=ResolutionResult.CHANGED
        )

    def _try_call_on_reentry() -> None:
        """
        Call `on_reentry_resolution()` on the parent node after its child finishes.

        This is used to apply propagated slots or update intermediate resolution state.
        """
        if len(resolution_context.call_stack) >= 2:
            parent = resolution_context.call_stack[-2]
            parent.obj.on_reentry_resolution(*args)

    def _handle_abort_if_needed(sub_outcome: ResolutionOutcome) -> bool:
        """
        Handle an ABORT signal by removing or replacing the current intent in the stack.

        Args:
            sub_outcome (ResolutionOutcome):
                An outcome returned from resolution to check.

        Returns:
            bool:
                True if the abort was handled and traversal should continue.
        """
        if sub_outcome.result is not ResolutionResult.ABORT:
            return False

        while resolution_context.call_stack:
            if isinstance(resolution_context.call_stack[-1].obj, ListElement):
                break

            resolution_context.call_stack.pop()
        else:
            raise RuntimeError("ABORT: no ListElement node found in call stack")

        assert resolution_context.call_stack

        parent = resolution_context.call_stack[-1]

        print("--> Clearing slot, intent and clarifying question due to abort condition")
        resolution_context.questions_being_clarified.clear()
        resolution_context.slot = None
        resolution_context.intent = None

        if sub_outcome.node is not None:
            parent.obj.update_child(parent.idx - 1, sub_outcome.node)
            resolution_context.call_stack.append(
                ResolutionContextStackElement(sub_outcome.node, 0)
            )
            sub_outcome.node.pre_resolution(*args)
        else:
            parent.obj.remove_child(parent.idx - 1)

        return True

    def _process_current_node(current: ResolutionContextStackElement) -> bool:
        """
        Perform resolution on a DSL node and handle any special outcomes.

        Args:
            current (ResolutionContextStackElement):
                The current call stack element being resolved.

        Returns:
            bool:
                True if resolution completed and traversal should continue.
        """
        outcome = current.obj.do_resolution(*args)

        if _handle_abort_if_needed(outcome):
            return True

        if outcome.result is ResolutionResult.INTERACTION_REQUESTED:
            raise StopIteration(outcome)

        current.obj.post_resolution(*args)

        if outcome.result is ResolutionResult.NEW_DSL_NODES:
            new_node = outcome.node[0]
            sub_outcome = _replace_current_node_with(new_node, outcome)

            if _handle_abort_if_needed(sub_outcome):
                return True

            new_node.pre_resolution(*args)
            _try_call_on_reentry()
            return True

        _try_call_on_reentry()
        resolution_context.call_stack.pop()
        return True

    while resolution_context.call_stack:
        current = resolution_context.call_stack[-1]

        try:
            if current.obj.is_leaf():
                if _process_current_node(current):
                    continue
            else:
                children = current.obj.get_children()
                if current.idx >= len(children):
                    if _process_current_node(current):
                        continue
                else:
                    child = children[current.idx]
                    resolution_context.call_stack.append(ResolutionContextStackElement(child, 0))
                    child.pre_resolution(*args)
                    current.idx += 1

        except StopIteration as interrupt:
            return interrupt.value

    return ResolutionOutcome()


class Resolver:

    _resolution_context: ResolutionContext
    _runtime_context: LLMRuntimeContext
    _root_dsl_elements: ListElement

    def __init__(self, runtime_context: LLMRuntimeContext, prompt: str):
        self._runtime_context = runtime_context
        self._resolution_context = ResolutionContext()
        self._process_user_prompt(prompt)
        self._resolution_context.call_stack.clear()
        self._resolution_context.call_stack.append(
            ResolutionContextStackElement(self._root_dsl_elements, 0)
        )

    def __call__(self, interaction_reply: Interaction | None) -> ResolutionOutcome:
        return resolve(
            self._runtime_context, self._resolution_context, AbortBehavior.SKIP, interaction_reply
        )

    def _process_user_prompt(self, prompt: str):

        answer = call_airlock_model_server(
            model=Model.Phi4MiniInstruct,
            adapter="intent-sequencer",
            messages=[
                    Message(
                        role=Role.system,
                        content=self._runtime_context.system_prompt_intent_sequencer
                    ),
                    Message(
                        role=Role.user,
                        content=prompt
                    )
            ],
            parameters=GenerationParameters(
                max_new_tokens=1024,
                do_sample=False
            ),
            container_name="dev-phi"
        )

        self._resolution_context.llm_call_logs.append(
            LLMCallLog(
                description="main",
                system_prompt=self._runtime_context.system_prompt_intent_sequencer,
                assistant=prompt,
                answer=answer
            )
        )

        self._root_dsl_elements = parse_dsl(answer)

    def fully_resolve_in_text_mode(self):
        interaction_reply = None
        while True:
            outcome = self(interaction_reply)
            if outcome.result is ResolutionResult.UNCHANGED:
                break
            assert outcome.result is ResolutionResult.INTERACTION_REQUESTED
            print(f"< {outcome.interaction.message}")
            answer = InteractionAnswer(content=input("> "))
            interaction_reply = Interaction(outcome.interaction, answer)
