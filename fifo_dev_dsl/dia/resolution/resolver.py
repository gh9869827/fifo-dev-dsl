import copy
from fifo_tool_airlock_model_env.common.models import GenerationParameters, Message
from fifo_tool_airlock_model_env.sdk.client_sdk import call_airlock_model_server
from fifo_dev_dsl.dia.dsl.parser.parser import parse_dsl
from fifo_dev_dsl.dia.dsl.elements.abort import Abort
from fifo_dev_dsl.dia.dsl.elements.abort_with_new_dsl import AbortWithNewDsl
from fifo_dev_dsl.dia.dsl.elements.base import DslBase
from fifo_dev_dsl.dia.dsl.elements.element_list import ListElement
from fifo_dev_dsl.dia.dsl.elements.propagate_slots import PropagateSlots
from fifo_dev_dsl.dia.resolution.context import ResolutionContext, ResolutionContextStackElement
from fifo_dev_dsl.dia.resolution.llm_call_log import LLMCallLog
from fifo_dev_dsl.dia.resolution.enums import ResolutionResult
from fifo_dev_dsl.dia.resolution.interaction import Interaction, InteractionAnswer
from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome
from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
from fifo_dev_dsl.common.logger import get_logger

logger = get_logger(__name__)


def resolve(
    runtime_context: LLMRuntimeContext,
    resolution_context: ResolutionContext,
    interaction: Interaction | None,
) -> ResolutionOutcome:
    """
    Resolve the current DSL structure using a resumable stack-based traversal.

    Supports mutation, user interaction, and reentry propagation hooks across a tree of DSL
    elements.

    Args:
        runtime_context (LLMRuntimeContext):
            The runtime context with environment and tool references.

        resolution_context (ResolutionContext):
            The resolution context holding the call stack and current processing state.

        interaction (Interaction | None):
            User interaction data passed in from outside resolution.

    Returns:
        ResolutionOutcome:
            The final result of the DSL tree resolution.
    """

    args = (runtime_context, resolution_context, interaction)

    def _replace_current_node_with(
        outcome: ResolutionOutcome
    ) -> ResolutionOutcome:
        """
        Replace the current node in the stack with a new DSL node.

        Args:
            outcome (ResolutionOutcome):
                The originating outcome containing propagated slots or supporting data.

        Returns:
            ResolutionOutcome:
                NEW_DSL_NODES if the replacement occurred successfully, or ABORT.
        """
        assert outcome.nodes is not None

        core_dsl_elements: list[DslBase] = []

        for element in outcome.nodes:
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

        assert len(core_dsl_elements) > 0 # for now

        if len(core_dsl_elements) == 1:
            new_node = core_dsl_elements[0]
        else:
            new_node = ListElement(core_dsl_elements)

        parent.obj.update_child(parent.idx - 1, new_node)

        logger.trace(f"--> in {parent} replacing {current.obj} by {new_node}")

        resolution_context.call_stack[-1] = ResolutionContextStackElement(new_node, 0)

        return ResolutionOutcome(
            result=ResolutionResult.NEW_DSL_NODES,
            nodes=[new_node]
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

        logger.trace("--> Clearing slot, intent and clarifying question due to abort condition")
        resolution_context.questions_being_clarified.clear()
        resolution_context.reset_state()

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
            sub_outcome = _replace_current_node_with(outcome)

            if _handle_abort_if_needed(sub_outcome):
                return True

            assert sub_outcome.nodes is not None and len(sub_outcome.nodes) == 1

            sub_outcome.nodes[0].pre_resolution(*args)
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
    """
    Orchestrates the resolution of a DSL expression tree.

    A `Resolver` manages the entire lifecycle of a DSL-based interaction, from natural
    language input to fully-resolved symbolic intent. It holds references to both:

    - `LLMRuntimeContext`: the tool/query registry, prompt templates, and model configuration
    - `ResolutionContext`: the evolving resolution state, call stack, and interaction history

    It can be initialized in two modes:
    - From a user prompt (`prompt`): invokes the intent sequencer to generate the initial DSL tree
    - From an existing DSL tree (`dsl`): used to resume resolution (e.g., after injecting an error
    handler)

    Resolution proceeds step-by-step via the `__call__` method, which wraps the `resolve(...)`
    function. Each step may:
    - mutate the DSL tree (e.g., expand a placeholder)
    - request user input (e.g., ask for a missing slot)
    - trigger an abort (e.g., user cancels the current intent)

    These outcomes are represented by the `ResolutionResult` enum and handled internally
    by the resolver loop.

    Once resolution completes, the resulting DSL tree can be evaluated using the `Evaluator`.
    If evaluation encounters a recoverable runtime failure (e.g., an intent executes but
    fails due to unsatisfied preconditions like insufficient inventory), the evaluator
    injects an `IntentRuntimeErrorResolver` node into the tree. This enables the system
    to pause, prompt the user for corrective input, and resume resolution with the updated tree.

    Use `fully_resolve_in_text_mode()` to drive an interactive resolution loop via standard
    input/output.

    Example:
    ```
    resolver = Resolver(runtime_context, prompt="give me 3 screws")
    while True:
        resolver.fully_resolve_in_text_mode()
        tree = resolver.dsl_elements
        result = Evaluator(runtime_context, tree).evaluate()
        if result.status != EvaluationStatus.ABORTED_RECOVERABLE:
            break
        resolver = Resolver(runtime_context, dsl=tree)
    ```

    Attributes:
        _runtime_context (LLMRuntimeContext):
            Contains tool registry, query sources, prompt templates, and model configuration.

        _resolution_context (ResolutionContext):
            Tracks resolution progress including stack, active slots, known values,
            clarification logs, and LLM call history.

        _root_dsl_elements (ListElement):
            The root of the current symbolic DSL tree.
    """

    _resolution_context: ResolutionContext
    _runtime_context: LLMRuntimeContext
    _root_dsl_elements: ListElement

    def __init__(self,
                 runtime_context: LLMRuntimeContext,
                 prompt: str | None = None,
                 dsl: ListElement | None = None):
        """
        Initialize a new resolver.

        A `Resolver` can be created either from a raw user prompt or from an existing DSL tree:
        - If `prompt` is provided, the intent sequencer model is invoked to generate the initial
          DSL.
        - If `dsl` is provided, it will be used directly as the root tree (commonly after a failed
          evaluation).

        Args:
            runtime_context (LLMRuntimeContext):
                The runtime environment containing tools, query sources, and prompt templates.

            prompt (str | None):
                Raw user input to interpret and convert into a symbolic DSL tree.

            dsl (ListElement | None):
                A pre-parsed symbolic DSL tree, typically used when resuming resolution.

        Raises:
            ValueError:
                If neither `prompt` nor `dsl` is provided.
        """
        self._runtime_context = runtime_context
        self._resolution_context = ResolutionContext()

        if dsl is not None:
            self._root_dsl_elements = copy.deepcopy(dsl)
        elif prompt is not None:
            self._process_user_prompt(prompt)
        else:
            raise ValueError("Either a prompt or a parsed DSL must be provided.")

        self._resolution_context.call_stack.clear()
        self._resolution_context.call_stack.append(
            ResolutionContextStackElement(self._root_dsl_elements, 0)
        )

    def __call__(self, interaction_reply: Interaction | None) -> ResolutionOutcome:
        """
        Advance the resolution process by one step.

        This method drives the resolution loop by forwarding the current state and any user
        response to the core `resolve(...)` function. It applies any resulting mutations
        to the DSL tree or pauses for interaction as needed.

        Args:
            interaction_reply (Interaction | None):
                The user's reply to the previous question (if any). Pass `None` on the first call
                or to proceed without new input.

        Returns:
            ResolutionOutcome:
                Result of the resolution step, including whether resolution changed the tree,
                paused for interaction, or triggered an abort.
        """
        return resolve(
            self._runtime_context, self._resolution_context, interaction_reply
        )

    @property
    def dsl_elements(self) -> ListElement:
        """
        Return a deep copy of the root DSL tree.

        This is typically used to pass the DSL to the `Evaluator`, which may mutate
        the tree (e.g. by injecting recovery nodes on failure). Returning a copy ensures
        that internal state within the resolver remains isolated.

        Returns:
            ListElement:
                A copy of the current DSL tree managed by this resolver.
        """
        return copy.deepcopy(self._root_dsl_elements)

    def _process_user_prompt(self, prompt: str):
        """
        Translate the user's natural language input into a symbolic DSL tree.

        This method calls the intent sequencer model with the provided prompt
        and stores the generated DSL output as the root of the resolution tree.
        It also logs the full interaction — including the system prompt, user input,
        and model output — into the `llm_call_logs` for traceability.

        Args:
            prompt (str):
                Raw user request to be interpreted and converted into DSL.

        Side Effects:
            - Updates `_root_dsl_elements` with the parsed DSL tree.
            - Appends a new entry to `_resolution_context.llm_call_logs`.
        """

        answer = call_airlock_model_server(
            model=self._runtime_context.base_model,
            adapter=self._runtime_context.intent_sequencer_adapter,
            messages=[
                Message.system(self._runtime_context.system_prompt_intent_sequencer),
                Message.user(prompt)
            ],
            parameters=GenerationParameters(
                max_new_tokens=1024,
                do_sample=False
            ),
            container_name=self._runtime_context.container_name,
            host=self._runtime_context.host
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
        """
        Drive the resolution process interactively using stdin/stdout.

        This utility method repeatedly advances the DSL resolution by calling the
        resolver until no further user input is needed. If an interaction is required
        (e.g. to fill a slot using ASK), the question is printed to the console,
        and the user's response is read from standard input.

        This is primarily intended for testing or demos, not production use.

        Behavior:
            - Calls `self(...)` with the last interaction response (or None at first).
            - Expects resolution to either finish or request a single interaction.
            - Repeats until no further interaction is needed (`UNCHANGED` result).

        Raises:
            AssertionError:
                If an unexpected resolution result is returned (i.e. not INTERACTION_REQUESTED
                or UNCHANGED), or if an interaction is missing when required.
        """

        interaction_reply = None
        while True:
            outcome = self(interaction_reply)
            if outcome.result is ResolutionResult.UNCHANGED:
                break
            assert outcome.result is ResolutionResult.INTERACTION_REQUESTED
            assert outcome.interaction is not None
            print(f"< {outcome.interaction.message}")
            answer = InteractionAnswer(content=input("> "))
            interaction_reply = Interaction(outcome.interaction, answer)
