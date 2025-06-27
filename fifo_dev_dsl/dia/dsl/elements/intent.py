from __future__ import annotations
from typing import TYPE_CHECKING, Any

from dataclasses import dataclass

from fifo_dev_dsl.dia.dsl.elements.base import make_dsl_container
from fifo_dev_dsl.dia.dsl.elements.slot import Slot
from fifo_dev_dsl.common.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_dsl.dia.resolution.interaction import Interaction
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext

@dataclass
class Intent(make_dsl_container(Slot)):
    """
    A DSL node that calls a named tool with structured parameters.

    An Intent represents a call to a tool, with arguments provided as an ordered
    list of :class:`Slot` objects. Each slot maps a name to a value, which may
    itself be another DSL expression. This enables nested calls and composable logic.

    During evaluation, the runtime context resolves the tool by name and invokes
    it with the evaluated slot values.

    Attributes:
        name (str):
            The name of the tool to invoke.
    """

    name: str

    def __init__(self, name: str, slots: list[Slot]):
        super().__init__(slots)
        self.name = name

    def __eq__(self, other: Any) -> bool:
        return (
                isinstance(other, self.__class__)
            and self.name == other.name
            and super().__eq__(other)
        )

    @property
    def slots(self) -> list[Slot]:
        """
        Return the list of slots contained in the intent.

        This is a convenience property that delegates to `get_items()`,
        typically used to access the key-value argument pairs of the intent
        in the order they were defined.

        Returns:
            list[Slot]:
                A list of Slot objects representing named arguments.
        """
        return self.get_items()

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of the intent.

        Formats the intent as a function-style call with named slot arguments,
        such as `set_task_due_date(task_uid=42, due_date="tomorrow at noon")`.

        Returns:
            str:
                A string representing the intent and its slots in DSL form.
        """
        parts = [slot.to_dsl_representation() for slot in self._items]
        return f"{self.name}({', '.join(parts)})"

    def _propagate_slots(self,
                         resolution_context: ResolutionContext):

        assert resolution_context.slot is None

        for propagated_slots in resolution_context.take_propagated_slots():
            pslots = propagated_slots.to_dict()
            updated: set[str] = set()

            for slot in self.get_items():
                pslot_value = pslots.get(slot.name)
                if pslot_value is not None:
                    logger.trace(
                        f"--> propagating slots {slot.name}, "
                        f"{slot.value} replaced by {pslot_value} "
                    )
                    slot.value = pslot_value
                    updated.add(slot.name)

            # process the unconsumed propagated slots
            for name, value in pslots.items():
                if name not in updated:
                    self._items.append(Slot(name, value))

    def pre_resolution(
        self,
        runtime_context: LLMRuntimeContext,
        resolution_context: ResolutionContext,
        interaction: Interaction | None,
    ) -> None:
        super().pre_resolution(runtime_context, resolution_context, interaction)
        resolution_context.entering_intent(self)

    def post_resolution(
        self,
        runtime_context: LLMRuntimeContext,
        resolution_context: ResolutionContext,
        interaction: Interaction | None,
    ) -> None:
        super().post_resolution(runtime_context, resolution_context, interaction)
        resolution_context.exiting_intent()

    def on_reentry_resolution(
        self,
        runtime_context: LLMRuntimeContext,
        resolution_context: ResolutionContext,
        interaction: Interaction | None,
    ) -> None:
        super().on_reentry_resolution(
            runtime_context, resolution_context, interaction
        )
        self._propagate_slots(resolution_context)

    def eval(self,
             runtime_context: LLMRuntimeContext) -> Any:
        """
        Evaluate this intent by invoking the named tool with the evaluated slot values as arguments.

        During evaluation, the runtime context resolves the tool by name and
        calls it with arguments obtained by evaluating each slot. If any slot
        or nested value is unresolved, evaluation will fail with a RuntimeError.

        Args:
            runtime_context (LLMRuntimeContext):
                Execution context providing tool access, query sources, and runtime helpers.


        Returns:
            Any:
                The result returned by the invoked tool.

        Raises:
            RuntimeError: If any slot or nested value is not resolved.
        """

        tool = runtime_context.get_tool(self.name)

        args = {
            slot.name: tool.tool_docstring.get_arg_by_name(slot.name).pytype.cast(
                slot.value.eval(runtime_context), allow_scalar_to_list=True
            )
            for slot in self._items
        }

        if tool.tool_docstring.return_type is None:
            tool(**args)
            return None

        ret = tool.tool_docstring.return_type.cast(tool(**args))

        return ret
