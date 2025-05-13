from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from common.introspection.docstring import MiniDocStringType
from common.llm.dia.resolution.interaction import Interaction
from common.llm.dia.resolution.outcome import ResolutionOutcome

if TYPE_CHECKING:
    from common.llm.dia.resolution.resolver import AbortBehavior
    from common.llm.dia.resolution.context import ResolutionContext
    from common.llm.dia.runtime.context import LLMRuntimeContext


class DslBase:
    """
    Base class for all DSL elements.

    Each DSL element represents a node in the intent structure and can optionally
    participate in resolution during a given resolution wave (e.g., user interaction,
    runtime queries), or in evaluation (producing a final value).

    Subclasses override `resolve()` to participate in specific resolution phases,
    and `eval()` if they produce a concrete value during execution.
    """

    def represent_content_as_text(self) -> str | None:
        """
        Represent the content of this DSL node as a string, if possible.

        This method is used to pass resolved or unresolved values to an LLM in text form,
        for example, when querying external knowledge, resolving slots, or sequencing intents.

        Not all DSL nodes can be meaningfully represented as text. If this node or any of its
        children cannot, this method should return None.

        By default, returns None. Override in subclasses where textual representation is possible.

        Returns:
            str | None:
                A string representation of this node's content, or None if not representable.
        """
        return None


    def pretty_print_dsl(self, indent: int = 0) -> None:
        """
        Recursively print the DSL tree in a readable, indented format.

        Args:
            indent (int):
                Current indentation level (used internally for recursion).
        """
        prefix = "  " * indent
        print(f"{prefix}{repr(self)}")

        for child in self.get_children():
            child.pretty_print_dsl(indent + 1)

    def get_children(self) -> list[DslBase]:
        """
        Return a list of DslBase child elements.

        Returns:
            list[tuple[str, DslBase]]:
                The ordered list of child key-value pairs.
        """
        return []

    def is_leaf(self) -> bool:
        """
        Return True if this node has no children.

        Returns:
            bool:
                True if the node is a leaf, False otherwise.
        """
        return not self.get_children()

    def update_child(self, index: int, new_child: DslBase) -> None:
        """
        Replace the child at the given index with a new value.

        Args:
            index (int):
                Index of the child to replace.

            new_child (DslBase):
                The new node to insert.
        """
        raise RuntimeError(f"{self.__class__.__name__} is a leaf node; it cannot update children.")

    def insert_child(self, index: int, new_child: DslBase) -> None:
        """
        Insert a new child node at the specified index.

        Args:
            index (int):
                Index at which to insert the new child.

            new_child (DslBase):
                The new node to insert.
        """
        raise RuntimeError(f"{self.__class__.__name__} is a leaf node; it cannot insert children.")

    def remove_child(self, index: int) -> None:
        """
        Remove the child at the specified index.

        Args:
            index (int):
                Index of the child to remove.
        """
        raise RuntimeError(f"{self.__class__.__name__} is a leaf node; it cannot remove children.")

    def is_abort_boundary(self) -> bool:
        """
        Return True if this node represents a boundary for abort pruning.

        Returns:
            bool:
                True if this node defines an abort scope boundary.
        """
        return False

    def _log_resolution(self,
                        label: str,
                        resolution_context: ResolutionContext):
        pad = "  " * len(resolution_context.call_stack)
        print(f"{pad}[{label:<8}] {self}")

    def pre_resolution(self,
                    runtime_context: LLMRuntimeContext,
                    resolution_context: ResolutionContext,
                    abort_behavior: AbortBehavior,
                    interaction: Interaction | None):
        _ = runtime_context, abort_behavior, interaction
        self._log_resolution(" → pre", resolution_context)

    def do_resolution(self,
                    runtime_context: LLMRuntimeContext,
                    resolution_context: ResolutionContext,
                    abort_behavior: AbortBehavior,
                    interaction: Interaction | None) -> ResolutionOutcome:
        _ = runtime_context, abort_behavior, interaction
        self._log_resolution("⚙️  do   ", resolution_context)
        return ResolutionOutcome()

    def post_resolution(self,
                        runtime_context: LLMRuntimeContext,
                        resolution_context: ResolutionContext,
                        abort_behavior: AbortBehavior,
                        interaction: Interaction | None):
        _ = runtime_context, abort_behavior, interaction
        self._log_resolution(" ← post", resolution_context)

    def on_reentry_resolution(self,
                            runtime_context: LLMRuntimeContext,
                            resolution_context: ResolutionContext,
                            abort_behavior: AbortBehavior,
                            interaction: Interaction | None):
        _ = runtime_context, abort_behavior, interaction
        self._log_resolution(" ↺ visit", resolution_context)

    def is_resolved(self) -> bool:
        """
        Indicates whether this DSL element is fully resolved.

        A resolved element has no remaining ASK, QUERY_FILL, or QUERY_USER nodes
        and is ready for evaluation or execution. Subclasses override this when
        their resolution state depends on internal fields.

        Returns:
            bool:
                True if the element is resolved, False otherwise.
        """
        return True

    def eval(self,
             runtime_context: LLMRuntimeContext,
             value_type: MiniDocStringType | None = None) -> Any:
        """
        Evaluate the DSL element and return its final value.

        Only fully resolved elements should be evaluated. Subclasses override this
        to return a concrete value (e.g., a string, number, or list). If called on
        an unresolved element or one that does not support evaluation, a RuntimeError
        is raised.

        Args:
            runtime_context (LLMRuntimeContext):
                Runtime context providing access to tools, sources, and LLM prompts.

            value_type (Optional[MiniDocStringType]):
                Optional hint about the expected type of the result.

        Returns:
            Any:
                The concrete value resulting from evaluation.

        Raises:
            RuntimeError:
                If the element cannot be evaluated.
        """
        raise RuntimeError("Expression cannot be evaluated, be sure it is resolvable/resolved.")


T = TypeVar("T", bound=DslBase)

class DslContainerBase(DslBase, Generic[T], ABC):

    _items: list[T]

    def __init__(self, items: list[T]):
        """
        Initialize a container node with an ordered list of items.

        Args:
            items (list[T]):
                A list of container items to store internally.
        """
        self._items: list[T] = items

    def is_resolved(self) -> bool:
        return all(val.is_resolved() for val in self._items)

    def get_items(self) -> list[T]:
        """
        Return the internal list of stored items.

        Returns:
            list[T]:
                The internal list used by this container.
        """
        return self._items

    def get_children(self) -> list[DslBase]:
        """
        Return the list of DslBase children in traversal order.

        Returns:
            list[DslBase]:
                The list of child nodes.
        """
        return [item for item in self._items]

    def is_leaf(self) -> bool:
        """
        Return False; container nodes are never leaves.

        Returns:
            bool:
                Always False.
        """
        return False

    def update_child(self, index: int, new_child: DslBase) -> None:
        """
        Replace the child at the given index with a new value.

        Args:
            index (int):
                Index of the child to replace.
            new_child (DslBase):
                The new node to insert.
        """
        self._items[index] = new_child

    def insert_child(self, index: int, new_child: DslBase) -> None:
        """
        Insert a new child node at the specified index.

        Args:
            index (int):
                Index at which to insert the new child.
            new_child (DslBase):
                The new node to insert.
        """
        self._items.insert(index, new_child)

    def remove_child(self, index: int) -> None:
        """
        Remove the child at the specified index.

        Args:
            index (int):
                Index of the child to remove.
        """
        self._items.pop(index)

    def represent_content_as_text(self) -> str | None:
        """
        Represent the content of this DSL node as a string, if possible.

        This method is used to pass resolved or unresolved values to an LLM in text form,
        for example, when querying external knowledge, resolving slots, or sequencing intents.

        Not all DSL nodes can be meaningfully represented as text. If this node or any of its
        children cannot, this method should return None.

        Returns:
            str | None:
                A list that is a string representation of this node's content, or None if not
                all items are representable. The list is formatted as `[...]`.
        """
        repr_elements = []

        for v in self.get_children():
            r = v.represent_content_as_text()
            if r is None:
                return None
            repr_elements.append(r)

        return f"[{','.join(repr_elements)}]"

    def __repr__(self) -> str:
        n = len(self.get_items())
        item_str = "item" if n == 1 else "items"
        return f"{self.__class__.__name__}({n} {item_str})"
