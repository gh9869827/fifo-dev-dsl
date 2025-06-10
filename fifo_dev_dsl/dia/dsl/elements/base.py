from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, Type, TypeVar, cast

from fifo_dev_common.typeutils.strict_cast import strict_cast
from fifo_dev_dsl.dia.resolution.outcome import ResolutionOutcome

if TYPE_CHECKING:  # pragma: no cover
    from fifo_dev_common.introspection.mini_docstring import MiniDocStringType
    from fifo_dev_dsl.dia.resolution.interaction import Interaction
    from fifo_dev_dsl.dia.resolution.context import ResolutionContext
    from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext


class DslBase:
    """
    Base class for all DSL elements.

    Each DSL element represents a node in the intent structure and can optionally
    participate in resolution during a given resolution wave (e.g., user interaction,
    runtime queries), or in evaluation (producing a final value).

    Subclasses override `resolve()` to participate in specific resolution phases,
    and `eval()` if they produce a concrete value during execution.
    """

    def to_dsl_representation(self) -> str:
        """
        Return a DSL-style string representation of this node.

        Used when rendering nodes into prompt text for LLMs, especially in
        resolution or clarification contexts. Subclasses may override to provide
        domain-specific formatting.

        Returns:
            str:
                A DSL-compatible string representation.
        """
        return repr(self)

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
            list[DslBase]:
                Ordered list of immediate child nodes.
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

    def _log_resolution(self,
                        label: str,
                        resolution_context: ResolutionContext):
        pad = "  " * len(resolution_context.call_stack)
        print(f"{pad}[{label:<8}] {self}")

    def pre_resolution(
        self,
        runtime_context: LLMRuntimeContext,
        resolution_context: ResolutionContext,
        interaction: Interaction | None,
    ) -> None:
        _ = runtime_context, interaction
        self._log_resolution(" → pre", resolution_context)

    def do_resolution(
        self,
        runtime_context: LLMRuntimeContext,
        resolution_context: ResolutionContext,
        interaction: Interaction | None,
    ) -> ResolutionOutcome:
        _ = runtime_context, interaction
        self._log_resolution("⚙️  do   ", resolution_context)
        return ResolutionOutcome()

    def post_resolution(
        self,
        runtime_context: LLMRuntimeContext,
        resolution_context: ResolutionContext,
        interaction: Interaction | None,
    ) -> None:
        _ = runtime_context, interaction
        self._log_resolution(" ← post", resolution_context)

    def on_reentry_resolution(
        self,
        runtime_context: LLMRuntimeContext,
        resolution_context: ResolutionContext,
        interaction: Interaction | None,
    ) -> None:
        _ = runtime_context, interaction
        self._log_resolution(" ↺ visit", resolution_context)

    def is_resolved(self) -> bool:
        """
        Determine whether this DSL element is fully resolved.

        Nodes remain unresolved while they contain interactive placeholders such
        as ``ASK``, ``QUERY_FILL``, ``QUERY_USER`` or ``QUERY_GATHER``. Subclasses
        may track additional state and override this method accordingly. Once no
        such placeholders remain, the element is ready for evaluation.

        Returns:
            bool:
                True if the element is resolved, False otherwise.
        """
        return True

    def eval(
        self,
        runtime_context: LLMRuntimeContext,
        value_type: MiniDocStringType | None = None,
    ) -> Any:
        """
        Evaluate this DSL element and return its runtime value.

        Subclasses implement the actual evaluation logic. If the element is not
        resolved, this method must raise a RuntimeError.

        Args:
            runtime_context (LLMRuntimeContext):
                Execution context providing tool access, query sources and runtime helpers.

            value_type (MiniDocStringType | None):
                Optional expected return type used to cast or interpret the result.

        Returns:
            Any:
                The value produced by evaluating this DSL node.

        Raises:
            RuntimeError: If the node is not resolved.
        """
        raise NotImplementedError("DslBase.eval must be implemented by subclasses")


T = TypeVar("T", bound=DslBase)

class DslContainerBase(DslBase, Generic[T], ABC):
    """
    Base class for nodes that group other DSL elements.

    This class is used to represent a collection of child DSL nodes (e.g., slots in an intent,
    values in a list) and provides utilities for traversal and evaluation.

    Used by DSL elements such as:
        - `Intent`, which contains `Slot` items.
        - `ListValue`, which contains `DSLValueBase` items.

    Attributes:
        _items (list[T]):
            A list of child DSL nodes stored in the order they were provided.
    """

    _items: list[T]

    def __init__(self, items: list[T]):
        """
        Initialize a container node with an ordered list of items.

        Args:
            items (list[T]):
                A list of container items to store internally.
        """
        self._items: list[T] = items

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DslContainerBase):
            return NotImplemented
        return self._items == cast(DslContainerBase[T], other)._items

    def to_dsl_representation(self) -> str:
        """
        Return the DSL-style representation of a generic container node.

        Returns:
            str:
                A string in DSL list syntax containing the representation of all items,
                e.g., '[x=1, y=foo()]'.
        """
        items = ", ".join([i.to_dsl_representation() for i in self.get_items()])
        return f"[{items}]"

    def is_resolved(self) -> bool:
        """
        Return ``True`` only if every child item is resolved.

        Container nodes propagate their resolved state from the elements they
        contain.  If any child reports ``False`` for :py:meth:`is_resolved`, the
        container itself is unresolved.

        Returns:
            bool:
                ``True`` when all items are resolved.
        """
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
        self._items[index] = strict_cast(self._expected_type(), new_child)

    def insert_child(self, index: int, new_child: DslBase) -> None:
        """
        Insert a new child node at the specified index.

        Args:
            index (int):
                Index at which to insert the new child.
            new_child (DslBase):
                The new node to insert.
        """
        self._items.insert(index, strict_cast(self._expected_type(), new_child))

    def remove_child(self, index: int) -> None:
        """
        Remove the child at the specified index.

        Args:
            index (int):
                Index of the child to remove.
        """
        self._items.pop(index)

    def __repr__(self) -> str:
        n = len(self.get_items())
        item_str = "item" if n == 1 else "items"
        return f"{self.__class__.__name__}({n} {item_str})"

    def _expected_type(self) -> Type[T]:
        """
        Return the expected type of child elements stored in this container.

        This method is used by `DslContainerBase` to enforce strict runtime type
        validation when inserting, updating, or replacing child nodes. It must
        return the exact subclass of `DslBase` that this container is designed
        to hold (i.e., the same type parameter `T` used in `DslContainerBase[T]`).

        Returns:
            Type[T]: 
                The class object representing the allowed type of child nodes.
        """
        raise NotImplementedError("Subclasses must define _expected_type()")


def make_dsl_container(expected_type: Type[T]) -> type[DslContainerBase[T]]:
    """
    Create a typed subclass of `DslContainerBase` for a specific DSL node type.

    This factory returns a new subclass of `DslContainerBase[T]` that is preconfigured
    to accept only child nodes of type `expected_type`. It automatically implements the
    `_expected_type()` method required by the base class, ensuring strict runtime
    validation via `strict_cast`.

    This approach eliminates type duplication in subclasses and guarantees both
    runtime safety and static typing support without requiring decorators or manual overrides.

    Args:
        expected_type (Type[T]):
            The DSL node type that this container should enforce for its children.

    Returns:
        type[DslContainerBase[T]]:
            A new subclass of `DslContainerBase` bound to `T` and ready to use
            as a base class for custom containers.

    Example:
        >>> class ListValue(make_dsl_container(DSLValueBase)):
        ...     pass

    """
    # `Any` avoids a Pylance warning about reusing the outer TypeVar `_CT`
    class _GeneratedContainer(DslContainerBase[Any]):
        def _expected_type(self) -> Type[T]:
            return expected_type
    _GeneratedContainer.__name__ = f"{expected_type.__name__}ListBase"
    return _GeneratedContainer
