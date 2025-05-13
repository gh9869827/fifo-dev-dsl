from __future__ import annotations
from dataclasses import dataclass

from common.llm.dia.dsl.elements.base import DslBase
from common.llm.dia.dsl.elements.element_list import ListElement


@dataclass
class AbortWithNewDsl(DslBase):
    """
    Aborts the current resolution path and replaces it with the new DSL elements.

    Like `Abort`, this node result in an abort condition, signaling that the
    current element is no longer valid. However, the new DSL elements are installed in place of the
    current aborted one.

    This is useful for graceful redirection, for example, when an item is unavailable and a fallback
    is suggested.

    Params:
        new_dsl (ListElement):
            New DSL elements to install as a replacement for the aborted one.
    """

    new_dsl: ListElement
