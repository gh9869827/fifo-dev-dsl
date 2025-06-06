from __future__ import annotations
from fifo_dev_dsl.dia.dsl.elements.base import DslBase, make_dsl_container


class ListElement(make_dsl_container(DslBase)):
    """
    A container node that holds a sequence of heterogeneous DSL elements.

    This class is used when multiple DSL nodes need to be grouped and treated
    as a single subtree. It is commonly used in constructs like `AbortWithNewDsl`
    to replace the current resolution path with an ordered set of new DSL nodes.

    Example:
        Used in an abort redirect:
            AbortWithNewDsl(
                ListElement([
                    Intent(name="action_1", slots=[]),
                    Intent(name="action_2", slots=[]),
                ])
            )
    """
