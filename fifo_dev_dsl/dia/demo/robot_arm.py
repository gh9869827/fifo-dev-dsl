from fifo_dev_common.introspection.tool_decorator import tool_handler, tool_query_source
from fifo_dev_dsl.dia.resolution.resolver import Resolver
from fifo_dev_dsl.dia.runtime.context import LLMRuntimeContext
from fifo_dev_dsl.dia.runtime.evaluation_outcome import EvaluationStatus
from fifo_dev_dsl.dia.runtime.evaluator import Evaluator
from fifo_dev_dsl.dia.runtime.exceptions import ApiErrorAbortAndResolve

class RobotArm:

    def __init__(self):
        # Initialize inventory as a dict of screws indexed by length
        self.inventory = {
            8: {"count": 4},
            10: {"count": 20},
            11: {"count": 3},
            12: {"count": 14},
            16: {"count": 2}
        }

    @tool_handler("shutdown")
    def shutdown(self):
        """
        Shutdown the system.
        """
        print("[Shutdown]")

    @tool_handler("retrieve_screw")
    def retrieve_screw(self, count: int, length: int):
        """
        Retrieve screws of a specific length from the inventory.

        Args:
            count (int):
                Number of screws to retrieve.

            length (int):
                Length of the screws to retrieve.
        """
        print(f"[retrieve_screw] Request: count={count}, length={length}")

        if length not in self.inventory:
            raise ApiErrorAbortAndResolve(f"No screws of length {length} found in inventory.")

        screw = self.inventory[length]
        if screw["count"] >= count:
            screw["count"] -= count
            print(f"[retrieve_screw] Retrieved {count} screws of length {length}. "
                  f"Remaining: {screw['count']}")
            if screw["count"] == 0:
                del self.inventory[length]
        else:
            raise ApiErrorAbortAndResolve(
                f"Not enough screws of length {length}. Requested {count}, "
                f"available {screw['count']}."
            )

    @tool_handler("organize")
    def organize(self):
        """
        Organize the screws that are on the table.
        """
        print("[organize] Organizing screws on the table.")

    @tool_handler("initialize_components")
    def initialize_components(self, components: list[str]):
        """
        Initialize one or more of the following components: table, gripper, camera, shelf.

        Args:
            components (list[str]):
                One or more of the components to initialize.
        """
        print(f"[initialize_components] Initializing components: {components}")

    @tool_query_source("inventory")
    def get_inventory(self) -> str:
        """
        Returns the inventory of screws, including their length and count.

        Useful to answer user queries about screw specifications or quantities,
        such as to resolve 'all' when the user asks for 'give me all screws'.

        Returns:
            str:
                The serialized inventory.
        """
        lines = ["inventory:"]
        for length, screw in self.inventory.items():
            lines.append(f"  - length: {length}")
            lines.append(f"    count: {screw['count']}")
        return "\n".join(lines)

if __name__ == "__main__":

    robot = RobotArm()

    runtime_context = LLMRuntimeContext(
        container_name="phi",
        intent_sequencer_adapter="dia-intent-sequencer-robot-arm-adapter",
        tools=[
            robot.retrieve_screw,
            robot.initialize_components,
            robot.organize,
            robot.shutdown
        ],
        query_sources=[
            robot.get_inventory
        ]
    )

    print("> ready for command")

    USER_PROMPT = "give me one of the longest screws you have."

    resolver = Resolver(runtime_context, prompt=USER_PROMPT)

    while True:
        # Step 1: resolve DSL (slot filling or follow-up questions)
        resolver.fully_resolve_in_text_mode()
        dsl_elements = resolver.dsl_elements
        dsl_elements.pretty_print_dsl()

        # Step 2: evaluate resolved DSL (may succeed or fail)
        evaluator = Evaluator(runtime_context, dsl_elements)
        outcome = evaluator.evaluate()

        # Step 3: recovery loop if needed
        if outcome.status != EvaluationStatus.ABORTED_RECOVERABLE:
            break  # success or unrecoverable failure

        resolver = Resolver(runtime_context, dsl=dsl_elements)
