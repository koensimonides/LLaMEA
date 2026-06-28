from collections.abc import Callable
from llamea.solution import Solution

class Operator:
    """
    Represents an operator that can be applied to a solution, such as mutation or crossover.
    Each operator has an associated weight that can be static or dynamically computed based on the solution's state.
    The operator can also have a method to update its weight based on the fitness improvement of the evolved solution compared to its parents.
    """

    def __init__(
        self,
        id,
        task_message,
        parent_count=1,
        weight_source: float | Callable[[str, Solution], float] = 1.0,
        update_callback: Callable[[str, float], None] | None = None,
    ):
        """
        Initializes an individual with optional attributes.

        Args:
            id (str): Unique identifier for this operator.
            task_message (str): A message describing the task to be performed by this operator, used in prompt construction.
            parent_count (int): The number of parent solutions this operator requires (e.g., 1 for mutation, 2 for crossover).
            weight_source (float or Callable): A static weight value or a function that computes the weight based on the solution's state.
            update_callback (Callable or None): An optional function that updates the operator's weight based on the fitness improvement.
        """

        self.id = id
        self.task_message = task_message
        self.parent_count = parent_count
        self.weight_source = weight_source
        self.update_callback = update_callback

    def get_weight(self, solution: Solution) -> float:
        """
        Retrieves the weight for this operator, which can be a static value or computed based on the solution.

        Args:
            solution (Solution): The solution for which to compute the weight.
        Returns:
            float: The computed weight for this operator.
        """

        if callable(self.weight_source):
            return self.weight_source(self.id, solution)
        return self.weight_source
    
    def update_weight(self, fitness_delta: float):
        """
        Updates the weight of this operator based on the obtained fitness delta.

        Args:
            fitness_delta (float): The difference in fitness between the evolved and parent solutions.
        """
        if self.update_callback is not None:
            self.update_callback(self.id, fitness_delta)
        