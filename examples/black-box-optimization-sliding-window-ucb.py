from collections import deque
import math
import os
import pickle
from typing import Callable

from llamea.solution import Solution
from llamea.operator import Operator
import numpy as np
from ioh import get_problem, logger

from llamea import Gemini_LLM, LLaMEA
from llamea.utils import prepare_namespace, clean_local_namespace
from misc import OverBudgetException, aoc_logger, correct_aoc

class SlidingWindowUCBState:
    def __init__(self, operator_ids, window=50, c=1.0):
        self.window = window
        self.c = c
        self.history = deque()
        self.counts = {op: 0 for op in operator_ids}
        self.sums = {op: 0.0 for op in operator_ids}
        self.t = 0
        self.operator_ids = operator_ids

    def _ucb_score(self, op_id):
        n = self.counts[op_id]
        if n == 0:
            return float("inf")

        mean = self.sums[op_id] / n
        bonus = self.c * math.sqrt(
            math.log(max(1, min(self.t, self.window))) / n
        )
        return mean + bonus

    def _best_operator(self):
        # Ensure each operator tried once
        for op in self.operator_ids:
            if self.counts[op] == 0:
                return op

        return max(self.operator_ids, key=self._ucb_score)

    def get_(self, op_id):
        best = self._best_operator()
        return 1.0 if op_id == best else 0.0

    def update(self, op_id, reward):
        self.t += 1
        reward = max(0.0, reward)

        if len(self.history) == self.window:
            old_op, old_reward = self.history.popleft()
            self.counts[old_op] -= 1
            self.sums[old_op] -= old_reward

        self.history.append((op_id, reward))
        self.counts[op_id] += 1
        self.sums[op_id] += reward

if __name__ == "__main__":
    # Execution code starts here
    api_key = os.getenv("GOOGLE_API_KEY")
    ai_model = "gemini-2.5-flash"
    experiment_name = "pop1-5"
    llm = Gemini_LLM(api_key, ai_model)

    # We define the evaluation function that executes the generated algorithm (solution.code) on the BBOB test suite.
    # It should set the scores and feedback of the solution based on the performance metric, in this case we use mean AOCC.
    def evaluateBBOB(solution, explogger=None):
        auc_mean = 0
        auc_std = 0

        code = solution.code
        algorithm_name = solution.name
        feedback=""
        possible_issue = None
        local_ns = {}
        try:
            global_ns, possible_issue = prepare_namespace(code, allowed=["numpy"], logger=explogger)
            exec(code, global_ns, local_ns)
            local_ns = clean_local_namespace(local_ns, global_ns)

        except Exception as e:
            if possible_issue:
                feedback = f" {possible_issue}."
            solution.set_scores(float("-inf"), feedback, e)
            return solution

        aucs = []

        algorithm = None
        for dim in [5]:
            budget = 2000 * dim
            l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
            for fid in np.arange(1, 25):
                for iid in [1, 2, 3]:  # , 4, 5]
                    problem = get_problem(fid, iid, dim)
                    problem.attach_logger(l2)

                    for rep in range(3):
                        np.random.seed(rep)
                        try:
                            algorithm = local_ns[algorithm_name](
                                budget=budget, dim=dim
                            )
                            algorithm(problem)
                        except OverBudgetException:
                            pass

                        auc = correct_aoc(problem, l2, budget)
                        aucs.append(auc)
                        l2.reset(problem)
                        problem.reset()
        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)

        feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.4f} with standard deviation {auc_std:0.4f}."

        print(algorithm_name, algorithm, auc_mean, auc_std)
        solution.add_metadata("aucs", aucs)
        solution.set_scores(auc_mean, feedback)

        return solution



    # The task prompt describes the problem to be solved by the LLaMEA algorithm.
    task_prompt = """
    The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
    The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
    Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description with the main idea.
    """

    operator_ids = [
        "correct", "simplify", "simplify_2p",
        "basic", "structure", "new",
        "new_2p", "refactor"
    ]

    ucb_state = SlidingWindowUCBState(
        operator_ids=operator_ids,
        window=20,
        c=1.2
    )

    def ucb_weight(op_id: str, solution: Solution) -> float:
        return ucb_state.score(op_id)

    def ucb_update(op_id: str, delta_fitness: float):
        reward = max(0.0, delta_fitness) # positive only reward
        ucb_state.update(op_id, reward)

    operators = [
        Operator(
            id="correct",
            task_message="Correct any mistakes in the selected algorithm.",
            parent_count=1,
            weight_source=ucb_weight,
            update_callback=ucb_update,
        ),
        Operator(
            id="simplify",
            task_message="Refine and simplify the selected algorithm.",
            parent_count=1,
            weight_source=ucb_weight,
            update_callback=ucb_update,
        ),
        Operator(
            id="simplify_2p",
            task_message="Simplify and recombine the selected algorithms.",
            parent_count=2,
            weight_source=ucb_weight,
            update_callback=ucb_update,
        ),
        Operator(
            id="basic",
            task_message="Refine the strategy of the selected algorithm.",
            parent_count=1,
            weight_source=ucb_weight,
            update_callback=ucb_update,
        ),
        Operator(
            id="structure",
            task_message="Modify the selected algorithm by introducing a meaningful structural change that alters its overall search behavior.",
            parent_count=1,
            weight_source=ucb_weight,
            update_callback=ucb_update,
        ),
        Operator(
            id="new",
            task_message="Generate a new algorithm that is different from the algorithms you have tried before.",
            parent_count=1,
            weight_source=ucb_weight,
            update_callback=ucb_update,
        ),
        Operator(
            id="new_2p",
            task_message="Generate a new algorithm that is different from the selected two optimization methods.",
            parent_count=2,
            weight_source=ucb_weight,
            update_callback=ucb_update,
        ),
        Operator(
            id="refactor",
            task_message="Generate a new algorithm using the core ideas from the selected algorithm.",
            parent_count=1,
            weight_source=ucb_weight,
            update_callback=ucb_update,
        ),
    ]


    for experiment_i in range(3):
        # A 3+3 strategy
        es = LLaMEA(
            evaluateBBOB,
            n_parents=3,
            n_offspring=3,
            llm=llm,
            task_prompt=task_prompt,
            experiment_name=experiment_name,
            operators=operators,
            elitism=True,
            HPO=False,
            budget=50
        )
        print(es.run())
