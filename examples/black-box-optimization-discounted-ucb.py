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

import math

class DiscountedUCBState:
    def __init__(self, operator_ids, gamma=0.95, c=1.0):
        self.gamma = gamma
        self.c = c

        self.sum_rewards = {op: 0.0 for op in operator_ids}
        self.sum_weights = {op: 0.0 for op in operator_ids}
        self.total_weight = 0.0

    def score(self, op_id):
        w = self.sum_weights[op_id]
        if w == 0.0:
            return float("inf")

        mean = self.sum_rewards[op_id] / w
        bonus = self.c * math.sqrt(
            math.log(max(1.0, self.total_weight)) / w
        )
        return mean + bonus

    def update(self, op_id, reward):
        # discount everything
        for k in self.sum_rewards:
            self.sum_rewards[k] *= self.gamma
            self.sum_weights[k] *= self.gamma

        self.total_weight *= self.gamma

        # add new reward
        self.sum_rewards[op_id] += reward
        self.sum_weights[op_id] += 1.0
        self.total_weight += 1.0

    def snapshot(self):
        rows = []
        for op in self.sum_rewards:
            w = self.sum_weights[op]
            mean = self.sum_rewards[op] / w if w > 0 else 0.0
            score = self.score(op)
            rows.append((op, score, mean, w))

        rows.sort(key=lambda x: x[1], reverse=True)

        print("\n[UCB] operator scores:")
        for op, s, m, w in rows:
            print(f"  {op:12s} score={s:6.3f} mean={m:6.3f} n={w:5.1f}")

if __name__ == "__main__":
    # Execution code starts here
    api_key = os.getenv("GOOGLE_API_KEY")
    ai_model = "gemini-2.5-flash"
    experiment_name = "bbob-discounted-ucb-1-1"
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

    OPERATOR_DEFS = [
        ("correct",     "Correct any mistakes in the selected algorithm.", 1),
        ("simplify",    "Refine and simplify the selected algorithm.", 1),
        ("basic",       "Refine the strategy of the selected algorithm.", 1),
        ("structure",   "Modify the selected algorithm by introducing a meaningful structural change that alters its overall search behavior.", 1),
        ("new",         "Generate a new algorithm that is different from the algorithms you have tried before.", 1),
        ("refactor",    "Generate a new algorithm using the core ideas from the selected algorithm.", 1),
    ]

    operator_ids = [op_id for op_id, _, _ in OPERATOR_DEFS]

    ucb_state = DiscountedUCBState(
        operator_ids=operator_ids,
        gamma=0.95, # Higher: less discounting of past rewards, .95: half life of ~13 iterations
        c=0.8 # Higher: more exploration
    )

    _last_print_gen = {"printed": False}

    def ucb_weight(op_id: str, solution: Solution) -> float:
        if not _last_print_gen["printed"]:
            ucb_state.snapshot()
            _last_print_gen["printed"] = True

        return ucb_state.score(op_id)

    def ucb_update(op_id: str, delta_fitness: float):
        reward = max(0.0, delta_fitness)
        ucb_state.update(op_id, reward)
        _last_print_gen["printed"] = False

    operators = [
        Operator(
            id=op_id,
            task_message=msg,
            parent_count=pc,
            weight_source=ucb_weight,
            update_callback=ucb_update,
        )
        for op_id, msg, pc in OPERATOR_DEFS
    ]

    for experiment_i in range(3):
        # A 1+1 strategy
        es = LLaMEA(
            evaluateBBOB,
            n_parents=1,
            n_offspring=1,
            llm=llm,
            task_prompt=task_prompt,
            experiment_name=experiment_name,
            operators=operators,
            elitism=True,
            HPO=False,
            budget=50
        )
        print(es.run())
