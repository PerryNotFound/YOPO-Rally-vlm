from abc import abstractmethod
from typing import Any, Literal

import casadi as ca


class OptimizationError(RuntimeError):
    pass


class OptimizerBase:
    def bake_optim_func(self, method: Literal["sqpmethod", "ipopt"], expand: bool):
        self._setup_optim_variables()
        self._setup_optim_constraints()
        self._setup_optim_cost()
        self._setup_optim_solver(method, expand)

    def _setup_optim_variables(self):
        self._opti = ca.Opti()

    def _setup_optim_solver(self, method: str, expand: bool):
        opti_settings: dict[str, Any] = {
            "print_time": 0,
            "error_on_fail": True,
            "expand": expand,
        }
        match method:
            case "sqpmethod":
                opti_settings["print_header"] = False
                opti_settings["print_iteration"] = False
                opti_settings["print_status"] = False
                opti_settings["qpsol_options.printLevel"] = "none"
                opti_settings["qpsol"] = "qpoases"
            case "ipopt":
                opti_settings["ipopt.print_level"] = 0
            case _:
                raise ValueError(f"Unknown method: {method}")
        self._opti.solver(method, opti_settings)

    @abstractmethod
    def _setup_optim_constraints(self): ...

    @abstractmethod
    def _setup_optim_cost(self): ...
