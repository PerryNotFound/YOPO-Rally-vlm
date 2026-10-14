from typing import Literal

import casadi as ca
import numpy as np
from typing_extensions import override

from .optimizer_base import OptimizationError, OptimizerBase


class MPCLocalPlanner(OptimizerBase):
    def setup_opti_params(
        self,
        v_max: float,
        omega_max: float,
        acc_max: float,
        alpha_max: float,
        sampling_time: float = 0.2,
        horizon: int = 10,
        Q=np.diag([1, 1, 0.5]),
        R=np.diag([0.1, 0.05]),
        runge_kutta=False,
    ):
        self._Ts = sampling_time
        self._N = horizon
        self._v_max = v_max
        self._omega_max = omega_max
        self._acc_max = acc_max
        self._alpha_max = alpha_max
        self._Q = Q
        self._R = R
        self._runge_kutta = runge_kutta

    def solve_optim(
        self,
        x_ref: np.ndarray,
        x_cur: np.ndarray,
        x_init_guess: np.ndarray,
        u_init_guess: np.ndarray,
    ):
        assert x_ref.shape == self._opti_states_ref.shape, "x_ref shape mismatch"
        assert x_init_guess.shape == self._opti_states.shape, (
            "x_init_guess shape mismatch"
        )
        assert u_init_guess.shape == self._opti_controls.shape, (
            "u_init_guess shape mismatch"
        )

        try:
            x_ref_relative = x_ref.copy()
            x_ref_relative[:, :2] -= x_cur[:2]
            x_cur_relative = np.zeros_like(x_cur)
            x_cur_relative[2] = x_cur[2]
            x_init_guess_relative = x_init_guess.copy()
            x_init_guess_relative[:, :2] -= x_cur[:2]
            opti_res: dict[str, ca.DM] = self._opti_func(
                x_ref=x_ref_relative,
                x_cur=x_cur_relative,
                u_init_guess=u_init_guess,
                x_init_guess=x_init_guess_relative,
            )
            u: np.ndarray = opti_res["u"].toarray(simplify=True)
            x: np.ndarray = opti_res["x"].toarray(simplify=True)
            x[:, :2] += x_cur[:2]
        except RuntimeError as e:
            print(f"Error in solve_opti: {e}")
            u = np.zeros(self._opti_controls.shape)
            x = np.zeros(self._opti_states.shape)
            raise OptimizationError(f"Error in solve_optim: {e}") from e
        return u, x

    def solve_optim_warm_start(self, x_ref: np.ndarray, x_cur: np.ndarray):
        if not (hasattr(self, "_x_init_guess") and hasattr(self, "_u_init_guess")):
            self._x_init_guess = np.zeros(self._opti_states.shape)
            self._u_init_guess = np.zeros(self._opti_controls.shape)
        else:
            self._x_init_guess = np.vstack(
                (self._x_init_guess[1:], self._x_init_guess[-1])
            )
            self._u_init_guess = np.vstack(
                (self._u_init_guess[1:], self._u_init_guess[-1])
            )
        u, x = self.solve_optim(
            x_ref=x_ref,
            x_cur=x_cur,
            x_init_guess=self._x_init_guess,
            u_init_guess=self._u_init_guess,
        )
        self._x_init_guess, self._u_init_guess = x, u
        return u, x

    @override
    def bake_optim_func(
        self, method: Literal["sqpmethod", "ipopt"] = "sqpmethod", expand=True
    ):
        super().bake_optim_func(method, expand)

    @override
    def _setup_optim_variables(self):
        super()._setup_optim_variables()
        self._opti_controls: ca.MX = self._opti.variable(self._N, 2)
        self._v: ca.MX = self._opti_controls[:, 0]
        self._omega: ca.MX = self._opti_controls[:, 1]

        self._opti_states: ca.MX = self._opti.variable(self._N + 1, 3)
        self._x: ca.MX = self._opti_states[:, 0]
        self._y: ca.MX = self._opti_states[:, 1]
        self._theta: ca.MX = self._opti_states[:, 2]

        self._opti_states_ref: ca.MX = self._opti.parameter(self._N + 1, 3)
        # use state_cur to pass the current state to the optimizer, instead of storing it as the first element of state_ref
        self._opti_state_cur: ca.MX = self._opti.parameter(1, 3)

    @override
    def _setup_optim_constraints(self):
        # init constraints
        self._opti.subject_to(self._opti_states[0, :] == self._opti_state_cur)

        # dynamic constraints
        for i in range(self._N):
            x = self._opti_states[i, :]
            u = self._opti_controls[i, :]
            h = self._Ts
            if not self._runge_kutta:
                # forward euler
                x_next = x + self.dynamic_model(x, u).T * h
            else:
                # runge-kutta 4th order
                k1 = self.dynamic_model(x, u).T
                k2 = self.dynamic_model(x + k1 / 2, u + h / 2).T
                k3 = self.dynamic_model(x + k2 / 2, u + h / 2).T
                k4 = self.dynamic_model(x + k3, u + h).T
                x_next = x + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6
            self._opti.subject_to(self._opti_states[i + 1, :] == x_next)

        # control constraints
        # velocity
        self._opti.subject_to(self._opti.bounded(-self._v_max, self._v, self._v_max))
        # omega
        self._opti.subject_to(
            self._opti.bounded(-self._omega_max, self._omega, self._omega_max)
        )
        # acceleration
        self._opti.subject_to(
            self._opti.bounded(
                -self._acc_max * self._Ts,
                self._v[:-1] - self._v[1:],
                self._acc_max * self._Ts,
            )
        )
        # angular acceleration
        self._opti.subject_to(
            self._opti.bounded(
                -self._alpha_max * self._Ts,
                self._omega[:-1] - self._omega[1:],
                self._alpha_max * self._Ts,
            )
        )

    @override
    def _setup_optim_cost(self):
        obj = 0
        for i in range(self._N + 1):
            state_error: ca.MX = self._opti_states[i, :] - self._opti_states_ref[i, :]
            position_cost = ca.mtimes(
                [state_error[:, :2], self._Q[:2, :2], state_error[:, :2].T]
            )

            # Reference: https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_orientation_cost.html
            theta_cost = self._Q[2, 2] * (1 - ca.cos(state_error[:, 2]))
            obj += position_cost + theta_cost
        for i in range(self._N):
            control_cost = self._opti_controls[i, :]
            obj += ca.mtimes([control_cost, self._R, control_cost.T])
        self._opti.minimize(obj)

    @override
    def _setup_optim_solver(self, method: str, expand: bool):
        super()._setup_optim_solver(method, expand)

        self._opti_func = self._opti.to_function(
            "opti_func",
            [
                self._opti_states_ref,
                self._opti_state_cur,
                self._opti_controls,
                self._opti_states,
            ],
            [self._opti_controls, self._opti_states],
            ["x_ref", "x_cur", "u_init_guess", "x_init_guess"],
            ["u", "x"],
        )

    @staticmethod
    def dynamic_model(
        x: np.ndarray | ca.MX, u: np.ndarray | ca.MX
    ) -> np.ndarray | ca.MX:
        v = u[0]
        omega = u[1]
        x_dot = v * np.cos(x[2])
        y_dot = v * np.sin(x[2])
        theta_dot = omega
        if isinstance(x, np.ndarray):
            return np.array([x_dot, y_dot, theta_dot])
        else:
            return ca.vertcat(x_dot, y_dot, theta_dot)

    @staticmethod
    def shift_movement(
        x0: np.ndarray, u: np.ndarray, T_sim: float, noise_mean=0.0, noise_std=0.1
    ) -> np.ndarray:
        x_next = x0 + MPCLocalPlanner.dynamic_model(x0, u) * T_sim
        noise = np.random.normal(noise_mean, noise_std, x0.shape) * T_sim
        x_next += noise
        # limit the angle between -pi and pi
        x_next[2] = (x_next[2] + np.pi) % (2 * np.pi) - np.pi
        return x_next

    @property
    def N(self):
        return self._N

    @property
    def Ts(self):
        return self._Ts
