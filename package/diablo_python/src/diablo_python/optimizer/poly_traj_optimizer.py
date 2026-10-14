from typing_extensions import override

import casadi as ca
import numpy as np

from diablo_python.geotiff import GeoTiffCostLut

from .optimizer_base import OptimizationError, OptimizerBase


class PolyTrajOptimizer(OptimizerBase):
    def setup_optim_params(
        self,
        v_max: float,
        t_end: float,
        look_ahead_distance: float,
        cost_lut: GeoTiffCostLut,
        sampling_number: int = 30,
    ):
        self._v_max = v_max
        self._sampling_number = sampling_number
        self._t_end = t_end
        self._look_ahead_distance = look_ahead_distance
        self._cost_lut = cost_lut

    def solve_optim(
        self,
        pos_start: np.ndarray,
        vel_start: np.ndarray,
        goal: np.ndarray,
        theta_limit: np.ndarray,
        pos_end_init_guess: np.ndarray,
        vel_end_init_guess: np.ndarray,
    ):
        assert pos_start.shape == (2,), "pos_start shape mismatch"
        assert vel_start.shape == (2,), "vel_start shape mismatch"
        assert goal.shape == (2,), "goal shape mismatch"
        assert theta_limit.shape == (2,), "theta_limit shape mismatch"
        assert pos_end_init_guess.shape == (2,), "pos_end_init_guess shape mismatch"
        assert vel_end_init_guess.shape == (2,), "vel_end_init_guess shape mismatch"

        try:
            opti_res: dict[str, ca.DM] = self._opti_func(
                p_start=pos_start,
                grad_start=vel_start * self._t_end,
                goal=goal,
                sector_theta_bounds=theta_limit,
                p_end_init_guess=pos_end_init_guess,
                grad_end_init_guess=vel_end_init_guess * self._t_end,
            )
            p_end: np.ndarray = opti_res["p_end"].toarray(simplify=True)
            v_end: np.ndarray = (
                opti_res["grad_end"].toarray(simplify=True) / self._t_end
            )
            cost: float = opti_res["cost"].toarray(simplify=True)
        except RuntimeError as e:
            print(f"Eror in solve_opti: {e}")
            p_end = np.zeros(2)
            v_end = np.zeros(2)
            cost = np.inf
            raise OptimizationError(f"Error in solve_optim: {e}") from e
        return p_end, v_end, cost

    # TODO: Make a seperate func to solve the candidate optimization with casadi's parallel optimization
    def solve_optim_candidate(
        self,
        pos_start: np.ndarray,
        vel_start: np.ndarray,
        goal: np.ndarray,
        candidate_num: int = 6,
        candidate_theta_center: float = 0,
        candidate_theta_range: float = np.pi / 2,
        candidate_theta_optim_limit: float = np.pi / 4,
    ):
        assert pos_start.shape == (2,), "pos_start shape mismatch"
        assert vel_start.shape == (2,), "vel_start shape mismatch"
        assert goal.shape == (2,), "goal shape mismatch"

        if not hasattr(self, "_p_end_init_guess_candidate"):
            self._p_end_init_guess_candidate = np.zeros((candidate_num, 2))

        v_end_init_guess_candidate = np.zeros((candidate_num, 2))
        theta_optim_limit_candidate = np.zeros((candidate_num, 2))

        # The camera view cone diagram:
        #       left    right
        #         ----------
        #         \        /
        #          \      /
        #           \    /
        #            \  /
        #             \/
        # The camera image is from left to right, so the theta should be clockwise from the center
        theta = np.linspace(
            candidate_theta_center + candidate_theta_range / 2,
            candidate_theta_center - candidate_theta_range / 2,
            candidate_num + 1,
        )
        theta = (theta[1:] + theta[:-1]) / 2
        theta_vec = np.stack([np.cos(theta), np.sin(theta)], axis=1)

        self._p_end_init_guess_candidate = (
            pos_start + self._look_ahead_distance * theta_vec
        )
        v_end_init_guess_candidate = self._v_max / 2 * theta_vec
        # To ensure the sector constraint view cone exist
        assert (
            candidate_theta_optim_limit < np.pi
        ), "The theta limit should be less than pi, or the sector constraint view cone will not exist"
        theta_optim_limit_candidate = np.stack(
            [
                theta - candidate_theta_optim_limit / 2,
                theta + candidate_theta_optim_limit / 2,
            ],
            axis=1,
        )

        pos_start = np.tile(pos_start, (candidate_num, 1))
        vel_start = np.tile(vel_start, (candidate_num, 1))
        goal = np.tile(goal, (candidate_num, 1))

        p_end_candidate = np.zeros((candidate_num, 2))
        v_end_candidate = np.zeros((candidate_num, 2))
        cost_candidate = np.zeros((candidate_num, 1))

        for i in range(candidate_num):
            p_end, v_end, cost = self.solve_optim(
                pos_start[i],
                vel_start[i],
                goal[i],
                theta_optim_limit_candidate[i],
                self._p_end_init_guess_candidate[i],
                v_end_init_guess_candidate[i],
            )

            p_end_candidate[i] = p_end
            v_end_candidate[i] = v_end
            cost_candidate[i] = cost

        return p_end_candidate, v_end_candidate, cost_candidate

    @override
    def bake_optim_func(self, method="ipopt", expand=False):
        super().bake_optim_func(method, expand)

    @override
    def _setup_optim_variables(self):
        super()._setup_optim_variables()
        self._goal = self._opti.parameter(2)
        self._p_start = self._opti.parameter(2)
        self._grad_start = self._opti.parameter(2)
        self._v_start = self._grad_start / self._t_end

        self._sector_theta_bounds = self._opti.parameter(2)

        self._p_end = self._opti.variable(2)
        self._grad_end = self._opti.variable(2)
        self._v_end = self._grad_end / self._t_end

        self._G_H = ca.hcat(
            [self._p_start, self._grad_start, self._grad_end, self._p_end]
        ).T
        # fmt: off
        self._M_H = ca.MX(ca.DM([[1, 0, -3, 2], 
                           [0, 1, -2, 1], 
                           [0, 0, -1, 1], 
                           [0, 0, 3, -2]]))
        # fmt: on
        # NOTICE: t is the normalized time in [0, 1]
        self._hermite_curve = (
            lambda tau: self._G_H.T @ self._M_H @ ca.vcat([1, tau, tau**2, tau**3])
        )
        self._hermite_curve_grad = (
            lambda tau: self._G_H.T @ self._M_H @ ca.vcat([0, 1, 2 * tau, 3 * tau**2])
        )

    @override
    def _setup_optim_constraints(self):
        # TODO: apply velocity constraints to each point in the trajectory
        # velocity constraints
        self._opti.subject_to(ca.sumsqr(self._v_end) <= self._v_max**2)

        # p_end location constraints
        # limit the end point to be within a certain sector
        delta_pos = self._p_end - self._p_start
        sector_theta_center = (
            self._sector_theta_bounds[0] + self._sector_theta_bounds[1]
        ) / 2
        half_sector_theta = (
            self._sector_theta_bounds[1] - self._sector_theta_bounds[0]
        ) / 2
        n = ca.vertcat(ca.cos(sector_theta_center), ca.sin(sector_theta_center))
        # distance constraint
        self._opti.subject_to(
            self._opti.bounded(0, ca.dot(delta_pos, n), self._look_ahead_distance * 2)
        )
        # angle constraint
        # Reference: https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_gaze_target_constraint.html

        # The half_sector_theta diagram:
        #         ---------
        #         \   |   /
        #          \  |  /
        #           \ | /
        #            \|/
        # The half_sector_theta should be less than pi/2, otherwise the view cone will not exist, and the constraint will be invalid
        self._opti.subject_to(
            ca.dot(delta_pos, n) ** 2
            >= ca.cos(half_sector_theta) ** 2 * ca.sumsqr(delta_pos)
        )

    @override
    def _setup_optim_cost(self):
        costmap_weight = 1
        control_weight = 1
        goal_weight = 1

        cost_lut_for_xy = self._cost_lut.lut_for_xy
        self._cost: ca.MX = 0
        dT = self._t_end / self._sampling_number
        for i in range(self._sampling_number + 1):
            tau = i / self._sampling_number
            p = self._hermite_curve(tau)
            v = self._hermite_curve_grad(tau) / self._t_end

            # costmap cost
            self._cost += costmap_weight * cost_lut_for_xy(p[0], p[1]) * dT

            # control cost
            self._cost += control_weight * ca.sumsqr(v) * dT
            # TODO: Calculate the curvature cost

        # goal cost
        self._cost += ca.sumsqr(self._p_end - self._goal) * goal_weight

        self._opti.minimize(self._cost)

    @override
    def _setup_optim_solver(self, method: str, expand: bool):
        super()._setup_optim_solver(method, expand)

        self._opti_func: ca.Function = self._opti.to_function(
            "opti_func",
            [
                self._p_start,
                self._grad_start,
                self._goal,
                self._sector_theta_bounds,
                self._p_end,
                self._grad_end,
            ],
            [
                self._p_end,
                self._grad_end,
                self._cost,
            ],
            [
                "p_start",
                "grad_start",
                "goal",
                "sector_theta_bounds",
                "p_end_init_guess",
                "grad_end_init_guess",
            ],
            ["p_end", "grad_end", "cost"],
        )

    @property
    def look_ahead_distance(self):
        return self._look_ahead_distance

    @property
    def sampling_number(self):
        return self._sampling_number

    @property
    def t_end(self):
        return self._t_end

    @property
    def p_end_init_guess_candidate(self):
        return self._p_end_init_guess_candidate

    @property
    def v_max(self):
        return self._v_max
