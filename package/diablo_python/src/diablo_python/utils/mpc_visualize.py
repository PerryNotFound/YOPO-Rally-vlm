import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


# reference: https://github.com/tomcattiger1230/CasADi_MPC_MHE_Python/blob/master/MPC/draw.py
class Draw_MPC_tracking:
    def __init__(
        self,
        robot_states: np.ndarray,
        init_state: np.ndarray,
        reference_states: np.ndarray,
        prediction_states: np.ndarray,
        rob_diam=0.3,
        export_fig=False,
        interval=100,
    ):
        self.init_state = init_state
        self.robot_states = robot_states
        self.rob_radius = rob_diam
        self.reference_states = reference_states
        self.prediction_states = prediction_states
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
        self.ax.set_aspect("equal")
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(
            self.fig,
            self.animation_loop,
            range(len(self.robot_states)),
            init_func=self.animation_init,
            interval=interval,
            repeat=False,
        )

        plt.grid("--")
        if export_fig:
            self.ani.save("tracking.gif", writer="imagemagick")
        plt.show()

    def animation_init(
        self,
    ):
        # draw the initial position of the robot
        self.init_robot_position = Circle(
            self.init_state[:2].tolist(), self.rob_radius, color="r", fill=False
        )
        self.ax.add_artist(self.init_robot_position)
        self.robot_body = Circle(
            self.init_state[:2].tolist(), self.rob_radius, color="r", fill=False
        )
        self.ax.add_artist(self.robot_body)
        self.robot_arr = mpatches.Arrow(
            self.init_state[0],
            self.init_state[1],
            self.rob_radius * np.cos(self.init_state[2]),
            self.rob_radius * np.sin(self.init_state[2]),
            width=0.2,
            color="r",
        )
        self.ax.add_patch(self.robot_arr)

        self.reference_traj = plt.plot([], [], "-b")[0]
        self.prediction_traj = plt.plot([], [], "--g")[0]

        return (
            self.init_robot_position,
            self.robot_body,
            self.robot_arr,
        )

    def animation_loop(self, indx):
        position = self.robot_states[indx][:2]
        orientation = self.robot_states[indx][2]
        self.robot_body.center = position
        self.robot_arr.remove()
        self.robot_arr = mpatches.Arrow(
            position[0],
            position[1],
            self.rob_radius * np.cos(orientation),
            self.rob_radius * np.sin(orientation),
            width=0.2,
            color="r",
        )
        self.ax.add_patch(self.robot_arr)

        self.reference_traj.set_data(
            self.reference_states[indx][:, 0],
            self.reference_states[indx][:, 1],
        )
        self.prediction_traj.set_data(
            self.prediction_states[indx][:, 0],
            self.prediction_states[indx][:, 1],
        )

        return (
            self.robot_arr,
            self.robot_body,
            self.reference_traj,
            self.prediction_traj,
        )
