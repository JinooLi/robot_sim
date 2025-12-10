import numpy as np
from typing import Callable
from interface import Controller, State, ControlType, CLBFGenerator


class MyCLBFGenerator(CLBFGenerator):
    def __init__(
        self,
        unsafe_region_center: np.ndarray,
        unsafe_region_radius: float,
        unsafe_region_margin: float,
        barrier_gain: float,
        Lyapunov_center: np.ndarray,
    ):
        self.unsafe_region_center = unsafe_region_center
        self.unsafe_region_radius = unsafe_region_radius
        self.unsafe_region_margin = unsafe_region_margin
        self.barrier_gain = barrier_gain
        self.Lyapunov_center = Lyapunov_center

    def get_lambda_W_and_gradW(
        self,
    ) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
        W_func = lambda x: self._W(x, self.barrier_gain)
        dW_dx_func = lambda x: self._dW_dx(x, self.barrier_gain)
        return W_func, dW_dx_func

    def _sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def _dsigmoid(self, s):
        sig = self._sigmoid(s)
        return sig * (1 - sig)

    def _Circ(self, x: np.ndarray):
        return -(
            (x - self.unsafe_region_center) @ (x - self.unsafe_region_center)
            - (self.unsafe_region_radius + self.unsafe_region_margin) ** 2
        )

    def _dCirc_dx(self, x: np.ndarray):
        return -2 * (x - self.unsafe_region_center - self.unsafe_region_margin)

    def _barrier_function(self, x: np.ndarray, theta: float = 10.0):
        return self._sigmoid(theta * self._Circ(x))

    def _dB_dx(self, x: np.ndarray, theta: float = 10.0):
        return self._dsigmoid(theta * self._Circ(x)) * self._dCirc_dx(x) * theta

    def _V(self, x: np.ndarray):
        return 0.5 * (x - self.Lyapunov_center) @ (x - self.Lyapunov_center)

    def _dV_dx(self, x: np.ndarray):
        return x - self.Lyapunov_center

    def _W(self, x: np.ndarray, theta: float = 10.0):
        return self._V(x) + self._barrier_function(x, theta)

    def _dW_dx(self, x: np.ndarray, theta: float = 10.0):
        return self._dV_dx(x) + self._dB_dx(x, theta)


class MyController(Controller):
    def __init__(
        self,
        clbf_generator: MyCLBFGenerator,
    ):
        super().__init__()
        self.clbf_generator = clbf_generator
        self.W_func, self.dW_dx_func = clbf_generator.get_lambda_W_and_gradW()
        self.pre_pos = np.zeros(7)
        self.ee_target_pos = clbf_generator.Lyapunov_center
        self.pre_torques = np.zeros(7)
        self.set_control_type(ControlType.VELOCITY)

    def control(self, state: State, t) -> np.ndarray:
        """제어입력을 만든다.

        Args:
            state: 현재 state
            t: 현재 시뮬레이션 시간

        Returns:
            np.ndarray: 제어 입력
        """

        velo = self.velocity_control(state, t)

        return velo

    def random_input_generator(self):
        dp = np.random.uniform(-0.05, 0.05, size=self.robot_info.ctrl_joint_number)
        pos = self.pre_pos + dp
        if self.control_type == ControlType.POSITION:
            pos = np.clip(
                pos,
                self.robot_info.joint_angle_min[: self.robot_info.ctrl_joint_number],
                self.robot_info.joint_angle_max[: self.robot_info.ctrl_joint_number],
            )
        elif self.control_type == ControlType.VELOCITY:
            vel = dp * self.robot_info.control_frequency
            vel = np.clip(
                vel,
                -np.array(
                    self.robot_info.velocity_limits[: self.robot_info.ctrl_joint_number]
                ),
                np.array(
                    self.robot_info.velocity_limits[: self.robot_info.ctrl_joint_number]
                ),
            )
            pos = vel
        elif self.control_type == ControlType.TORQUE:
            pos = np.random.uniform(
                -np.array(
                    self.robot_info.torque_limits[: self.robot_info.ctrl_joint_number]
                ),
                np.array(
                    self.robot_info.torque_limits[: self.robot_info.ctrl_joint_number]
                ),
            )
        self.pre_pos = pos.copy()
        return pos

    def velocity_control(self, state: State, t) -> np.ndarray:
        cjn = self.robot_info.ctrl_joint_number

        J = self.J_linear(state.positions[:cjn], 11)  # end-effector의 Jacobian

        k_ee = 1

        # J_pinv = J.T @ np.linalg.inv(J @ J.T)
        J_pinv = np.linalg.pinv(J)
        end_effector_control = J_pinv @ (-self.dW_dx_func(state.ee_position)) * k_ee

        N = np.eye(cjn) - J_pinv @ J

        k_N = 0.73

        p_null = np.zeros(cjn)
        for i in range(cjn):
            J_link = self.J_linear(state.positions[:cjn], i)
            pos = self.get_pos_of_joint(i)
            J_pinv_link = J_link.T @ np.linalg.inv(
                J_link @ J_link.T + 1e-6 * np.eye(3)
            )  # np.linalg.pinv(J_link) <- 이거 쓰면 기존 행렬이 거의 singular일 때 터진다.
            p_null += J_pinv_link @ self.dW_dx_func(pos)

        null_space_control = -N @ p_null * k_N  # (k_N * state.velocities[:cjn])

        dq = end_effector_control + null_space_control

        dq = np.clip(
            dq,
            -0.2 * self.robot_info.velocity_limits[:cjn],
            0.2 * self.robot_info.velocity_limits[:cjn],
        )

        return dq


if __name__ == "__main__":
    from simulation import RobotSim

    clbf_Gen = MyCLBFGenerator(
        unsafe_region_center=np.array([0.0, 0.0, 0.6]),
        unsafe_region_radius=0.3,
        unsafe_region_margin=0.05,
        barrier_gain=100.0,
        Lyapunov_center=np.array([0.5, 0.5, 0.5]),
    )

    controller = MyController(clbf_generator=clbf_Gen)

    sim = RobotSim(
        controller=controller,
        gravity=-9.81,
        time_frequency=1000.0,
        control_frequency=100.0,
        simulation_duration=10.0,
    )

    print(sim.get_robot_info())
    sim.simulate()
    sim.save_simulation_data(name="log_traj")
    print("시뮬레이션 데이터 저장 완료.")
    sim.visualize(file_name="log_traj", fps=50)
    print("시뮬레이션 재생 완료.")
