import numpy as np

from interface import Controller, State, ControlType


class MyController(Controller):
    def __init__(self, target_ee_pos: np.ndarray = np.array([-0.3, -0.3, 0.8])):
        super().__init__()
        self.pre_pos = np.zeros(7)
        self.ee_target_pos = target_ee_pos
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

        velo = self.velocity_control(state, t, self.ee_target_pos)

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

    def velocity_control(self, state: State, t, target_pose: np.ndarray) -> np.ndarray:
        cjn = self.robot_info.ctrl_joint_number

        J = self.J_linear(state.positions[:cjn])

        J_pinv = J.T @ np.linalg.inv(J @ J.T)

        N = np.eye(cjn) - J_pinv @ J

        k_ee = 1

        end_effector_control = J_pinv @ (target_pose - state.ee_position) * k_ee

        k_N = 1

        null_space_control = N @ (-k_N * state.positions[:cjn])

        dq = end_effector_control + null_space_control

        dq = np.clip(
            dq,
            -0.5 * self.robot_info.velocity_limits[:cjn],
            0.5 * self.robot_info.velocity_limits[:cjn],
        )

        return dq


# 이거 하려면
# 1. inverse kinematics 컨트롤러 만들어야 함
# 1-1. dynamic model 필요. Matrix, Coriolis, Gravity 등등 아 그냥 가져오자.


if __name__ == "__main__":
    from simulation import RobotSim

    controller = MyController(target_ee_pos=np.array([-0.6, -0.6, 0.1]))

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
