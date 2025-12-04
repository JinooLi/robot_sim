import numpy as np

from interface import Controller, State, ControlType


class MyController(Controller):
    def __init__(self, control_type: ControlType):
        super().__init__(control_type)
        self.pre_pos = np.zeros(7)

    def control(self, state: State, t) -> np.ndarray:
        """제어입력을 만든다.

        Args:
            state: 현재 state
            t: 현재 시뮬레이션 시간

        Returns:
            np.ndarray: 제어 입력
        """

        return self.random_input_generator() + self.g(
            state.positions[: self.robot_info.ctrl_joint_number]
        )

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


# 이거 하려면
# 1. inverse kinematics 컨트롤러 만들어야 함
# 1-1. dynamic model 필요. Matrix, Coriolis, Gravity 등등 아 그냥 가져오자.


if __name__ == "__main__":
    from simulation import RobotSim

    controller = MyController(ControlType.TORQUE)

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
