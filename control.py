from test_pybullet import RobotSim
import pybullet as p
import numpy as np


class MyRobotSim(RobotSim):
    def __init__(self, gravity, time_frequency, control_frequency, simulation_duration):
        super().__init__(
            gravity, time_frequency, control_frequency, simulation_duration
        )
        self.pre_pos = np.zeros(self.ctrl_joint_number)

    def controller(self, state, t) -> np.ndarray:
        """제어입력을 만든다.

        Args:
            state: 현재 state
            t: 현재 시뮬레이션 시간

        Returns:
            np.ndarray: 제어 입력
        """
        q = np.array([s[0] for s in state])
        dq = np.array([s[1] for s in state])
        joint_indices = list(range(self.ctrl_joint_number))
        robot_id = self.robotId

        dp = np.random.uniform(-0.05, 0.05, size=self.ctrl_joint_number)
        pos = self.pre_pos + dp
        pos = np.clip(
            pos,
            self.joint_angle_min[: self.ctrl_joint_number],
            self.joint_angle_max[: self.ctrl_joint_number],
        )
        self.pre_pos = pos.copy()

        pos = np.array([1.0, 0.1, -1.0, 0, 0, 0, 0])
        return pos


if __name__ == "__main__":
    sim = MyRobotSim(
        gravity=-9.81,
        time_frequency=1000.0,
        control_frequency=100.0,
        simulation_duration=10.0,
    )
    print(sim.get_robot_info())
    sim.set_control_type(control_type=p.TORQUE_CONTROL)
    sim.simulate()
    sim.save_simulation_data(name="log_traj")
    print("시뮬레이션 데이터 저장 완료.")
    sim.visualize(file_name="log_traj", fps=50)
    print("시뮬레이션 재생 완료.")
