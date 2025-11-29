from test_pybullet import RobotSim
import pybullet as p
import numpy as np


class MyRobotSim(RobotSim):
    def controller(self, state) -> np.ndarray:
        """제어입력을 만든다.

        Args:
            state: 현재 state

        Returns:
            np.ndarray: 제어 입력
        """
        q = np.array([s[0] for s in state])
        dq = np.array([s[1] for s in state])
        joint_indices = list(range(self.ctrl_joint_number))
        robot_id = self.robotId
        q_des = np.array(
            [2, 1, 1, -0.5, 0.5, 0.2, 0.1, 0, 0, 0, 0, 0]
        )  # 목표 관절 각도
        torq = self.calculate_control_torque(
            robot_id,
            joint_indices,
            q,
            dq,
            q_des,
        )

        return torq

    def calculate_control_torque(
        self, robot_id, joint_indices, q_curr, dq_curr, q_des, dq_des=None
    ):
        """
        PD 제어 + 중력 보상 토크를 계산하는 함수

        Args:
            robot_id (int): 로봇의 PyBullet Body ID
            joint_indices (list): 제어하려는 관절 인덱스 리스트 (예: [0,1,2,3,4,5,6])
            q_curr (np.array): 현재 관절 각도 (rad)
            dq_curr (np.array): 현재 관절 각속도 (rad/s)
            q_des (np.array): 목표 관절 각도 (rad)
            dq_des (np.array): 목표 관절 각속도 (rad/s), None일 경우 0으로 가정

        Returns:
            np.array: 계산된 제어 토크 (N*m)
        """

        # 1. 목표 속도가 없으면 0으로 설정 (Set Point Regulation)
        if dq_des is None:
            dq_des = np.zeros_like(q_curr)

        # 2. 제어 게인 설정 (Panda 로봇에 맞춰 튜닝된 값)
        # Kp: 위치 오차에 대한 복원력 (Stiffness)
        # Kd: 속도 오차에 대한 댐핑 (Damping)
        Kp = np.array([80.0, 80.0, 80.0, 80.0, 50.0, 50.0, 20.0])
        Kd = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 2.0])

        # 3. 오차 계산 (Error Calculation)
        print("q_des:", q_des)
        print("q_curr:", q_curr)
        print("dq_des:", dq_des)
        print("dq_curr:", dq_curr)
        q_err = q_des - q_curr
        dq_err = dq_des - dq_curr

        # 4. 중력 보상 (Gravity Compensation) 계산
        # PyBullet의 inverseDynamics는 모든 관절(Finger 포함)의 입력을 요구하므로 리스트 구성 필요
        num_total_joints = len(p.getJoint)

        # 전체 관절의 현재 위치/속도 리스트 생성 (초기화)
        # 로봇의 현재 상태를 완벽히 반영하기 위해 전체 상태를 읽어오는 것이 좋으나,
        # 성능을 위해 팔 부분만 업데이트하고 나머지는 0 또는 현재 상태 유지로 가정
        all_joint_pos = [0.0] * num_total_joints
        all_joint_vel = [0.0] * num_total_joints
        all_joint_acc = [0.0] * num_total_joints
        # 중력항만 구하기 위해 가속도는 0으로 설정

        # 제어 대상인 팔 관절(joint_indices)의 값만 현재 값으로 채워 넣음
        for i, j_idx in enumerate(joint_indices):
            all_joint_pos[j_idx] = q_curr[i]
            all_joint_vel[j_idx] = dq_curr[i]

        # 역동역학(Inverse Dynamics) 계산 -> 반환값은 전체 관절 토크
        full_torques = p.calculateInverseDynamics(
            robot_id, all_joint_pos, all_joint_vel, all_joint_acc
        )

        # 우리가 필요한 팔 관절(7-DOF)의 중력 토크만 추출
        gravity_comp = np.array([full_torques[i] for i in joint_indices])

        # 5. 최종 제어 법칙: tau = Kp*e + Kd*de + G(q)
        tau = (Kp * q_err) + (Kd * dq_err) + gravity_comp

        tau = tau[:, : self.ctrl_joint_number]

        return tau


if __name__ == "__main__":
    sim = MyRobotSim(
        gravity=-9.81,
        time_frequency=1000.0,
        control_frequency=100.0,
        simulation_duration=5.0,
    )
    print(sim.get_robot_info())
    sim.set_control_type(control_type=p.TORQUE_CONTROL)
    sim.simulate()
    sim.save_simulation_data(name="log_traj")
    print("시뮬레이션 데이터 저장 완료.")
    sim.visualize(file_name="log_traj", fps=50)
    print("시뮬레이션 재생 완료.")
