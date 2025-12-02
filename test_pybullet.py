import os
import time

import numpy as np
import pybullet as p
import pybullet_data
from numba import njit
from abc import ABC, abstractmethod


class RobotSim(ABC):
    def __init__(
        self,
        gravity: float,
        time_frequency: float,
        control_frequency: float,
        simulation_duration: float,
    ):
        """시뮬레이션 환경 초기화

        Args:
            gravity (float): 중력가속도 (m/s^2)
            time_frequency (float): 시뮬레이션 주파수 (Hz)
            control_frequency (float): 제어 주파수 (Hz) - 시뮬레이션 주파수보다 반드시 작거나 같아야함.
            simulation_duration (float): 시뮬레이션 시간 (t)
        """
        p.connect(p.DIRECT)
        # 로봇(Frank-Emika Panda) 불러오기
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

        self.gravity_const = gravity  # 중력 가속도 (m/s^2)
        self.time_frequency = time_frequency  # 시뮬레이션 주파수 (Hz)
        self.control_frequency = control_frequency  # 제어 주파수 (Hz)
        self.simulation_duration = simulation_duration  # 시뮬레이션 지속 시간 (초)
        self.dt = 1.0 / time_frequency  # 시뮬레이션 타임스텝 (초)
        self.ctrl_dt = 1.0 / control_frequency  # 제어 주기 (초)
        self.joint_number = p.getNumJoints(self.robotId)  # 로봇 조인트 개수
        self.ctrl_joint_number = 7  # 제어할 조인트 개수
        self.control_type = p.POSITION_CONTROL  # 기본 제어 타입

        # 로봇 조인트 물리 한계치 정보 불러오기
        self.joint_angle_max = []
        self.joint_angle_min = []
        self.torque_limits = []
        self.velocity_limits = []
        for j in range(self.joint_number):
            joint_info = p.getJointInfo(self.robotId, j)
            self.joint_angle_min.append(joint_info[8])
            self.joint_angle_max.append(joint_info[9])
            self.torque_limits.append(joint_info[10])
            self.velocity_limits.append(joint_info[11])

        self.log_traj = []
        p.disconnect()

    def get_robot_info(self) -> tuple:
        """로봇의 조인트 개수 및 한계치 정보를 반환한다.

        Returns:
            joint_number (int): 로봇 조인트 개수
            joint_angle_max (list): 조인트 최대 각도 리스트
            joint_angle_min (list): 조인트 최소 각도 리스트
            torque_limits (list): 토크 한계 리스트
            velocity_limits (list): 속도 한계 리스트
        """
        return (
            self.joint_number,
            self.joint_angle_max,
            self.joint_angle_min,
            self.torque_limits,
            self.velocity_limits,
        )

    def _get_robot_state(self):
        """로봇의 현재 상태를 반환한다.

        Returns:
            state: 현재 로봇 상태
        """
        state = p.getJointStates(self.robotId, range(self.joint_number))
        return state

    def _put_input_to_sim(self, u: np.ndarray):
        """로봇을 원하는 제어 입력과 control type에 맞게 제어한다.

        Args:
            u (np.ndarray): 제어 입력 벡터
        """
        # 예: 특정 조인트(관절) 위치 제어
        # Panda 로봇의 조인트 인덱스는 URDF 구조에 따라 다름
        for j in range(self.joint_number):
            if self.control_type == p.TORQUE_CONTROL:
                p.setJointMotorControl2(self.robotId, j, self.control_type, force=u[j])
            elif self.control_type == p.VELOCITY_CONTROL:
                p.setJointMotorControl2(
                    self.robotId, j, self.control_type, targetVelocity=u[j]
                )
            else:
                p.setJointMotorControl2(
                    self.robotId, j, self.control_type, targetPosition=u[j]
                )

    def set_control_type(self, control_type):
        """제어 입력 타입 지정

        Args:
            control_type: p.POSITION_CONTROL, p.VELOCITY_CONTROL, p.TORQUE_CONTROL 등
        """
        self.control_type = control_type

    @abstractmethod
    def controller(self, state, t) -> np.ndarray:
        """제어기 함수

        Args:
            state: _description_
            t: current simulation time

        Returns:
            np.ndarray[self.ctrl_joint_number]: 제어할 관절의 개수에 해당하는 제어 입력 벡터
        """
        pass

    def control(self, state, t) -> np.ndarray:
        """제어 입력을 관절 수에 맞게 확장하는 함수

        Args:
            state: 현재 state
            t: current simulation time

        Returns:
            np.ndarray: 제어 입력
        """
        u = self.controller(state, t)
        np_array = np.zeros(self.ctrl_joint_number)
        for i in range(self.ctrl_joint_number):
            np_array[i] = u[i]
        for i in range(self.joint_number - self.ctrl_joint_number):
            np_array = np.append(np_array, 0.0)
        return np_array

    def simulate(self):
        """제어 입력을 계산하는 시간까지 고려하여 시뮬레이션을 돌리는 함수"""
        # 시뮬레이션 엔진 연결 (DIRECT 모드)
        p.connect(p.DIRECT)
        # 기본 데이터 경로 추가 (예제 로봇 URDF 파일 등)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        # 중력가속도 설정
        p.setGravity(0, 0, self.gravity_const)

        # 만약 토크 제어라면 기본 모터 비활성화
        if self.control_type == p.TORQUE_CONTROL:
            for j in range(self.joint_number):
                p.setJointMotorControl2(self.robotId, j, p.VELOCITY_CONTROL, force=0)

        # 시뮬레이션 타임스텝 설정
        p.setTimeStep(self.dt)
        # 시뮬레이션 루프
        t = 0.0
        u = np.zeros(self.joint_number)
        while t <= self.simulation_duration:
            # 현재 로봇 상태 불러오기
            state = self._get_robot_state()

            # 제어 입력 계산
            tic = time.perf_counter()
            u_new: np.ndarray = self.control(state, t)
            toc = time.perf_counter()

            # 계산 시간에 따른 딜레이 스텝 수 계산
            calculation_time = toc - tic
            delay_steps = int(calculation_time / self.dt)

            # 딜레이 스텝만큼 이전 제어 입력으로 시뮬레이션 진행
            for _ in range(delay_steps):
                if t > self.simulation_duration:
                    break
                self._put_input_to_sim(u)
                p.stepSimulation()
                current_pos = [
                    state[0]
                    for state in p.getJointStates(
                        self.robotId, range(self.joint_number)
                    )
                ]
                self.log_traj.append(current_pos)
                t += self.dt

            # 남은 제어 주기 동안 새로운 제어 입력으로 시뮬레이션 진행
            for _ in range(int(self.ctrl_dt / self.dt) - delay_steps):
                if t > self.simulation_duration:
                    break
                self._put_input_to_sim(u_new)
                p.stepSimulation()
                current_pos = [
                    state[0]
                    for state in p.getJointStates(
                        self.robotId, range(self.joint_number)
                    )
                ]
                self.log_traj.append(current_pos)
                t += self.dt

            u = u_new

        p.disconnect()
        print(f"시뮬레이션 종료: {t:.2f}초")

    def save_simulation_data(self, name: str = "log_traj"):
        """시뮬레이션 데이터 저장"""
        if len(self.log_traj) == 0:
            print("시뮬레이션 데이터가 없습니다.")
            return

        if not os.path.exists("sim_data"):
            os.makedirs("sim_data", exist_ok=True)
            with open("sim_data/.gitignore", "w") as f:
                f.write("*")
        np.save(f"sim_data/{name}.npy", np.array(self.log_traj))
        np.save(
            f"sim_data/{name}_env.npy",
            np.array(
                [
                    self.gravity_const,
                    self.time_frequency,
                    self.control_frequency,
                    self.simulation_duration,
                ]
            ),
        )

    def visualize(self, file_name, fps: int = 60):
        """저장된 시뮬레이션 데이터를 불러와서 재생한다.

        Args:
            fps (int, optional): 재생 속도. Defaults to 60.
        """

        # 저장된 데이터 불러오기
        if not os.path.exists(f"sim_data/{file_name}.npy"):
            print("시뮬레이션 데이터가 없습니다.")
            return
        trajectory = np.load(f"sim_data/{file_name}.npy")
        env_data = np.load(f"sim_data/{file_name}_env.npy")
        self.__init__(*env_data.tolist())

        if fps >= self.time_frequency:
            raise ValueError("fps는 시뮬레이션 주파수보다 작아야 합니다.")
        sample_rate = int(1.0 / (self.dt * fps))
        steps = len(trajectory)

        # 시각화 환경 셋팅
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # GUI 잡동사니(메뉴 등) 숨기기 -> 영상 찍을 때 깔끔함
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        p.loadURDF("plane.urdf")
        robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        num_joints = p.getNumJoints(robotId)

        # 재생 루프
        print("재생 시작...")

        tic = time.time()
        for i in range(int(steps * self.dt * fps)):
            sim_tic = time.perf_counter()
            current_pos = trajectory[i * sample_rate]

            # 물리 엔진을 쓰는 게 아니라, 관절 위치를 강제로 덮어씌움 (Reset)
            for j in range(int(num_joints)):
                p.resetJointState(robotId, j, current_pos[j])

            # 화면 갱신을 위해 stepSimulation을 호출하되,
            # 실제 물리 연산이 목적이 아니므로 단순히 렌더링 트리거 역할만 함
            p.stepSimulation()
            sim_toc = time.perf_counter()

            # 시뮬레이션 계산 시간
            sim_time = sim_toc - sim_tic

            # 사람 눈에 실제 속도처럼 보이게 대기
            time.sleep(sample_rate * self.dt - sim_time)
        toc = time.time()
        print(
            f"재생 시간: {toc - tic:.2f}초 (실제 시뮬레이션 시간: {steps * self.dt:.2f}초)"
        )

        print("재생 종료")
        time.sleep(2)
        p.disconnect()


class MyRobotSim(RobotSim):
    def controller(self, state) -> np.ndarray:
        """제어입력을 만든다.

        Args:
            state: 현재 state

        Returns:
            np.ndarray: 제어 입력
        """
        # 간단한 예: 모든 조인트를 0.1 라디안 위치로 이동
        np_array = np.zeros(self.ctrl_joint_number)
        for i in range(self.ctrl_joint_number):
            np_array[i] = 0.1
        return np_array


if __name__ == "__main__":
    sim = MyRobotSim(
        gravity=-9.81,
        time_frequency=1000.0,
        control_frequency=20.0,
        simulation_duration=5.0,
    )
    print(sim.get_robot_info())
    sim.set_control_type(control_type=p.POSITION_CONTROL)
    sim.simulate()
    sim.save_simulation_data(name="log_traj2")
    print("시뮬레이션 데이터 저장 완료.")
    sim.visualize(file_name="log_traj2", fps=50)
    print("시뮬레이션 재생 완료.")
