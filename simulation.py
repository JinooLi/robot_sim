import os
import time

import numpy as np
import pybullet as p
import pybullet_data
import pinocchio as pin

from interface import Controller, RobotInfo, State, ControlType, Simulator


class RobotSim(Simulator):
    def __init__(
        self,
        controller: Controller = None,  # type: ignore
        gravity: float = 0,
        time_frequency: float = 0,
        control_frequency: float = 0,
        simulation_duration: float = 0,
    ):
        """시뮬레이션 환경 초기화

        Args:
            controller (Controller, optional): 제어기 객체. Defaults to None.
            gravity (float): 중력가속도 (m/s^2)
            time_frequency (float): 시뮬레이션 주파수 (Hz)
            control_frequency (float): 제어 주파수 (Hz) - 시뮬레이션 주파수보다 반드시 작거나 같아야함.
            simulation_duration (float): 시뮬레이션 시간 (t)
        """
        if controller == None:
            return
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
        self.set_control_type(controller.control_type)

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

        self.controller = controller
        robot_info = self.get_robot_info()
        self._make_kinematic_functions()
        self.controller.set_robot_info(
            robot_info, self.M_func, self.C_func, self.g_func, self.J_linear
        )

        self.log_traj = []
        p.disconnect()

    def get_robot_info(self) -> RobotInfo:
        """로봇의 조인트 개수 및 한계치 정보를 반환한다.

        Returns:
            RobotInfo:\n
                joint_number (int): 로봇 조인트 개수\n
                joint_angle_max (list): 조인트 최대 각도 리스트\n
                joint_angle_min (list): 조인트 최소 각도 리스트\n
                torque_limits (list): 토크 한계 리스트\n
                velocity_limits (list): 속도 한계 리스트\n
                control_frequency (float): 제어 주파수 (Hz)
        """
        info = RobotInfo(
            joint_number=self.joint_number,
            ctrl_joint_number=self.ctrl_joint_number,
            joint_angle_min=np.array(self.joint_angle_min),
            joint_angle_max=np.array(self.joint_angle_max),
            velocity_limits=np.array(self.velocity_limits),
            torque_limits=np.array(self.torque_limits),
            control_frequency=self.control_frequency,
        )
        return info

    def _get_robot_state(self) -> State:
        """로봇의 현재 상태를 반환한다.

        Returns:
            state: 현재 로봇 상태
        """
        p_state = p.getJointStates(self.robotId, range(self.joint_number))
        state = State(
            positions=np.array([s[0] for s in p_state]),
            velocities=np.array([s[1] for s in p_state]),
            ee_position=np.array(p.getLinkState(self.robotId, 11)[0]),
            ee_orientation=np.array(p.getLinkState(self.robotId, 11)[1]),
        )
        return state

    def _put_input_to_sim(self, u: np.ndarray):
        """로봇을 원하는 제어 입력과 control type에 맞게 제어한다.

        Args:
            u (np.ndarray): 제어 입력 벡터
        """
        # 예: 특정 조인트(관절) 위치 제어
        # Panda 로봇의 조인트 인덱스는 URDF 구조에 따라 다름
        if self.p_control_type == p.TORQUE_CONTROL:
            for i in range(self.joint_number):
                p.setJointMotorControl2(
                    self.robotId, i, self.p_control_type, force=u[i]
                )
        elif self.p_control_type == p.VELOCITY_CONTROL:
            for i in range(self.joint_number):
                p.setJointMotorControl2(
                    self.robotId, i, self.p_control_type, targetVelocity=u[i]
                )
        else:
            for i in range(self.joint_number):
                p.setJointMotorControl2(
                    self.robotId, i, self.p_control_type, targetPosition=u[i]
                )

    def ctrl_type2pct(self, control_type: ControlType):
        """제어 타입 string을 받고 제어 타입 변수를 반환한다.

        Args:
            control_type (ControlType): ControlType.POSITION, ControlType.VELOCITY, ControlType.TORQUE 중 하나

        Returns:
            p.control_type: 제어 타입 변수
        """
        if control_type == ControlType.POSITION:
            return p.POSITION_CONTROL
        elif control_type == ControlType.VELOCITY:
            return p.VELOCITY_CONTROL
        elif control_type == ControlType.TORQUE:
            return p.TORQUE_CONTROL
        else:
            raise ValueError(f"{control_type}은 지원하지 않는 제어 타입입니다.")

    def set_control_type(self, control_type: ControlType):
        """제어 입력 타입 지정

        Args:
            control_type (ControlType):  ControlType.POSITION, ControlType.VELOCITY, ControlType.TORQUE 중 하나
        """
        self.p_control_type = self.ctrl_type2pct(control_type)

    def pct2ctrl_type(self, p_control_type) -> ControlType:
        """제어 타입 변수를 받고 제어 타입 string을 반환한다.

        Returns:
            str: "position", "velocity", "torque" 중 하나
        """
        if p_control_type == p.POSITION_CONTROL:
            return ControlType.POSITION
        elif p_control_type == p.VELOCITY_CONTROL:
            return ControlType.VELOCITY
        elif p_control_type == p.TORQUE_CONTROL:
            return ControlType.TORQUE
        else:
            raise ValueError(f"{self.p_control_type}은 지원하지 않는 제어 타입입니다.")

    def _make_kinematic_functions(self):
        """역기구학 제어를 위한 동적 모델 함수 생성"""
        urdf_path = os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()

        extend = lambda arr: np.append(arr, [0.0, 0.0])

        self.M_func = lambda q: np.array(pin.crba(model, data, extend(q)), np.float64)[
            :-2, :-2
        ]

        self.C_func = lambda q, v: np.array(
            pin.computeCoriolisMatrix(model, data, extend(q), extend(v)), np.float64
        )[:-2, :-2]

        self.g_func = lambda q: np.array(
            pin.computeGeneralizedGravity(model, data, extend(q)), np.float64
        )[:-2]

        self.J_linear = lambda q: self.get_J_linear(extend(q))

    def get_J_linear(self, q) -> np.ndarray:
        q_list = [float(val) for val in q]
        dq_list = [0.0] * len(q_list)
        ddq_list = [0.0] * len(q_list)

        J_linear, _ = p.calculateJacobian(
            self.robotId,
            11,
            [0.0, 0.0, 0.0],
            q_list,
            dq_list,
            ddq_list,
        )

        J_linear = np.array(J_linear)
        J_linear = J_linear[:, : self.ctrl_joint_number]
        return J_linear

    def control(self, state: State, t) -> np.ndarray:
        """제어 입력을 관절 수에 맞게 확장하는 함수

        Args:
            state: 현재 state
            t: current simulation time

        Returns:
            np.ndarray: 제어 입력
        """
        u = self.controller.control(state, t)
        np_array = np.zeros(self.ctrl_joint_number)
        for i in range(self.ctrl_joint_number):
            np_array[i] = u[i]
        for i in range(self.joint_number - self.ctrl_joint_number):
            np_array = np.append(np_array, 0.0)
        return np_array

    def draw_debug_point(self, position, color=[1, 0, 0], size=1, lifeTime=0):
        """
        position: [x, y, z] 좌표
        color: [r, g, b] (0~1 사이 값)
        size: 십자가 크기
        lifeTime: 유지 시간 (0이면 영구 유지, 양수면 초 단위 후 사라짐)
        """
        x, y, z = position

        # X축 선
        p.addUserDebugLine(
            [x - size, y, z], [x + size, y, z], lineColorRGB=color, lifeTime=lifeTime
        )
        # Y축 선
        p.addUserDebugLine(
            [x, y - size, z], [x, y + size, z], lineColorRGB=color, lifeTime=lifeTime
        )
        # Z축 선
        p.addUserDebugLine(
            [x, y, z - size], [x, y, z + size], lineColorRGB=color, lifeTime=lifeTime
        )

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
        if self.p_control_type == p.TORQUE_CONTROL:
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

        Raises:
            ValueError: fps는 시뮬레이션 주파수보다 작아야 합니다.
        """

        # 저장된 데이터 불러오기
        if not os.path.exists(f"sim_data/{file_name}.npy"):
            print("시뮬레이션 데이터가 없습니다.")
            return
        trajectory = np.load(f"sim_data/{file_name}.npy")

        env_data = np.load(f"sim_data/{file_name}_env.npy")
        self.gravity_const = env_data[0]
        self.time_frequency = env_data[1]
        self.control_frequency = env_data[2]
        self.simulation_duration = env_data[3]
        self.dt = 1.0 / self.time_frequency
        self.ctrl_dt = 1.0 / self.control_frequency

        if fps >= self.time_frequency:
            raise ValueError("fps는 시뮬레이션 주파수보다 작아야 합니다.")
        sample_rate = int(1.0 / (self.dt * fps))
        steps = len(trajectory)

        p.connect(p.GUI)
        # 시각화 환경 셋팅
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.gravity_const)

        # GUI 잡동사니(메뉴 등) 숨기기 -> 영상 찍을 때 깔끔함
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        p.loadURDF("plane.urdf")
        robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        num_joints = p.getNumJoints(robotId)
        self.draw_debug_point(self.controller.ee_target_pos)

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

    def plot_trajectory(self, file_name, joint_indices: list):
        """저장된 시뮬레이션 데이터를 불러와서 지정한 조인트들의 궤적을 플로팅한다.

        Args:
            file_name (str): 저장된 시뮬레이션 데이터 파일 이름
            joint_indices (list): 플로팅할 조인트 인덱스 리스트
        """

        # 저장된 데이터 불러오기
        if not os.path.exists(f"sim_data/{file_name}.npy"):
            print("시뮬레이션 데이터가 없습니다.")
            return
        env_data = np.load(f"sim_data/{file_name}_env.npy")
        dt = 1.0 / env_data[1]
        trajectory = np.load(f"sim_data/{file_name}.npy")

        time_array = np.arange(0, len(trajectory)) * dt

        import matplotlib.pyplot as plt

        plt.figure()
        for joint_idx in joint_indices:
            joint_positions = trajectory[:, joint_idx]
            plt.plot(time_array, joint_positions, label=f"Joint {joint_idx}")

        plt.title("Joint Trajectories")
        plt.xlabel("Time (s)")
        plt.ylabel("Joint Position (rad)")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":

    class TestController(Controller):
        def control(self, state: State, t: float) -> np.ndarray:
            """제어입력을 만든다.

            Args:
                state: 현재 state
                t: 현재 시뮬레이션 시간

            Returns:
                np.ndarray: 제어 입력
            """
            # 간단한 예: 모든 조인트를 0.1 라디안 위치로 이동
            np_array = np.zeros(self.robot_info.ctrl_joint_number)
            for i in range(self.robot_info.ctrl_joint_number):
                np_array[i] = 0.1
            return np_array

    controller = TestController(control_type=ControlType.POSITION)
    sim = RobotSim(
        gravity=-9.81,
        time_frequency=1000.0,
        control_frequency=20.0,
        simulation_duration=5.0,
        controller=controller,
    )
    print(sim.get_robot_info())
    sim.simulate()
    sim.save_simulation_data(name="log_traj2")
    print("시뮬레이션 데이터 저장 완료.")
    sim.visualize(file_name="log_traj2", fps=50)
    print("시뮬레이션 재생 완료.")
    sim.plot_trajectory(file_name="log_traj2", joint_indices=[0, 1, 2])
