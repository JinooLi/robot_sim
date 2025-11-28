import time

import pybullet as p
import pybullet_data
import numpy as np
from numba import njit

# 시뮬레이션 설정
gravity_const = -9.81  # 중력 가속도 (m/s^2)
time_frequency = 240  # 시뮬레이션 주파수 (Hz)
control_frequency = 100  # 제어 주파수 (Hz)
time_step = 1.0 / time_frequency  # 240Hz
simulation_duration = 10.0  # 시뮬레이션 지속 시간 (초)
control_time_step = 1.0 / control_frequency  # 제어 주기 (초)

# 1. 시뮬레이션 엔진 연결 (GUI 모드)
p.connect(p.GUI)

# 2. 기본 데이터 경로 추가 (바닥, 예제 로봇 등)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3. 중력 설정
p.setGravity(0, 0, gravity_const)

# 4. 바닥과 로봇(Frank-Emika Panda) 불러오기
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

num_joints = p.getNumJoints(robotId)
joint_angle_max = []
joint_angle_min = []
torque_limits = []
velocity_limits = []
for j in range(num_joints):
    joint_info = p.getJointInfo(robotId, j)
    joint_angle_max.append(joint_info[8])
    joint_angle_min.append(joint_info[9])
    torque_limits.append(joint_info[10])
    velocity_limits.append(joint_info[11])
print(f"로봇 조인트 개수: {num_joints}")
print(f"조인트 최대 각도: {joint_angle_max}")
print(f"조인트 최소 각도: {joint_angle_min}")
print(f"토크 한계: {torque_limits}")
print(f"속도 한계: {velocity_limits}")


@njit
def is_time_to_control(
    current_time: float, control_time_step: float, previous_control_time: float
) -> tuple[bool, float]:
    if current_time - previous_control_time >= control_time_step:
        previous_control_time = current_time
        return True, current_time
    else:
        return False, previous_control_time


def control(state) -> np.ndarray:
    # 간단한 예: 모든 조인트를 0.1 라디안 위치로 이동
    np_array = np.zeros(num_joints)
    for i in range(num_joints):
        np_array[i] = 0.1
    return np_array


def control_robot(u: np.ndarray, control_type):
    # 예: 특정 조인트(관절) 위치 제어
    # Panda 로봇의 조인트 인덱스는 URDF 구조에 따라 다름
    for j in range(num_joints):
        if joint_angle_max[j] <= u[j] or u[j] <= joint_angle_min[j]:
            u[j] = -u[j]
        p.setJointMotorControl2(robotId, j, control_type, targetPosition=u[j])


# 5. 시뮬레이션 루프
log_traj = []
previous_control_time = 0.0
t = 0.0
u = np.zeros(num_joints)
for i in range(int(simulation_duration * time_frequency)):
    state = p.getJointStates(robotId, range(num_joints))

    is_control_time, previous_control_time = is_time_to_control(
        t, control_time_step, previous_control_time
    )

    tic = time.perf_counter()
    u_new: np.ndarray = control(state)
    toc = time.perf_counter()
    calculation_time = toc - tic
    delay_steps = int(calculation_time / time_step)

    t = i * time_step
    for _ in range(delay_steps):
        control_robot(u, control_type=p.POSITION_CONTROL)
        p.stepSimulation()
        current_pos = [
            state[0] for state in p.getJointStates(robotId, range(num_joints))
        ]
        log_traj.append(current_pos)
        time.sleep(time_step)

    for _ in range(int(control_time_step / time_step) - delay_steps):
        control_robot(u_new, control_type=p.POSITION_CONTROL)
        p.stepSimulation()
        current_pos = [
            state[0] for state in p.getJointStates(robotId, range(num_joints))
        ]
        log_traj.append(current_pos)
        time.sleep(time_step)

    u = u_new

np.save("sim_data/log_traj.npy", np.array(log_traj))
print(f"시뮬레이션 종료: {t:.2f}초")
p.disconnect()
