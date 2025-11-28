import pybullet as p
import pybullet_data
import numpy as np
import time

# 1. 저장된 데이터 불러오기
trajectory = np.load("sim_data/log_traj.npy")
steps = len(trajectory)

# 2. 시각화 환경 셋팅
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 중요: GUI 잡동사니(메뉴 등) 숨기기 -> 영상 찍을 때 깔끔함
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
num_joints = p.getNumJoints(robotId)

# 3. 재생 루프
print("재생 시작...")
# 재생 속도 (원래 시뮬레이션 타임스텝에 맞춤, 예: 1/240 또는 1/1000)
dt_replay = 1.0 / 240.0

for i in range(steps):
    current_pos = trajectory[i]

    # [핵심] 물리 엔진을 쓰는 게 아니라, 관절 위치를 강제로 덮어씌움 (Reset)
    for j in range(num_joints):
        p.resetJointState(robotId, j, current_pos[j])

    # 화면 갱신을 위해 stepSimulation을 호출하되,
    # 실제 물리 연산이 목적이 아니므로 단순히 렌더링 트리거 역할만 함
    p.stepSimulation()

    # 사람 눈에 실제 속도처럼 보이게 대기
    time.sleep(dt_replay)

print("재생 종료")
time.sleep(2)
p.disconnect()
