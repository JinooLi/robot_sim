import test_pybullet

if __name__ == "__main__":
    sim = test_pybullet.RobotSim()
    file_name = "log_traj"
    sim.visualize(file_name=file_name, fps=30)
    print("시뮬레이션 재생 완료.")
