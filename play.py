import test_pybullet

if __name__ == "__main__":
    sim = test_pybullet.MyRobotSim(
        gravity=-9.81,
        time_frequency=1000.0,
        control_frequency=20.0,
        simulation_duration=5.0,
        control_type="position",
    )
    file_name = "log_traj"
    sim.visualize(file_name=file_name, fps=30)
    print("시뮬레이션 재생 완료.")
