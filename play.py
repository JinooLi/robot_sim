import simulation

if __name__ == "__main__":
    sim = simulation.RobotSim()
    file_name = "log_traj"
    sim.visualize(file_name=file_name, fps=30)
    print("시뮬레이션 재생 완료.")
    sim.plot_trajectory(file_name=file_name, joint_indices=[0, 1, 2, 3, 4, 5, 6])
