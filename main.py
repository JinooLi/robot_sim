from interface import ControlType, Simulator
from control import MyController
from simulation import RobotSim


def run_sim(sim: Simulator):
    sim.simulate()
    data_name = "log_traj"
    sim.save_simulation_data(name=data_name)
    print("시뮬레이션 데이터 저장 완료.")
    sim.visualize(file_name=data_name, fps=50)
    print("시뮬레이션 재생 완료.")
    sim.plot_trajectory(file_name=data_name, joint_indices=[0, 1, 2, 3, 4, 5, 6])


if __name__ == "__main__":
    controller = MyController(ControlType.TORQUE)
    sim = RobotSim(
        controller=controller,
        gravity=-9.81,
        time_frequency=1000.0,
        control_frequency=100.0,
        simulation_duration=10.0,
    )
    run_sim(sim)
