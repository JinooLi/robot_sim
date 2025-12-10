from interface import Simulator
from control import MyController, MyCLBFGenerator
from simulation import RobotSim

import numpy as np


def run_sim(sim: Simulator):
    sim.simulate()
    data_name = "log_traj"
    sim.save_simulation_data(name=data_name)
    print("시뮬레이션 데이터 저장 완료.")
    sim.visualize(file_name=data_name, fps=50)
    print("시뮬레이션 재생 완료.")
    sim.plot_trajectory(file_name=data_name, joint_indices=[0, 1, 2, 3, 4, 5, 6])


if __name__ == "__main__":

    clbf_gen = MyCLBFGenerator(
        unsafe_region_center=np.array([0.3, 0.3, 0.8]),
        unsafe_region_radius=0.2,
        unsafe_region_margin=0.05,
        barrier_gain=200,
        Lyapunov_center=np.array([0.5, 0.5, 0.5]),
    )

    controller = MyController(clbf_generator=clbf_gen)
    sim = RobotSim(
        controller=controller,
        gravity=-9.81,
        time_frequency=1000.0,
        control_frequency=100.0,
        simulation_duration=30.0,
    )
    run_sim(sim)
