from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass
class RobotInfo:
    joint_number: int
    ctrl_joint_number: int
    joint_angle_min: np.ndarray
    joint_angle_max: np.ndarray
    velocity_limits: np.ndarray
    torque_limits: np.ndarray
    control_frequency: float


@dataclass
class State:
    positions: np.ndarray
    velocities: np.ndarray
    ee_position: np.ndarray
    ee_orientation: np.ndarray


class ControlType(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"


class Controller(ABC):
    def __init__(self, control_type: ControlType):
        self.control_type = control_type

    def set_robot_info(self, info: RobotInfo, M, C, g):
        self.robot_info = info
        self.M = M
        self.C = C
        self.g = g

    @abstractmethod
    def control(self, state: State, t: float) -> np.ndarray:
        pass


class Simulator(ABC):
    @abstractmethod
    def __init__(
        self,
        controller: Controller,
        gravity: float,
        time_frequency: float,
        control_frequency: float,
        simulation_duration: float,
    ) -> None:
        super().__init__()

    @abstractmethod
    def simulate():
        pass

    @abstractmethod
    def visualize(file_name: str, fps: int):
        pass

    @abstractmethod
    def plot_trajectory(file_name: str, joint_indices: list[float]):
        pass
