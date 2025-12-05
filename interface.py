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
    """로봇의 상태를 나타낸다.

    attrs:
        positions (np.ndarray): 관절 각도
        velocities (np.ndarray): 관절 속도
        ee_position (np.ndarray): 말단 위치
        ee_orientation (np.ndarray): 말단 자세

    """

    positions: np.ndarray
    velocities: np.ndarray
    ee_position: np.ndarray
    ee_orientation: np.ndarray


class ControlType(Enum):
    """제어 타입을 나타낸다.

    Args:
        POSITION: 위치 제어
        VELOCITY: 속도 제어
        TORQUE: 토크 제어
    """

    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"


class Controller(ABC):

    def set_robot_info(self, info: RobotInfo, M, C, g, J_linear):
        self.robot_info = info
        self.M = M
        self.C = C
        self.g = g
        self.J_linear = J_linear

    @abstractmethod
    def control(self, state: State, t: float) -> np.ndarray:
        pass

    def set_control_type(self, control_type: ControlType):
        self.control_type = control_type


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
    def simulate(self):
        pass

    @abstractmethod
    def visualize(self, file_name: str, fps: int):
        pass

    @abstractmethod
    def plot_trajectory(self, file_name: str, joint_indices: list[float]):
        pass

    @abstractmethod
    def save_simulation_data(self, name: str):
        pass
