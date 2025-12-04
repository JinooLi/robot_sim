from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class RobotInfo:
    joint_number: int
    ctrl_joint_number: int
    joint_angle_min: np.ndarray
    joint_angle_max: np.ndarray
    velocity_limits: np.ndarray
    torque_limits: np.ndarray
    control_type_str: str
    control_frequency: float


@dataclass
class State:
    positions: np.ndarray
    velocities: np.ndarray


class Controller(ABC):
    def set_robot_info(self, info: RobotInfo):
        self.robot_info = info

    @abstractmethod
    def control(self, state: State, t: float, M, C, g) -> np.ndarray:
        pass
