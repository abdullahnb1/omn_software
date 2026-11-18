"""MecanumSim: Model-Agnostic 4-Wheel Mecanum AMR Simulation Platform in MuJoCo."""

__version__ = "1.0.0"

from mecanum_sim.config import (
    SimConfig,
    RobotConfig,
    load_config,
    list_available_robots,
)

__all__ = [
    "__version__",
    "SimConfig",
    "RobotConfig",
    "load_config",
    "list_available_robots",
]

