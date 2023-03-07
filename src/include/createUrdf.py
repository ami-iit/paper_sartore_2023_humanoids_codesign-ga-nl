from urdfModifiers.core.fixedOffsetModifier import FixedOffsetModifier
from urdfModifiers.core.modification import Modification
from urdfModifiers.utils import *
from urdfModifiers.geometry import *


class createUrdf:
    def __init__(self, original_urdf_path) -> None:
        self.original_urdf_path = original_urdf_path
        self.dummy_file = "no_gazebo_plugins.urdf"
        self.robot, self.gazebo_plugin_text = utils.load_robot_and_gazebo_plugins(
            self.original_urdf_path, self.dummy_file
        )

    def modify_lengths(self, length_multipliers: dict):
        for name, modification in length_multipliers.items():
            fixed_offset_modifier = FixedOffsetModifier.from_name(name, self.robot)
            fixed_offset_modifications = Modification()
            fixed_offset_modifications.add_dimension(modification, absolute=False)
            # Apply the modifications
            fixed_offset_modifier.modify(fixed_offset_modifications)

    def modify_densities(self, densities: dict):
        for name, modification in densities.items():
            fixed_offset_modifier = FixedOffsetModifier.from_name(name, self.robot)
            fixed_offset_modifications = Modification()
            fixed_offset_modifications.add_density(modification, absolute=True)
            # Apply the modifications
            fixed_offset_modifier.modify(fixed_offset_modifications)

    def write_urdf_to_file(self, urdf_path_out):
        """Saves the URDF to a valid .urdf file, also adding the gazebo_plugins"""
        self.robot.save(urdf_path_out)
        lines = []
        with open(urdf_path_out, "r") as f:
            lines = f.readlines()
            last_line = lines.pop()
            lines = lines + self.gazebo_plugin_text
            lines.append(last_line)

        with open(urdf_path_out, "w") as f:
            f.writelines(lines[1:])

    def reset_modifications(self):
        self.robot, self.gazebo_plugin_text = utils.load_robot_and_gazebo_plugins(
            self.original_urdf_path, self.dummy_file
        )
