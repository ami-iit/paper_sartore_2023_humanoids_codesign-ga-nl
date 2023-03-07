from adam.core import link_parametric
import casadi as cs
import numpy as np
import math


def get_initial_human_configuration(joints_name_list):
    s_init = {
        "jL5S1_rotx": 0,
        "jL5S1_roty": 0,
        "jL4L3_rotx": 0,
        "jL4L3_roty": 0,
        "jL1T12_rotx": 0,
        "jL1T12_roty": 0,
        "jT9T8_rotx": 0,
        "jT9T8_roty": 0,
        "jT9T8_rotz": 0,
        "jT1C7_rotx": 0,
        "jT1C7_roty": 0,
        "jT1C7_rotz": 0,
        "jC1Head_rotx": 0,
        "jC1Head_roty": 0,
        "jRightC7Shoulder_rotx": 0,
        "jRightShoulder_rotx": 0,
        "jRightShoulder_roty": 0,
        "jRightShoulder_rotz": 90,
        "jRightElbow_roty": 0,
        "jRightElbow_rotz": 0,
        "jRightWrist_rotx": 0,
        "jRightWrist_rotz": 0,
        "jLeftC7Shoulder_rotx": 0,
        "jLeftShoulder_rotx": 0,
        "jLeftShoulder_roty": 0,
        "jLeftShoulder_rotz": -90,
        "jLeftElbow_roty": 0,
        "jLeftElbow_rotz": 0,
        "jLeftWrist_rotx": 0,
        "jLeftWrist_rotz": 0,
        "jRightHip_rotx": 0,
        "jRightHip_roty": -25,
        "jRightHip_rotz": 0,
        "jRightKnee_roty": 50,
        "jRightKnee_rotz": 0,
        "jRightAnkle_rotx": 0,
        "jRightAnkle_roty": -25,
        "jRightAnkle_rotz": 0,
        "jLeftHip_rotx": 0,
        "jLeftHip_roty": -25,
        "jLeftHip_rotz": 0,
        "jLeftKnee_roty": 50,
        "jLeftKnee_rotz": 0,
        "jLeftAnkle_rotx": 0,
        "jLeftAnkle_roty": -25,
        "jLeftAnkle_rotz": 0,
        "jLeftBallFoot_roty": 0,
        "jRightBallFoot_roty": 0,
    }
    s_init_out = [s_init[item] * math.pi / 180 for item in joints_name_list]
    return s_init_out
