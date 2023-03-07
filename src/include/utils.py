import numpy as np
import casadi as cs
import math
import idyntree.bindings as iDynTree


def FromQuaternionToMatrix():
    f_opts = (dict(jit=False, jit_options=dict(flags="-Ofast")),)
    # Quaternion variable
    H = cs.SX.eye(4)
    q = cs.SX.sym("q", 7)
    quat_normalized = q[:4] / cs.norm_2(q[:4])
    R = (
        cs.SX.eye(3)
        + 2 * quat_normalized[0] * cs.skew(quat_normalized[1:4])
        + 2 * cs.mpower(cs.skew(quat_normalized[1:4]), 2)
    )

    H[:3, :3] = R
    H[:3, 3] = q[4:7]
    H = cs.Function("H", [q], [H])
    return H


def TransfMatrixError():
    H = cs.SX.sym("H", 4, 4)
    H_des = cs.SX.sym("H_des", 4, 4)
    error_rot = cs.SX.ones(3)
    error_pos = cs.SX.ones(3)
    R = H[:3, :3]
    R_des = H_des[:3, :3]
    p = H[:3, 3]
    p_des = H_des[:3, 3]

    Temp = cs.mtimes(R, cs.transpose(R_des))
    error_rot = SkewVee(Temp)
    error_pos = p - p_des
    error_rot = cs.Function("error_rot", [H, H_des], [error_rot])
    error_pos = cs.Function("error_pos", [H, H_des], [error_pos])
    return error_rot, error_pos


def SkewVee(X):
    X_skew = 0.5 * (X - cs.transpose(X))
    x = cs.vertcat(-X_skew[1, 2], X_skew[0, 2], -X_skew[0, 1])
    return x


def ComputeOriginalDensity(kinDyn, link_name):
    link_original = get_element_by_name(link_name, kinDyn.robot_desc)
    mass = link_original.inertial.mass
    volume = 0
    visual_obj = link_original.visuals[0]

    if hasattr(visual_obj.geometry, "size"):
        width = link_original.visuals[0].geometry.size[0]
        depth = link_original.visuals[0].geometry.size[2]
        height = link_original.visuals[0].geometry.size[1]
        volume = width * depth * height

    elif hasattr(visual_obj.geometry, "length"):
        length = link_original.visuals[0].geometry.length
        radius = link_original.visuals[0].geometry.radius
        volume = math.pi * radius ** 2 * length
    elif hasattr(visual_obj.geometry, "radius"):
        radius = link_original.visuals[0].geometry.radius
        volume = 4 * (math.pi * radius ** 3) / 3
    return mass / volume


def compute_minimum_number_of_parameters(part_name_list):
    link_name_list = []
    minimum_number_param = 0
    tot_number_parameters = 0
    if "Arms" in part_name_list:
        link_name_list += [
            "l_shoulder_1",
            "l_shoulder_2",
            "l_shoulder_3",
            "l_elbow_1",
            "r_shoulder_1",
            "r_shoulder_2",
            "r_shoulder_3",
            "r_elbow_1",
        ]
        minimum_number_param += 4
        tot_number_parameters += 8
    if "Legs" in part_name_list:
        link_name_list += [
            "r_upper_leg",
            "r_lower_leg",
            "r_hip_1",
            "r_hip_2",
            "r_ankle_1",
            "r_ankle_2",
            "l_upper_leg",
            "l_lower_leg",
            "l_hip_1",
            "l_hip_2",
            "l_ankle_1",
            "l_ankle_2",
        ]
        minimum_number_param += 6
        tot_number_parameters += 12
    if "Torso" in part_name_list:
        link_name_list += ["torso_1", "torso_2", "root_link"]
        minimum_number_param += 3
        tot_number_parameters += 3
    return minimum_number_param, link_name_list, tot_number_parameters


def compute_total_motor_vector(link_name_list, part_name_list, vector_in):
    vector_out = np.zeros(len(link_name_list))
    index_start = 0
    index_start_min = 0
    if "Arms" in part_name_list:
        vector_out[index_start : index_start + 4] = vector_in[
            index_start_min : index_start_min + 4
        ]
        vector_out[index_start + 4 : index_start + 8] = vector_in[
            index_start_min : index_start_min + 4
        ]
        index_start += 8
        index_start_min += 4
    if "Legs" in part_name_list:
        vector_out[index_start : index_start + 6] = vector_in[
            index_start_min : index_start_min + 6
        ]
        vector_out[index_start + 6 : index_start + 12] = vector_in[
            index_start_min : index_start_min + 6
        ]
        index_start += 12
        index_start_min += 6
    if "Torso" in part_name_list:
        vector_out[index_start : index_start + 3] = vector_in[
            index_start_min : index_start_min + 3
        ]
    return vector_out


def compute_minimum_number_of_parameters_link(part_name_list):
    link_name_list = []
    number_of_parameters = 0
    if "Torso" in part_name_list:
        link_name_list += ["root_link", "torso_1", "torso_2", "chest"]
        number_of_parameters += 4
    if "Arms" in part_name_list:
        link_name_list += ["r_upper_arm", "r_forearm", "l_upper_arm", "l_forearm"]
        number_of_parameters += 2
    if "Legs" in part_name_list:
        link_name_list += ["r_hip_3", "r_lower_leg", "l_hip_3", "l_lower_leg"]
        number_of_parameters += 2
    return number_of_parameters, link_name_list


def create_length_multiplier_with_sym_CaSadi_only_principal(
    part_name_list, link_name_list, lenght, density
):
    length_tot = cs.MX.ones(len(link_name_list), 3)
    density_tot = cs.MX.zeros(len(link_name_list))
    n_end = 0
    n_end_min = 0
    if "Arms" in part_name_list:
        length_tot[:2, 0] = lenght[:2]
        length_tot[2:4, 0] = lenght[:2]
        density_tot[:2] = density[:2]
        density_tot[2:4] = density[:2]
        n_end = 4
        n_end_min = 2
    if "Legs" in part_name_list:
        length_tot[n_end : n_end + 2, 0] = lenght[n_end_min : n_end_min + 2]
        length_tot[n_end + 2 : n_end + 4, 0] = lenght[n_end_min : n_end_min + 2]
        density_tot[n_end : n_end + 2] = density[n_end_min : n_end_min + 2]
        density_tot[n_end + 2 : n_end + 4] = density[n_end_min : n_end_min + 2]
    return density_tot, length_tot


def create_length_multiplier_only_principal(part_name_list, link_name_list, lenght):
    length_tot = np.ones([len(link_name_list), 3])
    # density_tot = cs.MX.zeros(len(link_name_list))
    n_end = 0
    n_end_min = 0
    if "Torso" in part_name_list:
        length_tot[:4, 0] = lenght[:4]
        n_end += 4
        n_end_min += 4
    if "Arms" in part_name_list:
        length_tot[n_end : n_end + 2, 0] = lenght[n_end_min : n_end_min + 2]
        length_tot[n_end + 2 : n_end + 4, 0] = lenght[n_end_min : n_end_min + 2]
        # density_tot[:2] = density[:2]
        # density_tot[2:4] = density[:2]
        n_end += 4
        n_end_min += 2
    if "Legs" in part_name_list:
        length_tot[n_end : n_end + 2, 0] = lenght[n_end_min : n_end_min + 2]
        length_tot[n_end + 2 : n_end + 4, 0] = lenght[n_end_min : n_end_min + 2]
        # density_tot[n_end:n_end+2] = density[n_end_min:n_end_min+2]
        # density_tot[n_end+2:n_end+4] = density[n_end_min:n_end_min+2]
    return length_tot


def create_length_multiplier_density(part_name_list, link_name_list, density):
    density_tot = np.ones(len(link_name_list))
    # density_tot = cs.MX.zeros(len(link_name_list))
    n_end = 0
    n_end_min = 0
    if "Torso" in part_name_list:
        density_tot[:4] = density[:4]
        n_end += 4
        n_end_min += 4
    if "Arms" in part_name_list:
        density_tot[n_end : n_end + 2] = density[n_end_min : n_end_min + 2]
        density_tot[n_end + 2 : n_end + 4] = density[n_end_min : n_end_min + 2]
        # density_tot[:2] = density[:2]
        # density_tot[2:4] = density[:2]
        n_end += 4
        n_end_min += 2
    if "Legs" in part_name_list:
        density_tot[n_end : n_end + 2] = density[n_end_min : n_end_min + 2]
        density_tot[n_end + 2 : n_end + 4] = density[n_end_min : n_end_min + 2]
        # density_tot[n_end:n_end+2] = density[n_end_min:n_end_min+2]
        # density_tot[n_end+2:n_end+4] = density[n_end_min:n_end_min+2]
    return density_tot


def create_vector_sym(part_name_list, link_name_list, lenght):
    length_tot = np.ones([len(link_name_list), 3])
    n_end = 0
    n_end_min = 0

    if "Arms" in part_name_list:
        length_tot[:2, :] = lenght[:2, :]
        length_tot[2:4, :] = lenght[:2, :]
        n_end = 4
        n_end_min = 2
    if "Legs" in part_name_list:
        length_tot[n_end : n_end + 2, :] = lenght[n_end_min : n_end_min + 2, :]
        length_tot[n_end + 2 : n_end + 4, :] = lenght[n_end_min : n_end_min + 2, :]

    return length_tot


def create_reduced_hardware_parameters_vector(
    part_name_list, vector_hardware, minimum_n_parameters, number_columns
):
    if number_columns == 1:
        vector_out = np.empty(minimum_n_parameters)
        n_end = 0
        n_end_min = 0
        if "Torso" in part_name_list:
            vector_out[:4] = vector_hardware[:4]
            n_end += 4
            n_end_min += 4
        if "Arms" in part_name_list:
            vector_out[n_end_min : n_end_min + 2] = vector_hardware[n_end : n_end + 2]
            n_end_min += 2
            n_end += 4
        if "Legs" in part_name_list:
            vector_out[n_end_min : n_end_min + 2] = vector_hardware[n_end : n_end + 2]

            n_end_min += 2
            n_end += 4

    else:
        vector_out = np.empty([minimum_n_parameters, number_columns])
        n_end = 0
        n_end_min = 0
        if "Torso" in part_name_list:
            vector_out[:4, :] = vector_hardware[:4, :]
            n_end += 4
            n_end_min += 4
        if "Arms" in part_name_list:
            vector_out[n_end_min : n_end_min + 2, :] = vector_hardware[
                n_end : n_end + 2, :
            ]
            n_end_min += 2
            n_end += 4
        if "Legs" in part_name_list:
            vector_out[n_end_min : n_end_min + 2, :] = vector_hardware[
                n_end : n_end + 2, :
            ]

            n_end_min += 2
            n_end += 4
    return vector_out


def create_length_multiplier_with_sym(part_name_list, link_name_list, lenght, density):
    length_tot = np.empty([len(link_name_list), 3])
    density_tot = np.empty(len(link_name_list))
    n_end = 0
    n_end_min = 0
    if "Arms" in part_name_list:
        length_tot[:4, :] = lenght[:4, :]
        length_tot[4:8, :] = lenght[:4, :]
        density_tot[:4] = density[:4]
        density_tot[4:8] = density[:4]
        n_end = 8
        n_end_min = 4
    if "Legs" in part_name_list:
        length_tot[n_end : n_end + 5, :] = lenght[n_end_min : n_end_min + 5, :]
        length_tot[n_end + 5 : n_end + 10, :] = lenght[n_end_min : n_end_min + 5, :]
        density_tot[n_end : n_end + 5] = density[n_end_min : n_end_min + 5]
        density_tot[n_end + 5 : n_end + 10] = density[n_end_min : n_end_min + 5]
    return density_tot, length_tot


def create_length_multiplier_with_sym_links(
    part_name_list, link_name_list, lenght, density
):
    length_tot = np.empty([len(link_name_list), 3])
    density_tot = np.empty(len(link_name_list))
    n_end = 0
    n_end_min = 0
    if "Arms" in part_name_list:
        length_tot[:2, :] = lenght[:2, :]
        length_tot[2:4, :] = lenght[:2, :]
        density_tot[:2] = density[:2]
        density_tot[2:4] = density[:2]
        n_end = 4
        n_end_min = 2
    if "Legs" in part_name_list:
        length_tot[n_end : n_end + 2, :] = lenght[n_end_min : n_end_min + 2, :]
        length_tot[n_end + 2 : n_end + 4, :] = lenght[n_end_min : n_end_min + 2, :]
        density_tot[n_end : n_end + 2] = density[n_end_min : n_end_min + 2]
        density_tot[n_end + 2 : n_end + 4] = density[n_end_min : n_end_min + 2]
    return density_tot, length_tot


def ComputeTotalMassLinkChain(
    linkNameList, kinDyn, density, lengths_multipliers_vector
):
    mass_tot = 0.0
    ParametricLinkList = kinDyn.link_name_list
    for item in linkNameList:
        if item in ParametricLinkList:
            mass_temp = kinDyn.get_link_mass(item)
            index = ParametricLinkList.index(item)
            mass_tot += mass_temp(density[index], lengths_multipliers_vector[index])
        else:
            mass_temp = kinDyn.get_element_by_name(
                item, kinDyn.robot_desc
            ).inertial.mass
            mass_tot = mass_tot + mass_temp
    return mass_tot


def zAxisAngle():
    H = cs.SX.sym("H", 4, 4)
    theta = cs.SX.sym("theta")
    theta = cs.dot([0, 0, -1], H[:3, 2]) - 1

    error = cs.Function("error", [H], [theta])
    return error


def GetIDynTreeTransfMatrix(s, H):

    N_DoF = len(s)
    print(N_DoF)
    s_idyntree = iDynTree.VectorDynSize(N_DoF)
    pos_iDynTree = iDynTree.Position()
    R_iDynTree = iDynTree.Rotation()

    for i in range(N_DoF):
        s_idyntree.setVal(i, s[i])

    R_iDynTree.FromPython(H[:3, :3])
    for i in range(3):
        pos_iDynTree.setVal(i, float(H[i, 3]))
        for j in range(3):
            R_iDynTree.setVal(j, i, float(H[j, i]))

    return s_idyntree, pos_iDynTree, R_iDynTree
