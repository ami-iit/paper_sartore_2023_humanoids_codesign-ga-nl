import pygad
import numpy as np
import pickle
from adam.casadi.computations import KinDynComputations
import include.utils as eCubUtils
from include.motorType import motorType
from include.dataBaseFitnessFunction import DatabaseFitnessFunction
import os
from include.OptimizerMotorHumanRobot import OptimizerHumanRobotWithMotors as Optimizer

modifiedModel = "eCub"
originalModel = "iCub"

common_path = os.path.dirname(os.path.abspath(__file__))
urdf_path = common_path + "/models/model.urdf"
urdf_human_path_1 = common_path + "/models/humanSubject07_48dof.urdf"
urdf_human_path_2 = common_path + "/models/humanSubject01_48dof.urdf"
urdf_human_path_3 = common_path + "/models/humanSubject04_48dof.urdf"
urdf_box_path_1 = common_path + "/models/box_idyn01.urdf"
urdf_box_path_2 = common_path + "/models/box_idyn02.urdf"
urdf_box_path_3 = common_path + "/models/box_idyn03.urdf"


urdf_human_list = [urdf_human_path_1, urdf_human_path_2, urdf_human_path_3]
urdf_box_list = [urdf_box_path_1, urdf_box_path_2, urdf_box_path_3]
number_of_boxes = len(urdf_box_list)

results_folder = common_path + "/results/"
joints_ctrl_name_list = [
    "torso_pitch",
    "torso_roll",
    "torso_yaw",
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow",
    "l_wrist_prosup",
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow",
    "r_wrist_prosup",
    "l_hip_pitch",
    "l_hip_roll",
    "l_hip_yaw",
    "l_knee",
    "l_ankle_pitch",
    "l_ankle_roll",
    "r_hip_pitch",
    "r_hip_roll",
    "r_hip_yaw",
    "r_knee",
    "r_ankle_pitch",
    "r_ankle_roll",
]

joints_human_list = [
    "jT1C7_roty",
    "jT9T8_roty",
    "jL1T12_roty",
    "jL4L3_roty",
    "jL5S1_roty",
    "jLeftC7Shoulder_rotx",
    "jLeftShoulder_rotx",
    "jLeftShoulder_roty",
    "jLeftShoulder_rotz",
    "jLeftElbow_roty",
    "jLeftElbow_rotz",
    "jLeftWrist_rotx",
    "jLeftWrist_rotz",
    "jRightC7Shoulder_rotx",
    "jRightShoulder_rotx",
    "jRightShoulder_roty",
    "jRightShoulder_rotz",
    "jRightElbow_roty",
    "jRightElbow_rotz",
    "jRightWrist_rotx",
    "jRightWrist_rotz",
    "jLeftHip_rotx",
    "jLeftHip_roty",
    "jLeftKnee_roty",
    "jLeftAnkle_rotx",
    "jLeftAnkle_roty",
    "jLeftAnkle_rotz",
    "jRightHip_rotx",
    "jRightHip_roty",
    "jRightKnee_roty",
    "jRightAnkle_rotx",
    "jRightAnkle_roty",
    "jRightAnkle_rotz",
]

right_hand_frame_human = "RightHand"
left_hand_frame_human = "LeftHand"
left_foot_frame_human = "LeftToe"
right_foot_frame_human = "RightToe"
contact_frames_human = [
    left_foot_frame_human,
    right_foot_frame_human,
    left_hand_frame_human,
    right_hand_frame_human,
]
number_of_points = 3
root_link = "root_link"

s_initial_human_tot = pickle.load(
    open(common_path + "/outputKinematic/s_human.p", "rb")
)
quat_b_inital_human_tot = pickle.load(
    open(common_path + "/outputKinematic/quat_human.p", "rb")
)
s_initial_tot = pickle.load(open(common_path + "/outputKinematic/s_robot.p", "rb"))
quat_b_initial_tot = pickle.load(
    open(common_path + "/outputKinematic/quat_robot.p", "rb")
)
population_initial = pickle.load(
    open(common_path + "/outputKinematic/populationInitial.p", "rb")
)
part_name_list = ["Arms", "Legs", "Torso"]
part_name_list_motors = ["Arms", "Legs", "Torso"]
(
    n_motors,
    motor_name_list,
    tot_number_of_parameters_motor,
) = eCubUtils.compute_minimum_number_of_parameters(part_name_list=part_name_list_motors)
(
    n_links_param_all,
    link_name_list_param,
) = eCubUtils.compute_minimum_number_of_parameters_link(part_name_list=part_name_list)
n_param = n_links_param_all
link_name_list = [*link_name_list_param]
feasible_density_vector = [
    2129.2952964,
    1199.07622408,
    893.10763518,
    626.60271872,
    1661.68632652,
    727.43130782,
    600.50011475,
    2222.0327914,
]
motor_joint_map = {
    "r_upper_leg": "r_hip_roll",
    "r_hip_1": "r_hip_pitch",
    "r_hip_2": "r_hip_yaw",
    "r_ankle_1": "r_ankle_pitch",
    "r_ankle_2": "r_ankle_roll",
    "r_lower_leg": "r_knee",
}
motor_joint_map.update(
    {
        "l_upper_leg": "l_hip_roll",
        "l_hip_1": "l_hip_pitch",
        "l_hip_2": "l_hip_yaw",
        "l_ankle_1": "l_ankle_pitch",
        "l_ankle_2": "l_ankle_roll",
        "l_lower_leg": "l_knee",
    }
)
motor_joint_map.update(
    {
        "l_shoulder_1": "l_shoulder_pitch",
        "l_shoulder_2": "l_shoulder_roll",
        "l_shoulder_3": "l_shoulder_yaw",
        "l_elbow_1": "l_elbow",
    }
)
motor_joint_map.update(
    {
        "r_shoulder_1": "r_shoulder_pitch",
        "r_shoulder_2": "r_shoulder_roll",
        "r_shoulder_3": "r_shoulder_yaw",
        "r_elbow_1": "r_elbow",
    }
)
motor_joint_map.update(
    {"torso_1": "torso_yaw", "torso_2": "torso_roll", "root_link": "torso_pitch"}
)
(
    n_motors,
    motor_name_list,
    tot_number_motors,
) = eCubUtils.compute_minimum_number_of_parameters(part_name_list=part_name_list_motors)
kinDyn = KinDynComputations(urdf_path, joints_ctrl_name_list, root_link, link_name_list)

n_DoF = len(joints_ctrl_name_list)
n_DoF_min = 3 + 5 + 6
left_foot_frame_robot = "l_sole"
right_foot_frame_robot = "r_sole"
contact_frames_robot = [left_foot_frame_robot, right_foot_frame_robot]

## Motors Specs
motor_S = motorType()
motor_S.length = 0.075
motor_S.radius = 0.065
motor_S.mass = 0.750
motor_S.max_torque = 37
motor_S.I_m = 1e-4
motor_S.reduction_ratio = 1 / 100

motor_M = motorType()
motor_M.length = 0.092
motor_M.radius = 0.075
motor_M.mass = 0.950
motor_M.max_torque = 92
motor_M.I_m = 1e-3
motor_M.reduction_ratio = 1 / 160

motor_L = motorType()
motor_L.length = 0.090
motor_L.radius = 0.090
motor_L.mass = 1.45
motor_L.max_torque = 123
motor_L.I_m = 1e-3
motor_L.reduction_ratio = 1 / 160

motor_type_dict = {0: motor_S, 1: motor_M, 2: motor_L}


def compute_total_density(link_param_name_list, kinDyn):
    density = np.empty(len(link_param_name_list))
    for i in range(len(link_param_name_list)):
        density_i = eCubUtils.ComputeOriginalDensity(kinDyn, link_param_name_list[i])
        density[i] = density_i
    return density


def decode_chromosome(chromosome):
    length_mult = chromosome[:n_links_param_all]
    density_reduct = chromosome[n_links_param_all : 2 * n_links_param_all]
    motors = chromosome[2 * n_links_param_all :]
    motor_type_tot_all = eCubUtils.compute_total_motor_vector(
        motor_name_list, part_name_list_motors, motors
    )
    motor_limits_dict = {}
    inertia_motors_dict = {}
    reduction_ratio_dict = {}
    for i in range(tot_number_motors):
        motor_type_i = motor_type_dict[motor_type_tot_all[i]]
        motor_limits_dict.update(
            {motor_joint_map[motor_name_list[i]]: motor_type_i.max_torque}
        )
        inertia_motors_dict.update(
            {motor_joint_map[motor_name_list[i]]: motor_type_i.I_m}
        )
        reduction_ratio_dict.update(
            {motor_joint_map[motor_name_list[i]]: motor_type_i.reduction_ratio}
        )
    return (
        length_mult,
        density_reduct,
        motor_limits_dict,
        inertia_motors_dict,
        reduction_ratio_dict,
    )


def on_generation(ga_instance):
    generation_completed = ga_instance.generations_completed + 152
    filename_i = common_path + "/results/genetic" + str(generation_completed)
    ga_instance.save(filename_i)


def compute_fitness(sol, sol_idx):
    dataBase_instance = DatabaseFitnessFunction(
        common_path + "/results/fitnessDatabase"
    )
    fitness_value = dataBase_instance.get_fitness_value(chromosome=sol)
    # if already computed fitness value returing the value in the database
    if fitness_value is not None:
        print("Already comptued fitness")
        return fitness_value
    humans_taken_into_account = len(urdf_human_list)
    (
        lenght_multiplier,
        density_reduct,
        motor_limits_dict,
        inertia_dict,
        reduction_ratio_dict,
    ) = decode_chromosome(sol)
    torques = np.zeros(humans_taken_into_account * number_of_boxes)
    solver_failed = False
    index_torque = 0
    for human_item in range(humans_taken_into_account):
        urdf_human_path = urdf_human_list[human_item]
        kynDynHuman = KinDynComputations(urdf_human_path, joints_human_list, "Pelvis")
        for box_item in range(number_of_boxes):
            urdf_box_path = urdf_box_list[box_item]
            kynDynBox = KinDynComputations(urdf_box_path, [], "base_link")
            try:
                optimizer = Optimizer(
                    n_DoF,
                    "root_link",
                    kinDyn,
                    contact_frames_robot,
                    n_param,
                    link_name_list,
                    part_name_list,
                    kynDynHuman,
                    kynDynBox,
                    joints_ctrl_name_list,
                    joints_human_list,
                )
                optimizer.initialize_solver()
                optimizer.define_kin_din_functions()
                optimizer.define_search_variables()

                lenght_multiplier_tot = (
                    eCubUtils.create_length_multiplier_only_principal(
                        part_name_list, link_name_list, lenght_multiplier
                    )
                )
                density = eCubUtils.create_length_multiplier_density(
                    part_name_list, link_name_list, density_reduct
                )
                optimizer.set_initial_for_human_and_box(
                    quat_b_inital_human_tot,
                    s_initial_human_tot,
                    lenght_multiplier_tot,
                    density,
                )
                optimizer.set_initial_with_variable(
                    s_initial_tot, quat_b_initial_tot, lenght_multiplier, density_reduct
                )
                optimizer.populate_optimization_problem_both_agents(
                    inertia_dict, reduction_ratio_dict, motor_limits_dict
                )

                optimizer.set_references(density, lenght_multiplier_tot)
                optimizer.set_initial_torques(
                    reduction_ratio_dict,
                    motor_limits_dict,
                    lenght_multiplier_tot,
                    density,
                )

                optimizer.solve_optimization_problem()
                optimizer.get_solutions()
                tau_robot = optimizer.tau_opt
                tau_human = optimizer.sol.value(optimizer.tau_human)
                torques_i = 0.0

                for jj in range(optimizer.number_of_points):
                    torques_i += np.linalg.norm(tau_robot[:, jj]) + np.linalg.norm(
                        tau_human[:, jj]
                    )

                torques[index_torque] = torques_i
                index_torque += 1
            except:
                torques[index_torque] = 1e-10
                solver_failed = True
                index_torque += 1

    if solver_failed:
        fitness = 0.00000001
        dataBase_instance.update(sol, fitness)
        return fitness
    torques_max = np.max(torques)
    fitness = 100 * (1 / torques_max)  # Multiplied by 100 to have readable results
    dataBase_instance.update(sol, fitness)
    return float(fitness)


string_low_limit = "low"
string_upper_limit = "high"
string_step = "step"
bounds = []
bound_length = {string_low_limit: 0.6, string_upper_limit: 1.8, string_step: 0.1}

bound_motor = [0, 1, 2]
for _ in range(n_links_param_all):
    bounds.append(bound_length)
for _ in range(n_links_param_all):
    bounds.append(feasible_density_vector)
for _ in range(n_motors):
    bounds.append(bound_motor)

num_genes = 2 * n_links_param_all + n_motors
ga_instance = pygad.GA(
    num_generations=500,
    num_parents_mating=15,
    sol_per_pop=20,
    fitness_func=compute_fitness,
    gene_space=bounds,
    num_genes=num_genes,
    parent_selection_type="tournament",
    K_tournament=3,
    crossover_type="single_point",
    allow_duplicate_genes=True,
    on_generation=on_generation,
    initial_population=population_initial,
)
ga_instance.run()
