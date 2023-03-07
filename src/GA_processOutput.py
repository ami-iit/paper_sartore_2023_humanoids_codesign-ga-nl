import pygad
import numpy as np
import time
import casadi as cs
import pickle
from adam.casadi.computations import KinDynComputations
import matplotlib.pyplot as plt
import include.utils as eCubUtils
import idyntree.bindings as iDynTree
from include.motorType import motorType
from include.dataBaseFitnessFunction import DatabaseFitnessFunction
import os
from include.OptimizerMotorHumanRobot import OptimizerHumanRobotWithMotors as Optimizer
from include.createUrdf import createUrdf

modifiedModel = "eCub"
originalModel = "iCub"

common_path = os.path.dirname(os.path.abspath(__file__))
urdf_path = common_path + "/models/model.urdf"

urdf_human_path_1 = common_path + "/models/humanSubject07_48dof.urdf"
urdf_human_path_2 = common_path + "/models/humanSubject01_48dof.urdf"
urdf_human_path_3 = common_path + "/models/humanSubject04_48dof.urdf"
urdf_box_path = common_path + "/models/box_idyn01.urdf"

urdf_human_list = [urdf_human_path_1, urdf_human_path_2, urdf_human_path_3]

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

kynDynBox = KinDynComputations(urdf_box_path, [], "base_link")
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
    1661.69,
    727.431,
    600.864,
    2134.31,
    2129.3,
    1199.08,
    893.108,
    626.603,
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


def ComputeBasePosition(urdf_path):

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

    root_link = "root_link"
    left_foot_frame = "l_sole"

    kinDyn = KinDynComputations(urdf_path, joints_ctrl_name_list, root_link)

    w_H_torso = kinDyn.forward_kinematics_fun(root_link)
    w_H_leftFoot = kinDyn.forward_kinematics_fun(left_foot_frame)

    w_H_torso_num = np.array(w_H_torso(np.eye(4), np.zeros(len(joints_ctrl_name_list))))
    w_H_lefFoot_num = np.array(
        w_H_leftFoot(np.eye(4), np.zeros(len(joints_ctrl_name_list)))
    )
    w_H_init = np.linalg.inv(w_H_lefFoot_num) @ w_H_torso_num

    return w_H_init


def AddModel(viz, urdf_path, index, index_two, ChangeColor=[0, 0.0, 0.0, 1.0]):
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

    root_link = "root_link"

    wHb = ComputeBasePosition(urdf_path)
    mdlLoader = iDynTree.ModelLoader()
    mdlLoader.loadReducedModelFromFile(urdf_path, joints_ctrl_name_list, root_link)
    viz.addModel(mdlLoader.model(), urdf_path)
    H_b_vis = wHb
    H_b_vis[1, 3] = H_b_vis[1, 3] - index * 0.7
    H_b_vis[0, 3] = H_b_vis[0, 3] - index_two * 0.7
    T_b = iDynTree.Transform()
    [s_idyntree, pos, R_iDynTree] = eCubUtils.GetIDynTreeTransfMatrix(
        np.zeros(len(joints_ctrl_name_list)), H_b_vis
    )
    T_b.setPosition(pos)
    T_b.setRotation(R_iDynTree)
    T_b.setPosition(pos)
    viz.modelViz(urdf_path).setPositions(T_b, s_idyntree)
    viz.modelViz(urdf_path).setModelColor(
        iDynTree.ColorViz(iDynTree.Vector4_FromPython(ChangeColor))
    )


def diff_chromosome(list_gene_1, list_gene_2, max_values, min_values):
    range_values = np.array(min_values) - np.array(max_values)
    range_values[range_values == 0] = 1
    gene_1_norm = np.array(list_gene_1) / range_values
    gene_2_norm = np.array(list_gene_2) / range_values
    return list(gene_1_norm - gene_2_norm)


def compute_population_diversity(max_values, min_values, population):
    N = len(population)
    diversity = 0
    diff_max = np.array(diff_chromosome(max_values, min_values, max_values, min_values))
    norm_diff_max = np.sqrt(np.sum(diff_max ** 2))
    for i in range(N):
        chromo_i = population[i]
        for j in range(i + 1, N):
            chromo_j = population[j]
            diff_i_j = np.array(
                diff_chromosome(chromo_i, chromo_j, max_values, min_values)
            )
            norm_diff_i_j = np.sqrt(np.sum(diff_i_j ** 2))
            diversity += 2 * norm_diff_i_j / norm_diff_max / (N * (N - 1))
    return diversity


motor_type_dict = {0: motor_S, 1: motor_M, 2: motor_L}

## Data base fitness function
dataBase_instance = DatabaseFitnessFunction("fitnessDatabase")


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
        motor_type_tot_all,
    )


def on_generation(ga_instance):
    filename_i = (
        common_path + "/results/genetic" + str(ga_instance.generations_completed)
    )
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
    torques = np.zeros(humans_taken_into_account)
    solver_failed = False
    for i in range(humans_taken_into_account):
        urdf_human_path = urdf_human_list[i]
        try:
            kynDynHuman = KinDynComputations(
                urdf_human_path, joints_human_list, "Pelvis"
            )
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

            lenght_multiplier_tot = eCubUtils.create_length_multiplier_only_principal(
                part_name_list, link_name_list, lenght_multiplier
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
                reduction_ratio_dict, motor_limits_dict, lenght_multiplier_tot, density
            )

            optimizer.solve_optimization_problem()
            optimizer.get_solutions()
            tau_robot = optimizer.tau_opt
            tau_human = optimizer.sol.value(optimizer.tau_human)
            torques_i = 0.0

            for jj in range(optimizer.number_of_points):
                torques_i += 10 * np.linalg.norm(tau_robot[:, jj]) + np.linalg.norm(
                    tau_human[:, jj]
                )

            torques[i] = torques_i
        except:
            torques[i] = 1e-10
            solver_failed = True

    if solver_failed:
        fitness = 0.00000001
        dataBase_instance.update(sol, fitness)
        return fitness
    torques_max = np.max(torques)
    fitness = 1 / torques_max
    dataBase_instance.update(sol, fitness)
    return float(fitness)


common_path = os.path.dirname(os.path.abspath(__file__)) + "/results/GA_NL/"
number_generation = 151

try:
    os.mkdir(common_path + "/Images")
except:
    print("Images folder already exists")
try:
    os.mkdir(common_path + "/Models")
except:
    print("Models folder already exists")

ga_path = common_path + "/results/genetic" + str(number_generation)
ga_instance = pygad.load(ga_path)
print("ga_instance loaded")
pickle.dump(ga_instance.population, open("populationOutput.p", "wb"))
best_solution, fitness, idx = ga_instance.best_solution()
print(best_solution)
(
    length_mult,
    density_reduct,
    motor_limits_dict,
    inertia_motors_dict,
    reduction_ratio_dict,
    motor_type_tot_all,
) = decode_chromosome(best_solution)

for i in range(len(motor_type_tot_all)):
    motor_name = motor_joint_map[motor_name_list[i]]
    motor_type_i = motor_type_tot_all[i]
    type_motor_string = ""
    if motor_type_i == 0.0:
        type_motor_string = "S"
    if motor_type_i == 1.0:
        type_motor_string = "M"
    if motor_type_i == 2.0:
        type_motor_string = "L"
    print(motor_name, "=", type_motor_string)

string_low_limit = "low"
string_upper_limit = "high"
num_genes = 2 * n_links_param_all + n_motors
max_bound = np.zeros(num_genes)
min_bound = np.zeros(num_genes)
bounds = []
bound_length = {string_low_limit: 0.6, string_upper_limit: 1.8}
bound_motor = [0, 1, 2]
for i in range(n_links_param_all):
    max_bound[i] = 1.8
    min_bound[i] = 0.6
for i in range(n_links_param_all):
    max_bound[i + n_links_param_all] = max(feasible_density_vector)
    min_bound[i + n_links_param_all] = min(feasible_density_vector)
for i in range(n_motors):
    max_bound[i + 2 * n_links_param_all] = max(bound_motor)
    min_bound[i + 2 * n_links_param_all] = min(bound_motor)


index_best_population = np.zeros(number_generation)
diversity = np.zeros(number_generation)
fitness_mean = np.zeros(number_generation)
fitness_min = np.zeros(number_generation)
fitness_max = np.zeros(number_generation)
fitness_var = np.zeros(number_generation)
number_unf_tot = np.zeros(number_generation)

## Prepare Plot Fitness
for i in range(number_generation):
    ga_path = common_path + "/results/genetic" + str(i + 1)
    ga_instance = pygad.load(ga_path)
    diversity_i = compute_population_diversity(
        max_bound, min_bound, ga_instance.population
    )
    diversity[i] = diversity_i.sum()
    number_unf = 0
    fitness_i_tot = np.zeros(20)
    index_generation = 0

    for item in ga_instance.population:

        fitness_i = compute_fitness(item, 0)

        fitness_i_tot[index_generation] = fitness_i
        if fitness_i == 0.00000001:
            number_unf += 1

        index_generation += 1
    number_unf_tot[i] = number_unf
    index_best_population[i] = np.argmax(fitness_i_tot)

    fitness_max[i] = max(fitness_i_tot)
    fitness_min[i] = min(fitness_i_tot)
    fitness_mean[i] = np.mean(fitness_i_tot)
    fitness_var[i] = np.var(fitness_i_tot)

fitness_var_min = fitness_mean - fitness_var
fitness_var_max = fitness_mean + fitness_var


x = np.linspace(0, len(fitness_mean), len(fitness_mean))
plt.rcParams.update({"font.size": 100})
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"lines.linewidth": 2.5})
plt.rcParams.update({"axes.linewidth": 2})
plt.rcParams.update({"xtick.labelsize": 40})
plt.rcParams.update({"ytick.labelsize": 40})
plt.rcParams.update({"font.size": 28})
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"lines.linewidth": 60})

## Plotting Diversity
plt.figure()
plt.plot(diversity, linewidth="5")
plt.title("Diversity", fontsize="60")
plt.xlabel("Generation", fontsize="40")
# plt.show()
fig = plt.gcf()
fig.set_size_inches((21, 16), forward=False)
plt.savefig(common_path + "/Diverstiy.png")

## Plotting Fitness
plt.figure()
# plt.plot(x,fitness_min,  label='Min Fitness',linewidth = '4')
plt.plot(x, fitness_max, label="Max Fitness", color="mediumaquamarine", linewidth="4")
plt.plot(x, fitness_mean, label="Mean Fitness", color="steelblue", linewidth="4")
x_stop = np.linspace(0, len(fitness_mean), len(fitness_mean))
plt.fill_between(
    x, fitness_var_min, fitness_var_max, label="Variance", color="steelblue", alpha=0.2
)
plt.plot(
    x_stop,
    fitness_max[-1] * np.ones(len(x_stop)),
    color="k",
    label="Stop",
    linewidth="4",
    linestyle="dashed",
    alpha=0.5,
)

plt.legend(loc="lower right", fontsize="40")
plt.title("Fitness", fontsize="60")
plt.xlabel("Generation", fontsize="40")
# plt.show()
fig = plt.gcf()
fig.set_size_inches((21, 16), forward=False)
plt.savefig(common_path + "/Fitness.png")

## Plotting unfeasible items
plt.figure()
plt.plot(x, number_unf_tot, linewidth="2")
plt.title("Number Unfeasible Items")
fig = plt.gcf()
fig.set_size_inches((21, 16), forward=False)
plt.xlabel("Generation")
# plt.show()
fig = plt.gcf()
fig.set_size_inches((21, 16), forward=False)

plt.savefig(common_path + "/NumberUnfeasible.png")
index_temp = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
indexes = np.tile(index_temp, int(20 / index_temp.size))
difference_density_best = np.zeros(number_generation)
difference_motors_best = np.zeros(number_generation)
old_density_vector = None
old_motor_vector = None
torque_in = fitness_max[0]
torque_fin = fitness_max[-1]

## Plotting fitness per generation (For Video )
for gen_item in range(number_generation):
    plt.figure()
    fitness_max_i = fitness_max[:gen_item]
    fitness_mean_i = fitness_mean[:gen_item]
    fintess_var_i = fitness_var[:gen_item]
    fitness_var_min_i = fitness_mean_i - fintess_var_i
    fitness_var_max_i = fitness_mean_i + fintess_var_i
    x_i = np.linspace(0, len(fitness_mean_i), len(fitness_mean_i))
    plt.plot(
        x_i, fitness_max_i, label="Max Fitness", color="mediumaquamarine", linewidth="4"
    )
    plt.plot(
        x_i, fitness_mean_i, label="Mean Fitness", color="steelblue", linewidth="4"
    )
    x_stop = np.linspace(0, len(fitness_mean), len(fitness_mean))
    plt.fill_between(
        x_i,
        fitness_var_min_i,
        fitness_var_max_i,
        label="Variance",
        color="steelblue",
        alpha=0.2,
    )
    plt.plot(
        x_stop,
        fitness_max[-1] * np.ones(len(x_stop)),
        color="k",
        label="Stop",
        linewidth="4",
        linestyle="dashed",
        alpha=0.5,
    )

    plt.legend(loc="lower right", fontsize="40")
    plt.title("Fitness", fontsize="60")
    plt.xlabel("Generation", fontsize="40")
    plt.xlim([0, 140])
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches((21, 16), forward=False)
    name_fit = "/Fitness/Fitness_" + str(gen_item) + ".png"
    plt.savefig(common_path + name_fit)

## Plotting diversity per generation (for video)
for gen_item in range(number_generation):

    diversity_np = np.array(diversity)
    diveristy_i = diversity_np[:gen_item]
    plt.figure()
    plt.plot(diveristy_i, linewidth="5")
    plt.title("Diversity", fontsize="60")
    plt.xlabel("Generation", fontsize="40")
    # plt.show()
    plt.xlim([0, 140])
    plt.ylim([0.26, 0.44])
    fig = plt.gcf()
    fig.set_size_inches((21, 16), forward=False)
    name_fit = "/Diversity/Diversity_" + str(gen_item) + ".png"
    plt.savefig(common_path + name_fit)

# Create urdf
for n_gen in range(number_generation):
    ga_path = common_path + "/results/genetic" + str(n_gen + 1)
    ga_instance = pygad.load(ga_path)
    urdf_instance = createUrdf(original_urdf_path=urdf_path)
    directory_urdf_path = common_path + "/Models/Generation" + str(n_gen)
    urdf_path_best = directory_urdf_path + "/model_modified_best.urdf"
    os.mkdir(directory_urdf_path)
    for index_item in range(len(ga_instance.population)):
        item = ga_instance.population[index_item]
        motor_string = ""
        urdf_path_out = (
            directory_urdf_path + "/model_modified" + str(index_item) + ".urdf"
        )
        (
            length_mult,
            density_reduct,
            motor_limits_dict,
            inertia_motors_dict,
            reduction_ratio_dict,
            motor_type_tot_all,
        ) = decode_chromosome(item)
        if old_density_vector is None:
            old_density_vector = density_reduct
        if old_motor_vector is None:
            old_motor_vector = motor_type_tot_all
        if index_item == index_best_population[n_gen]:
            for index_motor in range(len(old_motor_vector)):
                if not (
                    old_motor_vector[index_motor] == motor_type_tot_all[index_motor]
                ):
                    difference_motors_best[n_gen] = 1
            for index_density in range(len(old_density_vector)):
                if not (
                    old_density_vector[index_density] == density_reduct[index_density]
                ):
                    difference_density_best[n_gen] = 1
            old_motor_vector = motor_type_tot_all
            old_density_vector = density_reduct
        length_mult_tot = eCubUtils.create_length_multiplier_only_principal(
            part_name_list, link_name_list, length_mult
        )
        density_tot = eCubUtils.create_length_multiplier_density(
            part_name_list, link_name_list, density_reduct
        )
        modification_length_dict = {}
        modification_density_dict = {}
        for link_index in range(len(link_name_list_param)):
            link_name = link_name_list_param[link_index]
            modification = length_mult_tot[link_index, :]
            density_i = density_tot[link_index]
            modification_length_dict.update({link_name: modification})
            modification_density_dict.update({link_name: density_i})
        urdf_instance.modify_lengths(modification_length_dict)
        urdf_instance.modify_densities(modification_density_dict)
        urdf_instance.write_urdf_to_file(urdf_path_out=urdf_path_out)
        if index_best_population[n_gen] == index_item:
            urdf_instance.write_urdf_to_file(urdf_path_out=urdf_path_best)
        urdf_instance.reset_modifications()

        for index_motor in range(len(motor_type_tot_all)):
            motor_name = motor_joint_map[motor_name_list[index_motor]]
            motor_type_i = motor_type_tot_all[index_motor]
            type_motor_string = ""
            if motor_type_i == 0.0:
                type_motor_string = " = S"
            if motor_type_i == 1.0:
                type_motor_string = " = M"
            if motor_type_i == 2.0:
                type_motor_string = " = L"
            motor_string = motor_string + motor_name + type_motor_string + "\n"

        if index_best_population[n_gen] == index_item:
            with open(directory_urdf_path + "/motor_type_best.txt", "w") as f:
                f.write(motor_string)
                f.write(str(motor_type_tot_all))
                f.write(str(length_mult_tot))
                f.write(str(density_tot))

        with open(
            directory_urdf_path + "/motor_type" + str(index_item) + ".txt", "w"
        ) as f:
            f.write(motor_string)
            f.write(str(motor_type_tot_all))

save_path_images = common_path + "/Images/"
time_now = time.time()
list_figure = []
old_color_type = [0.2, 1.0, 0.2, 0.2]
color_type_vector = []
color_type_vector.append([0.2, 1.0, 0.2, 0.2])
color_type_vector.append([1.0, 0.2, 0.2, 0.2])
color_type_vector.append([0.2, 0.2, 1.0, 0.2])
index_color_type = 0

## Visualize population output
for n_generation in range(number_generation):
    index_max = index_best_population[n_generation]
    viz = iDynTree.Visualizer()
    vizOpt = iDynTree.VisualizerOptions()
    vizOpt.winWidth = 1500
    vizOpt.winHeight = 1500
    viz.init(vizOpt)
    NDoF = 25
    env = viz.enviroment()
    env.setElementVisibility("floor_grid", True)
    env.setElementVisibility("world_frame", False)
    viz.setColorPalette("meshcat")
    env.setElementVisibility("world_frame", False)
    frames = viz.frames()
    cam = viz.camera()
    cam.setPosition(iDynTree.Position(5.5, 0, 4.0))
    viz.camera().animator().enableMouseControl(True)
    folder_visualization = common_path + "/Models/Generation" + str(n_generation)
    for item_number in range(20):
        urdf_path_out = (
            folder_visualization + "/model_modified" + str(item_number) + ".urdf"
        )
        if item_number == index_max:
            if (
                difference_density_best[n_generation]
                or difference_motors_best[n_generation]
            ):
                index_color_type += 1
                if index_color_type > 2:
                    index_color_type = 0
            AddModel(
                viz,
                urdf_path_out,
                indexes[item_number],
                2 * int(item_number / 10) - 1.5,
                color_type_vector[index_color_type],
            )
        else:
            AddModel(
                viz,
                urdf_path_out,
                indexes[item_number],
                2 * int(item_number / 10) - 1.2,
            )

    time_now = time.time()
    number_figure = 0
    while (time.time() - time_now) < 0.01 and viz.run():
        viz.draw()
    number_figure += 1

    file_name = save_path_images + "Population" + str(n_generation) + ".png"
    list_figure.append(file_name)
    viz.drawToFile(file_name)
