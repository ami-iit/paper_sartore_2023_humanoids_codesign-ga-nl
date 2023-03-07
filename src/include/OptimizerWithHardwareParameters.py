from include import ergoCubConfiguration
from include import utils as eCubUtils
import casadi as cs
import numpy as np


class OptimizerWithHardwareParameters:
    def __init__(
        self,
        N_DoF,
        root_link,
        kinDyn,
        contact_frame,
        minimum_number_parameters,
        link_name_list,
        part_name_list,
        number_of_points=2,
    ) -> None:
        self.joint_name_list = []
        self.index_parts = {}
        self.root_link = root_link
        self.N_DoF = N_DoF
        self.part_name_list = part_name_list
        self.link_name_list = link_name_list
        self.number_of_hardware_parameters = minimum_number_parameters
        self.number_of_points = number_of_points
        self.contact_frame = contact_frame
        self.kinDyn = kinDyn
        self.height_hands_start = [0.8, 1.0, 1.2]
        self.solver_initialize = False
        self.cost_function = 0.0
        self.define_kin_din_functions()
        self.fc = []

    def set_solver(self, solver):
        self.solver = solver
        self.solver_initialize = True
        self.initialize_parameter_search_variable()

    def initialize_parameter_search_variable(self):
        # Define the variable of the optimization problem
        self.s = self.solver.variable(
            self.N_DoF, self.number_of_points
        )  # joint positions
        self.quat_pose_b = self.solver.variable(
            7, self.number_of_points
        )  # base pose as quaternion
        self.lenght_multiplier = self.solver.parameter(
            self.number_of_hardware_parameters
        )
        self.densities = self.solver.parameter(self.number_of_hardware_parameters)
        (
            self.density_vector,
            self.lenght_multiplier_vector,
        ) = eCubUtils.create_length_multiplier_with_sym_CaSadi_only_principal(
            self.part_name_list,
            self.link_name_list,
            self.lenght_multiplier,
            self.densities,
        )
        self.tau = self.solver.variable(self.N_DoF, self.number_of_points)
        self.H_right_hand_star = self.solver.parameter(4, 4)
        self.H_left_hand_star = self.solver.parameter(4, 4)
        self.H_left_foot_star = self.solver.parameter(4, 4)
        self.H_right_foot_star = self.solver.parameter(4, 4)
        self.q_zero = self.solver.parameter(self.N_DoF)

    def initialize_solver(self):
        self.solver = cs.Opti()
        # Define the solver
        p_opts = {}
        s_opts = {"linear_solver": "ma27", "max_iter": 4000}
        self.solver.solver("ipopt", p_opts, s_opts)
        self.solver_initialize = True
        self.initialize_parameter_search_variable()

    def set_target_height_hands(self, height_hand_star):
        self.height_hands_start = height_hand_star

    def define_kin_din_functions(self):

        print("***** Defining kin and din functions*****")
        # Frames of Interest
        right_hand_frame = "r_hand"
        left_hand_frame = "l_hand"
        left_foot_frame = "l_sole"
        right_foot_frame = "r_sole"
        frame_heigth = "head"

        self.M = self.kinDyn.mass_matrix_fun()
        self.J_left = self.kinDyn.jacobian_fun(left_foot_frame)
        self.J_right = self.kinDyn.jacobian_fun(right_foot_frame)
        self.J_right_hand = self.kinDyn.jacobian_fun(right_hand_frame)
        self.J_left_hand = self.kinDyn.jacobian_fun(left_hand_frame)
        self.g = self.kinDyn.gravity_term_fun()
        self.B = cs.vertcat(cs.MX.zeros(6, self.N_DoF), cs.MX.eye(self.N_DoF))
        self.error_rot, self.error_pos = eCubUtils.TransfMatrixError()
        self.H_right_hand = self.kinDyn.forward_kinematics_fun(right_hand_frame)
        self.H_left_hand = self.kinDyn.forward_kinematics_fun(left_hand_frame)
        self.H_left_foot = self.kinDyn.forward_kinematics_fun(left_foot_frame)
        self.H_right_foot = self.kinDyn.forward_kinematics_fun(right_foot_frame)
        self.w_H_torso = self.kinDyn.forward_kinematics_fun(self.root_link)
        self.H_height = self.kinDyn.forward_kinematics_fun(frame_heigth)

        self.theta_left_foot_error = eCubUtils.zAxisAngle()

        self.CoM = self.kinDyn.CoM_position_fun()
        self.total_mass = self.kinDyn.get_total_mass()

        self.H_b_fun = eCubUtils.FromQuaternionToMatrix()

    def get_local_wrench(self, f_i, frame_orientation):
        X_c = cs.MX.zeros(6, 6)
        X_c[:3, :3] = frame_orientation
        X_c[3:, 3:] = frame_orientation

        f = X_c.T @ f_i

        return f

    def add_whrenches_constraint(
        self, f_i, add_torsional_friction=False, add_min_normal_constraint=False
    ):
        torsional_friction = 1 / 75
        static_friction = 1 / 3
        min_wrench = 0.0
        if add_torsional_friction:
            self.solver.subject_to(f_i[5] - torsional_friction * f_i[2] < 0)
            self.solver.subject_to(-f_i[5] - torsional_friction * f_i[2] < 0)
        # TODO: handle this repeated constraint in case both friction cone and local cop are active
        if add_min_normal_constraint:
            self.solver.subject_to(f_i[2] > min_wrench)
        self.solver.subject_to(f_i[0] < static_friction * f_i[2])
        self.solver.subject_to(-f_i[0] < static_friction * f_i[2])
        self.solver.subject_to(f_i[1] < static_friction * f_i[2])
        self.solver.subject_to(-f_i[1] < static_friction * f_i[2])

    def add_intrinsic_constraint(self, q_base_i, s_i, tau_i, motor_limits_dict):

        if not (self.solver_initialize):
            print(
                "Solver not initialized !! please either set the solver via set_solver or intialize the solver via initialize_solver"
            )
            return

        self.solver.subject_to(cs.sumsqr(q_base_i[:4]) == 1.0)
        for i in range(self.N_DoF):
            item = self.kinDyn.joints_list[i].name
            # if(item in motor_limits_dict):
            #     self.solver.subject_to(tau_i[i]< motor_limits_dict[item])
            #     self.solver.subject_to(tau_i[i]> -motor_limits_dict[item])
        self.solver.subject_to(s_i[3] == s_i[8])
        self.solver.subject_to(s_i[4] == s_i[9])
        self.solver.subject_to(s_i[5] == s_i[10])
        self.solver.subject_to(s_i[6] == s_i[11])
        self.solver.subject_to(s_i[16] == s_i[22])

    def compute_original_density(self):
        density_vector_min = []
        ## Compute original density
        for item in self.minimum_set_parameters_links:
            density_vector_min += [eCubUtils.ComputeOriginalDensity(self.kinDyn, item)]
        self.density_vector = ergoCubConfiguration.CreateTotalHardwareParameterVector(
            self.partsList, density_vector_min
        )
        self.density_vector_min = density_vector_min

    def set_references(self, density, lenght_multipliers):
        ## Desired quantities
        w_H_torso_num = self.w_H_torso(
            np.eye(4), self.s_initial[:, 0], density, lenght_multipliers
        )
        w_H_lefFoot_num = self.H_left_foot(
            np.eye(4), self.s_initial[:, 0], density, lenght_multipliers
        )
        w_H_init = cs.inv(w_H_lefFoot_num) @ w_H_torso_num
        # quat_b_initial = [1.0,0.0, 0.0, 0.0,-0.0489,-0.0648,0.65]
        self.w_H_lFoot_des = self.H_left_foot(
            w_H_init, self.s_initial[:, 0], density, lenght_multipliers
        )
        self.w_H_rFoot_des = self.H_right_foot(
            w_H_init, self.s_initial[:, 0], density, lenght_multipliers
        )
        self.w_H_lHand_des = self.H_left_hand(
            w_H_init, self.s_initial[:, 0], density, lenght_multipliers
        )
        self.w_H_rHand_des = self.H_right_hand(
            w_H_init, self.s_initial[:, 0], density, lenght_multipliers
        )
        # Setting the values
        self.solver.set_value(self.q_zero, np.zeros(self.N_DoF))

        self.solver.set_value(self.H_left_foot_star, self.w_H_lFoot_des)
        self.solver.set_value(self.H_right_foot_star, self.w_H_rFoot_des)
        self.solver.set_value(self.H_left_hand_star, self.w_H_lHand_des)
        self.solver.set_value(self.H_right_hand_star, self.w_H_rHand_des)

    def set_initial_with_variable(
        self, s_initial, quat_b_initial, lenght_multiplier, densities
    ):

        if not (self.solver_initialize):
            print(
                "Solver not initialized !! please either set the solver via set_solver or intialize the solver via initialize_solver"
            )
            return
        self.s_initial = s_initial
        self.quat_b_initial = quat_b_initial
        self.solver.set_initial(self.s, self.s_initial)
        self.solver.set_initial(self.quat_pose_b, self.quat_b_initial)
        self.solver.set_value(self.lenght_multiplier, lenght_multiplier)
        self.solver.set_value(self.densities, densities)

    def solve_optimization_problem(self):

        if not (self.solver_initialize):
            print(
                "Solver not initialized !! please either set the solver via set_solver or intialize the solver via initialize_solver"
            )
            return

        self.solver.minimize(self.cost_function)
        print("Solving")
        try:
            self.sol = self.solver.solve()

            print(self.solver.debug.value)
            print("Solved")
        except:
            print("WARNING NOT SOLVED PROBLEM DEBUG OUTPUT:")
            print(self.solver.debug.value)
            exit()

    def get_solutions(self):
        self.s_opt_all = self.sol.value(self.s)
        self.quat_b_opt = self.sol.value(self.quat_pose_b)
        self.lenght_multiplier_opt = self.sol.value(self.lenght_multiplier)
        self.density_opt = self.sol.value(self.densities)
        self.tau_opt = self.sol.value(self.tau)

    def set_solution(self, sol):
        self.sol = sol

    def get_joint_by_name(self, joint_name, robot):
        joint_list = [
            corresponding_joint
            for corresponding_joint in robot.joints
            if corresponding_joint.name == joint_name
        ]
        if len(joint_list) != 0:
            return joint_list[0]
        else:
            return None

    def add_reflected_inertia(self, M, reflectedInertia: dict, reductionRatio: dict):
        I_m = np.eye(self.N_DoF)
        Gamma = np.eye(self.N_DoF)
        for i in range(self.N_DoF):
            item = self.kinDyn.joints_list[i].name
            if item in reflectedInertia:
                I_m[i, i] = reflectedInertia[item]
                Gamma[i, i] = reductionRatio[item]
        M[6:, 6:] = M[6:, 6:] + np.linalg.inv(
            np.transpose(Gamma)
        ) @ I_m @ np.linalg.inv(np.transpose(Gamma))
        return M, Gamma
