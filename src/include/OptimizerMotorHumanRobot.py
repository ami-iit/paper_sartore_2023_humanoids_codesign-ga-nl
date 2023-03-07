from include.OptimizerWithHardwareParameters import (
    OptimizerWithHardwareParameters as OptimizerRobot,
)
import casadi as cs
import numpy as np
import include.utils as eCubUtils


class OptimizerHumanRobotWithMotors(OptimizerRobot):
    def __init__(
        self,
        N_DoF,
        root_link,
        kinDyn,
        contact_frame,
        minimum_number_parameters,
        link_name_list,
        part_name_list,
        kinDynHuman,
        kinDynBox,
        joint_name_list_robot,
        joint_name_list_human,
        number_of_points=3,
    ):
        self.joint_name_list_human = joint_name_list_human
        self.joint_name_list_robot = joint_name_list_robot
        self.kinDynHuman = kinDynHuman
        self.kinDynBox = kinDynBox
        self.compute_kin_din_box()
        self.compute_kin_din_human()
        super().__init__(
            N_DoF,
            root_link,
            kinDyn,
            contact_frame,
            minimum_number_parameters,
            link_name_list,
            part_name_list,
            number_of_points,
        )

    def define_search_variables(self):
        self.base_box = self.solver.variable(7, self.number_of_points)
        self.H_fun = eCubUtils.FromQuaternionToMatrix()
        self.f_tot = self.solver.variable(36, self.number_of_points)
        # Define the variable of the optimization problem
        self.s_human = self.solver.variable(
            self.kinDynHuman.NDoF, self.number_of_points
        )  # joint positions
        self.quat_pose_b_human = self.solver.variable(
            7, self.number_of_points
        )  # base pose as quaternion
        self.tau_human = self.solver.variable(
            self.kinDynHuman.NDoF, self.number_of_points
        )
        tot_N_DoF = self.kinDynHuman.NDoF + self.N_DoF + 18
        # self.nu_dot = self.solver.variable(tot_N_DoF, self.number_of_points)

    def add_wrench_constraint_human_robot(
        self, f_i, s_i_robot, s_i_human, H_b_i_robot, H_b_i_human
    ):
        f_i_robot_foot_left = f_i[:6]
        f_i_robot_foot_right = f_i[6:12]
        f_i_human_foot_left = f_i[12:18]
        f_i_human_foot_right = f_i[18:24]
        f_i_robot_hand_left = f_i[24:27]
        f_i_robot_hand_right = f_i[27:30]
        f_i_human_hand_left = f_i[30:33]
        f_i_human_hand_right = f_i[33:]
        H_left_foot_robot = self.H_left_foot(
            H_b_i_robot, s_i_robot, self.density_vector, self.lenght_multiplier_vector
        )
        H_right_foot_robot = self.H_right_foot(
            H_b_i_robot, s_i_robot, self.density_vector, self.lenght_multiplier_vector
        )
        H_left_foot_human = self.H_left_foot_human_fun(H_b_i_human, s_i_human)
        H_right_foot_human = self.H_right_foot_human_fun(H_b_i_human, s_i_human)
        H_left_hand_robot = self.H_left_hand(
            H_b_i_robot, s_i_robot, self.density_vector, self.lenght_multiplier_vector
        )
        H_right_hand_robot = self.H_right_hand(
            H_b_i_robot, s_i_robot, self.density_vector, self.lenght_multiplier_vector
        )
        H_left_hand_human = self.H_left_hand_human_fun(H_b_i_human, s_i_human)
        H_right_hand_human = self.H_right_hand_human_fun(H_b_i_human, s_i_human)
        self.add_whrenches_constraint(
            self.get_local_wrench(f_i_robot_foot_left, H_left_foot_robot[:3, :3])
        )
        self.add_whrenches_constraint(
            self.get_local_wrench(f_i_robot_foot_right, H_right_foot_robot[:3, :3])
        )
        self.add_whrenches_constraint(
            self.get_local_wrench(f_i_human_foot_left, H_left_foot_human[:3, :3])
        )
        self.add_whrenches_constraint(
            self.get_local_wrench(f_i_human_foot_right, H_right_foot_human[:3, :3])
        )

        ##for the hands adding zero for the torques since we are not considering them
        f_right_hand_human_temp = cs.vertcat(f_i_human_hand_right, np.zeros([3, 1]))
        f_left_hand_human_temp = cs.vertcat(f_i_human_hand_left, np.zeros([3, 1]))
        f_right_hand_robot_temp = cs.vertcat(f_i_robot_hand_right, np.zeros([3, 1]))
        f_left_hand_robot_temp = cs.vertcat(f_i_robot_hand_left, np.zeros([3, 1]))

        self.add_whrenches_constraint(
            self.get_local_wrench(f_right_hand_human_temp, H_right_hand_human[:3, :3]),
            True,
            True,
        )
        self.add_whrenches_constraint(
            self.get_local_wrench(f_left_hand_human_temp, H_left_hand_human[:3, :3]),
            True,
            True,
        )
        self.add_whrenches_constraint(
            self.get_local_wrench(f_right_hand_robot_temp, H_right_hand_robot[:3, :3]),
            True,
            True,
        )
        self.add_whrenches_constraint(
            self.get_local_wrench(f_left_hand_robot_temp, H_left_hand_robot[:3, :3]),
            True,
            True,
        )
        f_left_hand_human_local = self.get_local_wrench(
            f_left_hand_human_temp, H_left_hand_human[:3, :3]
        )
        f_right_hand_human_local = self.get_local_wrench(
            f_right_hand_human_temp, H_right_hand_human[:3, :3]
        )

        self.cost_function += 0.01 * cs.sumsqr(f_i[:30])
        self.cost_function += 0.1 * cs.sumsqr(f_i[30:])

    def compute_kin_din_human(self):
        # Frames of Interest
        right_hand_frame = "RightHand"
        left_hand_frame = "LeftHand"
        left_foot_frame = "LeftToe"
        right_foot_frame = "RightToe"
        frame_heigth = "head"

        self.M_human_fun = self.kinDynHuman.mass_matrix_fun()
        self.J_left_foot_human_fun = self.kinDynHuman.jacobian_fun(left_foot_frame)
        self.J_right_foot_human_fun = self.kinDynHuman.jacobian_fun(right_foot_frame)
        self.J_right_hand_human_fun = self.kinDynHuman.jacobian_fun(right_hand_frame)
        self.J_left_hand_human_fun = self.kinDynHuman.jacobian_fun(left_hand_frame)
        self.g_human_fun = self.kinDynHuman.gravity_term_fun()
        self.B_human = cs.vertcat(
            cs.MX.zeros(6, self.kinDynHuman.NDoF), cs.MX.eye(self.kinDynHuman.NDoF)
        )
        self.H_right_hand_human_fun = self.kinDynHuman.forward_kinematics_fun(
            right_hand_frame
        )
        self.H_left_hand_human_fun = self.kinDynHuman.forward_kinematics_fun(
            left_hand_frame
        )
        self.H_left_foot_human_fun = self.kinDynHuman.forward_kinematics_fun(
            left_foot_frame
        )
        self.H_right_foot_human_fun = self.kinDynHuman.forward_kinematics_fun(
            right_foot_frame
        )
        self.w_H_torso_human_fun = self.kinDynHuman.forward_kinematics_fun(
            self.kinDynHuman.root_link
        )
        self.H_b_fun_human_fun = eCubUtils.FromQuaternionToMatrix()
        self.CoM_human_fun = self.kinDynHuman.CoM_position_fun()
        self.total_mass_human_fun = self.kinDynHuman.get_total_mass()

    def compute_kin_din_box(self):

        box_frame_side_2_left = "side2_left_dummy_link"
        box_frame_side_2_right = "side2_right_dummy_link"
        box_frame_side_1_left = "side1_left_dummy_link"
        box_frame_side_1_right = "side1_right_dummy_link"

        self.M_box_fun = self.kinDynBox.mass_matrix_fun()
        self.H_box_fun = self.kinDynBox.forward_kinematics_fun("base_link")
        self.J_side_1_left_box_fun = self.kinDynBox.jacobian_fun(box_frame_side_1_left)
        self.J_side_1_right_box_fun = self.kinDynBox.jacobian_fun(
            box_frame_side_1_right
        )

        self.J_side_2_left_box_fun = self.kinDynBox.jacobian_fun(box_frame_side_2_left)
        self.J_side_2_right_box_fun = self.kinDynBox.jacobian_fun(
            box_frame_side_2_right
        )

        self.g_box_fun = self.kinDynBox.gravity_term_fun()

    def compute_total_matrices(
        self,
        s_robot,
        H_b_i_robot,
        s_human,
        H_b_i_human,
        quat_box,
        reflectedInertia,
        reductionRatio,
    ):

        H_b_i_box = self.H_fun(quat_box)
        N_DoF_human = self.kinDynHuman.NDoF
        N_DoF_robot = self.N_DoF

        # Computing Mass Matrix  Coupled
        # Mass matrix robot
        M_param_no_inertia = self.M(
            H_b_i_robot, s_robot, self.density_vector, self.lenght_multiplier_vector
        )
        M_param, Gamma = self.add_reflected_inertia(
            M_param_no_inertia, reflectedInertia, reductionRatio
        )
        Gamma_inv_T = cs.solve(
            cs.transpose(Gamma), cs.MX.eye(cs.transpose(Gamma).size1())
        )
        M_robot = M_param
        M_human = self.M_human_fun(H_b_i_human, s_human)
        M_box = self.M_box_fun(H_b_i_box, [])
        M = cs.vertcat(
            cs.horzcat(
                M_robot,
                np.zeros([N_DoF_robot + 6, N_DoF_human + 6]),
                np.zeros([N_DoF_robot + 6, 6]),
            ),
            cs.horzcat(
                np.zeros([N_DoF_human + 6, N_DoF_robot + 6]),
                M_human,
                np.zeros([N_DoF_human + 6, 6]),
            ),
            cs.horzcat(
                np.zeros([6, N_DoF_robot + 6]), np.zeros([6, N_DoF_human + 6]), M_box
            ),
        )

        ## Computing gravity effect coupled sumsqr
        g_human = self.g_human_fun(H_b_i_human, s_human)
        g_robot = self.g(
            H_b_i_robot, s_robot, self.density_vector, self.lenght_multiplier_vector
        )
        g_box = self.g_box_fun(H_b_i_box, [])
        g_temp = cs.vertcat(g_robot, g_human)
        g_out = cs.vertcat(g_temp, g_box)

        ## Computing B coupled
        B = cs.vertcat(
            cs.horzcat(self.B @ Gamma_inv_T, np.zeros([N_DoF_robot + 6, N_DoF_human])),
            cs.horzcat(np.zeros([N_DoF_human + 6, N_DoF_robot]), self.B_human),
        )
        B = cs.vertcat(B, np.zeros([6, N_DoF_robot + N_DoF_human]))

        ## Computing J coupled
        ## Human
        J_right_hand_human = self.J_right_hand_human_fun(H_b_i_human, s_human)
        J_left_hand_human = self.J_left_hand_human_fun(H_b_i_human, s_human)
        J_left_foot_human = self.J_left_foot_human_fun(H_b_i_human, s_human)
        J_right_foot_human = self.J_right_foot_human_fun(H_b_i_human, s_human)
        J_contact_feet_human = cs.vertcat(J_left_foot_human, J_right_foot_human)
        J_contact_hands_human = cs.vertcat(
            J_left_hand_human[:3, :], J_right_hand_human[:3, :]
        )

        ## Robot
        J_right_hand_robot = self.J_right_hand(
            H_b_i_robot, s_robot, self.density_vector, self.lenght_multiplier_vector
        )
        J_left_hand_robot = self.J_left_hand(
            H_b_i_robot, s_robot, self.density_vector, self.lenght_multiplier_vector
        )
        J_right_foot_robot = self.J_right(
            H_b_i_robot, s_robot, self.density_vector, self.lenght_multiplier_vector
        )
        J_left_foot_robot = self.J_left(
            H_b_i_robot, s_robot, self.density_vector, self.lenght_multiplier_vector
        )
        J_contact_feet_robot = cs.vertcat(J_left_foot_robot, J_right_foot_robot)
        J_contact_hand_robot = cs.vertcat(
            J_left_hand_robot[:3, :], J_right_hand_robot[:3, :]
        )

        ## Box
        J_side_1_left_box = self.J_side_1_left_box_fun(H_b_i_box, [])
        J_side_1_right_box = self.J_side_1_right_box_fun(H_b_i_box, [])
        J_side_2_left_box = self.J_side_2_left_box_fun(H_b_i_box, [])
        J_side_2_right_box = self.J_side_2_right_box_fun(H_b_i_box, [])

        J_side_1_box = cs.vertcat(J_side_1_left_box[:3, :], J_side_1_right_box[:3, :])
        J_side_2_box = cs.vertcat(J_side_2_left_box[:3, :], J_side_2_right_box[:3, :])

        ## J_tot
        J_first_line = cs.horzcat(
            J_contact_feet_robot, np.zeros([12, N_DoF_human + 6]), np.zeros([12, 6])
        )
        J_second_line = cs.horzcat(
            np.zeros([12, N_DoF_robot + 6]), J_contact_feet_human, np.zeros([12, 6])
        )
        J_third_line = cs.horzcat(
            J_contact_hand_robot, np.zeros([6, N_DoF_human + 6]), -J_side_1_box
        )
        J_fourth_line = cs.horzcat(
            np.zeros([6, N_DoF_robot + 6]), J_contact_hands_human, -J_side_2_box
        )
        J = cs.vertcat(J_first_line, J_second_line, J_third_line, J_fourth_line)

        return M, g_out, B, J

    def add_contact(self, s_robot, H_b_i_robot, s_human, H_b_i_human, quat_box):
        H_b_i_box = self.H_fun(quat_box)
        N_DoF_human = self.kinDynHuman.NDoF
        N_DoF_robot = self.N_DoF
        box_frame_side_2_left = "side2_left_dummy_link"
        box_frame_side_2_right = "side2_right_dummy_link"
        box_frame_side_1_left = "side1_left_dummy_link"
        box_frame_side_1_right = "side1_right_dummy_link"

        human_frame_right = "RightHand"
        human_frame_left = "LeftHand"

        robot_frame_right = "l_hand"
        robot_frame_left = "r_hand"

        H_right_human = self.kinDynHuman.forward_kinematics_fun(human_frame_right)
        H_left_human = self.kinDynHuman.forward_kinematics_fun(human_frame_left)
        H_right_robot = self.kinDyn.forward_kinematics_fun(robot_frame_right)
        H_left_robot = self.kinDyn.forward_kinematics_fun(robot_frame_left)

        H_side_1_left_box = self.kinDynBox.forward_kinematics_fun(box_frame_side_1_left)
        H_side_1_right_box = self.kinDynBox.forward_kinematics_fun(
            box_frame_side_1_right
        )
        H_side_2_left_box = self.kinDynBox.forward_kinematics_fun(box_frame_side_2_left)
        H_side_2_right_box = self.kinDynBox.forward_kinematics_fun(
            box_frame_side_2_right
        )
        H_right_human_param = H_right_human(H_b_i_human, s_human)
        H_left_human_param = H_left_human(H_b_i_human, s_human)

        H_right_robot_param = H_right_robot(
            H_b_i_robot, s_robot, self.density_vector, self.lenght_multiplier_vector
        )
        H_left_robot_param = H_left_robot(
            H_b_i_robot, s_robot, self.density_vector, self.lenght_multiplier_vector
        )

        H_side_1_left_box_param = H_side_1_left_box(H_b_i_box, [])
        H_side_1_right_box_param = H_side_1_right_box(H_b_i_box, [])

        H_side_2_left_box_param = H_side_2_left_box(H_b_i_box, [])
        H_side_2_right_box_param = H_side_2_right_box(H_b_i_box, [])

        ## Contact Position are one equal to the other
        self.solver.subject_to(
            H_right_robot_param[:3, 3] == H_side_1_left_box_param[:3, 3]
        )
        self.solver.subject_to(
            H_left_robot_param[:3, 3] == H_side_1_right_box_param[:3, 3]
        )
        self.solver.subject_to(
            H_right_human_param[:3, 3] == H_side_2_left_box_param[:3, 3]
        )
        self.solver.subject_to(
            H_left_human_param[:3, 3] == H_side_2_right_box_param[:3, 3]
        )

    def add_torque_minimization_both_agents(
        self,
        H_b_i_robot,
        s_robot_i,
        tau_robot_i,
        reflectedInertia,
        reductionRatio,
        H_b_i_human,
        s_human_i,
        tau_human_i,
        quat_box_i,
        f_i,
    ):
        M, g, B, J = self.compute_total_matrices(
            s_robot_i,
            H_b_i_robot,
            s_human_i,
            H_b_i_human,
            quat_box_i,
            reflectedInertia,
            reductionRatio,
        )
        tau = cs.vertcat(tau_robot_i, tau_human_i)
        self.cost_function += 0.01 * cs.sumsqr(tau)
        self.solver.subject_to(B @ tau + cs.transpose(J) @ f_i == g)

    def set_initial_for_human_and_box(
        self, quat_human_init, s_human_init, lenght_multiplier, densities
    ):
        self.s_human_init = s_human_init
        self.quat_b_initial_human = quat_human_init
        self.solver.set_initial(self.s_human, s_human_init)
        self.solver.set_initial(self.quat_pose_b_human, quat_human_init)
        self.f_initial = np.zeros([36, self.number_of_points])
        # print(self.total_mass(densities, lenght_multiplier)*9.81)
        for i in range(self.number_of_points):
            self.f_initial[2, i] = (
                self.total_mass(densities, lenght_multiplier) * 9.81
            ) / 2
            self.f_initial[8, i] = (
                self.total_mass(densities, lenght_multiplier) * 9.81
            ) / 2
            self.f_initial[14, i] = (self.total_mass_human_fun * 9.81) / 2
            self.f_initial[20, i] = (self.total_mass_human_fun * 9.81) / 2
            self.f_initial[26, i] = (self.kinDynBox.get_total_mass() * 9.81) / 4
            self.f_initial[29, i] = (self.kinDynBox.get_total_mass() * 9.81) / 4
            self.f_initial[32, i] = (self.kinDynBox.get_total_mass() * 9.81) / 4
            self.f_initial[35, i] = (self.kinDynBox.get_total_mass() * 9.81) / 4

        self.solver.set_initial(self.f_tot, self.f_initial)
        tot_N_DoF = self.kinDynHuman.NDoF + self.N_DoF + 18
        # self.solver.set_initial(self.nu_dot, np.zeros([tot_N_DoF,self.number_of_points]))

    def set_initial_torques(
        self, reflectedInertia, reductionRatio, lenght_multiplier_tot, densities
    ):
        torque_initial = []
        for i in range(self.number_of_points):
            s_robot_i = self.s_initial[:, i]
            s_human_i = self.s_human_init[:, i]
            H_b_i_human = self.base_pose_with_left_foot_human(s_human_i)
            w_H_torso_fun = self.kinDyn.forward_kinematics_fun(self.kinDyn.root_link)
            w_H_tors_num = w_H_torso_fun(
                np.eye(4), s_robot_i, densities, lenght_multiplier_tot
            )
            w_H_leftFoot = self.H_left_foot(
                np.eye(4), s_robot_i, densities, lenght_multiplier_tot
            )
            H_b_i_robot = cs.inv(w_H_leftFoot) @ w_H_tors_num
            quat_box_i = self.quat_box_initial[:, i]
            f_i = self.f_initial[:, i]
            # M,g,B,J = self.compute_total_matrices(s_robot_i,H_b_i_robot, s_human_i, H_b_i_human, quat_box_i, reflectedInertia, reductionRatio)
            H_b_i_box = self.H_fun(quat_box_i)
            N_DoF_human = self.kinDynHuman.NDoF
            N_DoF_robot = self.N_DoF

            # Computing Mass Matrix  Coupled
            # Mass matrix robot
            M_param_no_inertia = self.M(
                H_b_i_robot, s_robot_i, densities, lenght_multiplier_tot
            )
            M_param, Gamma = self.add_reflected_inertia(
                M_param_no_inertia, reflectedInertia, reductionRatio
            )
            # M_param = M_param_no_inertia
            Gamma_inv_T = cs.solve(
                cs.transpose(Gamma), cs.MX.eye(cs.transpose(Gamma).size1())
            )
            M_robot = M_param
            M_human = self.M_human_fun(H_b_i_human, s_human_i)
            M_box = self.M_box_fun(H_b_i_box, [])
            M = cs.vertcat(
                cs.horzcat(
                    M_robot,
                    np.zeros([N_DoF_robot + 6, N_DoF_human + 6]),
                    np.zeros([N_DoF_robot + 6, 6]),
                ),
                cs.horzcat(
                    np.zeros([N_DoF_human + 6, N_DoF_robot + 6]),
                    M_human,
                    np.zeros([N_DoF_human + 6, 6]),
                ),
                cs.horzcat(
                    np.zeros([6, N_DoF_robot + 6]),
                    np.zeros([6, N_DoF_human + 6]),
                    M_box,
                ),
            )
            ## Computing gravity effect coupled sumsqr
            g_human = self.g_human_fun(H_b_i_human, s_human_i)
            g_robot = self.g(H_b_i_robot, s_robot_i, densities, lenght_multiplier_tot)
            g_box = self.g_box_fun(H_b_i_box, [])
            g = cs.vertcat(g_robot, g_human, g_box)

            ## Computing B coupled
            B = cs.vertcat(
                cs.horzcat(
                    self.B @ Gamma_inv_T, np.zeros([N_DoF_robot + 6, N_DoF_human])
                ),
                cs.horzcat(np.zeros([N_DoF_human + 6, N_DoF_robot]), self.B_human),
            )
            B = cs.vertcat(B, np.zeros([6, N_DoF_robot + N_DoF_human]))

            ## Computing J coupled

            ## Human
            J_right_hand_human = self.J_right_hand_human_fun(H_b_i_human, s_human_i)
            J_left_hand_human = self.J_left_hand_human_fun(H_b_i_human, s_human_i)
            J_left_foot_human = self.J_left_foot_human_fun(H_b_i_human, s_human_i)
            J_right_foot_human = self.J_right_foot_human_fun(H_b_i_human, s_human_i)
            J_contact_feet_human = cs.vertcat(J_left_foot_human, J_right_foot_human)
            J_contact_hands_human = cs.vertcat(
                J_left_hand_human[:3, :], J_right_hand_human[:3, :]
            )

            ## Robot
            J_right_hand_robot = self.J_right_hand(
                H_b_i_robot, s_robot_i, densities, lenght_multiplier_tot
            )
            J_left_hand_robot = self.J_left_hand(
                H_b_i_robot, s_robot_i, densities, lenght_multiplier_tot
            )
            J_right_foot_robot = self.J_right(
                H_b_i_robot, s_robot_i, densities, lenght_multiplier_tot
            )
            J_left_foot_robot = self.J_left(
                H_b_i_robot, s_robot_i, densities, lenght_multiplier_tot
            )
            J_contact_feet_robot = cs.vertcat(J_left_foot_robot, J_right_foot_robot)
            J_contact_hand_robot = cs.vertcat(
                J_left_hand_robot[:3, :], J_right_hand_robot[:3, :]
            )

            ## Box
            J_side_1_left_box = self.J_side_1_left_box_fun(H_b_i_box, [])
            J_side_1_right_box = self.J_side_1_right_box_fun(H_b_i_box, [])

            J_side_2_left_box = self.J_side_2_left_box_fun(H_b_i_box, [])
            J_side_2_right_box = self.J_side_2_right_box_fun(H_b_i_box, [])

            J_side_1_box = cs.vertcat(
                J_side_1_left_box[:3, :], J_side_1_right_box[:3, :]
            )
            J_side_2_box = cs.vertcat(
                J_side_2_left_box[:3, :], J_side_2_right_box[:3, :]
            )

            ## J_tot
            J_first_line = cs.horzcat(
                J_contact_feet_robot, np.zeros([12, N_DoF_human + 6]), np.zeros([12, 6])
            )
            J_second_line = cs.horzcat(
                np.zeros([12, N_DoF_robot + 6]), J_contact_feet_human, np.zeros([12, 6])
            )
            J_third_line = cs.horzcat(
                J_contact_hand_robot, np.zeros([6, N_DoF_human + 6]), -J_side_1_box
            )
            J_fourth_line = cs.horzcat(
                np.zeros([6, N_DoF_robot + 6]), J_contact_hands_human, -J_side_2_box
            )
            J = cs.vertcat(J_first_line, J_second_line, J_third_line, J_fourth_line)
            JtF = np.transpose(np.array(J)) @ np.array(f_i) - g

            tau_initial_i = np.array(JtF[18:])
            torque_initial = cs.horzcat(torque_initial, tau_initial_i)
        tau_initial_robot = torque_initial[: self.N_DoF, :]
        tau_initial_human = torque_initial[self.N_DoF :, :]

    def populate_optimization_problem_both_agents(
        self, reflectedInertia, reductionRatio, motor_limits_dict
    ):
        self.add_box_constraint()
        self.add_constraint_human_feet()
        self.add_constraint_robot_feet()
        for i in range(self.number_of_points):
            s_i_robot = self.s[:, i]
            q_base_i_robot = self.quat_pose_b[:, i]
            H_b_i_robot = self.H_b_fun(q_base_i_robot)
            tau_i_robot = self.tau[:, i]
            s_i_human = self.s_human[:, i]
            f_i = self.f_tot[:, i]
            q_base_i_human = self.quat_pose_b_human[:, i]
            H_b_i_human = self.H_b_fun(q_base_i_human)
            tau_i_human = self.tau_human[:, i]
            q_box_i = self.base_box[:, i]
            H_b_box = self.H_b_fun(q_box_i)
            height_star = self.height_hands_start[i]
            self.add_intrinsic_constraint(
                q_base_i_robot, s_i_robot, tau_i_robot, motor_limits_dict
            )
            self.add_intrinsic_constraint_human(q_base_i_human, s_i_human)
            self.add_wrench_constraint_human_robot(
                f_i, s_i_robot, s_i_human, H_b_i_robot, H_b_i_human
            )
            self.add_constraint_kinematic_robot(H_b_i_robot, s_i_robot, height_star)
            self.add_kinematic_constraint_human(H_b_i_human, s_i_human, height_star)
            self.add_contact(s_i_robot, H_b_i_robot, s_i_human, H_b_i_human, q_box_i)
            self.add_torque_minimization_both_agents(
                H_b_i_robot,
                s_i_robot,
                tau_i_robot,
                reflectedInertia,
                reductionRatio,
                H_b_i_human,
                s_i_human,
                tau_i_human,
                q_box_i,
                f_i,
            )
            self.set_robot_joints_limits(s_i_robot)
            self.add_regularization_robot_joints()
            self.add_regularization_human_joints()
            self.fix_joints(s_i_robot=s_i_robot, s_i_human=s_i_human)

    def add_constraint_kinematic_robot(self, H_b_i_robot, s_i_robot, height_star):
        ## Feet
        H_param_left_Foot = self.H_left_foot(
            H_b_i_robot, s_i_robot, self.density_vector, self.lenght_multiplier_vector
        )
        H_param_right_foot = self.H_right_foot(
            H_b_i_robot, s_i_robot, self.density_vector, self.lenght_multiplier_vector
        )
        error_rot_left_foot = self.error_rot(H_param_left_Foot, self.H_left_foot_star)
        error_rot_right_foot = self.error_rot(
            H_param_right_foot, self.H_right_foot_star
        )
        z_alignment_error_fun = eCubUtils.zAxisAngle()
        error_z_left_foot = z_alignment_error_fun(H_param_left_Foot)
        error_z_right_foot = z_alignment_error_fun(H_param_right_foot)
        self.solver.subject_to(cs.sumsqr(error_rot_right_foot) == 0.0)
        self.solver.subject_to(cs.sumsqr(error_rot_left_foot) == 0.0)
        self.solver.subject_to(H_param_right_foot[2, 3] == 0.0)
        self.solver.subject_to(H_param_left_Foot[2, 3] == 0.0)
        leftFoot_H_right_foot = cs.inv(H_param_left_Foot) @ H_param_right_foot

        ## Hands

        weigth_right_hand = [0.0, 1.0]  # [rot,pos]
        weigth_left_hand = [0.0, 1.0]  # [rot,pos]

        # Target hands orientation
        # Right
        error_rot_right_hand = self.error_rot(
            self.H_right_hand(
                H_b_i_robot,
                s_i_robot,
                self.density_vector,
                self.lenght_multiplier_vector,
            ),
            self.H_right_hand_star,
        )
        cost_function_rotation_right_hand = weigth_right_hand[1] * cs.sumsqr(
            error_rot_right_hand
        )

        # Left
        error_rot_left_hand = self.error_rot(
            self.H_left_hand(
                H_b_i_robot,
                s_i_robot,
                self.density_vector,
                self.lenght_multiplier_vector,
            ),
            self.H_left_hand_star,
        )
        cost_function_rotation_left_hand = weigth_left_hand[1] * cs.sumsqr(
            error_rot_left_hand
        )
        # self.solver.subject_to(cs.sumsqr(error_rot_right_hand) == 0.0)
        # self.solver.subject_to(cs.sumsqr(error_rot_left_hand) == 0.0)
        # self.solver.subject_to(cost_function_rotation_left_hand == 0.0)
        # self.solver.subject_to(cost_function_rotation_right_hand == 0.0)
        # self.cost_function +=cost_function_rotation_left_hand + cost_function_rotation_right_hand

    def add_intrinsic_constraint_human(self, q_base_i_human, s_i_human):
        self.solver.subject_to(cs.sumsqr(q_base_i_human[:4]) == 1.0)
        self.solver.subject_to(s_i_human[9] == s_i_human[17])
        self.solver.subject_to(s_i_human[10] == s_i_human[18])
        self.solver.subject_to(s_i_human[12] == s_i_human[19])
        self.solver.subject_to(s_i_human[13] == s_i_human[20])
        self.solver.subject_to(s_i_human[23] == s_i_human[29])

    def add_constraint_human_feet(self):
        for i in range(self.number_of_points - 1):
            H_b_i_human = self.H_b_fun(self.quat_pose_b_human[:, i])
            H_b_i_plus_human = self.H_b_fun(self.quat_pose_b_human[:, i + 1])
            s_i_human = self.s_human[:, i]
            s_i_plus_human = self.s_human[:, i + 1]
            H_left_foot_i = self.H_left_foot_human_fun(H_b_i_human, s_i_human)
            H_left_foot_i_plus = self.H_left_foot_human_fun(
                H_b_i_plus_human, s_i_plus_human
            )
            H_right_foot_i = self.H_right_foot_human_fun(H_b_i_human, s_i_human)
            H_right_foot_i_plus = self.H_right_foot_human_fun(
                H_b_i_plus_human, s_i_plus_human
            )
            self.solver.subject_to(H_right_foot_i[:3, 3] == H_right_foot_i_plus[:3, 3])
            self.solver.subject_to(H_left_foot_i[:3, 3] == H_left_foot_i_plus[:3, 3])
            self.cost_function += cs.sumsqr(s_i_human - s_i_plus_human)

    def add_constraint_robot_feet(self):
        for i in range(self.number_of_points - 1):
            H_b_i_robot = self.H_b_fun(self.quat_pose_b[:, i])
            H_b_i_plus_robot = self.H_b_fun(self.quat_pose_b[:, i + 1])
            s_i_robot = self.s[:, i]
            s_i_plus_robot = self.s[:, i + 1]
            H_left_foot_i = self.H_left_foot(
                H_b_i_robot,
                s_i_robot,
                self.density_vector,
                self.lenght_multiplier_vector,
            )
            H_left_foot_i_plus = self.H_left_foot(
                H_b_i_plus_robot,
                s_i_plus_robot,
                self.density_vector,
                self.lenght_multiplier_vector,
            )
            H_right_foot_i = self.H_right_foot(
                H_b_i_robot,
                s_i_robot,
                self.density_vector,
                self.lenght_multiplier_vector,
            )
            H_right_foot_i_plus = self.H_right_foot(
                H_b_i_plus_robot,
                s_i_plus_robot,
                self.density_vector,
                self.lenght_multiplier_vector,
            )
            self.solver.subject_to(H_right_foot_i[:3, 3] == H_right_foot_i_plus[:3, 3])
            self.solver.subject_to(H_left_foot_i[:3, 3] == H_left_foot_i_plus[:3, 3])
            self.cost_function += cs.sumsqr(s_i_robot - s_i_plus_robot)

    def add_kinematic_constraint_human(self, H_b_i_human, s_i_human, height_star):

        weigth_right_hand = [0.0, 1.0]  # [rot,pos]
        weigth_left_hand = [0.0, 1.0]  # [rot,pos]

        # Target hands height
        # Right
        H_right_hands_param = self.H_right_hand_human_fun(H_b_i_human, s_i_human)
        error_right_hand_heights = cs.sumsqr(H_right_hands_param[2, 3] - height_star)

        # Target hands height
        # Left
        H_left_hands_param = self.H_left_hand_human_fun(H_b_i_human, s_i_human)
        error_left_hand_heights = cs.sumsqr(H_left_hands_param[2, 3] - height_star)
        error_left_hand_all = self.error_pos(H_left_hands_param, self.H_left_hand_star)

        # Target hands orientation
        # Right
        error_rot_right_hand = self.error_rot(
            self.H_right_hand_human_fun(H_b_i_human, s_i_human), self.H_right_hand_star
        )
        cost_function_rotation_right_hand = weigth_right_hand[1] * cs.sumsqr(
            error_rot_right_hand
        )
        # Left
        error_rot_left_hand = self.error_rot(
            self.H_left_hand_human_fun(H_b_i_human, s_i_human), self.H_left_hand_star
        )
        cost_function_rotation_left_hand = weigth_left_hand[1] * cs.sumsqr(
            error_rot_left_hand
        )
        # self.cost_function +=cost_function_rotation_left_hand + cost_function_rotation_right_hand
        ## Desired quantities
        # w_H_torso_num = self.w_H_torso_human_fun(np.eye(4), self.s_human_init[:,0])
        # w_H_lefFoot_num = self.H_left_foot_human_fun(np.eye(4), self.s_human_init[:,0])
        # w_H_init = cs.inv(w_H_lefFoot_num)@ w_H_torso_num
        w_H_init = self.H_b_fun(self.quat_b_initial_human[:, 0])
        w_H_lFoot_des = self.H_left_foot_human_fun(w_H_init, self.s_human_init[:, 0])
        w_H_rFoot_des = self.H_right_foot_human_fun(w_H_init, self.s_human_init[:, 0])
        # self.w_H_lHand_des = self.H_left_hand(w_H_init, self.s_human_init[:,0],density, lenght_multipliers)
        # self.w_H_rHand_des = self.H_right_hand(w_H_init, self.s_human_init[:,0], density, lenght_multipliers)

        # Feet
        H_param_left_Foot = self.H_left_foot_human_fun(H_b_i_human, s_i_human)
        H_param_right_foot = self.H_right_foot_human_fun(H_b_i_human, s_i_human)
        error_rot_left_foot = self.error_rot(H_param_left_Foot, w_H_lFoot_des)
        error_rot_right_foot = self.error_rot(H_param_right_foot, w_H_rFoot_des)
        self.solver.subject_to(cs.sumsqr(error_rot_left_foot) == 0.0)
        self.solver.subject_to(cs.sumsqr(error_rot_right_foot) == 0.0)
        # self.solver.subject_to(cs.sumsqr(error_rot_left_hand) == 0.0)
        # self.solver.subject_to(cs.sumsqr(error_rot_right_hand) == 0.0)
        z_alignment_error_fun = eCubUtils.zAxisAngle()
        error_left_foot = z_alignment_error_fun(H_param_left_Foot)
        error_right_foot = z_alignment_error_fun(H_param_right_foot)

        self.solver.subject_to(H_param_left_Foot[1, 3] == w_H_lFoot_des[1, 3])
        self.solver.subject_to(H_param_right_foot[1, 3] == w_H_rFoot_des[1, 3])
        self.solver.subject_to(H_param_left_Foot[0, 3] == w_H_lFoot_des[0, 3])
        self.solver.subject_to(H_param_right_foot[0, 3] == w_H_rFoot_des[0, 3])
        self.solver.subject_to(H_param_left_Foot[2, 3] == 0.0)
        self.solver.subject_to(H_param_right_foot[2, 3] == 0.0)
        left_foot_H_right_foot = cs.inv(H_param_left_Foot) @ H_param_right_foot
        # self.solver.subject_to(left_foot_H_right_foot[0,3]==0.0)
        # self.solver.subject_to(left_foot_H_right_foot[1,3]<0.1)
        # self.solver.subject_to(left_foot_H_right_foot[1,3]>-0.1)
        # self.cost_function += error_left_foo
        # self.cost_function += error_left_foot + error_right_foot
        # self.cost_function += cs.sumsqr(H_param_left_Foot[1,3]- w_H_lFoot_des[1,3]) + cs.sumsqr(H_param_right_foot[1,3]- w_H_rFoot_des[1,3])
        # self.solver.subject_to(error_left_foot == 0.0)
        # self.solver.subject_to(error_right_foot == 0.0)

    def add_box_constraint(self):

        quat_base_star = [
            9.99995289e-01,
            -8.99139270e-06,
            3.66289281e-07,
            3.06947462e-03,
            -4.16279437e-02,
            -1.86378435e-02,
            8.00000000e-01,
        ]
        quat_base = []
        for i in range(self.number_of_points):
            quat_base_temp = quat_base_star
            quat_base_temp[6] = self.height_hands_start[i]
            quat_base = cs.horzcat(quat_base, quat_base_temp)
        self.quat_box_initial = quat_base
        self.solver.set_initial(self.base_box, quat_base)
        H_base_star = self.H_fun(quat_base_star)
        height_star = self.height_hands_start
        for i in range(self.number_of_points):
            self.solver.subject_to(self.base_box[6, i] == height_star[i])
            self.solver.subject_to(cs.sumsqr(self.base_box[:4, i]) == 1.0)
            self.solver.subject_to(self.base_box[:4, i] == quat_base_star[:4])
        for i in range(self.number_of_points - 1):
            self.solver.subject_to(self.base_box[4, i] == self.base_box[4, i + 1])
            self.solver.subject_to(self.base_box[5, i] == self.base_box[5, i + 1])

    def base_pose_with_left_foot_human(self, s_i_human):

        w_H_torso_fun = self.kinDynHuman.forward_kinematics_fun(
            self.kinDynHuman.root_link
        )
        w_H_torso_num = w_H_torso_fun(np.eye(4), s_i_human)
        w_H_lefFoot_num = self.H_left_foot_human_fun(np.eye(4), s_i_human)
        w_H_b = cs.inv(w_H_lefFoot_num) @ w_H_torso_num
        return w_H_b

    def base_pose_with_left_foot_robot(self, s_i_robot):
        w_H_torso_fun = self.kinDyn.forward_kinematics_fun(self.kinDyn.root_link)
        w_H_tors_num = w_H_torso_fun(
            np.eye(4), s_i_robot, self.density_vector, self.lenght_multiplier_vector
        )
        w_H_leftFoot = self.H_left_foot(
            np.eye(4), s_i_robot, self.density_vector, self.lenght_multiplier_vector
        )
        w_H_b = cs.inv(w_H_leftFoot) @ w_H_tors_num
        return w_H_b

    def add_regularization_robot_joints(self):
        for i in range(self.number_of_points):
            s_i = self.s[:, i]
            s_i_ref = self.s_initial[:, i]
            quat_i = self.quat_pose_b[:, i]
            quat_i_ref = self.quat_b_initial[:, i]
            self.cost_function += 60 * cs.sumsqr(s_i - s_i_ref)
            self.cost_function += 100 * cs.sumsqr(quat_i - quat_i_ref)

    def add_regularization_human_joints(self):
        for i in range(self.number_of_points):
            s_i = self.s_human[:, i]
            s_i_ref = self.s_human_init[:, i]
            quat_i = self.quat_pose_b_human[:, i]
            quat_i_ref = self.quat_b_initial_human[:, i]
            self.cost_function += 60 * cs.sumsqr(s_i - s_i_ref)
            self.cost_function += 100 * cs.sumsqr(quat_i - quat_i_ref)

    def set_robot_joints_limits(self, s_i_robot):
        joint_limits = self.get_joint_limits_robot()
        name_limits = [
            "r_elbow",
            "l_elbow",
            "l_shoulder_yaw",
            "r_shoulder_yaw",
            "l_shoulder_roll",
            "r_shoulder_roll",
            "torso_pitch",
            "l_knee",
            "r_knee",
        ]
        for i in range(self.N_DoF):
            joint_name = self.joint_name_list_robot[i]
            limits = joint_limits[joint_name]
            lower_bound = limits[0]
            upper_bound = limits[1]
            if joint_name in name_limits:
                self.solver.subject_to(s_i_robot[i] < upper_bound)
                self.solver.subject_to(s_i_robot[i] > lower_bound)

    def get_joint_limits_robot(self):
        joints_limits = {
            "torso_pitch": [-0.3141592653589793, 0.5853981633974483],
            "torso_roll": [-0.4014257279586958, 0.4014257279586958],
            "torso_yaw": [-0.7504915783575618, 0.7504915783575618],
            "l_shoulder_pitch": [-1.53588974175501, 0.22689280275926285],
            "l_shoulder_roll": [0.20943951023931956, 2.8448866807507573],
            "l_shoulder_yaw": [-0.8726646259971648, 1.3962634015954636],
            "l_elbow": [0.3, 1.3089969389957472],
            "l_wrist_prosup": [-3.14, 3, 14],
            "r_shoulder_pitch": [-1.53588974175501, 0.22689280275926285],
            "r_shoulder_roll": [0.20943951023931956, 2.8448866807507573],
            "r_shoulder_yaw": [-0.8726646259971648, 1.3962634015954636],
            "r_elbow": [0.3, 1.3089969389957472],
            "r_wrist_prosup": [-3.14, 3, 14],
            "l_hip_pitch": [-0.7853981633974483, 2.007128639793479],
            "l_hip_roll": [-0.17453292519943295, 2.007128639793479],
            "l_hip_yaw": [-1.3962634015954636, 1.3962634015954636],
            "l_knee": [-1.2, -0.4],
            "l_ankle_pitch": [-0.7853981633974483, 0.7853981633974483],
            "l_ankle_roll": [-0.4363323129985824, 0.4363323129985824],
            "r_hip_pitch": [-0.7853981633974483, 2.007128639793479],
            "r_hip_roll": [-0.17453292519943295, 2.007128639793479],
            "r_hip_yaw": [-1.3962634015954636, 1.3962634015954636],
            "r_knee": [-1.2, -0.4],
            "r_ankle_pitch": [-0.7853981633974483, 0.7853981633974483],
            "r_ankle_roll": [-0.4363323129985824, 0.4363323129985824],
        }
        return joints_limits

    def get_joint_limits_human(self):
        joints_limits = {
            "jL5S1_rotx": [-0.610865, 0.610865],
            "jL5S1_roty": [-0.523599, 0.3],  # 1.309],
            "jL4L3_rotx": [-0.610865, 0.610865],
            "jL4L3_roty": [-0.523599, 0.3],  # 1.309],
            "jL1T12_rotx": [-0.610865, 0.610865],
            "jL1T12_roty": [-0.523599, 0.3],  # 1.309],
            "jT9T8_rotx": [-0.349066, 0.349066],
            "jT9T8_roty": [-0.261799, 0.3],  # 0.698132],
            "jT9T8_rotz": [-0.610865, 0.610865],
            "jT1C7_rotx": [-0.610865, 0.610865],
            "jT1C7_roty": [-0.959931, 0.3],  # 1.5708],
            "jT1C7_rotz": [-1.22173, 1.22173],
            "jC1Head_rotx": [-0.610865, 0.610865],
            "jC1Head_roty": [-0.436332, 0.174533],
            "jRightC7Shoulder_rotx": [-0.785398, 0.0872665],
            "jRightShoulder_rotx": [-2.35619, 1.5708],
            "jRightShoulder_roty": [-1.5708, 1.5708],
            "jRightShoulder_rotz": [-0.785398, 3.14159],
            "jRightElbow_roty": [-1.5708, 1.48353],
            "jRightElbow_rotz": [0, 2.53073],
            "jRightWrist_rotx": [-0.872665, 1.0472],
            "jRightWrist_rotz": [-0.523599, 0.349066],
            "jLeftC7Shoulder_rotx": [-0.0872665, 0.785398],
            "jLeftShoulder_rotx": [-1.5708, 2.35619],
            "jLeftShoulder_roty": [-1.5708, 1.5708],
            "jLeftShoulder_rotz": [-3.14159, 0.785398],
            "jLeftElbow_roty": [-1.5708, 1.48353],
            "jLeftElbow_rotz": [-2.53073, 0.0],
            "jLeftWrist_rotx": [-1.0472, 0.872665],
            "jLeftWrist_rotz": [-0.349066, 0.523599],
            "jRightHip_rotx": [-0.785398, 0.523599],
            "jRightHip_roty": [-2.0, -0.261799],  # [-0.261799, 2.0944],
            "jRightHip_rotz": [-0.785398, 0.785398],
            "jRightKnee_roty": [0.01, 2.35619],
            "jRightKnee_rotz": [-0.2, 0.2],  # [-0.698132, 0.523599],
            "jRightAnkle_rotx": [0, 0.785398],  # [-0.610865, 0.785398],
            "jRightAnkle_roty": [-0.523599, 0.872665],
            "jRightAnkle_rotz": [-0.436332, 0.872665],
            "jLeftHip_rotx": [-0.523599, 0.785398],
            "jLeftHip_roty": [-1.0, -0.261799],  # [-0.261799, 2.0944],
            "jLeftHip_rotz": [-0.785398, 0.785398],
            "jLeftKnee_roty": [0.01, 2.35619],
            "jLeftKnee_rotz": [-0.2, 0.2],  # [-0.523599, 0.698132],
            "jLeftAnkle_rotx": [-0.785398, 0],  # [-0.785398, 0.610865],
            "jLeftAnkle_roty": [-0.523599, 0.872665],
            "jLeftAnkle_rotz": [-0.872665, 0.436332],
        }

        return joints_limits

    def fix_joints(self, s_i_robot, s_i_human):
        self.solver.subject_to(s_i_robot[7] == 0.0)
        self.solver.subject_to(s_i_robot[12] == 0.0)
        self.solver.subject_to(s_i_human[19] == 0.0)
        self.solver.subject_to(s_i_human[20] == 0.0)
        self.solver.subject_to(s_i_human[11] == 0.0)
        self.solver.subject_to(s_i_human[12] == 0.0)
