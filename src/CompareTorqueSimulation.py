import pickle
import numpy as np
import matplotlib.pyplot as plt
import os 

common_path = os.path.dirname(os.path.abspath(__file__))
torque_original = pickle.load(open(common_path+"/results/Simulation/torque_robot_NL.p", "rb"))
torque_best = pickle.load(open(common_path+"/results/Simulation/torque_robot_GA.p", "rb"))

mean_robot_new = np.zeros(3)
var_robot_new = np.zeros(3)
mean_robot_old = np.zeros(3)
var_robot_old = np.zeros(3)

for i in range(3):
    torque_original_i = torque_original[:, i]
    torque_best_i = torque_best[:, i]
    mean_robot_new[i] = np.mean(torque_best_i)
    var_robot_new[i] = np.var(torque_best_i)
    mean_robot_old[i] = np.mean(torque_original_i)
    var_robot_old[i] = np.var(torque_original_i)

plt.rcParams.update({"font.size": 40})
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"lines.linewidth": 300})
plt.rcParams.update({"axes.linewidth": 2})
plt.rcParams.update({"xtick.labelsize": 40})
plt.rcParams.update({"ytick.labelsize": 40})

total_robot_torque_best_abs = abs(torque_best)
total_robot_torques_old_abs = abs(torque_original)
total_robot_torques_new = total_robot_torque_best_abs.sum()
total_robot_torques_old = total_robot_torques_old_abs.sum()

string_new = "GA"
string_old = "NL"
offset_string = 0.001
offset_center = 0.175
legend_1 = "Load at 0.8 m"
legend_2 = "Load at 1.0 m"
legend_3 = "Load at 1.2 m"

plt.figure

y_axis_1 = abs(mean_robot_new[0]) * np.ones(10)
x_1_new = np.linspace(2, 3, 10)

plt.plot(x_1_new, y_axis_1, label="_nolegend_", color="red", linewidth="3")
plt.text(
    x_1_new[2] + offset_center,
    y_axis_1[0] + var_robot_new[0] + offset_string,
    string_new,
)
plt.fill_between(
    x_1_new,
    y_axis_1 - var_robot_new[0],
    y_axis_1 + var_robot_new[0],
    alpha=0.2,
    label=legend_1,
    color="red",
)


x_1_old = np.linspace(0.5, 1.5, 10)
y_axis_old_1 = abs(mean_robot_old[0]) * np.ones(10)
plt.text(
    x_1_old[2] + offset_center,
    y_axis_old_1[0] + var_robot_old[0] + offset_string,
    string_old,
)
plt.plot(x_1_old, y_axis_old_1, label="_nolegend_", color="red", linewidth="3")
plt.fill_between(
    x_1_old,
    y_axis_old_1 - var_robot_old[0],
    y_axis_old_1 + var_robot_old[0],
    alpha=0.2,
    label="_nolegend_",
    color="red",
)

x_2_new = np.linspace(5.0, 6.0, 10)
y_axis_2 = abs(mean_robot_new[1]) * np.ones(10)
plt.plot(x_2_new, y_axis_2, label="_nolegend_", color="green", linewidth="3")
plt.text(
    x_2_new[2] + offset_center,
    y_axis_2[0] + var_robot_new[1] + offset_string,
    string_new,
)
plt.fill_between(
    x_2_new,
    y_axis_2 - var_robot_new[1],
    y_axis_2 + var_robot_new[1],
    alpha=0.2,
    label=legend_2,
    color="green",
)

y_axis_old_2 = abs(mean_robot_old)[1] * np.ones(10)
x_2_old = np.linspace(3.5, 4.5, 10)

plt.plot(x_2_old, y_axis_old_2, label="_nolegend_", color="green", linewidth="3")
plt.text(
    x_2_old[2] + offset_center,
    y_axis_old_2[0] + var_robot_old[1] + offset_string,
    string_old,
)
plt.fill_between(
    x_2_old,
    y_axis_old_2 - var_robot_old[1],
    y_axis_old_2 + var_robot_old[1],
    alpha=0.2,
    label="_nolegend_",
    color="green",
)

x_3_new = np.linspace(8.0, 9.0, 10)
y_axis_3 = abs(mean_robot_new[2]) * np.ones(10)

plt.plot(x_3_new, y_axis_3, label="_nolegend_", color="blue", linewidth="3")
plt.text(
    x_3_new[2] + offset_center,
    y_axis_3[0] + var_robot_new[2] + offset_string,
    string_new,
)
plt.fill_between(
    x_3_new,
    y_axis_3 - var_robot_new[2],
    y_axis_3 + var_robot_new[2],
    alpha=0.2,
    label=legend_3,
    color="blue",
)

x_3_old = np.linspace(6.5, 7.5, 10)
y_axis_old_3 = abs(mean_robot_old[2]) * np.ones(10)

plt.plot(x_3_old, y_axis_old_3, label="_nolegend_", color="blue", linewidth="3")

plt.text(
    x_3_old[2] + offset_center,
    y_axis_old_3[0] + var_robot_old[2] + offset_string,
    string_old,
)
plt.fill_between(
    x_3_old,
    y_axis_old_3 - var_robot_old[2],
    y_axis_old_3 + var_robot_old[2],
    alpha=0.2,
    label="_nolegend_",
    color="blue",
)

plt.title("Robot", fontsize="60")
plt.legend(loc="upper right", fontsize="30")
plt.xticks([])
plt.ylabel(r"$\displaystyle |\tau_m| \left[Nm\right]$", fontsize="60")
# plt.set_xticklabels([])
plt.show()


torque_original = pickle.load(open(common_path+"/results/Simulation/torque_human_NL.p", "rb"))
torque_best = pickle.load(open(common_path+"/results/Simulation/torque_human_GA.p", "rb"))

mean_human_new = np.zeros(3)
var_human_new = np.zeros(3)
mean_human_old = np.zeros(3)
var_human_old = np.zeros(3)

for i in range(3):
    torque_original_i = torque_original[:, i]
    torque_best_i = torque_best[:, i]
    mean_human_new[i] = np.mean((torque_best_i))
    var_human_new[i] = np.var((torque_best_i))
    mean_human_old[i] = np.mean((torque_original_i))
    var_human_old[i] = np.var((torque_original_i))

total_human_torque_best_abs = abs(torque_best)
total_human_torques_old_abs = abs(torque_original)
total_human_torques_new = total_human_torque_best_abs.sum()
total_human_torques_old = total_human_torques_old_abs.sum()
plt.figure
offset_string = 5
offset_center = +0.175
y_axis_1 = abs(mean_human_new[0]) * np.ones(10)
x_1_new = np.linspace(2, 3, 10)

plt.plot(x_1_new, y_axis_1, label="_nolegend_", color="red", linewidth="3")
plt.text(
    x_1_new[2] + offset_center,
    y_axis_1[0] + var_human_new[0] + offset_string,
    string_new,
)
plt.fill_between(
    x_1_new,
    y_axis_1 - var_human_new[0],
    y_axis_1 + var_human_new[0],
    alpha=0.2,
    label=legend_1,
    color="red",
)


x_1_old = np.linspace(0.5, 1.5, 10)
y_axis_old_1 = abs(mean_human_old[0]) * np.ones(10)
plt.text(
    x_1_old[2] + offset_center,
    y_axis_old_1[0] + var_human_old[0] + offset_string,
    string_old,
)
plt.plot(x_1_old, y_axis_old_1, label="_nolegend_", color="red", linewidth="2")
plt.fill_between(
    x_1_old,
    y_axis_old_1 - var_human_old[0],
    y_axis_old_1 + var_human_old[0],
    alpha=0.2,
    label="_nolegend_",
    color="red",
)

x_2_new = np.linspace(5.0, 6.0, 10)
y_axis_2 = abs(mean_human_new[1]) * np.ones(10)
plt.plot(x_2_new, y_axis_2, label="_nolegend_", color="green", linewidth="3")
plt.text(
    x_2_new[2] + offset_center,
    y_axis_2[0] + var_human_new[1] + offset_string,
    string_new,
)
plt.fill_between(
    x_2_new,
    y_axis_2 - var_human_new[1],
    y_axis_2 + var_human_new[1],
    alpha=0.2,
    label=legend_2,
    color="green",
)

y_axis_old_2 = mean_human_old[1] * np.ones(10)
x_2_old = np.linspace(3.5, 4.5, 10)

plt.plot(x_2_old, y_axis_old_2, label="_nolegend_", color="green", linewidth="3")
plt.text(
    x_2_old[2] + offset_center,
    y_axis_old_2[0] + var_human_old[1] + offset_string,
    string_old,
)
plt.fill_between(
    x_2_old,
    y_axis_old_2 - var_human_old[1],
    y_axis_old_2 + var_human_old[1],
    alpha=0.2,
    label="_nolegend_",
    color="green",
)

x_3_new = np.linspace(8.0, 9.0, 10)
y_axis_3 = abs(mean_human_new[2]) * np.ones(10)

plt.plot(x_3_new, y_axis_3, label="_nolegend_", color="blue", linewidth="3")
plt.text(
    x_3_new[2] + offset_center,
    y_axis_3[0] + var_human_new[2] + offset_string,
    string_new,
)
plt.fill_between(
    x_3_new,
    y_axis_3 - var_human_new[2],
    y_axis_3 + var_human_new[2],
    alpha=0.2,
    label=legend_3,
    color="blue",
)

x_3_old = np.linspace(6.5, 7.5, 10)
y_axis_old_3 = abs(mean_human_old[2]) * np.ones(10)

plt.plot(x_3_old, y_axis_old_3, label="_nolegend_", color="blue", linewidth="3")

plt.text(
    x_3_old[2] + offset_center,
    y_axis_old_3[0] + var_human_old[2] + offset_string,
    string_old,
)
plt.fill_between(
    x_3_old,
    y_axis_old_3 - var_human_old[2],
    y_axis_old_3 + var_human_old[2],
    alpha=0.2,
    label="_nolegend_",
    color="blue",
)

plt.title("Human", fontsize="60")
plt.legend(loc="upper right", fontsize="30")
plt.xticks([])
plt.ylabel(r"$\displaystyle |\tau | \left[Nm\right] $", fontsize="60")
plt.show()

# population_output = pickle.load(open("populationOutput.p","rb"))
# item_0 = population_output[0]
# item_1 = population_output[1]
# item_0[:8] = [0.9, 1.2, 1.6, 1.3, 0.6, 1.6, 1.0, 1.2]
# item_1[:8]= [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# item_1[8:16] = [2129.2952964,1199.07622408,893.10763518,626.60271872,1661.68632652,727.43130782, 600.50011475,2222.0327914 ]

# population_output[0] = item_0
# population_output[1] = item_1

# pickle.dump(population_output,open("populationInitial.p", "wb"))

print("new", total_robot_torques_new)
print("old", total_robot_torques_old)
print("%", (total_robot_torques_old * 100) / total_robot_torques_new)
print("%", (total_robot_torques_new * 100) / total_robot_torques_old)


diff_torques = abs(total_robot_torques_old - total_robot_torques_new)
print("% diff", (diff_torques * 100) / total_robot_torques_old)


print("Human")
print("new", total_human_torques_new)
print("old", total_human_torques_old)
print("%", (total_human_torques_old * 100) / total_human_torques_new)
print("%", (total_human_torques_new * 100) / total_human_torques_old)


diff_torques = abs(total_human_torques_old - total_human_torques_new)
print("% diff", (diff_torques * 100) / total_human_torques_old)
