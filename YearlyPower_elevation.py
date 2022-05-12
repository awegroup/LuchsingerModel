
#%% Imports, class definitions.
# =============================================================================
import numpy as np
from scipy import optimize as op
from scipy.integrate import quad
import scipy.interpolate
import matplotlib.pyplot as plt
import time
import seaborn as sns
from style import set_graph_style
import sys
from Cycle_elevation import *
from Environment import *
from System import *
# =============================================================================



def sol_calculation_new(s, e, sol_list1, graph=True, plot = True):
    start_time = time.time()
    energy_mwh_list =[]

    v_w_list = np.arange(s.cut_in_v_w, s.cut_out_v_w, 0.05)
    v_w_list1 = np.arange(s.cut_in_v_w, s.cut_out_v_w, 0.2)

    def power_per_vw(v_w,sol):
        # print(v_w, "vw_average_power")
        e.set_v_w(v_w)
        e.calculate_weibull_pdf(v_w,sol)
        c.run_simulation(s, e)
        return c.cycle_power, c.cycle_out_power

    for sol in sol_list1:
        # print("sol", sol)
        v_w_t_list = []
        v_w_p_list = []
        average_power_for_a_vw_list = []
        cycle_power_per_vw_list = []
        power_out_per_vw_list = []
        probability_list = []

        gamma_out_list = []
        gamma_in_list = []
        v_out_list = []
        v_in_list = []
        T_out_list = []
        T_in_list = []
        taus_g =[]
        taus_m =[]
        rpms_in =[]
        rpms_out =[]
        e.update_environment(sol)
        nominal_v_w_t = c.calculate_nominal_vw_t(s, v_w_list1, e.rho)
        nominal_v_w_p = c.calculate_nominal_vw_p(s, v_w_list1, e.rho)
        v_w_t_list.append(nominal_v_w_t)
        v_w_p_list.append(nominal_v_w_p)

        integrated_power1 =0
        for i in range(0, len(v_w_list)):
            delta_x = v_w_list[1] - v_w_list[0]

            cycle_power_per_vw_1, power_out_per_vw_1  = power_per_vw(v_w_list[i],sol)

            e.calculate_weibull_pdf(v_w_list[i],sol)
            average_power_for_a_vw = cycle_power_per_vw_1 *  e.g_w

            average_power_for_a_vw_list.append(average_power_for_a_vw/1000)
            cycle_power_per_vw_list.append(cycle_power_per_vw_1/1000)
            power_out_per_vw_list.append(power_out_per_vw_1/1000)

            gamma_out_list.append(c.gamma_out)
            gamma_in_list.append(c.gamma_in)
            v_out_list.append(c.v_out)
            v_in_list.append(c.v_in)
            probability_list.append(e.g_w)

            delta_x = v_w_list[1] - v_w_list[0]
            integrated_power1 += average_power_for_a_vw*delta_x

            T_out_list.append(c.tether_force_out)
            T_in_list.append(c.tether_force_in)
            taus_g.append(c.tau_g)
            taus_m.append(c.tau_m)
            rpms_in.append((c.v_in / s.drum_outer_radius) * 60 / (2 * np.pi))
            rpms_out.append((c.v_out / s.drum_outer_radius) * 60 / (2 * np.pi))


        torques_in = taus_m
        torques_out = taus_g

        # Calculate operational range for the generator
        gen_rpm_max = int(max(rpms_out))
        gen_rpm_min = int(min(rpms_out))

        gen_torque_max = int(max(torques_out))
        gen_torque_min = int(min(torques_out))

        # Calculate operational range for the motor
        mot_rpm_max = int(max(rpms_in))
        mot_rpm_min = int(min(rpms_in))

        mot_torque_max = int(max(torques_in))
        mot_torque_min = int(min(torques_in))
        # integrated_power = quad(average_power, s.cut_in_v_w, s.cut_out_v_w, limit=50, epsabs=1.49e-05, epsrel=1.49e-05)
        # print(integrated_power)
        # energy_mwh = integrated_power[0] * 24 * 1e-6
        energy_mwh1 = integrated_power1 * 24 * 1e-6

        # print(energy_mwh )
        # print(energy_mwh1 )
        # gs_mass = get_gs_mass(gen_torque_max, gen_rpm_max, mot_torque_max, mot_rpm_max)
        # system_mass = s.kite_mass + s.tether_mass + gs_mass + 5.3
        # print("System mass is {:.2f} kg.".format(system_mass))
        # print("Kite mass is {:.2f} kg.".format(s.kite_mass))

        end_time = time.time()
        print("Runtime of {:.1f} seconds.".format(end_time - start_time))
        # print("System produced {:.6f} MWh over the sols.".format(np.sum(energy_mwh)))
        print("System produced {:.6f} MWh over the sol(s).".format(np.sum(energy_mwh1)))
        print("System produces {:.6f} kW over the sol(s).".format(integrated_power1/1000))

        title = "Specifications of wind production for sol {:.0f}".format(sol)
        if graph:
            # set_graph_style()
            fig, ax = plt.subplots(5, 1, sharex=True)
            ax[0].set(title=title, ylabel="Kite power [kW]")
            sns.lineplot(x=v_w_list, y=cycle_power_per_vw_list, color="darkblue", ax=ax[0], label = r'$P_{c}$')
            sns.lineplot(x=v_w_list, y=power_out_per_vw_list, color="blue", ax=ax[0],  label = r'$P_{out}$')
            ax[0].axvline(nominal_v_w_t, 0, 1, color="k")
            ax[0].axvline(nominal_v_w_p, 0, 1, color="r")

            sns.lineplot(x=v_w_list, y=v_in_list, color="darkblue", ax=ax[1], label=r'$v_{in}$')
            sns.lineplot(x=v_w_list, y=v_out_list, color="blue", ax=ax[1], label=r'$v_{out}$')
            ax[1].axvline(nominal_v_w_t, 0, 1, color="k")
            ax[1].axvline(nominal_v_w_p, 0, 1, color="r")
            # ax[1].set(xlabel="Wind velocity", ylabel="[-]")
            plt.subplots_adjust(bottom=0.1)

            ax[2].axvline(nominal_v_w_t, 0, 1, color="k")
            ax[2].axvline(nominal_v_w_p, 0, 1, color="r")
            sns.lineplot(x=v_w_list, y=gamma_out_list, color="blue", ax=ax[2], label = r'$\gamma_{out}$')
            sns.lineplot(x=v_w_list, y=gamma_in_list, color="darkblue", ax=ax[2], label = r'$\gamma_{in}$')

            sns.lineplot(x=v_w_list, y=probability_list, color="darkblue", ax=ax[3])
            ax[3].set(ylabel="Probability [-]")

            sns.lineplot(x=v_w_list, y=average_power_for_a_vw_list, color="darkblue", ax=ax[4])
            ax[4].set(xlabel="Wind velocity", ylabel="Power times probability [kW]")
            energy_mwh_list.append(energy_mwh1)

    if plot == True:
        # PRINT OPERATIONAL RANGE
        print(
            f'The generator must operate at a rotational speed between {gen_rpm_max}-{gen_rpm_min} RPM and a torque between {gen_torque_max}-{gen_torque_min} Nm.\n')
        print(
            f'The motor must operate at a rotational speed between {mot_rpm_max}-{mot_rpm_min} RPM and a torque between {mot_torque_max}-{mot_torque_min} Nm.')

        # PLOT REELING SPEEDS AND (RE-)TRACTION FORCE AS A FUNCTION OF WIND SPEED
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(v_w_list, v_in_list, label="in")
        ax[0].plot(v_w_list, v_out_list, label="out")
        ax[0].legend()
        ax[1].plot(v_w_list, T_in_list, label="in")
        ax[1].plot(v_w_list, T_out_list, label="out")
        ax[1].legend()

        ax[0].set_ylabel('Reel speed [m/s]')
        ax[1].set_ylabel('Force [N]')
        ax[1].set_xlabel('Wind speed [m/s]')
        # PLOT ROTATIONAL SPEEDS AND TORQUES AS A FUNCTION OF WIND SPEED
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(v_w_list, rpms_in, label="in")
        ax[0].plot(v_w_list, rpms_out, label="out")
        ax[0].legend()
        ax[1].plot(v_w_list, torques_in, label="in", )
        ax[1].plot(v_w_list, torques_out, label="out")
        ax[1].legend()

        ax[0].set_ylabel('Rotational speed [RPM]')
        ax[1].set_ylabel('Torque [Nm]')
        ax[1].set_xlabel('Wind speed [m/s]')
        plt.show()
    return energy_mwh_list, gen_torque_max, gen_rpm_max, mot_torque_max, mot_rpm_max
    # return energy_mwh_list

#
# # Function definition to estimate the mass of the ground station based on the Alxion catalogues (alternators & direct drive).
# # Returns the total mass of the ground station. Brakes and spindle are fixed values that do not scale depending on the operational envelope.

def get_gs_mass(gen_torque_max, gen_rpm_max, mot_torque_max, mot_rpm_max):
    # Load technical information of generators and motors from Alxion catalogues
    generators = []
    motors = []

    with open('generators.csv') as f:
        for line in f:
            line = line.strip()
            line = line.split(',')
            line = [int(value) for value in line[:4]] + [float(line[-2])] + [line[-1]]
            generators.append(line)

    with open('motors.csv') as f:
        for line in f:
            line = line.strip()
            line = line.split(',')
            line = [int(round(float(value))) for value in line[:4]] + [float(line[-2])] + [line[-1]]
            motors.append(line)

    # Calculate mass of the generator part of the actuator
    generator_mass = 0

    if generator_mass == 0:
        for generator in generators:
            torque_range = range(int(generator[0]), int(generator[1]))
            rpm_range = range(int(generator[2]), int(generator[3]))

            if gen_torque_max in torque_range and gen_rpm_max in rpm_range and generator_mass == 0:
                generator_mass = generator[-2]
                print(f'The generator mass is estimated to be {round(generator_mass, 1)} kg.')
                print(f'The estimated generator mass corresponds to model {generator[-1]}.')

        for i in range(len(generators) - 1):
            torque_range = range(int(generators[i + 1][1]), int(generators[i][1]))
            rpm_range = range(int(generators[i + 1][2]), int(generators[i][3]))

            if gen_torque_max in torque_range and gen_rpm_max in rpm_range and generator_mass == 0:
                calculate_gen_mass = lambda tau: np.interp(tau, [generators[i + 1][1], \
                                                                 generators[i][1]],
                                                           [generators[i + 1][-2], generators[i][-2]])
                generator_mass = calculate_gen_mass(gen_torque_max)

                print(f'The generator mass is estimated to be {round(generator_mass, 1)} kg.')
                print(
                    f'The estimated generator mass is interpolated from models {generators[i + 1][-1]} and {generators[i][-1]}.\n')

    # Calculate mass of the motor part of the actuator
    motor_mass = 0

    if motor_mass == 0:
        for motor in motors:
            torque_range = range(int(motor[0]), int(motor[1]))
            rpm_range = range(int(motor[2]), int(motor[3]))

            if mot_torque_max in torque_range and mot_rpm_max in rpm_range and motor_mass == 0:
                motor_mass = motor[-2]
                print(f'The motor mass is estimated to be {round(motor_mass, 1)} kg.')
                print(f'The estimated motor mass corresponds to model {motor[-1]}.')

        for i in range(len(motors) - 1):
            torque_range = range(int(motors[i + 1][1]), int(motors[i][1]))
            rpm_range = range(int(motors[i + 1][2]), int(motors[i][3]))

            if mot_torque_max in torque_range and mot_rpm_max in rpm_range and motor_mass == 0:
                calculate_mot_mass = lambda tau: np.interp(tau, [motors[i + 1][1], \
                                                                 motors[i][1]], [motors[i + 1][-2], motors[i][-2]])
                motor_mass = calculate_mot_mass(gen_torque_max)

                print(f'The motor mass is estimated to be {round(motor_mass, 1)} kg.')
                print(f'The estimated motor mass is interpolated from models {motors[i + 1][-1]} and {motors[i][-1]}.')

                # If you want this code to calculate the mass of a ground station with a ludicrous operational envelope, it won't like it
    if generator_mass == 0 :
        print('The operational envelope of the motor station is unrealistic.')
        # raise Exception('The operational envelope of the motor station is unrealistic.')
    if motor_mass == 0:
        print('The operational envelope of the generator station is unrealistic.')
        # raise Exception('The operational envelope of the generator station is unrealistic.')
    # Specify masses of other elements in the ground station
    actuator_mass = generator_mass + motor_mass
    spindle_mass = 5.20 + 20
    drum_mass = s.drum_mass
    brake_mass = 3
    print("generator: {:.2f}, motor: {:.2f}, drum: {:.2f}".format(generator_mass, \
                                                                  motor_mass, drum_mass))
    return actuator_mass + spindle_mass + drum_mass + brake_mass


# Function definition to calculate the daily wind energy produced for the whole year.
# Also makes two .csv files and a pretty sexy graph
def annual_calculation(s, e, graph=True):
    start_time = time.time()
    energy_mwh1 = np.zeros(669)
    energy_mwh2 = np.zeros(669)
    v_w_0 = 20
    v_w_t_list = []
    v_w_p_list = []
    v_w_list = np.arange(s.cut_in_v_w,s.cut_out_v_w,0.05)
    v_w_list1 = np.arange(s.cut_in_v_w,s.cut_out_v_w,0.2)

    def average_power(v_w):
        # print(v_w,"vw_average_power")
        e.set_v_w(v_w)
        e.calculate_weibull_pdf(v_w,sol)
        c.run_simulation(s, e)
        return c.cycle_power*e.g_w
    
    for sol in e.sols:
        # print("sol",sol)
        e.update_environment(sol)
        nominal_v_w_t = c.calculate_nominal_vw_t(s, v_w_list1, e.rho)
        nominal_v_w_p = c.calculate_nominal_vw_p(s, v_w_list1,  e.rho)
        v_w_t_list.append(nominal_v_w_t)
        v_w_p_list.append(nominal_v_w_p)
        integrated_power1 = 0
        for i in range(0, len(v_w_list)):
            delta_x = v_w_list[1] - v_w_list[0]
            average_power_for_a_vw = average_power(v_w_list[i])
            integrated_power1 += average_power_for_a_vw*delta_x

        # integrated_power2 = quad(average_power, s.cut_in_v_w, s.cut_out_v_w, limit=50, epsabs=1.49e-05, epsrel=1.49e-05)
        # print(integrated_power1,"1")
        # print(integrated_power2,"2")

        energy_mwh1[sol] = integrated_power1*24*1e-6
        # energy_mwh2[sol] = integrated_power2[0] * 24 * 1e-6
        # energy_mwh1 *= np.cos(s.phi)**3
    power_sol = energy_mwh1 / 24 * 1e3
    # power_sol = integrated_power1
    name = "wind_energy_produced_AN_MCD_A" + str(s.projected_area) + "T" + str(
        round(s.nominal_tether_force, 0)) + "P" + str(round(s.nominal_generator_power, 0)) + "v" + str(
        round(s.reel_in_speed_limit, 0)) + "bi" + str(round(s.phi_in / np.pi * 180, 0)) + ".csv"

    np.savetxt(name, energy_mwh1, delimiter=",")

    np.savetxt("wind_power_produced_AN_MCD_A{:.1f}_elevation.csv".format(s.projected_area), power_sol, delimiter=",")

    energy_mwh = energy_mwh1

    end_time = time.time()
    print("Runtime of {:.1f} seconds.".format(end_time - start_time))
    print("System produced {:.5f} MWh over the year.".format(np.sum(energy_mwh)))

    if graph:
        set_graph_style()
        fig, ax = plt.subplots(2, 1, sharex=True)
        sns.lineplot(x = e.sols, y = energy_mwh*1000/24, color="darkblue", ax=ax[0])
        ax[0].set(title = "Daily wind energy production per year", ylabel = "Average sol power [kW]")
        sns.lineplot(x = e.sols, y = energy_mwh, color = "darkblue", ax=ax[1])
        ax[1].set(xlabel = "Sol", ylabel = "Energy Produced [MWh]")
        plt.subplots_adjust(bottom=0.1)
    
    return energy_mwh


#%% Main run - calculation for the whole year.
i = Input()
s = System(i)
e = Environment(i)
c = Cycle(s)

energy_mwh = annual_calculation(s, e, graph=True)

system_mass = s.kite_mass + s.tether_mass

print("System mass is {:.2f} kg.".format(system_mass))