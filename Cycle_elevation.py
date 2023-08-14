import numpy as np
from scipy import optimize as op
from scipy.integrate import quad
import scipy.interpolate
import matplotlib.pyplot as plt
import time
import seaborn as sns
from style import set_graph_style
import sys
from Environment import *
import os

def pause():
    programPause = input("Press the <ENTER> key to continue...")

class Cycle():
    """Cycle class, for running the analysis.
    Attributes:
        region_counter (integer): 1, 2 (limit tether force) or 3 (limit power), depending on the operational region [-]
        setting (integer): 2 or 3, depending on whether a 2-phase or 3-phase strategy is used [-]
    Methods:
        calculate_generator_eta(system): Calculates generator efficiency
            Attributes:
                tau_f_out (float): Friction torque
        run_simulation(System, Environment, setting = 2): Finds the kite state for reel-out \
            and reel-in phase for current System and Environment\
            Attributes:
                f_out (float): Dimensionless force factor for reel-out phase [-]
                f_in (float): Dimensionless force factor for reel-in phase [-]
                gamma_out_max (float): Maximum dimensionless reel-out velocity factor [-]
                gamma_in_max (float): Maximum dimensionless reel-in velocity factor [-]
                gamma_out (float): Dimensionless reel-out velocity factor [-]
                gamma_in (float): Dimensionless reel-in velocity factor [-]
                v_out (float): Reel-out speed [m/s]
                v_in (float): Reel-in speed [m/s]
                tether_force_out (float): Tether traction force [N]
                tether_force_in (float): Tether retraction force [N]
                torque_out (float): Mechanical torque for the reel-out phase [N m]
                torque_in (float): Mechanical torque for the reel-in phase [N m]
                rpm_out (float): Rotational speed at reel-out [rpm]
                rpm_in (float): Rotational speed at reel-in [rpm]
                rpm_out_max (float): Maxmum rotational speed at reel out [rpm]
                rpm_in_max (float): Maxmum rotational speed at reel in [rpm]
                cycle_energy (float): Energy produced over the whole cycle [J]
                cycle_out_time (float): Time taken for the reel-out phase [s]
                cycle_in_time (float): Time taken for the reel-in phase [s]
                cycle_time (float): Time taken for the whole cycle [s]
                cycle_power (float): Power produced during the whole cycle [W]
                cycle_out_power (float): Power that is produced during the reel-out phase [W]
                cycle_in_power (float): Power used to complete the reel-in phase [W]
                nominal_v_w (float): Wind speed at which the first limit is reached \
                    force limits are reached [m/s]
                nominal_gamma_out (float): Optimised dimensionless reel-out velocity \
                    parameter for the moment nominal_v_w was encountered.
                mu (float): Dimensionless velocity parameter [-]
                f_out_mu (float): Dimensionless force factor for reel-out phase, region 3 [-]
    """

    def __init__(self, system):
        self.cycle_max_time = 0

        self.rpm_out_max = system.reel_out_speed_limit / system.drum_outer_radius * 60 / 2 / np.pi
        self.rpm_in_max = system.reel_in_speed_limit / system.drum_outer_radius * 60 / 2 / np.pi

        system.brake_energy = 0.5 * system.drum_inertia * max(self.rpm_out_max * 2 * np.pi / 60, \
                                                              self.rpm_in_max * 2 * np.pi / 60) ** 2 / system.eta_brake


        self.f_out = (system.c_l_out ** 3) / (system.c_d_out ** 2)
        self.f_in = system.c_d_in

    def calculate_generator_eta(self, system):
        # self.gen_eta = 1
        self.pm_out1 = self.tether_force_out * self.v_out

        self.tau_f_out = system.gen_tau_s + (system.gen_c_f * system.drum_outer_radius * self.rpm_out * 2 * np.pi) / 60
        self.tau_g = self.torque_out - self.tau_f_out
        self.i_out = self.tau_g * system.gen_speed_constant
        self.le_out = 3 * system.gen_terminal_resistance * self.i_out ** 2 / system.gen_other_losses
        self.pm_out = self.tau_g * self.rpm_out * 2 * np.pi / 60
        # self.pe_out = self.pm_out - self.le_out - self.tau_f_out * self.rpm_out * 2 * np.pi / 60 * system.drum_outer_radius
        self.pe_out = self.pm_out - self.le_out
        self.gen_eta = self.pe_out / self.pm_out1

        # print("########################")
        # print(self.tau_f_out, "self.tau_f_out")
        # print(self.torque_out, "self.torque_out")
        # print(self.tau_g, "self.tau_g")
        # print(self.i_out, "self.i_out")
        # print(self.le_out, "self.le_out")
        # print(self.pm_out, "self.pm_out")
        # print(self.pe_out, "self.pe_out")

        # print(self.tau_f_out* self.rpm_out * 2 * np.pi/60* system.drum_outer_radius , "self.tau_f_out* self.rpm_out * 2 * np.pi/60* system.drum_outer_radius ")
        # print(self.gen_eta, "self.gen_eta")

        if self.gen_eta >= 1:
            print(self.gen_eta, "generator efficiency > 1")
            sys.exit()

    def calculate_motor_eta(self, system):
        # self.mot_eta = 1
        self.pm_in1 = self.tether_force_in * self.v_in
        self.tau_f_in = system.mot_tau_s + (system.mot_c_f * system.drum_outer_radius * self.rpm_in * 2 * np.pi) / 60
        self.tau_m = self.torque_in + self.tau_f_in
        self.i_in = self.tau_m * system.mot_speed_constant
        self.le_in = 3 * system.mot_terminal_resistance * self.i_in ** 2 / system.mot_other_losses
        self.pm_in = self.tau_m * self.rpm_in * 2 * np.pi / 60
        # self.pe_in = self.pm_in + self.le_in + self.tau_f_in * system.drum_outer_radius * self.rpm_in * 2 * np.pi / 60
        self.pe_in = self.pm_in + self.le_in
        self.mot_eta = self.pm_in1 / self.pe_in

        # print("########################")
        # print(self.tau_f_in, "self.tau_f_in")
        # print(self.torque_in, "self.torque_in")
        # print(self.tau_m, "self.tau_m")
        # print(self.i_in, "self.i_in")
        # print(self.le_in, "self.le_in")
        # print(self.pe_in, "self.pe_in")
        # print(self.tau_f_in * system.drum_outer_radius * self.rpm_in * 2 * np.pi/60, "tau_f_in * system.drum_outer_radius * self.rpm_in * 2 * np.pi/60")
        # print(self.mot_eta, "self.mot_eta")

        if self.mot_eta >= 1:
            print(self.mot_eta, "motor efficiency > 1")
            sys.exit()


    def calculate_nominal_vw_t(self, system,velocity_list, density):
        nominal_v_w_t = 0
        nominal_v_w_t0 = system.cut_out_v_w
        done = False
        for v_w in velocity_list:
            # print("nominal_v_w_t for loop with ",v_w, "m/s")
            loop1 = 0
            loop2 = 0
            # if v_w < nominal_v_w_t0:
            self.gamma_out_max = system.reel_out_speed_limit / v_w
            self.gamma_in_max = system.reel_in_speed_limit / v_w

            loop1 += 1
            # print("nominal_v_w_t while loop counter at ", loop1)

            def objective_function(x):
                gamma_out = x[0]
                gamma_in = x[1]
                f_c = ((np.cos(system.beta_out) - gamma_out) ** 2 - (self.f_in / self.f_out) * (1 + 2*np.cos(system.beta_in)*gamma_in+gamma_in**2) ) * \
                      ((gamma_out * gamma_in) / (gamma_out + gamma_in))
                return -f_c

            starting_point = (0.001, 0.001)
            bounds = ((0.001, self.gamma_out_max),
                      (0.001, self.gamma_in_max),)

            optimisation_result = op.minimize(objective_function, starting_point,
                                              bounds=bounds, method='SLSQP')
            gamma_out = optimisation_result['x'][0]
            gamma_in = optimisation_result['x'][1]
            v_out = v_w * gamma_out
            v_in = v_w * gamma_in

            tether_force_out = 0.5 * density * (v_w ** 2) * \
                                    system.projected_area * ((np.cos(system.beta_out) - gamma_out) ** 2) * self.f_out

            if gamma_out <= 0 or round( gamma_in,3)>round(self.gamma_in_max,3) or round( gamma_out,3)>round(self.gamma_out_max,3):
                print("R1")
                # if round( rpm_in,3)>round(rpm_in_max,3):
                #     print(" rpm_in> rpm_in_max", rpm_in,">",rpm_in_max)
                # if round(rpm_out,3)>round( rpm_out_max,3):
                #     print(" rpm_out> rpm_out_max",  rpm_out,">", rpm_out_max)
                if round(  gamma_in,3)>round( self.gamma_in_max,3):
                    print(" gamma_in> gamma_in_max",   gamma_in,">", self.gamma_in_max)
                if round( gamma_out,3)>round( self.gamma_out_max,3):
                    print(" gamma_out> gamma_out_max",  gamma_out,">", self.gamma_out_max)
                sys.exit()

            if tether_force_out >= system.nominal_tether_force:
                # print("if loop for force limit at", tether_force_out, "N")
                nominal_gamma_out_t0 =  gamma_out
                nominal_gamma_out_t = 0

                while round(nominal_gamma_out_t0,3) != round(nominal_gamma_out_t,3):
                    loop2 += 1
                    if loop2 > 10000:
                        sys.exit()
                    # print("nominal_gamma_out_t while loop counter at ", loop2)
                    nominal_gamma_out_t =  nominal_gamma_out_t0

                    nominal_v_w_t0 = np.sqrt(system.nominal_tether_force / 0.5 / density / system.projected_area / ((np.cos(system.beta_out) -  nominal_gamma_out_t) ** 2) /  self.f_out)
                    nominal_gamma_out_t0 = v_out/nominal_v_w_t0

                    # print(nominal_v_w_t0 ,"self.nominal_v_w_t0 ")
                    # print(v_out ,"self.v_out ")
                    # print(nominal_gamma_out_t0 ,"self.nominal_gamma_out_t0 ")

                # =============================================================================
                # print("Tether force limitation experienced at {:.1f} m/s." \
                #       .format(nominal_v_w_t0))
                # =============================================================================
                self.nominal_gamma_out_t = nominal_gamma_out_t0
                nominal_v_w_t = nominal_v_w_t0
                self.nominal_v_w_t = nominal_v_w_t0
                done = True
            if done == True:  # added an extra condition to exit the main loop
                break
        return  nominal_v_w_t


################################################################################################################################################################################
    def calculate_nominal_vw_p(self, system,velocity_list, density):
        nominal_v_w_p = 0
        nominal_v_w_p0 = system.cut_out_v_w
        done = False
        velocity_list1=np.where(velocity_list > self.nominal_v_w_t, velocity_list, 0)
        # print(velocity_list1,"velocity_list1")
        velocity_list2=np.trim_zeros(velocity_list1)
        # print(velocity_list2,"velocity_list2")

        for v_w in velocity_list2:
            # print("nominal_v_w_p for loop with ",v_w, "m/s")
            loop1 = 0
            loop2 = 0
            # if self.nominal_v_w_t < v_w <= nominal_v_w_p0:
            self.tether_force_out = system.nominal_tether_force

            mu = v_w /  self.nominal_v_w_t

            self.gamma_out = np.cos(system.beta_out) - ((np.cos(system.beta_out) - self.nominal_gamma_out_t) / mu)
            self.v_out = v_w*np.cos(system.beta_out) - self.nominal_v_w_t*np.cos(system.beta_out) + self.nominal_gamma_out_t * self.nominal_v_w_t

            self.torque_out = self.tether_force_out * system.drum_outer_radius
            self.rpm_out = (self.v_out / system.drum_outer_radius) * 60 / (2 * np.pi)


            loop1 += 1

            self.calculate_generator_eta(system)
            self.cycle_out_power = self.pe_out

            # print(self.cycle_out_power, "cycle_power for nominal_v_w_p")

            if  self.cycle_out_power >= system.nominal_generator_power:
                # print("if loop for power limit at", self.cycle_out_power, "W")
                nominal_gamma_out_p0 =  self.gamma_out
                nominal_gamma_out_p = 0
                nominal_v_w_p0 = v_w

                while round( nominal_gamma_out_p0,3) != round(nominal_gamma_out_p,3):
                    if loop2 > 10000:
                        print("Model does not converge due to incompatible input")
                        sys.exit()

                    loop2 += 1
                    # print("nominal_gamma_out_p while loop counter at ", loop2)
                    nominal_gamma_out_p =  nominal_gamma_out_p0

                    nominal_v_w_p0 = np.sqrt(system.nominal_tether_force / 0.5 / density / system.projected_area / ((np.cos(system.beta_out) -  nominal_gamma_out_p0) ** 2) /  self.f_out)
                    self.v_out = system.v_out_nominal_p
                    nominal_gamma_out_p0  = self.v_out/nominal_v_w_p0

                    # print(nominal_gamma_out_p0 ,"self.nominal_gamma_out_p0 ")
                    # print(self.v_out ,"self.v_out ")
                    # print(nominal_v_w_p0 ,"self.nominal_v_w_p0 ")

                # =============================================================================
                # print("Power generation limitation experienced at {:.4f} m/s." \
                #       .format(nominal_v_w_p0))
                # =============================================================================
                self.nominal_gamma_out_p = nominal_gamma_out_p0
                nominal_v_w_p = nominal_v_w_p0
                self.nominal_v_w_p = nominal_v_w_p
                done = True
            if done == True:  # added an extra condition to exit the main loop
                break
        return  nominal_v_w_p


################################################################################################################################################################################
################################################################################################################################################################################

    def run_simulation(self, system, environment):
        self.gamma_out_max = system.reel_out_speed_limit / environment.v_w
        self.gamma_in_max = system.reel_in_speed_limit / environment.v_w

    ################################################################################################################################################################################
        if environment.v_w < self.nominal_v_w_t:
            # print("in R1")
            def objective_function(x):
                gamma_out = x[0]
                gamma_in = x[1]
                f_c = ((np.cos(system.beta_out) - gamma_out) ** 2 - (self.f_in / self.f_out) * (1 + 2*np.cos(system.beta_in)*gamma_in+ gamma_in ** 2)) * \
                      ((gamma_out * gamma_in) / (gamma_out + gamma_in))
                return -f_c


            starting_point = (0.001, 0.001)
            bounds = ((0.001, self.gamma_out_max),
                      (0.001, self.gamma_in_max),)

            optimisation_result = op.minimize(objective_function, starting_point,
                                              bounds=bounds, method='SLSQP')
            self.gamma_out = optimisation_result['x'][0]
            self.gamma_in = optimisation_result['x'][1]

            self.v_out = environment.v_w * self.gamma_out
            self.v_in = environment.v_w * self.gamma_in

            self.tether_force_out = 0.5 * environment.rho * (environment.v_w ** 2) * \
                                    system.projected_area * ((np.cos(system.beta_out) - self.gamma_out) ** 2) * self.f_out
            self.torque_out = self.tether_force_out * system.drum_outer_radius
            self.rpm_out = (self.v_out / system.drum_outer_radius) * 60 / (2 * np.pi)

            self.tether_force_in = 0.5 * environment.rho * (environment.v_w ** 2) * \
                                   system.projected_area * (1 + 2*np.cos(system.beta_in)*self.gamma_in+ self.gamma_in ** 2) * self.f_in
            self.torque_in = self.tether_force_in * system.drum_outer_radius
            self.rpm_in = (self.v_in / system.drum_outer_radius) * 60 / (2 * np.pi)

            self.cycle_out_time = system.reeling_length / self.v_out
            self.cycle_in_time = system.reeling_length / self.v_in

            self.cycle_time = self.cycle_out_time + self.cycle_in_time

            self.calculate_generator_eta(system)
            self.cycle_out_power = self.pe_out

            self.calculate_motor_eta(system)
            self.cycle_in_power = self.pe_in + system.spindle_power + system.thermal_control_power

            self.cycle_power = (self.cycle_out_power * self.cycle_out_time - (self.cycle_in_power * self.cycle_in_time + system.brake_energy) / system.eta_energy_storage) / self.cycle_time

            # print(self.cycle_in_power, self.cycle_out_power, self.cycle_power, "cycle_power R1")

            if self.gamma_out <= 0 or round(self.rpm_in, 3) > round(self.rpm_in_max, 3) or round(self.rpm_out,
                                                                                                 3) > round(
                    self.rpm_out_max, 3) or round(self.gamma_in, 3) > round(self.gamma_in_max, 3) or round(
                    self.gamma_out, 3) > round(self.gamma_out_max, 3):
                print("R1")
                if round(self.rpm_in, 3) > round(self.rpm_in_max, 3):
                    print("self.rpm_in>self.rpm_in_max", self.rpm_in, ">", self.rpm_in_max)
                if round(self.rpm_out, 3) > round(self.rpm_out_max, 3):
                    print("self.rpm_out>self.rpm_out_max", self.rpm_out, ">", self.rpm_out_max)
                if round(self.gamma_in, 3) > round(self.gamma_in_max, 3):
                    print("self.gamma_in>self.gamma_in_max", self.gamma_in, ">", self.gamma_in_max)
                if round(self.gamma_out, 3) > round(self.gamma_out_max, 3):
                    print("self.gamma_out>self.gamma_out_max", self.gamma_out, ">", self.gamma_out_max)
                sys.exit()

    ################################################################################################################################################################################
        if self.nominal_v_w_t <= environment.v_w <= self.nominal_v_w_p:
            # print("in R2")
            self.gamma_in_max = system.reel_in_speed_limit / environment.v_w

            self.mu = environment.v_w / self.nominal_v_w_t

            self.gamma_out = np.cos(system.beta_out) - ((np.cos(system.beta_out) - self.nominal_gamma_out_t) / self.mu)
            self.v_out = environment.v_w * np.cos(system.beta_out) - self.nominal_v_w_t * np.cos(
                system.beta_out) + self.nominal_gamma_out_t * self.nominal_v_w_t

            self.tether_force_out = system.nominal_tether_force
            self.torque_out = self.tether_force_out * system.drum_outer_radius
            self.rpm_out = (self.v_out / system.drum_outer_radius) * 60 / (2 * np.pi)

            def objective_function(x):
                gamma_in = x[0]
                f_c_2 = ((1 / (self.mu ** 2)) * (np.cos(system.beta_out) - self.nominal_gamma_out_t) ** 2 - (self.f_in / self.f_out) * (1 + 2*np.cos(system.beta_in)*gamma_in+ gamma_in ** 2)) * \
                        ((gamma_in * (self.mu *np.cos(system.beta_out)- np.cos(system.beta_out) + self.nominal_gamma_out_t)) / (self.mu * gamma_in + self.mu *np.cos(system.beta_out)- np.cos(system.beta_out) + self.nominal_gamma_out_t))
                return -f_c_2

            starting_point = (0.001)
            bounds = ((0.001, self.gamma_in_max),)

            optimisation_result = op.minimize(objective_function, starting_point, bounds=bounds, method='SLSQP')

            self.gamma_in = optimisation_result['x'][0]
            self.v_in = environment.v_w * self.gamma_in
            self.tether_force_in = 0.5 * environment.rho * (environment.v_w ** 2) * \
                                   system.projected_area * (1 + 2 * np.cos(
                system.beta_in) * self.gamma_in + self.gamma_in ** 2) * self.f_in
            self.torque_in = self.tether_force_in * system.drum_outer_radius
            self.rpm_in = (self.v_in / system.drum_outer_radius) * 60 / (2 * np.pi)

            self.calculate_motor_eta(system)
            self.cycle_in_power = self.pe_in + system.spindle_power + system.thermal_control_power

            self.calculate_generator_eta(system)
            self.cycle_out_power = self.pe_out

            self.cycle_out_time = system.reeling_length / self.v_out
            self.cycle_in_time = system.reeling_length / self.v_in
            self.cycle_time = self.cycle_out_time + self.cycle_in_time

            self.cycle_power = (self.cycle_out_power * self.cycle_out_time - (self.cycle_in_power * self.cycle_in_time + \
                                           system.brake_energy) / system.eta_energy_storage) / self.cycle_time


            # print(self.cycle_in_power, self.cycle_out_power, self.cycle_power, "cycle_power R2")

            if self.mu < 1 or self.gamma_out <= 0:
                print("mu<1 in R2", self.mu)
                print("R2, gamma_out", self.gamma_out)
                sys.exit()
    ###############################################################################################################################################################################

        if environment.v_w >= self.nominal_v_w_p:
            # print("in R3")

            self.gamma_in_max = system.reel_in_speed_limit / environment.v_w
            self.tether_force_out = system.nominal_tether_force
            self.v_out = system.v_out_nominal_p
            self.cycle_out_power = system.nominal_generator_power

            self.gamma_out = self.v_out / environment.v_w
            self.mu = environment.v_w / self.nominal_v_w_p

            if self.gamma_out <= 0:
                print("R3")
                sys.exit()

            if self.mu < 1:
                print("mu<1 in R3")
                sys.exit()

            def objective_function(x):
                gamma_in = x[0]
                f_c_3 = ((1 / (self.mu ** 2)) * (np.cos(system.beta_out) - self.nominal_gamma_out_t) ** 2 - (self.f_in / self.f_out) * (1 + 2*np.cos(system.beta_in)*gamma_in+ gamma_in ** 2)) * \
                        ((self.nominal_gamma_out_p * gamma_in) / (self.nominal_gamma_out_p + self.mu * gamma_in))
                return -f_c_3

            starting_point = (0.001)
            bounds = ((0.001, self.gamma_in_max),)

            optimisation_result = op.minimize(objective_function, starting_point, bounds=bounds, method='SLSQP')
            self.gamma_in = optimisation_result['x'][0]

            self.v_in = environment.v_w * self.gamma_in
            self.tether_force_in = 0.5 * environment.rho * (environment.v_w ** 2) * \
                                   system.projected_area * (1 + 2 * np.cos(
                system.beta_in) * self.gamma_in + self.gamma_in ** 2) * self.f_in
            self.torque_in = self.tether_force_in * system.drum_outer_radius
            self.rpm_in = (self.v_in / system.drum_outer_radius) * 60 / (2 * np.pi)
            self.calculate_motor_eta(system)
            self.cycle_in_power = (self.pe_in + system.spindle_power + system.thermal_control_power) / system.eta_energy_storage

            self.cycle_out_time = system.reeling_length / self.v_out
            self.cycle_in_time = system.reeling_length / self.v_in
            self.cycle_time = self.cycle_out_time + self.cycle_in_time

            self.cycle_power = (self.cycle_out_power * self.cycle_out_time - self.cycle_in_power * self.cycle_in_time - \
                                           system.brake_energy / system.eta_energy_storage) / self.cycle_time
            # print(self.cycle_in_power, self.cycle_out_power, self.cycle_power, "cycle_power R3")

            if self.cycle_in_time > self.cycle_max_time:
                self.cycle_max_time = self.cycle_in_time

        if self.cycle_power < 0:
            self.cycle_power = 0
