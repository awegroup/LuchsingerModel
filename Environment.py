import numpy as np
from scipy import optimize as op
from scipy.integrate import quad
import scipy.interpolate
import matplotlib.pyplot as plt
import time
import seaborn as sns
from style import set_graph_style
import sys
from Input import *

class Environment(Input):
    """ Environment class.
    Attributes:
        rho (float): Atmospheric density [kg/m^3]
        h_0 (float): Surface roughness [m]
        kappa (float): von Karman constant [-]
        friction_velocity (float): Friction velocity [m/s]
        k (float): Weibull k parameter [-]
        u (float): Weibull u parameter [m/s]
        v_w (float): Wind speed (at operational height) [m/s]
        season (int): Season counter, 0 = spring, 1 = summer, 2 = autumn, 3 = winter, 4 = dust storm
        dust_storm (bool): If True, then it considers a winter with a dust storm, of length dust_storm_length
        dust_storm_length (int): Length of the dust storm. Should range from 35 to 70 sols.
        g_w (float): Probability density for the given wind speed [-]
        ls_min_temp (array): List of solar longitudes values at minimum temperature [deg]
        ls_max_temp (array): List of solar longitudes values at maximum temperature [deg]
        rho_min_temp (array): List of the density values corresponding to the minimum temperatures [kg/m^3]
        rho_min_temp (array): List of the density values corresponding to the maximum temperatures [kg/m^3]
        ls_to_sol (array): List that compares sols to their respective solar longitudes
        sol_hours (array): List of day martian hours per sol [h]
    Methods:
        set_v_w(v_w): Updates the wind velocity
        set_friction_velocity(sol): Updates the environment friction velocity based on the sol
        set_weibull_k(sol): Updates the Weibull k parameter, based on the sol
        set_weibull_u(system): Updates the Weibull u parameter, based on System.operational_height
        calculate_weibull_pdf(v_w): Calculates the probability of v_w for the current Weibull distribution
        rho_interpolated_min_temp(sol): Interpolation function of the densities at minimum temperatures
        rho_interpolated_max_temp(sol): Interpolation function of the densities at maximum temperatures
        update_environment(system, sol): Updates all sol-dependent attributes of the environment
    """

    def __init__(self,input):
        self.rho = 0.021
        self.h_0 = 0.0316
        self.kappa = 0.4
        self.k = 1.3
        self.v_w = 7
        self.season = 0

        self.ls_to_sol = np.loadtxt("Ls_new.txt", unpack=True)
        self.rho_min_list, self.rho_avg_list = np.loadtxt(input.density_file, unpack=True)
        self.ls_for_rho = np.arange(0, 375, 15)
        self.rho_min_interpolated = scipy.interpolate.interp1d(self.ls_for_rho, self.rho_min_list)
        self.rho_avg_interpolated = scipy.interpolate.interp1d(self.ls_for_rho, self.rho_avg_list)

        self.sol_hours = np.zeros(669)
        self.sol_hours = 24

        self.sols = np.arange(0, 669, 1)

        self.k_list = input.k_list
        self.u_list = input.u_list
        self.set_periods_weibull_k_and_u()

    def set_v_w(self, v_w):
        # print(v_w,"set_v_w")
        self.v_w = v_w

    def set_season(self, sol):
        if sol >= 0 and sol <= 194:  # Spring
            self.season = 0
        elif sol > 194 and sol <= 372:  # Summer
            self.season = 1
        elif sol > 372 and sol <= 514:  # Autumn
            self.season = 2
        elif sol > 514 and sol <= 669:  # Winter
            self.season = 3

    def set_weibull_k_and_u(self):
        if self.season == 0:  # Spring
            k = self.k_list[0]
            u = self.u_list[0]
        elif self.season == 1:  # Summer
            k = self.k_list[1]
            u = self.u_list[1]
        elif self.season == 2:  # Autumn
            k = self.k_list[2]
            u = self.u_list[2]
        elif self.season == 3:  # Winter
            k = self.k_list[3]
            u = self.u_list[3]

        self.k = k
        self.u = u

    def set_periods_weibull_k_and_u(self):
        self.numbered_periods = np.arange(0,len(self.k_list))
        print(self.numbered_periods,"self.numbered_periods")
        end_list = 360 + 360 / len(self.k_list)
        list_step = 360 / len(self.k_list)
        print(end_list,"self.end_list")
        print(list_step,"self.list_step")


        self.periods_start_ls = np.arange(0,end_list, list_step)
        print(self.periods_start_ls,"self.periods_start_ls")

        self.k_list_year =  np.zeros(669)
        self.u_list_year =  np.zeros(669)

        self.period_number_of_sols = np.zeros(669)
        for i in range(669):
            # print(i, "i")

            self.ls = self.ls_to_sol[i]
            for j in range(0, len(self.numbered_periods)):
                # print(j,"j")
                # if self.periods_start_ls[j]<= self.ls < self.periods_start_ls[j+1]:
                #     self.period_number_of_sols = self.numbered_periods[j]
                #     self.k_list_year[i] = self.k_list[j]
                #     self.u_list_year[i] = self.u_list[j]
                # elif self.ls >= self.periods_start_ls[-1]:
                #     self.k_list_year[i] = self.k_list[j]
                #     self.u_list_year[i] = self.u_list[j]
                if  self.ls >= self.periods_start_ls[j]:
                    print("self.ls", self.ls)
                    print("self.periods_start_ls", self.periods_start_ls[j])

                    self.period_number_of_sols = self.numbered_periods[j]
                    self.k_list_year[i] = self.k_list[j]
                    self.u_list_year[i] = self.u_list[j]
        print("self.k_list_year",self.k_list_year)
        print("self.u_list_year",self.u_list_year)

    def calculate_weibull_pdf(self, v_w,sol):
        # self.k = self.k_list_year[sol]
        # self.u = self.u_list_year[sol]
        self.g_w = (self.k / self.u) * ((v_w / self.u) ** (self.k - 1)) * \
                   np.exp(-(self.v_w / self.u) ** self.k)
        # print(self.v_w, "v_w_gw")
        # print(self.g_w,"g_w")

    # def calculate_weibull_pdf(self, v_w,sol):
    #     self.k = self.k_list_year[sol]
    #     self.u = self.u_list_year[sol]
    #     self.g_w = (self.k / self.u) * ((v_w / self.u) ** (self.k - 1)) * \
    #                np.exp(-(self.v_w / self.u) ** self.k)
    #     # print(self.v_w, "v_w_gw")
    #     # print(self.g_w,"g_w")

    def set_density(self, sol):
        """
        Attributes:
            rho_low (float): The low value of the atmospheric density at a certain sol [kg/m^3]
            rho_high (float): The high value of the atmospheric density at a certain sol [kg/m^3]
            rho_avg (float): The average value of the atmospheric density at a certain sol [kg/m^3]
        """
        self.ls = self.ls_to_sol[sol]
        self.rho_low = float(self.rho_min_interpolated(self.ls))
        self.rho_avg = float(self.rho_avg_interpolated(self.ls))
        self.rho = self.rho_avg

    def update_environment(self, sol):
        # print("update_environment")
        # self.set_season(sol)
        # self.set_density(sol)
        # self.set_weibull_k_and_u()

        self.k = self.k_list_year[sol]
        self.u = self.u_list_year[sol]
        self.set_density(sol)
