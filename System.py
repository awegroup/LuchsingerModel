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

class Kite(Input):
    """ Kite class.
    Attributes:
        c_l (float): Kite lift coefficient [-]
        c_d_k (float): Kite drag coefficient [-]
        flat_area (float):  Flattened kite area [m^2]
        flattening_factor (float): [-]
        projected_area (float): Effective kite aerodynamic area [m^2]
    """

    def __init__(self,input):
        # self.c_l = 0.6
        # self.c_d_k = 0.06
        # self.c_l_in = input.c_l_in
        self.c_d_k_in = input.c_d_k_in
        self.c_l_out = input.c_l_out
        self.c_d_k_out = input.c_d_k_out

        self.flattening_factor = 0.9
        self.area_density = 0.10802
        self.projected_area = input.projected_area
        self.kite_mass = self.projected_area / self.flattening_factor * self.area_density
        self.kite_thickness = 0.01
        self.kite_volume = self.projected_area / self.flattening_factor * self.kite_thickness

    def calculate_kite_mass(self):
        self.kite_mass = self.projected_area / self.flattening_factor * self.area_density
        print(self.kite_mass, "kite_mass")


class Tether(Input):
    """ Tether class.
    Attributes:
        d_tether (float): Tether diameter [m]
        c_d_c (float): Cylinder drag coefficient [-]
        tether_max_length (float): Maximum tether length [m]
        tether_min_length (float): Minimum operational tether length [m]
        operational_length (float): Operational tether length [m]
        reeling_length (float): The length of tether that is reeled out during one \
            reel-out or reel-in phase, respectively [m]
    Methods:
        calculate_operational_length(): Calculates the operational tether length based on \
            Tether class attributes.
        calculate_reeling_length(): Calculates the reeling length for a single reel-out or reel-in \
            phase in a cycle
    """

    def __init__(self, input):

        self.nominal_tether_force = input.nominal_tether_force

        self.c_d_c = input.c_d_c
        self.tether_max_length = 400
        self.tether_min_length = 200
        self.design_factor = 3
        self.weight_slope = 2000
        self.GRAVITATIONAL_CONSTANT = 3.711
        self.mbl = (self.nominal_tether_force * self.design_factor) / self.GRAVITATIONAL_CONSTANT
        self.mass_tether_perhm = self.mbl / self.weight_slope
        self.tether_mass = self.mass_tether_perhm * self.tether_max_length / 100
        self.diameter_slope = 1.16 / (np.pi * 0.002 ** 2)
        self.calculate_operational_length()
        self.calculate_reeling_length()
        self.calculate_tether_diameter()
        self.tether_volume = self.tether_max_length * self.d_tether ** 2 / 4

    def calculate_operational_length(self):
        self.operational_length = (self.tether_max_length - self.tether_min_length) / 2 + \
                                  self.tether_min_length

    def calculate_reeling_length(self):
        self.reeling_length = (self.tether_max_length - self.tether_min_length)


    def calculate_tether_diameter(self):
        self.max_break_load = self.nominal_tether_force * self.design_factor / \
                              self.GRAVITATIONAL_CONSTANT
        self.tether_mass_per_hundred_m = self.max_break_load / self.weight_slope
        self.tether_mass = self.tether_max_length * self.tether_mass_per_hundred_m / 100
        self.d_tether = 2 * np.sqrt(self.tether_mass_per_hundred_m / \
                                    (self.diameter_slope * np.pi))
        print(self.d_tether, "d_tether")


class System(Kite, Tether):
    """ System class, inheriting all attributes and methods from Kite() and Tether().
    Attributes:
        reel_out_speed_limit (float): Maximum reel-out speed achievable [m/s]
        reel_in_speed_limit (float): Maximum reel-in speed achievable [m/s]
        cut_in_v_w (float): Cut-in wind speed [m/s]
        cut_out_v_w (float): Cut-out wind speed [m/s]
        nominal_tether_force (float): Maximum designed for tether force [N]
        nominal_generator_power (float): Maximum designed for generator power [W]
        phi (float): Elevation angle [rad]
        overall_gs_efficiency (float): Overall ground-station mechanical-to-electrical\
            conversion efficiency
        drum_outer_radius (float): Outer radius of the drum on which the tether is wound [m]
        drum_inner_radius (float): Inner radius of the drum on which the shaft is connect [m]
        drum_density (float): Material density of the drum (Aluminium 7075-T6) [kg / m^3]
        drum_length (float): Length of the drum [m]
        drum_inertia (float): Mass moment of inertia of the drum [kg / m^2]
        gen_speed_constant (float): Generator inverse of torque constant [A / N m]
        gen_terminal_resistance (float): Generator terminal resistance [Ohm]
        gen_other_losses (float): Factor accounting for higher resistance at operating\
            frequency due to the skin effect, stray-load-losses and other not explicitly\
                modelled losses of the generator
        gen_tau_s (float): Generator static contribution to friction torque [N m]
        gen_c_f (float): Generator dynamic contribution to friction torque [N s]
        mot_speed_constant (float): Motor inverse of torque constant [A / N m]
        mot_terminal_resistance (float): Motor terminal resistance [Ohm]
        mot_other_losses (float): Factor accounting for higher resistance at operating\
            frequency due to the skin effect, stray-load-losses and other not explicitly\
                modelled losses of the motor
        mot_tau_s (float): Motor static contribution to friction torque [N m]
        mot_c_f (float): Motor dynamic contribution to friction torque [N s]
        eta_energy_storage (float): Supercapacitor efficiency [-]
        eta_brake (float): Brakes efficiency [-]
        brake_power (float): Power required to release the brakes [W]
        spindle_power (float): Power required for the spindle motor [W]
        thermal_control_power (float): Power required for the heating system [J]
        c_d (float): Drag coefficient of kite and tether [-]
        operational_height (float): Average height (z-coordinate) of operation [m]
    Methods:
        calculate_c_d(): Calculates drag coefficient of kite and tether based \
            on their inherited attributes
        calculate_operational_height(): Calculates average height \
            (z-coordinate) of operation
    """

    def __init__(self, input):
        Kite.__init__(self, input)
        Tether.__init__(self, input)

        self.nominal_generator_power = input.nominal_generator_power

        self.reel_out_speed_limit = input.reel_out_speed_limit

        self.reel_in_speed_limit = input.reel_in_speed_limit

        self.cut_in_v_w = input.cut_in_v_w
        self.cut_out_v_w = input.cut_out_v_w

        self.phi_in = input.phi_in
        self.phi_out = input.phi_out

        self.drum_outer_radius = input.drum_outer_radius
        self.drum_inner_radius = input.drum_inner_radius

        self.drum_length = (self.tether_max_length + 20) / (2 * np.pi * self.drum_outer_radius) * self.d_tether
        self.drum_volume = np.pi * (self.drum_outer_radius ** 2 - self.drum_inner_radius ** 2) * self.drum_length
        self.drum_density = 2810
        self.drum_mass = self.drum_volume * self.drum_density
        self.drum_inertia = 0.5 * self.drum_mass * (self.drum_outer_radius ** 2 + self.drum_inner_radius ** 2)

        self.gen_speed_constant = 1 / 12
        self.gen_terminal_resistance = 0.04 * 1.2
        self.gen_other_losses = 0.9
        self.gen_tau_s = 3.18
        self.gen_c_f = 0.799
        self.mot_speed_constant = 1 / 12  # 0.1898
        self.mot_terminal_resistance = 0.08 * 1.35
        self.mot_other_losses = 0.9
        self.mot_tau_s = 3.18
        self.mot_c_f = 0.799
        self.eta_energy_storage = 0.95
        self.eta_brake = 0.95
        self.spindle_power = 0
        self.thermal_control_power = 0

        self.calculate_c_d()
        self.calculate_operational_height()
        self.calculate_nominal_v_out_p()

    def calculate_c_d(self):
        self.c_d_in = self.c_d_k_in + \
                   0.25 * self.d_tether * self.operational_length * self.c_d_c * (1 / self.projected_area)
        self.c_d_out = self.c_d_k_out + \
                   0.25 * self.d_tether * self.operational_length * self.c_d_c * (1 / self.projected_area)

    def calculate_operational_height(self):
        self.operational_height = np.sin(self.phi_out) * self.operational_length


    def calculate_nominal_v_out_p(self):
        self.v_out_nominal_p = 8
        self.v_out1 =0
        while self.v_out_nominal_p != self.v_out1:
            self.v_out1 = self.v_out_nominal_p
            self.tether_force_out1 = self.nominal_tether_force

            self.torque_out1 = self.tether_force_out1 * self.drum_outer_radius
            self.rpm_out1 = (self.v_out1 / self.drum_outer_radius) * 60 / (2 * np.pi)
            self.tau_f_out1 = self.gen_tau_s + (self.gen_c_f * self.drum_outer_radius * self.rpm_out1 * 2 * np.pi) / 60
            self.tau_g1 = self.torque_out1 - self.tau_f_out1
            self.i_out1 = self.tau_g1 * self.gen_speed_constant
            self.le_out1 = 3 * self.gen_terminal_resistance * self.i_out1 ** 2 / self.gen_other_losses
            self.pe_out_nominal_p = self.nominal_generator_power
            self.pm_out_nominal_p = self.pe_out_nominal_p + self.le_out1 + self.tau_f_out1 * self.rpm_out1 * 2 * np.pi / 60 * self.drum_outer_radius

            self.gen_eta_nominal_p = self.pe_out_nominal_p / self.pm_out_nominal_p

            self.v_out_nominal_p = self.pe_out_nominal_p / self.tether_force_out1 / self.gen_eta_nominal_p

        # print(self.v_out_nominal_p,"self.v_out_nominal_p")
        # print(self.pm_out_nominal_p,"self.pm_out_nominal_p")
        # print(self.cycle_out_power1,"self.cycle_out_power1")
        # print(self.pe_out_nominal_p,"self.pe_out_nominal_p")
        # print(self.gen_eta_nominal_p,"self.gen_eta_nominal_p")
#
