import numpy as np

class Input:
    def __init__(self):

        self.c_d_c = 1.1
        self.c_d_k_in = 0.07
        self.c_l_out = 0.71
        self.c_d_k_out = 0.14


        # #Input for a 300m^2 kite
        self.projected_area = 300
        self.nominal_tether_force = 7200 #N
        self.nominal_generator_power = 98000 #W
        self.reel_out_speed_limit = 8
        # self.reel_in_speed_limit = 26
        self.reel_in_speed_limit = 31
        # Based on wind resource availability
        self.cut_in_v_w = 6
        self.cut_out_v_w = 40
        # Relevant only for energy losses
        self.phi_in = 45. * np.pi / 180
        self.phi_out = 25. * np.pi / 180.


        # #Input for a 250m^2 kite
        self.projected_area = 250
        self.nominal_tether_force = 6300 #N
        self.nominal_generator_power = 90000 #W

        self.reel_out_speed_limit = 8
        # self.reel_in_speed_limit = 26
        self.reel_in_speed_limit = 30

        # Based on wind resource availability
        self.cut_in_v_w = 6
        self.cut_out_v_w = 40

        # Relevant only for energy losses
        self.phi_in = 45. * np.pi / 180
        self.phi_out = 25. * np.pi / 180.

        # #Input for a 200m^2 kite
        # self.projected_area = 200
        # self.nominal_tether_force = 5100 #N
        # self.nominal_generator_power = 77000 #W
        #
        # self.reel_out_speed_limit = 8
        # # self.reel_in_speed_limit = 26
        # self.reel_in_speed_limit = 29
        #
        # # Based on wind resource availability
        # self.cut_in_v_w = 6
        # self.cut_out_v_w = 40
        #
        # # Relevant only for energy losses
        # self.phi_in = 45. * np.pi / 180
        # self.phi_out = 25. * np.pi / 180.

        # #
        # #Input for a 165^2 kite
        # self.projected_area = 165
        # self.nominal_tether_force = 4500 #N
        # self.nominal_generator_power = 68000 #W
        # self.reel_out_speed_limit = 8
        # self.reel_in_speed_limit = 28
        # # Based on wind resource availability
        # self.cut_in_v_w = 6
        # self.cut_out_v_w = 40
        # self.phi_in = 45. * np.pi / 180
        # self.phi_out = 25. * np.pi / 180.

        # #Input for a 150^2 kite
        # self.projected_area = 150
        # self.nominal_tether_force = 4000 #N
        # self.nominal_generator_power = 66000 #W
        # self.reel_out_speed_limit = 8
        # self.reel_in_speed_limit = 27
        # # Based on wind resource availability
        # self.cut_in_v_w = 6
        # self.cut_out_v_w = 40
        # self.phi_in = 45. * np.pi / 180
        # self.phi_out = 25. * np.pi / 180.


        # # Input for a 100m^2 kite
        # self.projected_area = 100
        # self.nominal_tether_force = 3100  # N
        # self.nominal_generator_power = 48000  # W
        # self.reel_out_speed_limit = 8
        # self.reel_in_speed_limit = 25
        # # Based on wind resource availability
        # self.cut_in_v_w = 10
        # self.cut_out_v_w = 40
        # self.phi_in = 45. * np.pi / 180
        # self.phi_out = 25. * np.pi / 180.


        # #Input for a 50m^2 kite
        # self.projected_area = 50
        # self.nominal_tether_force = 1700 #N
        # self.nominal_generator_power = 24000 #W
        # self.reel_out_speed_limit = 8
        # self.reel_in_speed_limit = 20
        # # Based on wind resource availability
        # self.cut_in_v_w = 6
        # self.cut_out_v_w = 40
        # # Relevant only for energy losses
        # self.phi_in = 45. * np.pi / 180
        # self.phi_out = 25. * np.pi / 180.

        self.drum_outer_radius = 0.50
        self.drum_inner_radius = 0.49

        # Values for Arsia North for 4 or 16 periods
        self.u_list = [20.02, 18.152,  22.013, 21.177]
        self.k_list = [2.648, 2.794,  2.972, 2.913]
        self.u_list = [19.452,18.002,17.635,17.827,18.714,18.422,19.158,21.479,23.478,23.919,23.042,20.999,20.025,20.534,21.249,21.273]
        self.k_list = [2.64,2.377,2.589,2.76,3.014,2.867,2.788,3.02,3.293,3.34,3.118,2.853,2.931,2.977,2.94,2.886]

        self.density_file = "density_AN_MCD.txt"

        print(f"self.projected_area = {self.projected_area} \n\
            self.nominal_tether_force = {self.nominal_tether_force} #N\n\
            self.nominal_generator_power = {self.nominal_generator_power} #W\n\
            self.reel_out_speed_limit ={self.reel_out_speed_limit} \n\
            self.reel_in_speed_limit ={self.reel_in_speed_limit}\n\
            self.cut_in_v_w = {self.cut_in_v_w}\n\
            self.cut_out_v_w = {self.cut_out_v_w}\n\
            self.phi_in = {self.phi_in}\n\
            self.phi_out = {self.phi_out}\n\
            number of periods = {len(self.k_list)}\n\
            ######################################")

        # Values for ref location
        # self.u_list = [14.208, 18.152, 16.884, 14.976]
        # self.k_list = [2.745, 2.954, 2.645, 2.769]
        # self.density_file = "density_MO_MCD.txt"


