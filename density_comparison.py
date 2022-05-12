import numpy as np
from scipy import optimize as op
from scipy.integrate import quad
import scipy.interpolate
import matplotlib.pyplot as plt
import time
import seaborn as sns
# from style import set_graph_style
import matplotlib
from matplotlib.font_manager import findfont, FontProperties

font = findfont(FontProperties(family=['sans-serif']))
matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
matplotlib.rcParams['font.family'] = "Arial"
iter = 30
palette = list(reversed(sns.color_palette("Spectral_r", iter).as_hex()))
# print(palette)
# width_px=1000
# new = Image.new(mode="RGB", size=(width_px,120))
#
# for i in range(iter):
#
#     newt = Image.new(mode="RGB", size=(width_px//iter,100), color=palette[i])
#     new.paste(newt, (i*width_px//iter,10))
# new.show()
color_extra_demand = palette[4]
color_wind_direct = palette[-2]
color_solar_direct = palette[9]
color_wind_battery = palette[-3]
color_solar_battery = palette[12]
color_wind_battery_extra = palette[-5]
color_solar_battery_extra =palette[18]


####GRAPHING
## Define text sizes for **SAVED** pictures (texpsize -- text export size)
texpsize= [26,28,30]

## Graphing Parameters
SMALL_SIZE  = texpsize[0]
MEDIUM_SIZE = texpsize[1]
BIGGER_SIZE = texpsize[2]

plt.style.use('grayscale')
plt.rc('font', size=MEDIUM_SIZE, family='Arial')    ## controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
plt.rc('text', usetex=False)
matplotlib.rcParams['lines.linewidth']  = 1.5
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor']   = 'white'
matplotlib.rcParams["legend.fancybox"]  = False

ls_min_temp, rho_min_temp, ls_max_temp, rho_max_temp = np.loadtxt("density.txt", unpack = True, delimiter = ',')
ls_to_sol = np.loadtxt("Ls_new.txt", unpack = True)
rho_interpolated_max_temp = scipy.interpolate.interp1d(ls_max_temp, rho_max_temp)
rho_interpolated_min_temp = scipy.interpolate.interp1d(ls_min_temp, rho_min_temp)

rho_min_list,  rho_avg_list = np.loadtxt("density_AN_MCD.txt", unpack=True)
ls_for_rho = np.arange(0,375,15)
rho_min_interpolated = scipy.interpolate.interp1d(ls_for_rho, rho_min_list)
rho_avg_interpolated = scipy.interpolate.interp1d(ls_for_rho, rho_avg_list)

ls_list = np.arange(0,361,1)
rho_max_temp_int = np.zeros(len(ls_list))
rho_avg_list_int = np.zeros(len(ls_list))

for ls in ls_list:
    rho_max_temp_int[ls] = float(rho_interpolated_max_temp(ls))
    rho_avg_list_int[ls] = float(rho_avg_interpolated(ls))


linewidth1 =3

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(14, 8))
ax[0,0].grid(True,which="major",color="#999999",alpha=0.75)
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--",alpha=0.50)
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')

ax[0,0].set_xlabel("Solar longitude $ls$ [$\degree$]")
ax[0,0].set_ylabel(r"Density $\rho$ [$kg/m^3$]")

# ax[0,0].plot(i.sols,energy_supply_sol, color=color_solar_battery_extra,alpha =0.6, linewidth = linewidth1)
ax[0,0].scatter(ls_for_rho, rho_avg_list, marker = 'x',s=200, label ='Sampled data', color=color_wind_direct,alpha = 0.9, linewidth = linewidth1)
ax[0,0].plot(ls_list,rho_avg_list_int, "--", label ='Interpolated data', color=color_wind_direct,alpha =1,  linewidth = linewidth1)
# ax[0,0].plot(i.sols,c.wind_energy_for_H_sol, color=color_wind_direct,alpha =alpha3, linewidth = linewidth1)
# ax[0,0].plot(i.sols,energy_supply_sol, color=color_solar_battery_extra,alpha =0.4, linewidth = linewidth1)

ax[0,0].set_xlim(0,360)
# ax[0,0].set_ylim(0, )

plt.legend(facecolor = "white", framealpha = 1, fancybox = True, edgecolor = "black", loc='best')
fig.savefig("density_AN.png", dpi= 300)
#
# plt.scatter(ls_max_temp, rho_max_temp* 1e-3, color = 'b', alpha = 0.8, marker = 'o', label = 'DM')
# plt.plot(ls_list, rho_max_temp_int* 1e-3,color = 'b', alpha = 0.8)
# plt.scatter(ls_for_rho, rho_avg_list, color = 'r', alpha = 0.8,marker = 'x',label = 'Ref. loc')
# plt.plot(ls_list, rho_avg_list_int,color = 'r', alpha = 0.8)
# # plt.xlim(0,360)
# plt.xlabel(r'Solar longitude [$^{\circ}$]')
# plt.ylabel(r'Density [$kg/m^3$]')
# plt.legend()
# plt.grid()
# plt.show()


sol_hours = np.genfromtxt('day_length.csv', delimiter=",")
sols = np.arange(0, len(sol_hours), 1)
rho_low_list1 = np.zeros(len(sols))
rho_high_list1 = np.zeros(len(sols))
rho_avg_list1 = np.zeros(len(sols))
sols_for_density_data1 = np.zeros(len(ls_min_temp))

rho_low_list2 = np.zeros(len(sols))
rho_avg_list2 = np.zeros(len(sols))
sols_for_density_data2 = np.zeros(len(ls_for_rho))
for sol in sols:
    ls = ls_to_sol[sol]
    rho_low_list1[sol] = float(rho_interpolated_max_temp(ls)) * 1e-3
    rho_high_list1[sol] = float(rho_interpolated_min_temp(ls)) * 1e-3
    rho_avg_list1[sol] = (rho_low_list1[sol] + rho_high_list1[sol]) / 2
#         self.rho = self.rho_low
#         print(sol, "sol")
#         print(self.rho, "rho")
#

    rho_low_list2[sol] = float(rho_min_interpolated(ls))
    rho_avg_list2[sol] = float(rho_avg_interpolated(ls))
    # print(self.rho_low, "rho_new")
    # print(sol, "sol")
    # print(self.rho, "rho")

# plt.plot(sols, rho_low_list1)
# plt.plot(sols, rho_avg_list2)
# plt.xlabel(r'Sols [-]')
# plt.ylabel(r'Density [$kg/m^3$]')
# plt.legend()
# plt.grid()
# plt.show()

# plt.scatter()
