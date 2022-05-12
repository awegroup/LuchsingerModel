import numpy as np
from scipy import optimize as op
from scipy.integrate import quad
import scipy.interpolate
import matplotlib.pyplot as plt
import time
import seaborn as sns
from style import set_graph_style
import sys

import seaborn as sns
import matplotlib
from matplotlib.font_manager import findfont, FontProperties
import matplotlib.pyplot as plt

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

sol_list = np.arange(0,669,1)
print(sol_list)
wind_energy_produced_AN_MCD_new_A50 =  np.loadtxt("wind_energy_produced_AN_MCD_A50T1600P25000v21bi45.0_fin.csv", unpack=True)
wind_energy_produced_AN_MCD_new_A100 = np.loadtxt("wind_energy_produced_AN_MCD_A100T3000P48000v25bi45.0_fin.csv", unpack=True)
wind_energy_produced_AN_MCD_new_A150 = np.loadtxt("wind_energy_produced_AN_MCD_A150T4100P65000v27bi45.0_fin.csv", unpack=True)
wind_energy_produced_AN_MCD_new_A200 = np.loadtxt("wind_energy_produced_AN_MCD_A200T5100P77000v29bi45.0_fin1.csv", unpack=True)
wind_energy_produced_AN_MCD_new_A250 = np.loadtxt("wind_energy_produced_AN_MCD_A250T6300P90000v30bi45.0_fin1.csv", unpack=True)
wind_energy_produced_AN_MCD_new_A300 = np.loadtxt("wind_energy_produced_AN_MCD_A300T7200P98000v31bi45.0_fin.csv", unpack=True)

# print(wind_energy_produced_AN_MCD_new_A50)
# plt.plot(sol_list, wind_energy_produced_AN_MCD_new_A300, '--', c = 'k',alpha = 0.9,label=r"300 $m^2$")
# plt.plot(sol_list, wind_energy_produced_AN_MCD_new_A250, '-', c = 'k',alpha = 0.8,label=r"250 $m^2$")
# plt.plot(sol_list, wind_energy_produced_AN_MCD_new_A200, ':', c = 'k',alpha = 0.7,label=r"200 $m^2$")
# plt.plot(sol_list, wind_energy_produced_AN_MCD_new_A150,'-.', c = 'k',alpha = 0.6, label=r"150 $m^2$")
# plt.plot(sol_list, wind_energy_produced_AN_MCD_new_A100,'--', c = 'k',alpha = 0.55, label=r"100 $m^2$")
# plt.plot(sol_list, wind_energy_produced_AN_MCD_new_A50, '-', c = 'k',alpha = 0.5, label=r"50 $m^2$")
#
# plt.legend()
# plt.xlabel('Martian sols')
# plt.ylabel(r'Electrical energy $P_{sol}$[kWh]')
# plt.xlim(0,668)
# plt.grid()
# plt.show()

linewidth1 =3
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(15, 9))
ax[0, 0].grid(True, which="major", color="#999999", alpha=0.75)
ax[0, 0].grid(True, which="minor", color="#DDDDDD", ls="--", alpha=0.50)
ax[0, 0].minorticks_on()
ax[0, 0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0, 0].tick_params(which='minor', length=5, width=2, direction='in')

ax[0, 0].set_xlabel('Martian sols')
ax[0, 0].set_ylabel(r'Electrical power per sol $P_{sol}$[kWh]')

ax[0, 0].plot(sol_list,wind_energy_produced_AN_MCD_new_A300*1000/24,'--',  label=r"300 $m^2$", color = 'k',alpha = 0.9, linewidth=linewidth1)
ax[0, 0].plot(sol_list,wind_energy_produced_AN_MCD_new_A250*1000/24,'-',  label=r"250 $m^2$", color = 'k',alpha = 0.8, linewidth=linewidth1)
ax[0, 0].plot(sol_list,wind_energy_produced_AN_MCD_new_A200*1000/24,':',  label=r"200 $m^2$", color = 'k',alpha = 0.7, linewidth=linewidth1)
ax[0, 0].plot(sol_list,wind_energy_produced_AN_MCD_new_A150*1000/24,'-.',  label=r"150 $m^2$", color = 'k',alpha = 0.6, linewidth=linewidth1)
ax[0, 0].plot(sol_list,wind_energy_produced_AN_MCD_new_A100*1000/24,'--',  label=r"100 $m^2$", color = 'k',alpha = 0.55, linewidth=linewidth1)
ax[0, 0].plot(sol_list,wind_energy_produced_AN_MCD_new_A50*1000/24,'-',  label=r"50 $m^2$", color = 'k',alpha = 0.5, linewidth=linewidth1)
ax[0, 0].set_xlim(0, 669)
ax[0,0].set_ylim(0, )

plt.legend(facecolor="white", framealpha=1, fancybox=True,ncol=2, edgecolor="black", loc='best')
fig.savefig("wind_power_for_kites.png", dpi=300)
