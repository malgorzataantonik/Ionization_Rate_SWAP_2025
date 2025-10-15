# Małgorzata Anotnik, Space Research Centre PAS (CBK PAN), Bartycka 18a, 00-716 Warsaw, Poland
# 01.07.2023
# Program to calculate energy-angle response function for rotation-average FOV

import numpy as np
from numpy import radians, sin, cos, arcsin, argmin, atan2
import matplotlib.pyplot as plt
from math import floor
from scipy.integrate import quad
from scipy.io import readsav
from scipy import interpolate

# function calculating response from given rotational angle (rho) and angular distance from the center of the FOV (eta)
def fun(rho, eta):

    # transitions from the new coordinates to the SWAP FOV theta and phi coordinates
    theta_calc = arcsin( sin(eta)*sin(rho) ) 
    phi_calc = atan2( sin(eta)*cos(rho), cos(eta) ) 

    # checking the counting conditions
    if (theta_calc < theta_max) and (theta_calc > theta_min) and (abs(phi_calc) <= FOV_phi):

        theta_pom = argmin(abs(theta - theta_calc)) # index of the nearest theta value
        ener_pom=ener_index
        # interpolation in energy steps
        if ener_index < (ener_steps_nr-1):
            ener_interp = interpolate.interp1d([floor(ener_pom), floor(ener_pom)+1], [energy_theta_av[floor(ener_pom),theta_pom],energy_theta_av[floor(ener_pom)+1,theta_pom]])
            resp=ener_interp(ener_index)
        else:
            resp=energy_theta_av[floor(ener_pom),theta_pom]
    else:
        resp = 0  # default value outside the FOV
    return resp


#############################################################################################################################

# SWAP FOV
FOV_phi = radians(138.)

#############################################################################################################################

# reading energy-angle response arrays (E_beam/E_step vs theta) from .sav data from file (H.A.Elliott et al.2016)

try:
    sav_data = readsav('C:\\Users\\mantonik\\New_Horizons\\data\\swap_calibration_model\\fin_arr_ebea_ang_eb_bg_corr.sav')
except:
    print("No data file (fin_arr_ebea_ang_eb_bg_corr.sav)")
    exit(1)

E_beam_E_Step = sav_data.s4.y[0] # energy steps

theta=radians(-sav_data.s4.x[0]) # theta angles [deg]
theta_min = min(theta)
theta_max = max(theta)

arr = sav_data.s4.arr[0] # trensmission data

#############################################################################################################################

# averaged energy-angle response function

energy_theta_av_copy = np.mean(arr[:,:,:].T, axis=2)
energy_theta_av = energy_theta_av_copy.copy()
mask = energy_theta_av < 0
energy_theta_av[mask] = 0
print(np.shape(energy_theta_av))
#############################################################################################################################

# new response function calculations

ener_steps_nr = 30
angular_steps_nr = 180
response_function = np.empty([ener_steps_nr, angular_steps_nr]) # assumed transmission grid (30 energy steps and 180 angular steps, as in .sav data)

for ener_index in range(0,ener_steps_nr): # loop for 30 energy steps
    print(ener_index)
    for eta_index in range(0,angular_steps_nr): # loop for 180 energy steps

        eta=radians(eta_index) # calculating radians of the angular distance from the center of the FOV (eta [0,pi])

        # integration by rotational angle rho = omega + phi (rho [0,2 pi])
        function_to_int = lambda rho: fun(rho, eta)
        result = quad(function_to_int,0.0,2*np.pi)

        response_function[ener_index, eta_index] = result[0]/(2*np.pi) # new normalized response function

#############################################################################################################################

# plot the new energy-angle response function for rotation-average FOV

plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 15
extent = 0,180, min(E_beam_E_Step[:,0]), max(E_beam_E_Step[:,0]) 
fig = plt.figure()
plt.imshow(response_function, interpolation='none',aspect='auto',extent=extent,origin='lower')
plt.xlim(0.1,180)
plt.xscale('log')
plt.xlabel(r'$ \eta $ [deg]')
plt.ylabel(r" $ E_{\mathrm{beam}}/E_{\mathrm{step}} $ ")
plt.colorbar()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tick_params(axis="y", labelsize=15)
plt.show()
