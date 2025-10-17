# Małgorzata Anotnik, Space Research Centre PAS (CBK PAN), Bartycka 18a, 00-716 Warsaw, Poland
# mantonik@cbk.waw.pl
# 17.10.2025
# Program to calculate and save parameters of solar wind from SWAP data

import numpy as np
from scipy.integrate import quad
from numpy import sin, cos, arcsin, sqrt, exp, pi, where, heaviside, array, empty, arcsin, arccos, tan, abs, cross, dot, log, sum, matrix
from scipy.optimize import minimize
import torch
from torch import Tensor
from torch import sqrt as torch_sqrt
from torch import sin as torch_sin
from torch import cos as torch_cos
from torch import where as torch_where
from torch import empty as torch_empty
from torch import min as torch_min
from torch import abs as torch_abs
from torch import argmin as torch_argmin
from torch import exp as torch_exp
from torch import pi as torch_pi
from torch.special import gammaincc
from torchquad import set_up_backend, Trapezoid, GaussLegendre
from torch import zeros as torch_zeros
from math import radians, gamma
import solarsystem
from astropy.coordinates import spherical_to_cartesian
from astropy.time import Time
import astropy.units as u
from numpy.linalg import norm
import numpy.polynomial.chebyshev as cheb 
import glob
import scipy.special as sc
from uncertainties import unumpy
from scipy.io import readsav
import matplotlib.pyplot as plt
import matplotlib

##############################################################################################################################################################################################################################################################
# functions

# kappa function
# input parameters: density (n), thermal velocity (v_th), kappa parameter (kappa), speed (w)
def kappa_distribution(n, v_th, kappa, w):
    kappa_d = n / ( (pi * v_th)**1.5 * (kappa - 1.5)**1.5)  * ( gamma(kappa+1)/gamma(kappa-0.5) ) * ( 1. + (1./(kappa-1.5)) * (w**2.)/(v_th) )**(-kappa-1.)
    return kappa_d

# generalized filled shell model
# input parameters: ionization rate normalized to 1 au (beta_0), cooling index (alpha), PUI injection speed (v_b), speed (w), density at the termination shock (n_TS), cavity size (lamb), the angle between the ISN inflow direction and the radial position (theta)
def filled_shell_model(beta_0, alpha, v_b, w, n_TS, lamb, theta):
    if beta_0 == 1: # survival probability (S) for He+ PUI is assumed as 1
        S = 1
    else:           # survival probability (S) for H+ PUI is calculated
        S  = torch_exp( -cxsig(v_b) * (NH_density(r*torch_ones_gamma) - NH_density(r*(w/v_b)**alpha)) )
    v_b_flag  = torch_where(w<v_b, 1.0, 0.0)
    fs_model = 1./(4.*pi) * (beta_0*r_0**2./u_p) * ( (alpha*S)/(r*v_b**3.) ) * (w/v_b)**(alpha-3.) * n_TS * torch_exp( - (lamb/r) * theta/(np.sin(theta)) * (w/v_b)**(-alpha) )
    return fs_model*v_b_flag


# function to calcucalte speed form velocity vector
# input parameters: bulk speed (v), velocity vector (u_x, u_y, u_z), ions inflow angles (eta, psi)
def v_to_w(v, u_x, u_y, u_z, eta, psi):
    w = torch_sqrt((v*torch_sin(eta)*torch_cos(psi)-u_x)**2. + (v*torch_sin(eta)*torch_sin(psi)-u_y)**2. + (v*torch_cos(eta)-u_z)**2.)
    return w

# function calculate tensor of SWAP response function for PUIs
# input parameter: angle (eta), velocity (v), E_beam/E_step (E_beam_E_step) from original energy-angle response function 
def response_eta_v(eta, v, E_beam_E_step):
   if torch_min(v) >= 0:
    eta_index = eta/torch_pi*(response_size_PUI-1)
    v_index=torch_empty(grid_size_PUI**3)
    for i_v in range(grid_size_PUI):
        arg_pom=torch_argmin(torch_abs(v[i_v*grid_size_PUI**2]-E_beam_E_step))
        v_index[i_v*grid_size_PUI**2:(i_v+1)*grid_size_PUI**2]=arg_pom
    resp=response_function_PUI[0,0,v_index.round().long(), eta_index.round().long()]
   else:
    resp = 0
   return resp

# function calculate tensor of SWAP response function for proton, alpha and He+
# input parameter: response function
def response_eta_v_grid(response_function):
    eta_index = torch.arange(grid_size_kappa).repeat_interleave(grid_size_kappa)          
    eta_index = eta_index.repeat(grid_size_kappa)                               
    v_index = torch.arange(grid_size_kappa).repeat_interleave(grid_size_kappa **2) 

    resp = response_function[0, 0, v_index, eta_index]
    return resp
    
# function calculates the rotation matrix
# input parameters: New Horizons vector (NH), Earth vector (Earth)
def rot_matrix(NH, Earth):
    z_prim = NH - Earth
    z_prim_vers = (z_prim)/norm(z_prim)
    x_prim = cross(z_prim_vers,[0,0,1])
    x_prim_vers = (x_prim)/norm(x_prim)
    y_prim_vers = cross(z_prim_vers,x_prim_vers)
    rot_matrix = matrix([x_prim_vers,y_prim_vers,z_prim_vers])
    return rot_matrix

# function calculates PUI models
def model_kappa(params, m_particle, bin_start, bin_stop, v_c, response_values, integration_domain, response_norm):
    n_protons, u_sw, T, kappa = params

    u_sw_vector = dot(rot_matrix(NH, Earth), NH_norm*u_sw)
    u_x =  u_sw_vector[0,0] - v_NH[0,0]
    u_y =  u_sw_vector[0,1] - v_NH[0,1]
    u_z =  u_sw_vector[0,2] - v_NH[0,2]

    result_H = empty(bin_stop-bin_start)
    for i in range(bin_start, bin_stop):
       result_H[i-bin_start] = integrate_jit_compiled_parts(lambda x: response_values * kappa_distribution(n_protons ,(2.*kb*T)/m_particle, 1/kappa, v_to_w(x[:,0], u_x,u_y, u_z, x[:,1], x[:,2])) * x[:,0]**3. * torch_sin(x[:,1]), integration_domain=integration_domain[i,:,:])
    model_counts = v_c[bin_start:bin_stop]*geometric_factor[bin_start:bin_stop]*g_t*result_H/(response_norm[bin_start:bin_stop])
    return model_counts*counts_per_s

# function calculates PUI models
def model_PUI(params, all_model, v_c, response_values, hs, grid_points, integration_domain, response_norm, u_x_model,u_y_model, u_z_model, n_H_TS, lamb, theta):
    beta_0, alpha, v_b = params
    beta_0 = 10**beta_0
    if all_model == 1: wh_all[:] = 1

    result_PUI_H = empty(64)
    for i in range(64):
       if wh_all[i] == 1:
        function_values, _ = integaration_method_PUI.evaluate_integrand(lambda x: response_values*  filled_shell_model(beta_0, alpha, v_b, v_to_w(x[:,0], u_x_model,u_y_model, u_z_model, x[:,1], x[:,2]), n_H_TS, lamb, theta) *  x[:,0]**3. * torch_sin(x[:,1]), grid_points[i,:,:])
        result_PUI_H[i] = integaration_method_PUI.calculate_result(function_values, 3, grid_size_PUI, hs[i,:], integration_domain[i,:,:])  
    model_counts = v_c*geometric_factor*g_t*result_PUI_H/(response_norm)
    return model_counts*counts_per_s

# The maximum likelihood method
# input parameters: model parameters (params), observations, counts to add (model_to_add), and model_nr (0 - proton, 1 - alpha, 2 - H PUI) 
def log_likelihood(params, observations, model_to_add, model_nr):
    if model_nr==0:
        model_values = model_kappa(params, m_p, bin_start_p, bin_stop_p, v_c, response_values_proton, integration_domain_proton, response_norm_H) + model_to_add
    if model_nr==1:
        model_values = model_kappa([params[0], params[1],params[2],result_proton.x[3]],m_He, bin_start_alpha,bin_stop_alpha,v_c_alpha, response_values_alpha, integration_domain_alpha, response_norm_alpha) + model_to_add
    if model_nr==2:
        model_values_all = model_PUI(params, 0, v_c, response_values_PUI_H, hs_PUI_H, grid_points_PUI_H, integration_domain_PUI, response_norm_H, u_x_model,u_y_model, u_z_model, n_H_TS, lamb, theta)
        model_values=model_values_all[wh_all == 1]+model_to_add
    L = -2*sum(observations * log(model_values) - model_values +  observations - observations*log(observations))
    return L

# the function generates the integration domains
# input parameters: velocity: v_min, v_max, and angle: eta_max
def integration_domain(v_min, v_max, eta_max):
    integration_domain = torch_zeros(64, 3, 2)
    integration_domain[:, 0, 0] = v_min
    integration_domain[:, 0, 1] = v_max
    integration_domain[:, 1, 0] = 0
    integration_domain[:, 1, 1] = radians(eta_max)
    integration_domain[:, 2, 0] = 0
    integration_domain[:, 2, 1] = 2 * np.pi
    return integration_domain

# distribution of ISN atoms
# input parameters: distance (r), density at the termination shock (n_TS), cavity size (lamb), the angle between the ISN inflow direction and the radial position (theta)
def n_r(r, n_TS, lamb, theta):
    n = n_TS * exp(-(lamb/r)*(theta/sin(theta)))
    return n

# column density of ISN hydrogen
# input parameter: distance (r)
def NH_density(r):
    N  = n_H_TS * ( r * torch_exp(- (lamb/r) * theta/sin(theta)) -  (lamb*theta)/sin(theta) * gamma_0*gammaincc(torch_empty_gamma, (lamb/r) * theta/sin(theta) ) )
    return N

# probability of production of an He+ ion from a solar wind He2+ ions
# input parameter: distance (r)
def prob_He2p_Hep(r):
  fun = lambda r_prim: n_r(r_prim, n_H_TS, lamb, theta)*sig_H_calc*v_rel_H + n_r(r_prim, n_He_TS, lamb_He, theta_He)*sig_He_calc*v_rel_He
  prob = quad(fun, 0, r) / u_p
  return prob[0]

# charge-exchange cross sections 
# input parameter: PUI injection speed (v_b)
def cxsig(v_b):
   energy_v_b_keV = 0.001 * 0.5*m_p*v_b*v_b/e
   cxsig_res = 1e-16 * 4.8569 * ( log(21.906/energy_v_b_keV + 31.487) / (1 + 0.12018*energy_v_b_keV + 4.1402e-6 * energy_v_b_keV**3.7524 +  8.8476e-12*energy_v_b_keV**6.1091)) * 1e-4   
   return cxsig_res

#*************************************************************************************************************************************************************************************************************************************************************

# saving results to .txt file
def save_to_file():
        
        f = open(result_path+str(dates_start[day_nr])+'_n_iter_'+str(n_iter)+'.txt', "w")
        f.write("# "+str(dates_list[day_nr])+"\n")
        f.write("# H+ fit results: u_sw [m s-1], T [K], 1_kappa, n [m-3], L \n")
        f.write(str(u_p)+' '+str(T_p)+' '+str(kappa)+' '+str(n_p)+' '+str(result_proton.fun)+'\n')
        f.write("# He2+ fit results: u_He [m s-1], T_He [K], n [m-3], L \n")
        f.write(str(u_alpha)+' '+str(T_alpha)+' '+str(n_alpha)+' '+str(result_alpha.fun)+'\n')
        f.write("# Hep model results: n_Hep [m-3]\n")
        f.write(str(n_Hep)+'\n')
        f.write("# PUI H+ fit results: alpha, v_b [m s-1], bkg_rate [s-1], beta_0 [s-1], L, bkg_counts \n")
        f.write(str(alpha)+' '+str(v_b_H)+' '+str(bkg_counts.n/counts_per_s)+' '+str(beta_PUI_H)+' '+str(result_PUI_H.fun)+' '+str(bkg_counts.n)+'\n')
        f.write("# PUI He+ calculations results: beta_0, beta_0_err, beta_wh_bin \n")
        f.write(str(beta_PUI_He.n) + ' ' + str(beta_PUI_He.s) + ' ' + str(wh_beta_bin)+'\n')
        f.write("# New Horizons velocities: v_x [m s-1], v_y [m s-1], v_z [m s-1] \n")
        f.write(str(v_NH[0,0])+' '+str(v_NH[0,1])+' '+str(v_NH[0,2])+'\n')
        f.write("# solar wind velocities (u_sw): u_sw_x [m s-1], u_sw_y [m s-1], u_sw_z [m s-1] \n")
        f.write(str(u_sw_vector_model[0,0])+' '+str(u_sw_vector_model[0,1])+' '+str(u_sw_vector_model[0,2])+'\n')
        f.write("# H+ model counts \n")
        f.write(' '.join(str(x) for x in model_proton_all)+'\n')
        f.write("# He2+ model counts \n")
        f.write(' '.join(str(x) for x in model_alpha_all)+'\n')
        f.write("# He+ model counts \n")
        f.write(' '.join(str(x) for x in model_Hep)+'\n')
        f.write("# PUI H model counts \n")
        f.write(' '.join(str(x) for x in model_PUI_H_all)+'\n')
        f.write("# PUI He model counts \n")
        f.write(' '.join(str(x) for x in beta_PUI_He.n*model_PUI_He)+'\n')
        f.close()

##############################################################################################################################################################################################################################################################

# start perameters
days_to_analyze = '2021-01-17*' # one example day; program the program works for plans 12a and 12b: from 2008-10-29 to 2023-03-09
print_output = 0    # 0 - no, 1 - yes
n_itertions = 2     # number of iterations
grid_size_PUI = 55  # grid size for calculating PUI distribution, the minimum recommended value is 40\

data_path = 'C:\\Users\\mantonik\\New_Horizons\\data\\'
histogram_data_path = 'C:\\Users\\mantonik\\New_Horizons\\results_histograms\\all_days\\'
result_path = 'C:\\Users\\mantonik\\New_Horizons\\results_parameters\\results_'

#*************************************************************************************************************************************************************************************************************************************************************
# constants

r_0 = 149597870700.0                # au [m]
m_p = 1.67262192369e-27             # proton mass [kg]
uu =  1.66053906892e-27             # atomic mass [kg]
m_He = 4.002602*uu                  # helium mass
e = 1.602176634e-19                 # elementary charge [C] 
kb = 1.380649e-23                   # Boltzmann constant [J/K = (kg*m2/s2) / K = (kg*m2)/(s2*K)]

lamb = 4.0 * r_0                    # hydrogen ionization cavity size [m]
lamb_He = 0.5 * r_0                 # helium ionization cavity size [m]
n_H_TS = 0.127e+6                   # density of the ISN hydrogen at the termination shock [m-3]
n_He_TS = 0.015e+6                  # density of the ISN helium at the termination shock [m-3]
v_H = 22000.                        # ISN hydrogen velocity [m/s]
v_He = 25400.                       # ISN helium velocity [m/s]
v_th_H = sqrt((2.*kb*7500)/m_p)     # ISN hydrogen thermal speed (T=7500 K)[m/s]
v_th_He= sqrt((2.*kb*7500)/m_He)    # ISN helium thermal speed (T=7500 K)[m/s]
ISN_H_lon = 252.2                   # ISN hydrogen inflow ecliptic longitude [deg]
ISN_H_lat = 9.0                     # ISN hydrogen inflow ecliptic latitude [deg]
ISN_He_lon = 255.7                  # ISN helium inflow ecliptic longitude [deg]
ISN_He_lat = 5.0                    # ISN hydrogen inflow ecliptic latitude [deg]


SWAP_FOV_phi = 139.
grid_size_kappa = 55
eta_max_proton = 55 # [deg]
eta_max_alpha = 80 # [deg]

#*************************************************************************************************************************************************************************************************************************************************************
# torch, integaration methods
torch.set_default_device('cuda')
set_up_backend("torch")
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
integaration_method_kappa = GaussLegendre()
integaration_method_PUI = Trapezoid()
integrate_jit_compiled_parts = integaration_method_kappa.get_jit_compiled_integrate(3, grid_size_kappa**3, backend="torch")

#*************************************************************************************************************************************************************************************************************************************************************
# loading necessary SWAP data: energy bins, geomatric facotr, efficiency, and New Horizons trajectory, E_beam/E_step from .sav data (H.A.Elliott et al.2016)

#energy bins
file_name=data_path + 'histogram_energy_bins.txt'
with open(file_name,'rt') as filedata:
    values = np.genfromtxt(file_name, comments='#', unpack=True)
energy_bins_12a=values[3,:] 
energy_bins_12b=values[4,:]

# geomatric facotr
file_name=data_path + 'geometric_factor.txt'
with open(file_name,'rt') as filedata:
    values = np.genfromtxt(file_name, comments='#', unpack=True)        #[km^2-sr-eV/eV]
geometric_factor=values*1.e+6*0.5 # [m^2-sr-m/s / m/s] km na metry oraz konwersja na predkosciowy czynnik geometryczny

# efficiency
file_name=data_path + 'efficiency.txt'
with open(file_name,'rt') as filedata:
    values = np.genfromtxt(file_name, comments='#', unpack=True,delimiter=',')
MET_efficiency = values[0,:]
efficiency = values[2,:]

# New Horizons trajectory
file_name=data_path + 'traj_NH_HAE_J2000.tab'
with open(file_name,'rt') as filedata:
    values = np.genfromtxt(file_name, dtype=str, unpack=True, delimiter=',')       
MET=values[0,:]
UTC=values[1,:]
NH_X=values[3,:]
NH_Y=values[4,:]
NH_Z=values[5,:]
NH_v_X=values[6,:]
NH_v_Y=values[7,:]
NH_v_Z=values[8,:]
NH_lat=values[9,:]
NH_lon=values[10,:]
NH_range=values[11,:]

# ISN hydrogen and helium inflow coordinates
ISN_H_inflow  = [ cos(radians(ISN_H_lat)) * cos(radians(ISN_H_lon)), cos(radians(ISN_H_lat)) * sin(radians(ISN_H_lon)), sin(radians(ISN_H_lat)) ]
ISN_He_inflow  = [ cos(radians(ISN_He_lat)) * cos(radians(ISN_He_lon)), cos(radians(ISN_He_lat)) * sin(radians(ISN_He_lon)), sin(radians(ISN_He_lat)) ]

# energy-angle response arrays (E_beam/E_step vs theta) from .sav data from file (H.A.Elliott et al.2016)
sav_data = readsav(data_path + 'fin_arr_ebea_ang_eb_bg_corr.sav')
E_beam_E_step = sav_data.s4.y[0] 
E_beam_E_step = E_beam_E_step[:,0]
E_beam_E_step_PUI=torch.linspace(min(E_beam_E_step), max(E_beam_E_step),300)

#*************************************************************************************************************************************************************************************************************************************************************
# load the energy-angle response function for rotation-average FOV
response_function_PUI = torch.from_numpy(np.float32(np.load(data_path + 'response_function_PUI.npy'))).unsqueeze(0).unsqueeze(0).to('cuda')
response_function_proton = torch.from_numpy(np.float32(np.load(data_path + 'response_function_proton.npy'))).unsqueeze(0).unsqueeze(0).to('cuda')
response_function_alpha = torch.from_numpy(np.float32(np.load(data_path + 'response_function_alpha.npy'))).unsqueeze(0).unsqueeze(0).to('cuda')
response_size_PUI=response_function_PUI.size()
response_size_PUI=response_size_PUI[3]

# transform response functions to tensors
response_values_proton = response_eta_v_grid(response_function_proton)
response_values_alpha = response_eta_v_grid(response_function_alpha)
response_values_Hep = response_eta_v_grid(response_function_alpha)

# declaring tensor arrays for PUI integration
grid_points_PUI_H = torch_zeros(64,grid_size_PUI**3, 3)
grid_points_PUI_He = torch_zeros(64,grid_size_PUI**3, 3)
hs_PUI_H = torch_zeros(64,3)
hs_PUI_He = torch_zeros(64,3)

###################################################################################################################### days loop (i_day) ###############################################################################################################################

# files with histogram-type observations
file_list = glob.glob(histogram_data_path + days_to_analyze + "*.txt")
dates_list = []
dates_start = []
for file in file_list:
    dates_list.append(file[file.find("20"):file.find(".txt")])
    date_start_str = str(dates_list[-1])
    dates_start.append(date_start_str[0:10])

day_nr = 0  # number of analyzed days

for i_day in dates_start:                       # days loop (i_day)
    year = i_day[0:4]
    month = i_day[5:7]
    day = i_day[8:10]

    print('Analyzed day:', i_day)

#*************************************************************************************************************************************************************************************************************************************************************
# data from file
#*************************************************************************************************************************************************************************************************************************************************************

# SWAP observations from file
    file_name=file_list[day_nr]
    with open(file_name,'rt') as filedata:
        line = filedata.readlines()[1]      # sweeps
        sweeps=int(line[line.index('=')+2:])

        values = np.genfromtxt(file_name, comments='#', unpack=True)  # observations

    observations_all=values[1,:]      # counts   
    counts_per_s = sweeps*0.39        # counts rate [s-1]
    observations_all_err=values[2,:]*sweeps  
    observations_all_unumpy = unumpy.uarray(observations_all, observations_all_err)
    if sweeps < 1000:
        day_nr = day_nr +1 
        continue
    if np.var(observations_all) == 0:
        day_nr = day_nr +1 
        continue

#*************************************************************************************************************************************************************************************************************************************************************
# trajectory, distance, inflow angles, NH velocity vecotr

    #trajectory
    for i in range(0,8077):
        data_test = UTC[i]
        flag=data_test.find(dates_start[day_nr])
        if flag == 1:
            MET_data_pom = MET[i]
            MET_data = int(MET_data_pom[MET_data_pom.find("/")+1:MET_data_pom.find(":")])
            NH = array([ float(NH_X[i]),  float(NH_Y[i]),  float(NH_Z[i])])
            NH_ecli = array([ float(NH_lat[i]),  float(NH_lon[i])])
            v_ecli = array([ float(NH_v_X[i])*1000.,  float(NH_v_Y[i])*1000.,  float(NH_v_Z[i])*1000.])
    # distance 
    r = norm(NH)*r_0
    # inflow angles
    theta = arcsin( norm(cross(NH, ISN_H_inflow)) / (norm(NH) * norm(ISN_H_inflow)) )
    theta_He = arcsin( norm(cross(NH, ISN_He_inflow)) / (norm(NH) * norm(ISN_He_inflow)) )
    # New Horizons normalized velocity vector (NH_norm)
    H = solarsystem.Heliocentric(year=int(year), month=int(month), day=int(day), hour=23, minute=60 )
    planets_dict=H.planets()
    pos=planets_dict['Earth']
    Earth = array(spherical_to_cartesian(pos[2],radians(pos[1]),radians(pos[0]) ) )
    v_NH = dot(rot_matrix(NH, Earth), v_ecli)              
    NH_norm = (NH)/norm(NH)
    # time-varying normalized SWAP efficiency (g_t)
    g_t = np.interp(MET_data, MET_efficiency, efficiency)/efficiency[0]

#*************************************************************************************************************************************************************************************************************************************************************
    # energy bins
    if Time(dates_start[day_nr]) < Time('2021-08-09T00:49:07'): 
        energy_bins = energy_bins_12a
    else: 
        energy_bins = energy_bins_12b

    # v_c, v_min, v_max for every energy bin
    v_c = sqrt(2.*energy_bins*e/m_p)
    v_c_alpha= sqrt(2.*2*energy_bins*e/m_He)
    v_c_He_PUI = sqrt(2.*energy_bins*e/m_He)
    v_min = Tensor(sqrt(2.*min(E_beam_E_step)*energy_bins*e/m_p))
    v_max = Tensor(sqrt(2.*max(E_beam_E_step)*energy_bins*e/m_p))
    v_min_alpha = Tensor(sqrt(2.*2*min(E_beam_E_step)*energy_bins*e/m_He))
    v_max_alpha = Tensor(sqrt(2.*2*max(E_beam_E_step)*energy_bins*e/m_He))
    v_min_He_PUI = Tensor(sqrt(2.*min(E_beam_E_step)*energy_bins*e/m_He))
    v_max_He_PUI = Tensor(sqrt(2.*max(E_beam_E_step)*energy_bins*e/m_He))
    v_min_Hep = Tensor(sqrt(2.*min(E_beam_E_step)*energy_bins*e/m_He))
    v_max_Hep = Tensor(sqrt(2.*max(E_beam_E_step)*energy_bins*e/m_He))

    # integration domains
    integration_domain_proton = integration_domain(v_min, v_max, eta_max_proton)
    integration_domain_alpha = integration_domain(v_min_alpha, v_max_alpha, eta_max_alpha)
    integration_domain_Hep = integration_domain(v_min_Hep, v_max_Hep, eta_max_alpha)
    integration_domain_PUI = integration_domain(v_min, v_max, SWAP_FOV_phi)
    integration_domain_PUI_He = integration_domain(v_min_He_PUI, v_max_He_PUI, SWAP_FOV_phi)

    # calculate integration grid and response function values for PUIs
    for i in range(0,64):
        grid_points_PUI_H[i,:,:], hs_PUI_H[i,:], n_per_dim = integaration_method_PUI.calculate_grid(grid_size_PUI**3, integration_domain_PUI[i,:,:])
        grid_points_PUI_He[i,:,:], hs_PUI_He[i,:], n_per_dim = integaration_method_PUI.calculate_grid(grid_size_PUI**3, integration_domain_PUI_He[i,:,:])
    response_values_PUI_H,_ = integaration_method_PUI.evaluate_integrand(lambda x: response_eta_v(x[:,1], x[:,0], torch_sqrt(2.*E_beam_E_step_PUI*energy_bins[0]*e/m_p)), grid_points_PUI_H[0,:,:])
    response_values_PUI_He, _ = integaration_method_PUI.evaluate_integrand(lambda x: response_eta_v(x[:,1], x[:,0], torch_sqrt(2.*E_beam_E_step_PUI*energy_bins[0]*e/m_He)), grid_points_PUI_He[0,:,:])

    # loading values ​​normalizing response functions
    if Time(dates_start[day_nr]) < Time('2021-08-09T00:49:07'): 
        response_norm_H = np.load(data_path + 'response_norm_H_12a.npy')
        response_norm_alpha = np.load(data_path + 'response_norm_alpha_12a.npy')
        response_norm_He_PUI = np.load(data_path + 'response_norm_He_PUI_12a.npy')
    else:
        response_norm_H = np.load(data_path + 'response_norm_H_12b.npy')
        response_norm_alpha = np.load(data_path + 'response_norm_alpha_12b.npy')
        response_norm_He_PUI = np.load(data_path + 'response_norm_He_PUI_12b.npy')

#*************************************************************************************************************************************************************************************************************************************************************


# H+ observations to fit 
    observations = observations_all[np.argmax(observations_all)-2: np.argmax(observations_all)+3]
    observations_err = observations_all_err[np.argmax(observations_all)-2: np.argmax(observations_all)+3]
    bin_start_p = np.argmax(observations_all)-2
    bin_stop_p = np.argmax(observations_all)+3


# finding He2+ maximum
    if observations_all[np.argmin(abs(energy_bins-energy_bins[np.argmax(observations_all)]*2))-1] > observations_all[np.argmin(abs(energy_bins-energy_bins[np.argmax(observations_all)]*2))]:
        bin_max_alpha = np.argmin(abs(energy_bins-energy_bins[np.argmax(observations_all)]*2))-1
    else:
        bin_max_alpha = np.argmin(abs(energy_bins-energy_bins[np.argmax(observations_all)]*2))

    if observations_all[bin_max_alpha-1] > observations_all[bin_max_alpha+1]:
        bin_start_alpha = bin_max_alpha - 2
        bin_stop_alpha = bin_max_alpha + 2
    else:
        bin_start_alpha = bin_max_alpha - 1
        bin_stop_alpha = bin_max_alpha + 3
    
    observations_alpha = observations_all[bin_start_alpha: bin_stop_alpha]
    observations_alpha_err = observations_all_err[bin_start_alpha: bin_stop_alpha]

# parameters guess and bounds: H+ [n_p, u_p, T_p, kappa] He2+ [n_alpha, u_alpha, T_alpha]
    u_p_guess = sqrt(2.*energy_bins[np.argmax(observations_all)]*e/m_p) + v_NH[0,2]
    initial_guess_proton = [10000, u_p_guess,  10000., 0.25]
    bounds_proton = [[10,100000], [u_p_guess - 20000,u_p_guess + 20000],[1000,200000],[0.01,0.66]]

    initial_guess_alpha = [200., u_p_guess,  50000.]
    bounds_alpha = [[1,20000],[u_p_guess - 20000,u_p_guess + 20000],[1000,2000000]]

###################################################################################################################### iteration loop (n_iter) ######################################################################################################################

    for n_iter in range(0,n_itertions):

        if n_iter == 1:
            bin_start_p = bin_start_p - 1 # new bins to fit proton model
            bin_stop_p = bin_stop_p + 1
            observations = observations_all[np.argmax(observations_all)-3: np.argmax(observations_all)+4]
            observations_err = observations_all_err[np.argmax(observations_all)-3: np.argmax(observations_all)+4]

# background
        if n_iter ==0:
            bkg_counts = np.mean(observations_all_unumpy[0:4])
        else:
            bkg_counts = np.mean(observations_all_unumpy[0:4] - model_PUI_H_all[0:4])

# minimalization of log-probability function: proton and alpha
        if n_iter == 0: # no H PUI model
            result_proton = minimize(log_likelihood, initial_guess_proton, args=(observations, [0,0,0,0,0], 0), method = 'Powell', bounds=bounds_proton,options={'ftol': 1e-3})
            result_alpha = minimize(log_likelihood, initial_guess_alpha, args=(observations_alpha, [0,0,0,0], 1), method = 'Powell', bounds=bounds_alpha,options={'ftol': 1e-3})
        else:           # we substract H PUI model
            result_proton = minimize(log_likelihood, initial_guess_proton, args=(observations, model_PUI_H_all[bin_start_p:bin_stop_p], 0), method = 'Powell', bounds=bounds_proton,options={'ftol': 1e-3})
            result_alpha = minimize(log_likelihood, initial_guess_alpha, args=(observations_alpha, model_PUI_H_all[bin_start_alpha:bin_stop_alpha], 1), method = 'Powell', bounds=bounds_alpha,options={'ftol': 1e-3})
    
        n_p = result_proton.x[0]
        u_p = result_proton.x[1]
        T_p = result_proton.x[2]
        kappa = result_proton.x[3]

        n_alpha = result_alpha.x[0]
        u_alpha = result_alpha.x[1]
        T_alpha = result_alpha.x[2]

# print the results
        if print_output == 1:
            print(result_proton)
            print(result_alpha)
        
# H+ and He2+ all models
        params_alpha = [n_alpha, u_alpha, T_alpha, kappa]
        model_proton_all = model_kappa(result_proton.x,m_p, 0, 64, v_c, response_values_proton, integration_domain_proton, response_norm_H)
        model_alpha_all = model_kappa(params_alpha,m_He, 0, 64, v_c_alpha, response_values_alpha, integration_domain_alpha, response_norm_alpha)

# data pionts to PUI He+ model
        wh_beta = 8*energy_bins[np.argmax(model_proton_all)]
        wh_beta_bin=np.argmin(abs(energy_bins-wh_beta))+1

# u_x, u_y, u_z for u_sw model
        # proton
        u_sw_vector_model = dot(rot_matrix(NH, Earth), NH_norm*u_p)
        u_x_model =  u_sw_vector_model[0,0] - v_NH[0,0]
        u_y_model =  u_sw_vector_model[0,1] - v_NH[0,1]
        u_z_model =  u_sw_vector_model[0,2] - v_NH[0,2]

        # alpha
        u_sw_vector_model_He = dot(rot_matrix(NH, Earth), NH_norm*u_alpha)
        u_x_model_He =  u_sw_vector_model_He[0,0] - v_NH[0,0]
        u_y_model_He =  u_sw_vector_model_He[0,1] - v_NH[0,1]
        u_z_model_He =  u_sw_vector_model_He[0,2] - v_NH[0,2]

# He+ model
        # He+ density estimation (n_Hep)
        v_rel_H = sqrt(4/pi * (v_th_H**2 + (2.*kb*T_alpha)/m_He) + u_p**2 + v_H**2 + 2*u_p*v_H*cos(theta))
        v_rel_He = sqrt(4/pi * (v_th_He**2 + (2.*kb*T_alpha)/m_He) + u_p**2 + v_He**2 + 2*u_p*v_He*cos(theta_He))
        energy_HkeV = 0.001 * 0.5*m_p*v_rel_H*v_rel_H/e
        logenergy_He=log(0.5*m_He*v_rel_He*v_rel_He/(4*e))
        sig_H_calc  = 1e-16 * 17.438 * ( exp(-2.1263/energy_HkeV)/(1+2.1401e-3*energy_HkeV**1.6498+2.6259e-6*energy_HkeV**3.5+2.4226e-11*energy_HkeV**5.4)  + ( (15.665 * exp(-7.9193 * energy_HkeV)) / energy_HkeV**-4.4053 ) ) * 1e-4
        sig_He_calc= exp(cheb.chebval(  ((logenergy_He - log(0.00024)) - (log(2e6) - logenergy_He)) / ( log(2e6) - log(0.00024) ) , (-83.5447/2, 0.758421, -0.673803, -3.68088, -2.02719, -0.148166, 0.0237951, -0.0738396, 0.243993)) )*1e-4
        n_Hep  = n_alpha * ( (1-exp(-prob_He2p_Hep(r))) / exp(-prob_He2p_Hep(r)) )
        
        # calculating He+ model
        params_Hep =[n_Hep, u_alpha, T_alpha, kappa]
        model_Hep = model_kappa(params_Hep, m_He, 0, 64, v_c_He_PUI, response_values_Hep, integration_domain_Hep, response_norm_He_PUI)

# PUI data points to fit 

        # If the sum of the core solar wind proton and alpha particle models exceeded 10% of the observed counts in a given bin, that bin is excluded from the fit
        model_proton_alpha = model_proton_all + model_alpha_all
        wh_model=where(observations_all >= model_proton_alpha*10,1,0)
        wh_alpha=where(energy_bins <= energy_bins[bin_start_alpha],1,0)
        wh_all = wh_model*wh_alpha

        wh_all[bin_start_p-1] = 0
        wh_all[bin_start_p-2] = 0
        wh_all[bin_stop_p:bin_start_alpha] = 0

        # observations near the theoretical cut-off energy (v_b)
        v_b = abs(v_H - (-u_z_model))
        wh_cutoff = where(energy_bins <= m_p*(u_p+v_b)**2/(2*e),energy_bins,0)
        wh_all[np.argmax(wh_cutoff) - 2:np.argmax(wh_cutoff)+2] = 1
        observations_cut_off=observations_all[np.argmax(wh_cutoff) - 2:np.argmax(wh_cutoff)+2]
        # additional bins with higher energies, provided their counts exceeded 10% of the maximum count among the already selected points
        for i_cut_off in range(np.argmax(wh_cutoff)+2,63):
            if observations_all[i_cut_off] >= 0.1*max(observations_cut_off): 
                wh_all[i_cut_off] = 1
            else:
                break
        
        n_bin_PUI = sum(wh_all)
        energy_bins_PUI = empty(n_bin_PUI)
        observations_PUI = empty(n_bin_PUI)
        model_to_add_PUI = empty(n_bin_PUI)
        observations_PUI_err = empty(n_bin_PUI)
        bins_PUI=[]
        j=0
        for i in range(64):
            if wh_all[i] == 1:
                bins_PUI.append(i)
                energy_bins_PUI[j] = energy_bins[i]
                if n_iter == 0:
                    observations_PUI[j] = observations_all[i]
                    model_to_add_PUI[j] = model_Hep[i]+model_proton_all[i]+model_alpha_all[i]+bkg_counts.n
                else:
                    observations_PUI[j] = observations_all[i]
                    model_to_add_PUI[j] = model_Hep[i]+model_proton_all[i]+model_alpha_all[i]+beta_PUI_He.n*model_PUI_He[i]+bkg_counts.n
                observations_PUI_err[j] = observations_all_err[i]
                j=j+1

# for gamma function
        torch_empty_gamma = torch.zeros(grid_size_PUI**3)+1e-5
        torch_ones_gamma = torch.ones(grid_size_PUI**3)
        gamma_0 = gamma(1e-5)

# H+ PUI model
        if n_iter == 0: v_b_H = v_b
    
        # initial guess and bounds: [beta, alpha, v_b]
        initial_guess_PUI_H = [-6.15, 2.0, v_b_H]
        bounds_PUI_H = [[-7.3,-5.3], [1. , 6. ],[0.8*(v_b_H),1.2*(v_b_H)]]
        
        # minimalization of log-probability function: H PUI
        result_PUI_H = minimize(log_likelihood, initial_guess_PUI_H, args=(observations_PUI, model_to_add_PUI, 2), method = 'Nelder-Mead', bounds=bounds_PUI_H, options={'fatol': 1e-2})
        # calculating all H PUI model
        model_PUI_H_all = model_PUI(result_PUI_H.x,1,v_c, response_values_PUI_H, hs_PUI_H, grid_points_PUI_H, integration_domain_PUI, response_norm_H, u_x_model,u_y_model, u_z_model, n_H_TS, lamb, theta)

        beta_PUI_H = 10**result_PUI_H.x[0]
        alpha = result_PUI_H.x[1]
        v_b_H = result_PUI_H.x[2]

        if print_output == 1: print(result_PUI_H)
        
# He+ PUI model
        # estimation of the helium PUI injection speed (v_b_He)
        v_b_He = v_b_H * ( ( abs(v_He - (-u_z_model)) ) / ( abs(v_H - (-u_z_model)) ) )
         
        # helium PUI model
        params_PUI_He = [0., alpha ,v_b_He]
        model_PUI_He=model_PUI(params_PUI_He,1,v_c_He_PUI, response_values_PUI_He, hs_PUI_He, grid_points_PUI_He , integration_domain_PUI_He, response_norm_He_PUI, u_x_model_He,u_y_model_He, u_z_model_He, n_He_TS, lamb_He, theta_He)
        
        # calculation of the helium ionization rate (beta_PUI_He)
        if wh_beta_bin < 64:
            beta_PUI_He = sum(observations_all_unumpy[wh_beta_bin:]-bkg_counts)/sum(model_PUI_He[wh_beta_bin:])
        else:
            beta_PUI_He = sum(observations_all_unumpy[-1]-bkg_counts)/sum(model_PUI_He[-1])

#*************************************************************************************************************************************************************************************************************************************************************
# Total spectrum
        spectrum_all = model_PUI_H_all + model_proton_all + model_alpha_all + beta_PUI_He.n*model_PUI_He + model_Hep + bkg_counts.n

# save to file
        save_to_file()

    day_nr = day_nr + 1 # end of the loop/day