# Ionization_Rate_SWAP
Ionization Rate of Interstellar Neutral Helium from New Horizons/SWAP Observations

Manuscript submitted to ApJ

## Main Files

- `SWAP_data.py` – program to read FITS files and save SWAP spectrum to .txt files
- `response_function.py` – program to calculate SWAP energy-angle response function for rotation-average FOV
- `solar_wind_parameters.py` – program to calculate and save parameters of solar wind from SWAP data

# SWAP_data.py

  > ## Requirements

   - Python 3.x
   - Libraries: numpy, astropy, pandas
   
  > ## Usage Instructions

  1. Download the New Horizons/SWAP and/or NH Kuiper Belt Extended Missions (KEM and KEM2)/SWAP calibrated data from [NASA PDS](https://pds-smallbodies.astro.umd.edu/data_sb/by_mission.shtml).
  2. Download table A1 from [D. J. McComas et al 2022 ApJ 934 147](https://iopscience.iop.org/article/10.3847/1538-4357/ac7956#apjac7956t4).
  3. Update the default start/stop times in the script:
     
     ```time_start = "YYYY-MM-DDThh:mm"```
     ```time_stop = "YYYY-MM-DDThh:mm"```

  4. Update the default paths in the script:
  
     for the data files (by default, data is in directories named the same as DATA_SET_ID, for example, 'nh-x-swap-3-plutocruise-v3.0'):

     ```data_path = 'path/to/data'```

     for table with SWAP energy bins (Tab. A1 from D. J. McComas et al. 2022):

     ```bins_file_name='path/to/energy_bins_table'```

     and for output directory:
   
     ```results_path = 'path/to/results'```

  5. Run the script.
   
# response_function.py

  > ## Requirements

   - Python 3.x
   - Libraries: numpy, scipy
   - SWAP energy-angle response arrays (E_beam/E_step vs theta) from .sav data (from H. A. Elliott et al. 2016)
  
  > ## Usage Instructions

  1. Update the default paths in the script:
  
     for the .sav energy-angle response arrays:

     ```sav_data = 'path/to/sav_data'```

  2. Run the script.

# solar_wind_parameters.py

  > ## Requirements

   - Python 3.9+
   - Libraries: numpy, astropy, scipy, pandas, torch, solarsystem, uncertainties
   - SWAP histogram data (generated from SWAP_data.py)
   - additional data files:
     - SWAP histogram energy bins (Tab. A1 from D. J. McComas et al. 2022)
     - SWAP geometric factor 
     - SWAP efficiency
     - SWAP trajectory (from NASA PDS) 
     - SWAP energy-angle response arrays (E_beam/E_step vs theta) from .sav data (from H. A. Elliott et al. 2016)
     - precalculated response functions (response_function_PUI.npy, response_function_proton.npy, response_function_alpha.npy)
     - precalculated values ​​normalizing response functions (response_norm_H_12a.npy, response_norm_alpha_12a.npy, response_norm_He_PUI_12a.npy, response_norm_H_12b.npy, response_norm_alpha_12b.npy, response_norm_He_PUI_12b.npy)

  > ## Usage Instructions

  1. Update the default days to analyze in the script (for example, days_to_analyze = '2021-01-*' to analyze all January 2021):
     
     ```days_to_analyze = "YYYY-MM-DD"```

  2. Update the default paths in the script:
  
     for the data files:

     ```data_path = 'path/to/data'```

     for SWAP histograms generated from SWAP_data.py:

     ```histogram_data_path ='path/to/histogram_data'```

     and for output directory:
   
     ```results_path = 'path/to/results'```

  3. Run the script.
