# Małgorzata Anotnik, Space Research Centre PAS (CBK PAN), Bartycka 18a, 00-716 Warsaw, Poland
# 26.09.2023
# Program to read FITS files and save SWAP spectrum to .txt files

from astropy.io import fits
import numpy as np
from astropy.time import Time
import glob
import astropy.units as u
import pandas as pd 

# time selection
# available from 2008-10-29T18:08 to 2023-03-09T18:08
time_start='2012-01-01T18:08'
time_stop='2023-01-31T18:08'

# path names
data_path = 'C:\\Users\\mantonik\\New_Horizons\\data\\'
results_path = 'C:\\Users\\mantonik\\New_Horizons\\results_histograms\\all_days\\'
bins_file_name='C:\\Users\\mantonik\\New_Horizons\\data\\histogram_energy_bins.txt'

# loop for all days with observations
days_series = pd.Series(pd.date_range(start=time_start, end=str(Time(time_stop)-1*u.day), freq="d"))
for i in days_series:                       # 'i' is analyzed date
    data_prev = str(Time(i)-1*u.day)
    data_next = str(Time(i)+1*u.day)

    month=str(i.month)
    if i.month <= 9: month='0'+str(i.month)
    day=str(i.day)
    if i.day <= 9: day='0'+str(i.day)

    time_search=str(i.year)+month+day       # date in the appropriate format
    time_search_prev = data_prev[0:4]+data_prev[5:7]+data_prev[8:10]
    time_search_next = data_next[0:4]+data_next[5:7]+data_next[8:10]

    start_stop_time = str(i.year)+'-'+month+'-'+day+'_'+data_next[0:4]+'-'+data_next[5:7]+'-'+data_next[8:10]

    # plan name for each day of observation: plutocruise, pluto, kemcrusie, kem1, kem2 (0, 2, 5, 12a, 12b)
    if Time(i) < Time('2008-10-29T10:02:34'): print("Error, no data")
    if Time(i) >= Time('2008-09-28T10:02:34'): catalog_name='nh-x-swap-3-plutocruise-v3.0'
    if Time(i) >= Time('2015-01-14T18:09:00'): catalog_name='nh-p-swap-3-pluto-v3.0'
    if Time(i) >= Time('2016-10-25T18:08:59'): catalog_name='nh-x-swap-3-kemcruise1-v2.0'
    if Time(i) >= Time('2018-08-14T20:49:30'): catalog_name='nh-a-swap-3-kem1-v6.0'
    if Time(i) >= Time('2022-04-24T18:08:58'): catalog_name='nh-a-swap-3-kem2-v1.0'
    if Time(i) > Time('2023-03-09T02:02:03'): print("Error, no data")

    # energy bins
    with open(bins_file_name,'rt') as filedata:
            values = np.genfromtxt(bins_file_name, comments='#', unpack=True)
    if Time(i) < Time('2008-09-28T10:02:34'): energy_bins=values[1,:]
    if Time(i) >= Time('2008-09-28T10:02:42'): energy_bins=values[2,:]
    if Time(i) >= Time('2008-10-29T03:04:34'): energy_bins=values[3,:]
    if Time(i) >= Time('2021-08-09T00:49:07'): energy_bins=values[4,:]

    # searching for data files

    files_prev = glob.glob(data_path+catalog_name+'\\data\\'+time_search_prev+'*\\swa*x586_sci.fit')
    files_ok = glob.glob(data_path+catalog_name+'\\data\\'+time_search+'*\\swa*x586_sci.fit')
    files_next = glob.glob(data_path+catalog_name+'\\data\\'+time_search_next+'*\\swa*x586_sci.fit')
    files = np.concatenate((files_prev, files_ok, files_next))

    if len(files) == 0: continue # if no data is available, the next day is checked

    # resetting data before each new day
    tot_count_rate_bin_all_day=0
    sweeps_all_day=0
    UTC_list = ''

    # loop through all files on a given day
    for file in files:
     fits_file_name=file

     # FITS data
     hdul = fits.open(fits_file_name)
     MET_start = hdul[0].header['BEGTIME']
     MET_end = hdul[0].header['ENDTIME']
     UTC_start_time=str(Time('2006-01-19T18:09', out_subfmt='date_hm') + MET_start * u.second)
     UTC_end_time=str(Time('2006-01-19T18:09', out_subfmt='date_hm') + MET_end * u.second)

     # selection of files analyzed within one day defined from 18:09 to 18:09 the following day
     if (UTC_start_time >= Time(i)) & (UTC_end_time <= (Time(i)+1*u.day)):
             
        UTC_start_stop_time = UTC_start_time[0:13]+'-'+UTC_start_time[14:16]+'_'+UTC_end_time[0:13]+'-'+UTC_end_time[14:16]
        # list of start and end observation times from each selected FITS file
        UTC_list = UTC_list + UTC_start_stop_time + ' '

        # defining whether it is Unnormalized Histogram (UH) or High Time Resolution Histogram (HTRH)
        hist_type=0     # control variable specifying the histogram type in the FITS file
        if len(hdul[0].data) == 2048: hist_type='UH' 
        if len(hdul[0].data) == 47: hist_type='HTRH'
        if hist_type == 0: 
            print('Error, no .fits file')
            continue

        #  UH data
        if hist_type == 'UH':
            tot_count_rate_bin_all_day=np.flip(hdul[1].data[1984:]) + tot_count_rate_bin_all_day # all counts in 64 energy bins summed from several files
            sweeps_all_day=hdul[0].header['SMPLSCNT'] + sweeps_all_day # Number of 64-second samples used, summed

        # HTRH data
        if hist_type == 'HTRH':
            sweeps=hdul[0].header['SMPLSCNT'] # sweeps number
            tot_count_rate_bin_all_file=0.0
            for hist_nr in range(0,47):
                if hdul[1].data[hist_nr] == 0.0: continue
                if sweeps-hist_nr*29 >= 29: sweeps_hist=29
                if sweeps-hist_nr*29 < 29: sweeps_hist=sweeps-hist_nr*29
                tot_count_rate_bin=np.flip(hdul[0].data[hist_nr,:])
                count_rate_hz=tot_count_rate_bin/(sweeps_hist*0.39)
                # estimation of uncertainties
                tot_count_rate_bin_err=np.sqrt(tot_count_rate_bin)/(sweeps_hist)
                count_rate_hz_err=np.sqrt(tot_count_rate_bin)/(sweeps_hist*0.39)

                # saving Hight Time Resolution data from one histogram to a table
                header='Energy bin [eV/q]           Total counts            Total counts error \nsweeps = '+str(sweeps_hist)
                np.savetxt(results_path+UTC_start_stop_time+'_'+str(int(hdul[1].data[hist_nr]))+'.txt', np.c_[energy_bins,tot_count_rate_bin,tot_count_rate_bin_err], header=header)

                # summing data from the entire file
                tot_count_rate_bin_all_file=tot_count_rate_bin+tot_count_rate_bin_all_file
             
            # data for the whole day from several files
            tot_count_rate_bin_all_day=tot_count_rate_bin_all_file + tot_count_rate_bin_all_day # all counts in 64 energy bins summed from several files
            sweeps_all_day=sweeps + sweeps_all_day # Number of 64-second samples used, summed

        hdul.close() # closing FITS file

     else:
        hdul.close() # zamkniecie pliku FITS
        continue

    if sweeps_all_day == 0: continue

    # estimation of uncertainties for the whole day
    tot_count_rate_bin_all_day_err=np.sqrt(tot_count_rate_bin_all_day)/(sweeps_all_day) 
    count_rate_hz_all_day=tot_count_rate_bin_all_day/(sweeps_all_day*0.39) 
    count_rate_hz_all_day_err=np.sqrt(tot_count_rate_bin_all_day)/(sweeps_all_day*0.39)


    # saving data from whole day 
    header='Energy bin [eV/q]           Total counts            Total counts error \nsweeps all day = '+str(sweeps_all_day) + '\n' + UTC_list
    np.savetxt(results_path+start_stop_time+'.txt', np.c_[energy_bins,tot_count_rate_bin_all_day,tot_count_rate_bin_all_day_err], header=header)



