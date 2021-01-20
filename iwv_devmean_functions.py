# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import math
import time
import os,glob
import re

def offzen_data(t,azi,prw_offzen,elevation,flag):
    """
    Written by Lennéa Hayo, 2020-11-13
    
    Organizes data, collected at an elevation angle of ca. 30 (off zenith), on a fixed grid of 48x72. 
    Every scan is made up of two rounds of data, which are meaned and air mass corrected. Data where the instrument
    "looks" directly into the sun is exchanged for NaN.
    
    Parameters: 
        t: time data in seconds since 1970-01-01 00:00:00 UTC
        azi: sensor azimuth angle (0=North, 90=East, 180=South, 270=West)
        prw_offzen: off zenith path integrated water vapor 
        elevation: retrieval elevation angle
        
    Returns:
        azimuth: sensor azimuth angle (48x72)
        time_ofday: time when data was taken in (hours + portion of minutes)
        airmass_corr: air mass corrected data
    """
    data_list = list(zip(t,elevation,prw_offzen,azi,flag))
    azi_pre = np.zeros((48,72))
    airmass_corr = np.full((48,72),np.nan)
    time_ofday = np.stack([np.full(72, x) for x in np.array(range(48))/2])
    
    if all(i > 30 for i in elevation):
        raise Exception('Dataset does not include scans')

    i = 0
    j = 0
    line_advanced = False
    last_time = data_list[0][0]
    print_warning = True

    for data in data_list:
        result_time = time.gmtime(data[0])

        if data[1] > 30:
            last_time = data[0]
            if line_advanced: 
                continue
            i += 1
            j = 0
            line_advanced = True
            continue
        else:    
            line_advanced = False

        if data[0] - last_time > 20*60 or (i==0 and j==0 and (result_time.tm_hour + (result_time.tm_min/60))!=0.):
            while (result_time.tm_hour + (result_time.tm_min/60))*2 - 1 > i:
                i += 1
            j = 0
            
        #organizing data onto 48x72 grid and taking the mean of two scans
        if j < 72:
            azi_pre[i,j] = data[3]
            time_ofday[i,j] = result_time.tm_hour + (result_time.tm_min/60)
            if data[4] & 0b1000000:
                airmass_corr[i,j] = np.nan
            else:    
                airmass_corr[i,j] = np.sin(np.deg2rad(data[1]))*data[2]
        elif j < 144:
            if data[4] & 0b1000000:
                airmass_corr[i,j-72] = np.nan
            else:
                airmass_corr[i,j-72] = (np.sin(np.deg2rad(data[1]))*data[2] + airmass_corr[i,j-72])/2
        elif print_warning:
            print('Warning: array too long, data taken out')
            print_warning=False
            airmass_corr[i] = np.tile(np.nan,(1,72))
        j += 1
        if azi_pre[i].all() != 0 and 'azimuth' not in locals():
            azimuth = np.tile(azi_pre[i],(48,1)) 
    
    #safety, should not arise
    if not 'azimuth' in locals():
        raise Exception('Azimuth not defined. No scan data available')
    return azimuth, time_ofday, airmass_corr   



    
    
def azi_sort(azimuth, airmass_corr):
    """
    Sorts the azimuth and corresponding IWV data, so that the Azimuth increases from 0-360.
    
    Parameters:
        azimuth: sensor azimuth angle (48x72)
        airmass_corr: airmass corrected data 
        
    Returns:
        iwv_azi_sort: sorted sensor azimuth angle
        airmass_corr: equally sorted airmass corrected data
    """
    iwv_azi_sort = np.zeros(shape=(48,72))
    for i in range(len(azimuth)):
        zipped_lists = zip(azimuth[i], airmass_corr[i])
        sorted_pairs = sorted(zipped_lists)

        tuples = zip(*sorted_pairs)
        iwv_azi_sort[i], airmass_corr[i] = [ list(single) for single in  tuples]
        
    return iwv_azi_sort, airmass_corr

def dev_mean(time_ofday,azimuth,airmass_corr,filename,titledate):
    """
    Written by Lennéa Hayo, 2020-11-13
    
    Plots the deviation from the mean over each scan. Creates a plot for the entire day.
    
    Parameters:
        time_ofday: time when data was taken in (hours + portion of minutes)
        azimuth: sensor azimuth angle (48x72)
        airmass_corr: air mass corrected data
        filename: path where file is saved
        titledate: title for plot, only shows date as yyyymmdd
        
    Returns:    
        iwv_dev_mean: deviation from mean of each scan as array 
    """
    iwv_val_mean = np.nanmean(airmass_corr,axis=1)
    iwv_val_mean = np.reshape(iwv_val_mean,(48,1))

    iwv_dev_mean = [[]] #gives deviation of iwv values from mean 
    for i in range(len(airmass_corr)):
        iwv_dev_mean.append([])
        for j in range(len(airmass_corr[i])):
            iwv_dev_mean[i].append(float(airmass_corr[i][j] - iwv_val_mean[i]))

    #find which is higher, min or max, then make the bigger number vmin and vmax in plot 
    max_dev = []
    min_dev = []
    for i in range(len(iwv_dev_mean)-1):
        max_dev.append(max(iwv_dev_mean[i]))
        min_dev.append(min(iwv_dev_mean[i]))
    if abs(np.nanmax(max_dev)) > abs(np.nanmin(min_dev)):
        abs_highest = abs(np.nanmax(max_dev))
    else:
        abs_highest = abs(np.nanmin(min_dev))
    #print(abs_highest)        


    xticks = [0,90,180,270,360]
    xlabels = ['N','E', 'S', 'W', 'N']

    plt.figure(figsize=(6.4,8))
    plt.pcolormesh(azimuth,time_ofday,iwv_dev_mean[:-1], cmap='seismic',vmin=-abs_highest,vmax=abs_highest)
    plt.xticks(xticks,xlabels)           
    plt.xlabel('Azimuth angle')
    plt.ylabel('Time of day [UTC]')
    plt.title('{}'.format(titledate))
    plt.rc('axes', labelsize=12) 
    plt.colorbar(label=r'deviation from the IWV mean per timestep in (kg m$^{-2}$)')
    plt.savefig(filename)
    
    return iwv_dev_mean

def dev_mean_data(time_ofday,azimuth,airmass_corr):
    """
    Written by Lennéa Hayo, 2020-12-18
    
    Creates deviation from mean data.
    
    Parameters:
        time_ofday: time when data was taken in (hours + portion of minutes)
        azimuth: sensor azimuth angle (48x72)
        airmass_corr: air mass corrected data
        
    Returns:    
        iwv_dev_mean: deviation from mean of each scan as array
    """
    iwv_val_mean = np.nanmean(airmass_corr,axis=1)
    iwv_val_mean = np.reshape(iwv_val_mean,(48,1))

    iwv_dev_mean = [[]] #gives deviation of iwv values from mean 
    for i in range(len(airmass_corr)):
        iwv_dev_mean.append([])
        for j in range(len(airmass_corr[i])):
            iwv_dev_mean[i].append(float(airmass_corr[i][j] - iwv_val_mean[i]))
    return iwv_dev_mean  

def monthly_devmean(path): 
    """
    Written by Lennéa Hayo, 2020-12-18
    
    Creates monthly deviation of mean for every azimuth angle.
    
    Parameters:
        path: path to directory including month, e.g. '/data/obs/site/nya/nyhat/l2/2020/02'
        
    Returns:    
        azimuth: sensor azimuth angle (48x72)
        monthly_mean: vertical mean over entire moth
        amc_devmean_std: standard deviation of data
    """
    
    #generate file with all files containing sequence
    files = []
    sequence = './**/*prw*.nc'
    os.chdir(path)
    for file in glob.glob(sequence, recursive=True):
        files.append(os.path.realpath(file))
    files = np.array(files)
    print(files)
    
    amc_month = np.full((1,72),np.nan)
    #generate data for every day
    for file in files: 
        iwv_data = Dataset(file)
        iwv_data.variables.keys()
        t = np.array(iwv_data['time'])
        azi = np.array(iwv_data['azi'])
        elevation = np.array(iwv_data['ele'])
        prw_offzen = np.array(iwv_data['prw_off_zenith'])
        
        azimuth, time_ofday, airmass_corr = offzen_data(t,azi,prw_offzen,elevation)
        azimuth, airmass_corr = azi_sort(azimuth, airmass_corr)
        iwv_dev_mean = dev_mean_data(time_ofday, azimuth, airmass_corr)
        amc_month = np.vstack((amc_month,iwv_dev_mean[:-1]))    
    
    monthly_mean = np.nanmean(amc_month,axis=0)
    amc_devmean_std = np.nanstd(amc_month,axis=0)
    return azimuth, monthly_mean, amc_devmean_std


def daily_devmean_scan(filename):
    """
    Written by Lennéa Hayo, 2020-11-20
    
    Creates plot daily deveation from mean plot.
    
    Parameters: 
        filename: 'path to file and file.nc' 
    
    """
    dir_path = os.path.dirname(os.path.realpath(filename))
    savefilename = os.path.basename(os.path.realpath(filename))
    
    pattern = re.compile('v00_(\d{8})')
    titledate = pattern.findall(savefilename)[0]
    savefilename = '{}/{}_devmean.png'.format(dir_path,titledate)
    iwv_data = Dataset(filename)
    iwv_data.variables.keys()
    t = np.array(iwv_data['time'])
    azi = np.array(iwv_data['azi'])
    elevation = np.array(iwv_data['ele'])
    prw_offzen = np.array(iwv_data['prw_off_zenith'])
    flag = np.array(iwv_data['flag'])
    
    azimuth, time_ofday, airmass_corr = offzen_data(t,azi,prw_offzen,elevation,flag)
    azimuth, airmass_corr = azi_sort(azimuth, airmass_corr)
    
    #plotting function
    dev_mean(time_ofday,azimuth,airmass_corr,savefilename,titledate)