#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Mon Jan 16 15:44:10 2023

Functions for analysing MITgcm output. 
Output should be loaded using Functions_load_output and used as inputs here. 

@author: gemma
"""

import xarray as xr
import numpy as np
from scipy import stats, signal
from math import isnan
import math
import bisect
from Functions_load_output import load_experiment_ds, get_bathymetry_and_troughs


# %% Location dictionaries

# Plotting regions for making 2d maps
plot_regions = {'AS_far':[-68,-76.5,-135,-90],\
                'AS_long':[-66,-76.5,-135,-90],\
                'AS_near':[-70,-76.5,-120,-95],\
                'bathy_map':[-70,-76.5,-120,-95],\
                'ASBS':[-68,-76.5,-130,-75],\
                'PIB':[-74.5,-76,-105,-98],\
                'Inner Shelf':[-74,-76,-110,-98],\
                'full_model_domain':[-65.5,-76.5,-140,-65.1]}

# Regions for calculating TC at troughs
#idxes corresponding to lat1,lat2,lon1,lon2
trough_idx_dict = {'WOT':[240,246,256,268],'COT':[220,221,310,328],\
                'EOT':[205,206,392,470],'MIT':[310,311,395,418],\
                'MOT':[260,261,384,405],'Thwaites':[355,356,401,412],\
                'PIG':[351,368,466,470],'Thwaites_East':[352,354,422,425],\
                'PITT':[332,342,405,425]}

# Regions for calculating TC depth by lon
lon_cross_dict = {'CT_1':[-113.65,-69,-75.5],\
                  'CT_2':[-110.5,-69,-75.5],\
                  'ET_1':[-108,-69,-75.5],\
                  'ET_2':[-105.7,-69,-75.5],\
                  'ET_3':[-103,-69,-75.5],\
                  'ET_4':[-100.5,-69,-75.5]}

# These are narrower sections of just one shelf break trough
#'CT_cross':[-71.8,-115,-112],'ET_cross':[-71.4,-109,-100],\
#'SB0':[-71.1,-117,-98],\
lat_cross_dict = {\
                  # 'SB_1':[-71.3,-117,-98],\
                  'lat_1':[-71.6,-117,-98],\
                  'lat_2':[-72.5,-117,-98],\
                  'lat_3':[-74,-117,-98],\
                  'lat_4':[-74.8,-117,-98],\
                  # 'IS_2':[-75.0,-117,-98],\
                  'lat_5':[-75.4,-117,-98]}

# For map region analyses, e.g. shelf-break undercurrent strength, total-shelf means, heat budget. 
analysis_region_dict = {'thesis_shelf_box':[-71.6,-76,-111,-100],
                        'naughten_shelf_box':[-70.8,-76,-115,-100],
                        'shelf_break':[-70.9,-71.9,-115,-100],
                        'inner_shelf_corner':[-75,-76,-109,-98],
                        'PIG_shelf':[-75.2,-75.9,-101.5,-98],
                        'ase_domain':[-70,-76,-115,-100]}
                    # naughten shelf box is a bigger shelf-box that is used to search for shelf-break

# Simple analyses for model grid-------------------------------------------

def calculate_distance(coord1, coord2):
    """
    Calculate distance in m between two grid points (called in calc_grid_sizes func below)
    Either two lats or two lons (not yet capable of calculating for different lat/lon pairs)
    
    Inputs
    ------
    coord1: float, lat or lon
    coord2: float, lat or lon (must be lat or lon like coord1)
    
    Returns
    -------
    distance: float
        distance between two points, in m
    """
    
    # Convert latitude or longitude values from degrees to radians
    coord1_rad = math.radians(coord1)
    coord2_rad = math.radians(coord2)

    # Earth radius in kilometers
    earth_radius = 6371.0

    # Calculate the difference (in lats or lons)
    dcoord = coord2_rad - coord1_rad

    # Calculate the distance using the Haversine formula
    distance = 2 * earth_radius * math.asin(math.sqrt(math.sin(dcoord/2)**2))
    
    distance_m = distance*1000

    return distance_m


def calc_grid_areas(data):
    """
    Calculates areas (in m^2) of each grid cell in the data array
    Uses function above to calculate distance between lats and lons
    Data is the input so that it only performs the calc on the data provided (i.e. on the shelf, not always a simple region)

    Inputs
    ------
    data: n-dimensional xr data array (xr ds may also work)
        any dimensions ok as long as contains lat and lon coords
        i.e., exp T. must contain coordinates lat or lon (need to alter for other coordinate names)

    Outputs
    -------
    grid_areas: np array of shape (n_lats-1,n_lons-1)
        contains grid area in m^2 of each grid point in input data array

    """
    n_lats, n_lons = len(data.lat), len(data.lon)
    #n_lats,n_lons = 4,4
    grid_areas = np.zeros((n_lats-1,n_lons-1))

    # calculate area in each grid point in selected region
    for i in range(n_lats-1):

        # Calculate lat distance of grid cell
        lat1,lat2 = data.lat[i].values,data.lat[i+1].values
        dist_i = calculate_distance(lat1,lat2)
        if i == 0:
            print('distance between first set of lats:', lat1, lat2, ':', 
                  dist_i, 'meters')

        for j in range(n_lons-1):
            lon1, lon2 = data.lon[j].values, data.lon[j+1].values
            dist_j = calculate_distance(lon1, lon2)
            if i == 0 and j == 0:
                print('distance between first set of lons:', lon1, lon2, ',', 
                      dist_j, 'meters')

            cell_area = dist_i * dist_j
            grid_areas[i, j] = cell_area

    return grid_areas


# LOADING DATA------------------------------------------------------------------------------------

# Load single variable  data for all experiments and save in a data array (takes about 40 secs per run)
def get_ensemble_data_for_ic(vname, runs, region, prefix, month_start_idx = 0, n_months = 60, depth=None):
    
    """
    ***TO DO***
    Figure out how to use dask to save xr data arrays faster (like xmitgcm)
    
    Make a data array containing all data from a set of runs 
    Calls load_experiment_ds to load the state or EXF file for each run, 
    and saves the data from each run into one master array. 
    
    Searches casper directory first then scratch directory.
    
    Intended for conducting analyses on ensembles of data, such as calculating
    ensemble mean or anomalies relative to an ensemble mean. 
    
    **Uncomment code adjusting for sine and cosine if necessary. 
    It looks the same visaully so I commented it out for simply plotting maps. 
    Probably want to uncomment for doing calculations like heat transport. 
    
    Inputs
    ------
    vname : str
        variable name corresponding to var_meta_dict definited in function
    runs : list of strings 
        contains experiment names matching Datasets.py
    region : str
        region to average data over matching keys in plot_regions defined at top
    month_start_idx : int, optional
        idx of first timestep you want. 
        I.e. if you want the 5th year of data (and runs start in feb), set to 47 and set n_months to 12
    n_months : int, optional
        each dataset needs to be the same size for turning into a data array
        if month_start_idx is not specified, takes first n_months of data
        
    
    Returns
    ------
    all_run_data_da: xr data array with the run as the first coordinate
        grid data is named lat and lon
        (i.e., lat_at_f_loc is saved as just lat if it's not already called lat')
        2d states have shape (n_runs, n_times, n_lats, n_lons)
        3d datasets have shape (n_runs, n_times, n_depths, n_lats, n_lons)
    
    """
    
    
    print('Getting',vname,'data for all runs...')
    n_runs = len(runs)
    if region in plot_regions.keys():
        lat1,lat2,lon1,lon2 = plot_regions[region]
        reg_type = 'map'
    elif region in lon_cross_dict.keys():
        lon,lat1,lat2 = lon_cross_dict[region]
        reg_type = 'lon_cross'
    elif region in lat_cross_dict.keys():
        lat,lon1,lon2 = lat_cross_dict[region]
        reg_type = 'lat_cross'
        
    var_meta_dict = {
                     #state_3d_set2
                     'T':['THETA','degC','lat','lon'],
                     'S':['SALT','psu','lat','lon'],
                     'U':['UVELMASS','m/s','lat','lon_at_f_loc'],
                     'V':['VVELMASS','m/s','lat_at_f_loc','lon'],
                     'W':['WVELMASS','m/s','lat','lon'],
                         
                     #EXF_forcing_set2
                     'uwind':['EXFuwind','m/s','lat','lon'],
                     'vwind':['EXFvwind','m/s','lat','lon'],
                     'taux':['EXFtaux','N/m^2','lat','lon'],
                     'tauy':['EXFtauy','N/m^2','lat','lon'],
                     'pressure':['EXFpress','N/m^2','lat','lon'],
                     'atemp':['EXFatemp','degC','lat','lon'],
        
                     
                         
                     #state_2d_set2
                     'ETAN':['ETAN','m','lat','lon'],\
                     'SIarea':['SIarea','m^2/m^2','lat','lon'],\
                     'SIheff':['SIheff','m','lat','lon'],\
                     'MXLDEPTH':['MXLDEPTH','m','lat','lon'],\
                     'oceQnet':['oceQnet','W/m^2','lat','lon'],\
                     'oceFWflx':['oceFWflx','kg/m^2/s','lat','lon'],\
                     'oceTAUX':['oceTAUX','N/m^2','lat_at_f_loc','lon'],\
                     'oceTAUY':['oceTAUY','N/m^2','lat','lon_at_f_loc'],\
                     
                     #budg2d_zflux_set2
                     'SIdAbATO':['SIdAbATO','m^2/m^2/s','lat','lon'],\
                     'SIdAbATC':['SIdAbATC','m^2/m^2/s','lat','lon'],\
                     'SIdAbOCN':['SIdAbOCN','m^2/m^2/s','lat','lon'],\
                     'SIdA':['SIdA','m^2/m^2/s','lat','lon'],\
                     'oceFWflx':['oceFWflx','kg/m^2/s','lat','lon'],
                     'SIatmFW':['SIatmFW','kg/m^2/s','lat','lon'],
                     'SIaQbOCN':['SIaQbOCN','m/s','lat','lon'],
                     'SIaQbATC': ['SIaQbATC','m/s','lat','lon'],
                     'SIaQbATO': ['SIaQbATC','m/s','lat','lon'],
        
                    
                     #iceshelf_state_set2
                     'SHIfwFlx':['SHIfwFlx','kg m-2 s-1','lat','lon'],
                     'SHIhtFlx':['SHIhtFlx','W m-2','lat','lon']
                     }

    """
    All output vars:
    EXF_forcing_set2: 'EXFhs   ', 'EXFhl   ', 'EXFlwnet', 'EXFswnet', 'EXFlwdn ',
                  'EXFswdn ','EXFqnet ','EXFtaux ','EXFtauy ','EXFuwind',
                  'EXFvwind','EXFwspee','EXFatemp','EXFaqh ','EXFevap ',
                  'EXFpreci','EXFsnow ','EXFempmr','EXFpress','EXFroff ',
                  'EXFroft ',
    state_2d_set2: 'ETAN    ','SIarea  ','SIheff ','SIhsnow ',
                 'DETADT2 ','PHIBOT  ','sIceLoad',
                 'MXLDEPTH',
                 'SIatmQnt','SIatmFW ','oceQnet ','oceFWflx',
                 'oceTAUX ','oceTAUY ',
                 'ADVxHEFF','ADVyHEFF','DFxEHEFF','DFyEHEFF',
                 'ADVxSNOW','ADVySNOW','DFxESNOW','DFyESNOW',
                 'SIuice  ','SIvice  ',
    budg2d_zflux_set2: 'oceFWflx','SIatmFW ','TFLUX   ','SItflux ',
                   'SFLUX   ','oceQsw  ','SIareaPR', 'SIareaPT',
                   'SIheffPT','SIhsnoPT','SIaQbOCN', 'SIaQbATC',
                   'SIaQbATO','SIdHbOCN','SIdSbATC', 'SIdSbOCN',
                   'SIdHbATC','SIdHbATO','SIdHbFLO', 'SIdAbATO',
                   'SIdAbATC','SIdAbOCN','SIdA',
                   # SIatmFW: Net freshwater flux from atmosphere & land (+=down)
    iceshelf_state_set2: 'SHIfwFlx','SHIhtFlx'
    """
    vname_full, units, lat_name, lon_name = var_meta_dict[vname]
    
    # Appending each run to a list is faster than saving them in a numpy array or data array
    ens_data_list = []
    for i in range(n_runs):
        print(i)
        run_ds = load_experiment_ds(runs[i],'all',prefix)
        if reg_type == 'map':
            run_ds_reg = run_ds.sel(lat = slice(lat1,lat2), lon = slice(lon1,lon2),\
                                lat_at_f_loc = slice(lat1,lat2),\
                                lon_at_f_loc = slice(lon1,lon2))

        elif reg_type == 'lon_cross':
            run_ds_reg = run_ds.sel(lon = lon, lon_at_f_loc = lon, method='nearest')
            run_ds_reg = run_ds_reg.sel(lat = slice(lat1,lat2),lat_at_f_loc = slice(lat1,lat2))
            
        elif reg_type == 'lat_cross':
            run_ds_reg = run_ds.sel(lat = lat, lat_at_f_loc = lat, method = 'nearest')
            run_ds_reg = run_ds_reg.sel(lon = slice(lon1,lon2),lon_at_f_loc = slice(lon1,lon2))
                
        run_data_var = run_ds_reg[vname_full]
        run_data_5yrs = run_data_var.isel(time = slice(month_start_idx,month_start_idx+n_months))
        if i == 0:
            print('saving date from these times for 1st run:\n',run_data_5yrs.time.values)
        if depth != None:
            run_data = run_data_5yrs.sel(depth = depth, method = 'nearest')
        else:
            run_data = run_data_5yrs.squeeze()
            
        lats = run_ds_reg.lat.values
        lons = run_ds_reg.lon.values

        #data has shape times, lat, lon. handle mismatched lengths if necessary due to selection rounding errors
        if reg_type == 'lon_cross':
            if lats.shape[0] > run_data.shape[1]:
                lats = lats[0:-1]
            if lats.shape[0] < run_data.shape[1]:
                run_data = run_data[:,0:-1,:]
        elif reg_type == 'lat_cross':
            if lons.shape[0] > run_data.shape[2]:
                lons = lons[0:-1]
            if lons.shape[0] < run_data.shape[2]:
                run_data = run_data[:,:,0:-1]
        elif reg_type=='map':
            if lons.shape[0] < run_data.shape[-1]:
                run_data = run_data[:,:,0:-1]
            if lats.shape[0] > run_data.shape[1]:
                lats = lats[0:-1]

        print(run_data.shape)
        
        ens_data_list.append(run_data)
    
    print('Saving as xr data array...')
    if reg_type == 'map':
        all_run_data_da = xr.DataArray(data = ens_data_list, dims = ['run','time','lat','lon'],\
                            coords = dict(run = runs, time = run_data_5yrs.time,\
                                            lat=lats,\
                                            lon=lons),\
                            attrs = dict(units = units,standard_name = vname_full))

    elif reg_type == 'lon_cross':
        all_run_data_da = xr.DataArray(data = ens_data_list, dims = ['run','time','depth','lat'],\
                            coords = dict(run = runs, time = run_data_5yrs.time,\
                                            depth = run_ds_reg.depth.values,lat=run_ds_reg.lat.values),\
                            attrs = dict(units = units,standard_name = vname_full))
    elif reg_type == 'lat_cross':
        all_run_data_da = xr.DataArray(data = ens_data_list, dims = ['run','time','depth','lon'],\
                            coords = dict(run = runs, time = run_data_5yrs.time,\
                                            depth = run_ds_reg.depth.values,lon=run_ds_reg.lon.values),\
                            attrs = dict(units = units,standard_name = vname_full))

    
    
    return all_run_data_da


def load_composite_data(vname, regions, warm_runs, cool_runs, prefix, month_start_idx, n_months,depth=None):
    """
    For a given variable and region and time period, load data to calculate composite means
    and their differences. 
    This function is very simple, it just organizes data loaded from get_ensemble_data. 
    Get_ensemble_data_for_ic which loads xr data arrays of runs with the same initial conditions
    Averages the runs over the specific time period, removing the run and time dimensions
    
    Called in Figure_composite_depth_profiles 
    Have not tested for making composite maps. 
    
    Inputs
    ------
    vname: str
        i.e. T, S, U, or V
        must be key in var_meta_dict in get_ensemble_data_for_ic
    regions: str
        must be key in a region dict above
    warm_runs: list of strings for first set of runs to average for warm composite
        containing run names as defined in Dataests.py
    cool_runs: aame as warm runs but for cool composite
    prefix: str pointing to filename
        i.e. 'state_3d_set2'
    month_start_idx: int
        index of first month to select for averaging (i.e. 48 for avging the 5th year)
    n_months: int
        num of months to select after month_start_idx for averaging
        (i.e., 12 to average over a full year). 
        It prints the months used so you can check that it's indexing correctly.
    depth (optional): float or int
        for specifying depth for plotting maps at a given depth (rather than a cross section)
    
    Output
    ------
    3 lists of length n_regions, each contianing items of shape corresponding to regional data
        i.e., for lat_profiles, each item has shape (n_depths, n_lons)
        
    
    
    """
    
    warm_list = []
    cool_list = []
    diff_list = []
    
    for region in regions:

        print(region)

        # Get warm data
        warm_runs_da = get_ensemble_data_for_ic(vname, warm_runs, region, prefix, \
                                                    month_start_idx = month_start_idx, \
                                                    n_months = n_months, depth=depth)
        # Average over runs
        warm_run_mean = np.mean(warm_runs_da,axis=0)
        # Average over times
        warm_yr5_mean = np.mean(warm_run_mean,axis=0)
        # Append
        warm_list.append(warm_yr5_mean)

        # Repeat for cool
        cool_runs_da = get_ensemble_data_for_ic(vname, cool_runs, region, prefix, \
                                                    month_start_idx = month_start_idx,\
                                                    n_months = n_months,depth=depth)
        cool_run_mean = np.mean(cool_runs_da,axis=0)
        cool_yr5_mean = np.mean(cool_run_mean,axis=0)
        cool_list.append(cool_yr5_mean)

        # Calc diff
        diff_list.append(warm_yr5_mean - cool_yr5_mean)

    print('Data retrieval complete!')
    
    return warm_list, cool_list, diff_list

# %% Calculate thermocline depth timseries

def calc_tc_depth_tseries(exp_list, loc, tc_temp,n_months=None):
    """
    Calculate 1d thermocline depth for a set of experiments in a specific region over time.
    Experiments are either the control run or a set of exps with the same IC. 
    
    Inputs
    ------
    exp_list : List of strings
        Contains experiment names following names in Datasets.py
        E.g., exp_set_dict['ic_2001'] or ['control'] or ['run_forc_erai_2015_rep_ic_1995',...]
    loc : str
        region over which to average. 
        Corresponds to keys in trough_idx_dict
    tc_temp : int
        temp in deg C for thermocline depth (1°C is typical)
    n_months: int
        only need to specify if the run doesn't have all 5 years (or if control run is no longer 312 months)
    
    Outputs
    -------
    tc_depths_all_runs : numpy array of shape (n_experiments, n_months)
        contains thermocline depth data for all experiments in set
    times_dt64 : list of length n_months
        contains monthly dates associated with each tc depth
        dates are formatted numpy.datetime64, which are plottable 
    
    """

    lat1,lat2,lon1,lon2 = trough_idx_dict[loc]
    # Note you have hard coded the num of months so that you can load times consistently. 
    # If you want to change # months, have to associate it with a different set of times.
    if exp_list == ['control']:
        n_months = 312
    elif n_months == None:
        n_months = 60 #if n_months not provided, assume full run
    # Set up np array to populate if sensitivity experiments
    tc_depths_all_runs = np.zeros((len(exp_list),n_months))
    print('tcd shape is now:',tc_depths_all_runs.shape)

    for i in range(len(exp_list)):

        exp = exp_list[i]
        print(i, 'calculating tc depth for exp:',exp)
        
        # Load exp T dataset
        exp_T = load_experiment_ds(exp, 'all', 'state_3d_set2').THETA

        # Avg over trough location, restrict to 5 years, and drop surface layers in case warm 
        exp_T_loc = exp_T.isel(lat = slice(lat1,lat2), lon = slice(lon1,lon2), \
                               time = slice(0,n_months), depth = slice(5,-1))
        exp_T_trough = exp_T_loc.mean('lat')
        exp_T_trough = exp_T_trough.mean('lon') #now has shape n_months, n_depths

        # For each month, calc TC depth
        exp_tc_depths = []
        nan_count = 0
        for month in range(n_months):
            
            # print(month)
            T_month = exp_T_trough[month]
            # print(T_month.values)

            # search Temps for CDW (instances >= 1)
            T_month_arr = T_month.values
            cdw_indices = list(filter(lambda x: T_month_arr[x] >= 1, range(len(T_month_arr))))
            
            try:
                # Get indexes of top of CDW and the temp just above it to interpolate tc depth
                cdw_top_idx = cdw_indices[0]
                cdw_top_temp = T_month_arr[cdw_top_idx]
                cdw_top_depth = float(T_month[cdw_top_idx].depth)

                # Get depth of layer just above CDW layer
                ww_bot_temp = T_month_arr[cdw_top_idx - 1]
                ww_bot_depth = float(T_month[cdw_top_idx-1].depth)

                # Interpolate between layers surrounding tc_temp
                t_interp = np.linspace(ww_bot_temp,cdw_top_temp,200) #increasing array
                tcd_interp_idx = bisect.bisect_left(t_interp,tc_temp)

                # Get corresponding interpolated depth at this index
                depth_interp = np.linspace(ww_bot_depth, cdw_top_depth,200) #decreasing array
                tcd_final = depth_interp[tcd_interp_idx]

            # if no CDW at this timestep, set to nan
            except:
                if month == 0:
                    print('Exception passed at first timestep! Check code.')
                tcd_final = np.nan
                nan_count += 1

            exp_tc_depths.append(tcd_final)

        if nan_count == len(exp_T_trough):
            print('Warning! All thermocline depths are nans. Look into why exceptions were passed.')

        try:
            # this will error if there aren't enough months of data in the run 
            tc_depths_all_runs[i,:] = exp_tc_depths
            print('appending tc_depths to exp_tc_depths, which now has shape',tc_depths_all_runs.shape)
        except:
            tc_depths_all_runs = np.array(exp_tc_depths)
            print('exception passed; setting tc_depths_all_runs to array of shape',tc_depths_all_runs.shape)
    
    times = exp_T_loc.time.values
    times_dt64 = [np.datetime64(x) for x in times]
    
    return tc_depths_all_runs, times_dt64


# Calculate on shelf CDW volume timeseries

def calc_cdw_volume_tseries(run, region, cdw_min_temp, grid_areas, n_months = None):
    """
    Calculate CDW volume timeseries for a single run using specified on shelf definition (either box or fancier region).
    
    Inputs
    ------
    run: str
        experiment name from Datasets.py
    region: str
        either a key in analysis_region_dict (above) or something more complex (TBD)
    cdw_min_temp: float
        cutoff for CDW min temp
        only includes water warmer than this temp in calculation
    grid_areas: 2d np array
        contains areas of grid sizes in meters squared, must match the area corresponding to loc input
        provide it to avoid having to recalculate grid sizes for each experiment
    n_months: int, optional
        if you want to restrict number of months of data to look at (typical value is 60 for 5 yrs)
        
    Outputs
    -------
    times: 1d np array of times in datetime64 format (for easy plotting)
    cdw_vol: 1d np array of floats in cubic meters
        contains on-shelf cdw volumes over time for this experiment
    """
    # Load experiment output as xr ds
    run_ds = load_experiment_ds(run, 'all', 'state_3d_set2')
    if n_months != None:
        run_ds = run_ds.isel(time = slice(0,n_months))
    
    # Only look at data between 200m and 1500m depth to exclude deep ocean and surface water
    exp_T_raw = run_ds.THETA
    exp_T_mid_depths = exp_T_raw.sel(depth=slice(-200,-1500))
    
    # Select data from shelf region

    lat1,lat2,lon1,lon2 = analysis_region_dict[region]
    exp_T = exp_T_mid_depths.sel(lat = slice(lat1,lat2),lon=slice(lon1,lon2))
    
    # mask values south of the shelf break (defined as 1750m contour) for naughten box
    if region == 'naughten_shelf_box':
        # get bathymetry to cutoff values in box where depth > 1750
        land_ice_ds = get_bathymetry_and_troughs()
        shelf_box_ds = land_ice_ds.sel(lat = slice(lat1,lat2),lon=slice(lon1,lon2))
        shelf_mask = shelf_box_ds['bathy'] < 1750
        exp_T = exp_T.where(shelf_mask)
    
        
    # Create binary mask for water cooler than CDW temp 
    exp_CDW_mask = xr.where(exp_T > cdw_min_temp, 1, 0) #shape (time, depth, lat, lon)
    exp_CDW_mask = exp_CDW_mask.values #make it an np array
    print('masked CDW water with shape',exp_CDW_mask.shape)

    # Calculate thicknesses of each layer in the model
    depths = exp_T.depth.values
    layer_thicknesses = np.diff(-depths) #(in meters)
    
    # Calculate volume of CDW for each timestep
    
    n_times = exp_T.shape[0]
    n_lats, n_lons = len(exp_T.lat), len(exp_T.lon)
    print('n_lats,nlons=',n_lats,n_lons)
    # n_times,n_lats,n_lons = 3,3,3
    cdw_vol = np.zeros((n_times))
    
    for t in range(n_times):
        # if t %5 == 0:
        #     print('working on t...',t)
        
        # start counter for cdw on the shelf
        shelf_cdw_vol = 0
        
        # for each cell, calculate volume of cdw
        for i in range(n_lats):
            for j in range(n_lons):
                # get thicknesses of CDW with dot product of array containining CDW depths (1's) and array containing depth thicknesses
                # this sums the total thicknesses of CDW in the cell
                cdw_thickness = np.dot(exp_CDW_mask[t,:,i,j][0:-1],layer_thicknesses)
                # print(cdw_thickness, grid_areas.shape)
                # multiply thickness by grid area to get volume
                cell_cdw_vol = cdw_thickness * grid_areas[i,j]
                # add cell cdw vol to shelf counter
                shelf_cdw_vol += cell_cdw_vol
                
        cdw_vol[t] = shelf_cdw_vol
        
    times = exp_T.time.values
    times_dt64 = [np.datetime64(x) for x in times]
        
    return times_dt64, cdw_vol

def calc_cdw_heat_content_tseries(run, region, cdw_min_temp, shelf_grid_areas, n_months = None):
    """
    Calculate heat content timeseries of CDW for a single run using specified on shelf definition (either box or fancier region). (only calculates for CDW; searches for water warmer than cdw_min_temp)
    
    Inputs
    ------
    run: str
        experiment name from Datasets.py
    region: str
        either a key in analysis_region_dict (above) or something more complex (TBD)
    cdw_min_temp: float
        cutoff for CDW min temp
        only includes water warmer than this temp in calculation
    shelf_grid_areas: 2d np array
        contains areas of grid sizes, must match the area corresponding to loc input
        provide it to avoid having to recalculate grid sizes for each experiment
    n_months: int, optional
        if you want to restrict number of months of data to look at (typical value is 60 for 5 yrs)
        
    Outputs
    -------
    times: 1d np array of times in datetime64 format (for easy plotting)
    heat_content: 1d np array of floats
        contains on-shelf heat contents over time for this experiment
    """
    # Load experiment output as xr ds
    run_ds = load_experiment_ds(run, 'all', 'state_3d_set2')
    if n_months != None:
        run_ds = run_ds.isel(time = slice(0,n_months))
    
    # Only look at data between 200m and 1500m depth to exclude deep ocean and surface water
    exp_T_raw = run_ds.THETA
    exp_T_mid_depths = exp_T_raw.sel(depth=slice(-200,-1500))
    
    # Select data from shelf region
    if region in analysis_region_dict.keys():
        lat1,lat2,lon1,lon2 = analysis_region_dict[region]
        exp_T = exp_T_mid_depths.sel(lat = slice(lat1,lat2),lon=slice(lon1,lon2))
    else:
        pass
        # To do: add code for defining more complex shelf region
    
    # mask everything's that not CDW but keep values 
    exp_CDW_vals = exp_T.where(exp_T > cdw_min_temp)
    exp_CDW_vals_cels = exp_CDW_vals.values
    #convert to Kelvin
    exp_CDW_vals = exp_CDW_vals_cels + 273.15

    # Calculate thicknesses of each layer in the model
    depths = exp_T.depth.values
    layer_thicknesses = np.diff(-depths)
    
    # calculate heat content (everything not masked)
    n_times = exp_T.shape[0]
    n_lats, n_lons = len(exp_T.lat), len(exp_T.lon)
    # n_times,n_lats,n_lons = 3,3,3
    print('n_lats,nlons=',n_lats,n_lons)
    heat_content = np.zeros((n_times))
    rho = 1026 #1026 kg/m^3
    C = 3990 #specific heat of seawater (J/kgK)

    for t in range(n_times):

        # start counter for heat content on the shelf
        shelf_heat_content = 0

        # for each cell, calculate volume of cdw
        for i in range(n_lats):
            for j in range(n_lons):
                # get integral of temps over all depths
                # use nansum bc dot product cant handle nans
                temp_integral_ij = np.nansum(exp_CDW_vals[t,:,i,j][0:-1]*layer_thicknesses)
                # multiply heat content by grid area to get volume
                temp_integral_cell = temp_integral_ij * shelf_grid_areas[i,j]
                # multiply by c and p to get heat content
                heat_content_cell = rho * C * temp_integral_cell
                # add cell heat content to shelf counter
                shelf_heat_content += heat_content_cell
            
        heat_content[t] = shelf_heat_content
        
    times = exp_T.time.values
    times_dt64 = [np.datetime64(x) for x in times]
        
    return times_dt64, heat_content


# Calculate undercurrent strength

def calc_undercurrent_strength(run_name,region):
    """
    Given an experiment name, load U data. Select data in a shelf-break region (defined in a analysis_region_dict in fda).
    Exclude the surface water (say top 50m; a fancier way to do this would be to select CDW layer by first searching for TCD and masking.)
    exclude really deep water as that is irrelevant, say >1000m (much deeper than shelf break)
    At each timestep, at each longitude in the shelf break region, select the velocity at the latitude where the *depth-avged* U is maxed. 
    The undercurrent strength at that timestep is the avg of those velocities (max U at each lat/lon combo). 

    Inputs
    ------
    run_name: str of experiment run name for reasing mds files
        e.g., 'run_forc_erai_2015_rep_ic_1995'
    region: str of region to search for max depth-avgd undercurrent velocity
        corresponds to analysis_region_dict key at top of fda

    Outputs
    -------
    uc_tseries: 1d np array of floats containing undercurrent strength
    """
    # get experiment U data in specified region
    lat1, lat2, lon1, lon2 = analysis_region_dict[region]
    exp_U = load_experiment_ds(run_name, 'all', 'state_3d_set2').UVELMASS
    exp_U_sb = exp_U.sel(lat=slice(lat1, lat2), lon_at_f_loc=slice(lon1, lon2), \
                         depth=slice(-50, -1000))
    exp_U_depth_avg = exp_U_sb.mean(dim='depth')

    # for each t, for each lon, find lat of U and store avg U in column 
    n_lons = len(exp_U_depth_avg.lon_at_f_loc)
    uc_tseries = []
    # search first 60 points for 5 years
    for t in range(60):
        print('t = ', t)
        U_t = exp_U_depth_avg.isel(time=t)
        U_max_all_lons = []
        # search for lat with max U at each lon
        for lon_i in range(n_lons):
            lon_i_u = U_t.isel(lon_at_f_loc=lon_i)
            max_u_lon_i = lon_i_u.max()
            U_max_all_lons.append(max_u_lon_i.values)
        # average depth-avgd max U for all lons to get a single U val for this time point
        mean_U_t = np.mean(U_max_all_lons)
        uc_tseries.append(mean_U_t)
    uc_tseries = np.array(uc_tseries)

    return uc_tseries


