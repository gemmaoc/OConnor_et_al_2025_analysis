#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Mon Jan 16 15:44:10 2023

Functions called for reading in mitgcm output from binary files.
To be run on Casper and reads in files on casper dir first, otherwise searches cheyenne scratch.

@author: gemma
"""

import xarray as xr
import numpy as np
import xmitgcm


def load_experiment_ds(exp_name, iters, prefix):
    """
    Basic function for reading in all output from an experiment for a given state. 
    Flips U and V vars as needed.
    
    Uses xmitgcm.open_mdsdataset to read an experiment's output 
    for any state or exf files as set up (i.e. state_3d_set2). 
    Based on set up for 5-year experiments or control run (noleap calendar, ref to 1992, etc.)
    
    Sets up grid vars (i.e. changes xc to lat and xg to lat_at_f_loc)
    And switches U and V components of variables that need switching/flipping. 
    You manually set up the switch, so for new vars that you haven't analyzed yet
    you need to add code to switch them.
    
    This is quite fast to run becuase it uses dask.

    Parameters
    ----------
    exp_name : str
        experiment name as listed in Datasets.py 
    iters : either 'all' or an int 
        if 'all', uses all iters avaailable
        if int given, uses iter num in file
    prefix : str
        state_3d_set2, state_2d_set2, or EXF_forcing_set2 are set up

    Returns
    -------
    run_ds : xr dataset containing data
        2d datasets have shape (n_times, n_lats, n_lons)
        3d datasets have shape (n_times, n_depths, n_lats, n_lons)
        Some of the vars are associated with lat_at_f_loc or funny things like that. 
        They are very similar to regular lat and lon and are quite annoying. 
        

    """
    # casper_dir = '/glade/campaign/univ/ulnl0002/Gemma/AS_climBC_5yr_exp/'
    casper_dir = '/glade/campaign/univ/uwas0134/Gemma/AS_climBC_5yr_exp/'
    run_path = casper_dir + exp_name + '/diags/'
    grid_dir = casper_dir + 'input/'
    
    # Load output as xarray data array (package has issue loading grid)
    # Uses dask so faster than doing this step yourself and creating an xr dataset
    
    # If runs aren't on Casper, try scratch
    if exp_name != 'control':
        try:
            run_ds = xmitgcm.open_mdsdataset(run_path, \
                 grid_dir = grid_dir,\
                 iters = iters,\
                 prefix = prefix, delta_t=120, ref_date = '1992-01-01',\
                 calendar='noLeap')
        except:
            print('exception passed when loading data... this experiment data may only be stored on scratch drive!')
            run_path = '/glade/scratch/gemmao/MITgcm/experiments/AS_climBC_5yr_exp/' + exp_name + '/diags/'
            run_ds = xmitgcm.open_mdsdataset(run_path, \
                 grid_dir = grid_dir,\
                 iters = iters,\
                 prefix = prefix, delta_t=120, ref_date = '1992-01-01',\
                 calendar='noLeap')
    else:
        run_path = '/glade/campaign/univ/uwas0134/Gemma/AS_climBC_control/run/diags/'
        run_ds = xmitgcm.open_mdsdataset(run_path, \
                 grid_dir = grid_dir,\
                 iters = iters,\
                 prefix = prefix, delta_t=120, ref_date = '1992-01-01',\
                 calendar='noLeap')
    # You have suppressed a warning that available_diagnostics.log cannot be found
    # so it uses the default. See code here: /Users/gemma/opt/anaconda3/envs/env_2023/lib/python3.10/site-packages/xmitgcm/mds_store.py
    
    
    # Load grid (see documentation in saved PDF: mitgcm.pdf)
    # XC, YC - grid cell center point locations
    x_fname = grid_dir+'/XC.data'
    f = open(x_fname,'r')
    xc = np.fromfile(f,dtype = '>f')
    xc_rs = xc.reshape([896,-1]) #array of shape 896,416
    lons = xc_rs[:,0] #nd array of shape 896
    
    y_fname = grid_dir+'YC.data'
    f = open(y_fname,'r')
    yc = np.fromfile(f,dtype = '>f')
    yc_rs = yc.reshape([896,-1])
    lats = yc_rs[0,:] #ndarray of shape 416
    
    # XG, YG - locations of grid cell vertices (used in some velocity fields)
    xg_fname = grid_dir+'XG.data'
    f = open(xg_fname,'r')
    xg = np.fromfile(f,dtype = '>f')
    xg_rs = xg.reshape([896,-1]) #array of shape 896,416
    lons_f = xg_rs[:,0] #nd array of shape 896
    
    yg_fname = grid_dir+'YG.data'
    f = open(yg_fname,'r')
    yg = np.fromfile(f,dtype = '>f')
    yg_rs = yg.reshape([896,-1]) #array of shape 896,416
    lats_f = yg_rs[0,:] #nd array of shape 896
    
    # RC, RF - vertical cell center and cell faces positions
    r_fname = grid_dir+'RC.data'
    f = open(r_fname,'r')
    depths = np.fromfile(f,dtype = '>f')
    
    rw_fname = grid_dir+'RF.data'
    f = open(rw_fname,'r')
    depths_w = np.fromfile(f,dtype = '>f')
    
    # Rename grid vars
    
    # # X and Y are technically lon and lat, but the ASBS grid is sideways. 
    run_ds = run_ds.rename({'XC':'lat', 'YC':'lon', 'Z':'depth',\
                            'XG':'lat_at_f_loc','YG':'lon_at_f_loc',\
                            'Zl':'depth_at_lower_w_loc'})
    run_ds['lat'] = lats
    run_ds['lon'] = lons
    run_ds['lat_at_f_loc'] = lats_f
    run_ds['lon_at_f_loc'] = lons_f
    run_ds['depth'] = depths
    run_ds['depth_at_lower_w_loc'] = depths_w[1:]
    
    # To do: chnge it so they're all on the same lat/lon grid and called lat/lon. They're basically the same. 
    if prefix == 'state_3d_set2':
    
        # Switch Uvel and Vvel because of rotated domain 
        # and switch sign of the new v (positive = northward). The new U is fine (positive = eastward) 
        run_ds = run_ds.rename({'UVELMASS':'VVELMASS','VVELMASS':'UVELMASS'})
        run_ds['VVELMASS'] = - run_ds['VVELMASS'] #The model's u is now called v, so take the opposite of v.
        run_ds['UVELMASS'].attrs['standard_name'] = 'UVELMASS'
        run_ds['UVELMASS'].attrs['long_name'] = 'Zonal Mass-Weighted Comp of Velocity (m/s)'
        run_ds['VVELMASS'].attrs['standard_name'] = 'VVELMASS'
        run_ds['VVELMASS'].attrs['long_name'] = 'Meridional Mass-Weighted Comp of Velocity (m/s)'
        
    elif prefix == 'EXF_forcing_set2':
        
        # Switch u and v components of wind speeds and stresses
        run_ds = run_ds.rename({'EXFtaux':'EXFtauy','EXFtauy':'EXFtaux',\
                                'EXFuwind':'EXFvwind','EXFvwind':'EXFuwind'})
        run_ds['EXFtaux'].attrs['standard_name'] = 'EXFtaux'
        run_ds['EXFtaux'].attrs['long_name'] = 'zonal surface wind stress, >0 increases uVel'
        run_ds['EXFtauy'].attrs['standard_name'] = 'EXFtauy'
        run_ds['EXFtauy'].attrs['long_name'] = 'meridional surface wind stress, >0 increases vVel'
        run_ds['EXFuwind'].attrs['standard_name'] = 'EXFuwind'
        run_ds['EXFuwind'].attrs['long_name'] = 'zonal 10-m wind speed, >0 increases uVel'
        run_ds['EXFvwind'].attrs['standard_name'] = 'EXFvwind'
        run_ds['EXFvwind'].attrs['long_name'] = 'meridional 10-m wind speed, >0 increases vVel'
        
        # Take opposite of data now labeld as y or v
        run_ds['EXFtauy'] = - run_ds['EXFtauy'] 
        run_ds['EXFvwind'] = - run_ds['EXFvwind']
        
    elif prefix == 'state_2d_set2':
        
        # Switch u and v components of OCEAN stresses (need to do for other u and v comp. vars if you use those later)
        run_ds = run_ds.rename({'oceTAUX':'oceTAUY','oceTAUY':'oceTAUX'})
        run_ds['oceTAUX'].attrs['standard_name'] = 'oceTAUX'
        run_ds['oceTAUX'].attrs['long_name'] = 'zonal surface wind stress, >0 increases uVel'
        run_ds['oceTAUY'].attrs['standard_name'] = 'oceTAUY'
        run_ds['oceTAUY'].attrs['long_name'] = 'meridional surf. wind stress, >0 increases vVel'
        
        # Take opposite of data now labeld as y 
        run_ds['oceTAUY'] = - run_ds['oceTAUY'] 
        
   
    # Swap order of lat and lon for easier plotting. want (time,lat,lon) for 3d or (time,depth, lat,lon) for 4d
    for var in run_ds.data_vars:
        dims = run_ds[var].dims
        if len(dims) == 4:
            run_ds[var] = run_ds[var].transpose(dims[0],dims[1],dims[3],dims[2])
        elif len(dims) == 3:
            run_ds[var] = run_ds[var].transpose(dims[0],dims[2],dims[1])
            
    
    # Uncomment to plot
    # plt.figure(figsize=(10,5))
    # ax1 = plt.subplot(121)
    # test_temp = run_ds.THETA[0,20]
    # test_temp.plot(ax=ax1)#,x='lon')
    # ax2 = plt.subplot(122)
    # test_u = run_ds.UVELMASS[0,20]
    # test_u.plot(ax=ax2,vmin=-0.1,vmax=0.1,cmap='RdBu_r')
    
    return run_ds




def get_bathymetry_and_troughs():
    
    """
    
    Gets 2d np arrays of bathymetry, ice shelves, and grounded ice 
    and their lat/lon locations
    
    Inputs
    ------
    
    None
    
    Returns
    ------
    land_ice_ds: xr dataset containing these vars:
        bathy: 2d np array. shape (416,896)
               bathymetry at each location. can be used to plot contours on top of maps.
        grounded_ice : 2d masked numpy array with binary values. shape (416,896)
               shows locations of grounded ice. 1 = grounded ice, mask = no ice
        all_ice : 2d masked numpy array with binary values. shape (416,896)
               shows locations of grounded ice and ice shelves. 
               1 = grounded OR ice shelf, mask (no value) = no ice
        Also contains the coordinates lat and lon, retrieved from XC and YC grid.

    """
    
    # grid_dir = '/glade/u/home/gemmao/MITgcm/experiments/AS_climBC_5yr_exp/input/'
    grid_dir = '/glade/campaign/univ/uwas0134/Gemma/AS_climBC_5yr_exp/input/'
    #get bathymetry to plot as contours
    b_fname = grid_dir + '1080_BATHY_2_rignot.bin'
    f_b = open(b_fname)
    bathy_raw = np.fromfile(f_b,dtype = '>f') #>f means big endian single precision float, 32-bit
    bathy = bathy_raw.reshape([896,416]) 
    bathy = -bathy #make them positive so contours are solid
    bathy_tp = np.swapaxes(bathy,0,1)
    
    #get extent of ice area, including ice shelves
    ice_fname = '1080_icetopo_fastice.bin'
    ice_file = open(grid_dir+ice_fname)
    ice_data_raw = np.fromfile(ice_file,dtype='float32')
    ice_data = ice_data_raw.reshape([896,416])
    ice_data_tp = np.swapaxes(ice_data,0,1)
    fill_val = ice_data_tp[0,0]
    #set all non-fill values to 1 so you can plot solid black for ice, rather than holy heights
    ice_data_tp[ice_data_tp!=fill_val]=1
    #mask fill values (index 0,0 is a fill value)
    all_ice_ma = np.ma.masked_where(ice_data_tp==fill_val,ice_data_tp)
    
    #get grounded ice only
    bathy_cp = bathy_tp
    bathy_cp[bathy_cp==0]=1
    #mask all other values so they don't cover up other data in plotting
    grounded_ice_ma = np.ma.masked_where(bathy_cp!=1,bathy_cp)


    # Get lons and lats to go with the data for plotting
    x_fname = 'XC.data'
    f = open(grid_dir + x_fname,'r')
    xc = np.fromfile(f,dtype = '>f')
    xc_rs = xc.reshape([896,-1]) 
    lon = xc_rs[:,0] 
    
    y_fname = 'YC.data'
    f = open(grid_dir + y_fname,'r')
    yc = np.fromfile(f,dtype = '>f')
    yc_rs = yc.reshape([896,-1])
    lat = yc_rs[0,:] 
    
    data_dict = {'bathy' : (['lat','lon'], bathy_tp), \
                 'all_ice' : (['lat','lon'], all_ice_ma), \
                 'grounded_ice' : (['lat','lon'], grounded_ice_ma) }
    coords_dict = dict(lat=lat,lon=lon)
    land_ice_ds = xr.Dataset(data_vars = data_dict, coords = coords_dict,\
        attrs = dict(description = 'Model bathymetry and ice extent as binary values'))
    
    return land_ice_ds

