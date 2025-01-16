#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Mon Jan 16 15:44:10 2023

Functions for making plots using output from Functions_data_analysis. 
Functions that can easily be transferred to various state files are here
rather than in specific figure notebooks.

@author: gemma
"""

import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors
import Datasets
from Functions_data_analysis import plot_regions, trough_idx_dict, \
lon_cross_dict, lat_cross_dict,analysis_region_dict, get_ensemble_data_for_ic
from Functions_load_output import get_bathymetry_and_troughs

def plot_bathymetry_and_locs(plot_region, highlight_locs, plot_velocity = False, velocity_depth = None):
    """
    Plot bathymetry as colored contours
    Plot ice extent in dark gray
    Plot highlight locs in white lines or boxes to showcase analysis regions
    **To do: set up adding boxes for trough regions (only shelf break lon lines are set up)
    
    Parameters
    ----------
    plot_region : str
        region key in plot_regions dict
    highlight_locs : list of strings
        strings must be keys in lon_cross_dict or trough_idx_dict (trough plotting not set up yet)
    plot_velocity: False (default) or True
        if true, calculates ensemble mean U and V in 1999 across ic_1995 experiments and plots quivers.
    velocity_depth: None (Default) or int
        depth at which to show velocity quivers
    """
    
    # Get plot region bounds
    lat1,lat2,lon1,lon2 = plot_regions[plot_region]
    lon_mid = (lon1+lon2)/2
    
    # get bathymtery and ice ds
    land_ice_ds = get_bathymetry_and_troughs()
    lons_og, lats_og = land_ice_ds.lon, land_ice_ds.lat #for plotting trough indxs (need to finish setting up)
    land_ice_ds = land_ice_ds.sel(lat=slice(lat1,lat2),lon=slice(lon1,lon2))
    lons, lats = land_ice_ds.lon, land_ice_ds.lat

    # Set up plot
    fig = plt.figure()
    fig.set_size_inches((4,2.7))
    if plot_region == 'model_domain':
        fig.set_size_inches((6,2.7))
    ax = fig.add_subplot(1,1,1,projection=ccrs.SouthPolarStereo(central_longitude=lon_mid)) #eentral_lon =-100 rotates antarctica so pacific is up
    grid_proj = ccrs.PlateCarree()
    
    # Plot bathymetry 
    blevs = np.arange(0,3.25,0.25)
    cmap_bupu = plt.get_cmap('YlGnBu')#BuPu
    new_cmap = colors.ListedColormap(cmap_bupu(np.linspace(0.33, 1, 256)))
    cf = ax.contourf(lons,lats,land_ice_ds.bathy/1000,blevs,transform=grid_proj,cmap=new_cmap,extend='both')#was viridis_r
    ax.contour(lons,lats,land_ice_ds.bathy/1000,blevs,colors='gray',transform=grid_proj,linewidths=0.2)
    plt.title('Bathymetry')
    # Add colorbar
    # cb_ax = fig.add_axes([0.25,0.15,0.6,0.025])#for horizontal at bottom
    cb_ax = fig.add_axes([0.78,0.18,0.03,0.6]) #for right side vertical
    cb = fig.colorbar(cf, cax=cb_ax, extend='both',orientation = 'vertical')  
    cb.set_label(label = 'Depth [km]', fontsize=8)
    cb.ax.tick_params(labelsize=6)
    cb.set_ticks(blevs[::4])#[::2]
    # cb.set_ticklabels(np.arange(0,3.25,1))
    
    # Plot ice
    ax.contourf(lons,lats,land_ice_ds.all_ice,transform=grid_proj,colors=['lightgray']*2,alpha=0.6,zorder=2)
    ax.contourf(lons,lats,land_ice_ds.grounded_ice,transform=grid_proj,cmap='binary_r',zorder=2)
    

    # Set shape of map to match shape of data rather than a rectangle
    rect = mpath.Path([[lon1, lat2], [lon2, lat2],
    [lon2, lat1], [lon1, lat1], [lon1, lat2]]).interpolated(50)
    proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    rect_in_target = proj_to_data.transform_path(rect)
    ax.set_boundary(rect_in_target)
    ax.set_xlim(rect_in_target.vertices[:,0].min(), rect_in_target.vertices[:,0].max())
    ax.set_ylim(rect_in_target.vertices[:,1].min(), rect_in_target.vertices[:,1].max())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linestyle='-', linewidth=0.1,color='gray', 
                      xlocs=np.arange(-140,-60,5),
                      ylocs=np.arange(-76,-60,1),draw_labels=True,\
                      x_inline=False,y_inline=False,rotate_labels=False)
    gl.top_labels=False
    gl.right_labels=False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    
    # Plot longitudes to highlight
    color = 'lightcoral' #lightcoral
    for loc in highlight_locs:
        if loc in lon_cross_dict.keys():
            lon = float(lon_cross_dict[loc][0])
            lat1,lat2 = lon_cross_dict[loc][1:]
            ax.plot([lon,lon],[lat1,-76],transform = grid_proj,lw=1,color=color,linestyle='--')
        elif loc in lat_cross_dict.keys():
            lat,lon1,lon2 = lat_cross_dict[loc]
            n=100
            lats = [lat]*n
            lons = np.linspace(lon1,lon2,n)
            ax.plot(lons,lats,transform=ccrs.PlateCarree(),lw=3,color=color)
            
        # add box to show on shelf region (does not follow curvature of lat lines...chatgpt suggestions have not worked)
        elif loc in analysis_region_dict.keys():
            lat1,lat2,lon1,lon2 = analysis_region_dict[loc]
            
            if loc == 'naughten_shelf_box':
                # create mask where shelf-break is found using 1750m isobath
                s_lat1,s_lat2,s_lon1,s_lon2 = analysis_region_dict[loc] #start with box defined
                shelf_box_ds = land_ice_ds.sel(lat = slice(s_lat1,s_lat2),lon=slice(s_lon1,s_lon2))
                #create mask in box everywhere shallower than 1750 contour (checked in Find_shelf_break_location.ipynb)
                shelf_mask = shelf_box_ds['bathy'] < 1750 
                # add shelf-mask to mask spanning plot domain 
                shelf_mask_plot_domain =  xr.DataArray(np.full((len(lats), len(lons)), False, dtype=bool),
                                                 dims=('lat', 'lon'),
                                                 coords={'lat': lats, 'lon': lons})
                shelf_lat1_idx = np.where(lats == shelf_mask.lat[0])[0][0]
                shelf_lon1_idx = np.where(lons == shelf_mask.lon[0])[0][0]
                shelf_n_lats, shelf_n_lons = shelf_mask.shape
                # shelf_lat1_idx, shelf_lon1_idx
                shelf_mask_plot_domain[shelf_lat1_idx:shelf_lat1_idx+shelf_n_lats,\
                                       shelf_lon1_idx:shelf_lon1_idx+shelf_n_lons] = shelf_mask
                ax.contour(lons,lats, shelf_mask_plot_domain, transform = grid_proj,levels=[0.5],\
                            colors=color, linewidths=1.25,zorder=3) 
            else:
                print('here')
                box = mpatches.Rectangle((lon1,lat1),(lon2-lon1),(lat2-lat1),
                               fill=False, edgecolor=color, linewidth=1.25,linestyle='--',\
                                         transform=ccrs.PlateCarree(),zorder=3)
                ax.add_patch(box)

        elif loc in trough_idx_dict:
            lat_i1,lat_i2,lon_i1,lon_i2 = trough_idx_dict[loc]
            lon1 = lons_og.isel(lon=lon_i1)
            lon2 = lons_og.isel(lon=lon_i2)
            lat1 = lats_og.isel(lat=lat_i1)
            lat2 = lats_og.isel(lat=lat_i2)
            box = mpatches.Rectangle((lon1, lat1), lon2 - lon1, lat2 - lat1,
                         edgecolor=color, fill=False, linewidth=1.25,transform=ccrs.PlateCarree())
                        #facecolor='cyan'
            ax.add_patch(box)
            
        else:
            print('Error with plotting highlight_locs!')

    if plot_velocity:
        runs = Datasets.exp_set_dict_wind_order['ic_1995']
        print('Retrieving U and V data in year 1999 for all 10 experiments to calculate velocities at depth:',velocity_depth)
        all_run_U_da = get_ensemble_data_for_ic('U', runs, plot_region, 'state_3d_set2', month_start_idx = 47,
                                            n_months = 12, depth=velocity_depth) 
        all_run_V_da = get_ensemble_data_for_ic('V', runs, plot_region, 'state_3d_set2', month_start_idx = 47,
                                            n_months = 12, depth=velocity_depth)
        # Calculate time-mean ens-mean avgs
        U_em = all_run_U_da.mean(dim = 'run')
        U_em_time_avg = U_em.mean(dim = 'time')
        V_em = all_run_V_da.mean(dim = 'run')
        V_em_time_avg = V_em.mean(dim = 'time')

        # Mask really large and really small values to avoid tiny and large quivers
        mag_em_levs = np.linspace(0,0.1,15)
        mag_em = (U_em_time_avg ** 2 + V_em_time_avg ** 2) ** 0.5
    
        # #mask small values and large values (Nan where condition is met)
        # mask = (mag_em < 0.01) | (mag_em > 0.1) # scale = .35
        mask = (mag_em < 0.01) | (mag_em > 0.1) # scale = 0.45
        U_em_masked = U_em_time_avg.where(~mask)
        V_em_masked = V_em_time_avg.where(~mask)

        quiv = ax.quiver(U_em.lon,U_em.lat, U_em_masked.values, V_em_masked.values,
                          transform = grid_proj, regrid_shape=18,
                          pivot = 'middle',scale=.45, width=0.01,
                          edgecolor='white',headwidth=4,headlength=6,
                          linewidth = 0.2,zorder=3)
        vec_len = 0.03
        ax.quiverkey(quiv, X=0.005, Y=-0.15, U=vec_len, label=str(vec_len)+' m/s', labelpos='S') # U = len to show in quiver key (in native units)
        ax.contour(lons,lats,land_ice_ds.bathy,blevs,colors='gray',transform=grid_proj,linewidths=0.2,zorder=1)
    
    fig.subplots_adjust(left=0.15,right=0.75,top=0.95,bottom=0.05,wspace=0.1,hspace=.05)

    return


def make_ensemble_subplots(vname, ens_data, ens_levs, cmap, region,title,\
                           save = None, data_x = None, data_y = None, 
                           q_scale = None,quiv_vname=None,vec_len=None):
    """
    Used for making ensemble subplots of filled contour maps for any 2d dataset. Option to plot quivers on top.
    
    """
    
    # Set up plot
    try:
        lons, lats = ens_data.lon, ens_data.lat
    except:
        lons, lats = data_x.lon, data_x.lat
    lat1,lat2,lon1,lon2 = plot_regions[region]
    lon_mid = (lon1+lon2)/2
    
    n_runs = len(ens_data)
    rows,cols = 3, 4 #need to change for other ic runs
    grid_proj = ccrs.PlateCarree()
    try:
        run_names = ens_data.run.values
        runs = [x.split('_')[3]+'_x5' for x in run_names]
    except:
        runs = Datasets.forcings
    
    fig = plt.figure()
    fig.set_size_inches((9,6.5))

    
    # Plot each run in ensemble
    for i in range(n_runs):
        print(i)
        
        run_anom = ens_data[i]
        # Plot and mask 0 values so they are not plotted as colors
        run_anom_mask = run_anom.where(run_anom != 0)

        ax = fig.add_subplot(rows, cols, i+1, projection=\
                             ccrs.SouthPolarStereo(central_longitude=lon_mid)) 
                             #central_lon =-100 rotates antarctica so pacific is up
        cf = ax.contourf(lons, lats, run_anom_mask, transform=grid_proj,\
                         levels=ens_levs, cmap=cmap, zorder=0, extend='both')
        plt.title(runs[i],fontsize=10)
        
        # If extra  and y data, plot as quivers
        if data_x is not None:
            
            data_x_i = data_x[i]
            data_y_i = data_y[i]
            data_x_ma = data_x_i.where(data_x_i != 0)
            data_y_ma = data_y_i.where(data_y_i != 0)
            q_lon,q_lat = data_x_ma.lon.values, data_x_ma.lat.values

            # smaller scale means longer arrows. 
            if q_scale == None:
                q_scales = {'T':0.4,'oceTAU':0.4,'wind':70,'U':.4,'mag':.2,'wind_mag':15,
                           'wind_speed_anom':7}# T 1.2
                q_scale = q_scales[quiv_vname]
            regrid_dict = {'AS_near':12,'AS_far':25,'ASBS':25,'full_model_domain':9,'AS_long':7}
            
            quiv = ax.quiver(q_lon,q_lat, data_x_ma.values, data_y_ma.values,\
                      transform = grid_proj, regrid_shape=regrid_dict[region],\
                      pivot = 'middle',scale=q_scale,\
                      width=0.02,headlength=4,headaxislength=3.5,\
                      minshaft=1,edgecolor='white', \
                      linewidth = 0.2,zorder=3)#lw .15, width .005, width=0.007
    ax.quiverkey(quiv, X=1.025, Y=0.0, U=vec_len, label=str(vec_len)+' m/s', labelpos='S')
            
    # Plot bathyemtry and ice locations in all subplots, and format map extent
    # Get bathymetry and ice data used on all plots
    land_ice_ds = get_bathymetry_and_troughs()
    land_ice_ds = land_ice_ds.sel(lat=slice(lat1,lat2),lon=slice(lon1,lon2))
    lons,lats = land_ice_ds.lon, land_ice_ds.lat
    blevs = (500,1000)
    
    for ax in fig.get_axes():
    
        # Plot bathymetry and ice
        ax.contour(lons,lats,land_ice_ds.bathy,blevs,colors='k',transform=grid_proj,linewidths=0.5,zorder=1)
        # Plot shelf break in thicker line
        ax.contour(lons,lats,land_ice_ds.bathy,(1000,),colors='k',transform=grid_proj,linewidths=1,zorder=1)
        white_cm = colors.ListedColormap(("lightgray","lightgray"))
        ax.contourf(lons,lats,land_ice_ds.all_ice,transform=grid_proj,cmap=white_cm,zorder=2)
        ax.contourf(lons,lats,land_ice_ds.grounded_ice,transform=grid_proj,cmap='binary_r',zorder=2)
        
        # Set shape of map to match shape of data rather than a rectangle
        rect = mpath.Path([[lon1, lat2], [lon2, lat2],
        [lon2, lat1], [lon1, lat1], [lon1, lat2]]).interpolated(50)
        proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
        rect_in_target = proj_to_data.transform_path(rect)
        ax.set_boundary(rect_in_target)
        ax.set_xlim(rect_in_target.vertices[:,0].min(), rect_in_target.vertices[:,0].max())
        ax.set_ylim(rect_in_target.vertices[:,1].min(), rect_in_target.vertices[:,1].max())
        gl = ax.gridlines(crs=ccrs.PlateCarree(), linestyle='--', linewidth=0.5,color='gray', 
                          xlocs=np.arange(-140,-60,20),
                          ylocs=np.arange(-76,-60,2),draw_labels=False,\
                          x_inline=False,y_inline=False,rotate_labels=False)
        gl.top_labels=False
        gl.right_labels=False
        gl.xlabel_style = {'size': 5}
        gl.ylabel_style = {'size': 5}
        
    fig.subplots_adjust(left=0.01,right=0.99,top=0.9,bottom=0.13,wspace=0.1,hspace=.05)
    plt.suptitle(title)
    
   
    # Add colorbar for runs
    cb_ax = fig.add_axes([0.43,0.08,0.2,0.02])
    cb = fig.colorbar(cf, cax=cb_ax, extend='both',orientation = 'horizontal')
    labs_dict = {'T':'Pot. Temp anomaly (degC)',\
                 'U':'U anomaly (m/s)',\
                 'V':'V anomaly (m/s)',\
                 'uwind':'uwind anomaly (m/s)',\
                 'vwind':'vwind anomaly (m/s)',\
                 'atemp':'Air Temp anomaly (degC)',\
                 'taux':'Taux (N/m^2)',\
                 'tauy':'Tauy (N/m^2)',\
                 'SIarea':'Sea ice area anomaly',\
                 'oceQnet':'oceQnet anomaly (W/m^2)',\
                 'oceTAUX':'oceTAUX anomaly (N/m^2)',\
                 'oceTAUY':'oceTAUY anomaly (N/m^2)',\
                 'mag':'Velocity Magnitude (m/s)',\
                 'wind_mag':'Wind Speed Anomaly (m/s)',\
                 'SIdA':'Sea ice area rate of change\nanomaly (m^2/m^2/s)',\
                 'SIdAbATO':'Sea ice area (ocn/atm)\nrate of change anomaly (m^2/m^2/s)',\
                 'SIdAbATC':'Sea ice area (atm/ice)\nrate of change anomaly (m^2/m^2/s)',\
                 'SIdAbOCN':'Sea ice area (ocn/ice)\nrate of change anomaly (m^2/m^2/s)',\
                 }
    cb.set_label(label = labs_dict[vname], fontsize=10)
    cb.ax.tick_params(labelsize=9)
    cb.set_ticks([ens_levs[0],0,ens_levs[-1]])
    cb.set_ticklabels([ens_levs[0],0,ens_levs[-1]])

    if save:
        plt.savefig(save, bbox_inches='tight', dpi=500)
        print('Saving figure as',save)
    
    return 

def make_ensemble_subplots_with_composites(vname, ens_data, ens_levs, cmap, region,title,
                           save = None, data_x = None, data_y = None, q_scale = None,quiv_vname=None):
    """
    Same as make_ensemble_subplots but also calculates and plots mean of first 4 
    and mean of 2nd four at end of first 2 rows.
    
    So order should be 4 warmest, 4 coolest, and then whatever. 
    Plots 12 plots wit 3 rows and 5 columns:
    Plots 4 datasets, mean of those 4 in first row. 
    Plots another 4 datasets, the mean of thsoe 4. 
    Then plots the remaining. 
    
    """
    
    # Set up plot
    try:
        lons, lats = ens_data.lon, ens_data.lat
    except:
        lons, lats = data_x.lon, data_x.lat
    lat1,lat2,lon1,lon2 = plot_regions[region]
    lon_mid = (lon1+lon2)/2
    
    n_runs = len(ens_data)
    rows,cols = 3, 5
    grid_proj = ccrs.PlateCarree()
    try:
        run_names = ens_data.run.values
        runs = [x.split('_')[3] for x in run_names]
    except:
        runs = Datasets.forcings
    
    #calculate composites for first 4 and second 4:
    warm_comp = ens_data[0:4].mean(dim='run')
    cool_comp = ens_data[5:9].mean(dim='run')
    if data_x is not None:
        warm_comp_x = data_x[0:4].mean(dim='run')
        warm_comp_y = data_y[0:4].mean(dim='run')
        cool_comp_x = data_x[5:9].mean(dim='run')
        cool_comp_y = data_y[5:9].mean(dim='run')
    
    fig = plt.figure()
    fig.set_size_inches((9,6))

    # Plot each run in ensemble
    run_counter = 0
    # iterate by subplot number
    for i in range(n_runs+2):
        print(i)
        
        if i == 4:
            plot_data = warm_comp
            subplot_title = 'warm composite'
        elif i == 9:
            plot_data = cool_comp
            subplot_title = 'cool composite'
        else:
            plot_data = ens_data[run_counter]
            subplot_title = runs[run_counter]
            if data_x is None:
                run_counter += 1
            
        # Plot and mask 0 values so they are not plotted as colors
        plot_data_mask = plot_data.where(plot_data != 0)

        ax = fig.add_subplot(rows, cols, i+1, projection=\
                             ccrs.SouthPolarStereo(central_longitude=lon_mid)) 
                             #central_lon =-100 rotates antarctica so pacific is up
        cf = ax.contourf(lons, lats, plot_data_mask, transform=grid_proj,\
                         levels=ens_levs, cmap=cmap, zorder=0, extend='both')
        if vname == 'SIarea':
            contour = ax.contour(lons,lats,plot_data_mask, transform=grid_proj,
                                 colors='k',zorder=3,linewidths=0.2,
                                 levels=cf.levels)
        plt.title(subplot_title,fontsize=10)
        
        # If extra  and y data, plot as quivers
        if data_x is not None:
            
            if i==4:
                data_x_i,data_y_i = warm_comp_x,warm_comp_y
            elif i == 9:
                data_x_i,data_y_i = cool_comp_x,cool_comp_y
            else:
                data_x_i = data_x[run_counter]
                data_y_i = data_y[run_counter]
                run_counter += 1
            data_x_ma = data_x_i.where(data_x_i != 0)
            data_y_ma = data_y_i.where(data_y_i != 0)
            q_lon,q_lat = data_x_ma.lon.values, data_x_ma.lat.values

            # smaller scale means longer arrows. 
            if q_scale == None:
                q_scales = {'T':0.4,'oceTAU':0.4,'wind':70,'U':.4,'mag':.2,'wind_mag':10,
                            'Velocity':0.1,'wind_speed_anom':20}# wind_ma 20
                q_scale = q_scales[quiv_vname]
            # smaller regrid = fewer arrows
            regrid_dict = {'AS_near':10,'AS_far':25,'ASBS':25,'full_model_domain':8,'AS_long':8}#as-near 25
            # arrow width
            width_dict = {'AS_near':.04,'AS_far':.02,'ASBS':.015,'full_model_domain':.01,'AS_long':.02}
            
            quiv = ax.quiver(q_lon,q_lat, data_x_ma.values, data_y_ma.values,\
                      transform = grid_proj, regrid_shape=regrid_dict[region],\
                      pivot = 'middle',scale=q_scale,\
                      width=width_dict[region],headlength=4,headaxislength=3.5,\
                      minshaft=2,edgecolor='white', \
                      linewidth = 0.2,zorder=3)#lw .15, width .005, width=0.007
            
    # Plot bathyemtry and ice locations in all subplots, and format map extent
    # Get bathymetry and ice data used on all plots
    land_ice_ds = get_bathymetry_and_troughs()
    land_ice_ds = land_ice_ds.sel(lat=slice(lat1,lat2),lon=slice(lon1,lon2))
    lons,lats = land_ice_ds.lon, land_ice_ds.lat
    blevs = (500,1000)
    
    for ax in fig.get_axes():
    
        # Plot bathymetry and ice
        if vname == 'SIarea':
            blevs = (1500,)
            ax.contour(lons,lats,land_ice_ds.bathy,blevs,colors='k',transform=grid_proj,linewidths=0.5,zorder=1)
        else:
            ax.contour(lons,lats,land_ice_ds.bathy,blevs,colors='k',transform=grid_proj,linewidths=0.5,zorder=1)

        # Plot shelf break in thicker line
        # ax.contour(lons,lats,land_ice_ds.bathy,(1000,),colors='k',transform=grid_proj,linewidths=1,zorder=1)
        white_cm = colors.ListedColormap(("lightgray","lightgray"))
        ax.contourf(lons,lats,land_ice_ds.all_ice,transform=grid_proj,cmap=white_cm,zorder=2)
        ax.contourf(lons,lats,land_ice_ds.grounded_ice,transform=grid_proj,cmap='binary_r',zorder=2)
        
        # Set shape of map to match shape of data rather than a rectangle
        rect = mpath.Path([[lon1, lat2], [lon2, lat2],
        [lon2, lat1], [lon1, lat1], [lon1, lat2]]).interpolated(50)
        proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
        rect_in_target = proj_to_data.transform_path(rect)
        ax.set_boundary(rect_in_target)
        ax.set_xlim(rect_in_target.vertices[:,0].min(), rect_in_target.vertices[:,0].max())
        ax.set_ylim(rect_in_target.vertices[:,1].min(), rect_in_target.vertices[:,1].max())
        # Uncomment for lat/lon lines
        # gl = ax.gridlines(crs=ccrs.PlateCarree(), linestyle='--', linewidth=0.5,color='gray', 
        #                   xlocs=np.arange(-140,-60,20),
        #                   ylocs=np.arange(-76,-60,2),draw_labels=False,\
        #                   x_inline=False,y_inline=False,rotate_labels=False)
        # gl.top_labels=False
        # gl.right_labels=False
        # gl.xlabel_style = {'size': 5}
        # gl.ylabel_style = {'size': 5}
        
    fig.subplots_adjust(left=0.01,right=0.99,top=0.9,bottom=0.13,wspace=0.1,hspace=.05)
    plt.suptitle(title)
    
   
    # Add colorbar for runs
    cb_ax = fig.add_axes([0.43,0.08,0.2,0.02])
    cb = fig.colorbar(cf, cax=cb_ax, extend='both',orientation = 'horizontal')
    labs_dict = {'T':'Pot. Temp anomaly (degC)',\
                 'U':'U anomaly (m/s)',\
                 'V':'V anomaly (m/s)',\
                 'uwind':'uwind anomaly (m/s)',\
                 'vwind':'vwind anomaly (m/s)',\
                 'atemp':'air temp anomaly (degC)',\
                 'pressure':'Pressure (N/m^2)',
                 'taux':'Taux (N/m^2)',\
                 'tauy':'Tauy (N/m^2)',\
                 'SIarea':'Sea ice area anomaly',\
                 'oceQnet':'Surface heat flux anomaly (W/m^2)',\
                 'oceTAUX':'oceTAUX anomaly (N/m^2)',\
                 'oceTAUY':'oceTAUY anomaly (N/m^2)',\
                 'mag':'Velocity Magnitude (m/s)',\
                 'wind_mag':'Wind Speed Anomaly (m/s)',\
                 'SIdA':'Sea ice area rate of change\nanomaly (m^2/m^2/s)',\
                 'SIdAbATO':'Sea ice area (ocn/atm)\nrate of change anomaly (m^2/m^2/s)',\
                 'SIdAbATC':'Sea ice area (atm/ice)\nrate of change anomaly (m^2/m^2/s)',\
                 'SIdAbOCN':'Sea ice area (ocn/ice)\nrate of change anomaly (m^2/m^2/s)'\
                 }
    cb.set_label(label = labs_dict[vname], fontsize=10)
    cb.ax.tick_params(labelsize=9)
    cb.set_ticks([ens_levs[0],0,ens_levs[-1]])
    cb.set_ticklabels([ens_levs[0],0,ens_levs[-1]])

    if save:
        plt.savefig(save, bbox_inches='tight', dpi=600)
        print('Saving figure as',save)
    
    return 

def make_contour_map(vname, data_2d, levs, cmap, region,title = None,
                     save = None, quiv_vname = None, data_x = None, 
                     data_y = None, q_scale = None,vec_len=1,regrid=None):
    """
    Make 1 contour map of 2d xr data array (vname is name for colors)
    Plots bathymetry contours on top
    Plots ice in gray
    Option to add quivers
    
    """

    
    # Set up plot
    lons, lats = data_2d.lon, data_2d.lat
    lat1,lat2,lon1,lon2 = plot_regions[region]
    lon_mid = (lon1+lon2)/2
    grid_proj = ccrs.PlateCarree()
    
    labs_dict = {'T':'Pot. Temp (°C)',\
                 'THETA':'Pot. Temp (°C)',\
                 'U':'U (m/s)',\
                 'V':'V (m/s)',\
                 'uwind':'uwind (m/s)',\
                 'vwind':'vwind (m/s)',\
                 'pressure':'Pressure (N/m^2)',
                 'taux':'Taux (N/m^2)',\
                 'tauy':'Tauy (N/m^2)',\
                 'SIarea':'Sea Ice Area (fractional)',\
                 'SIdA':'Sea Ice Area rate of change (m^2/m^2/s)',\
                 'SIheff':'Sea Ice height (m)',\
                 'oceQnet':'Surface heat flux (W/m^2)',\
                 'oceTAUX':'oceTAUX (N/m^2)',\
                 'oceTAUY':'oceTAUY (N/m^2)',\
                 'atemp':'Air Temp (°C)',\
                 'Bathymetry':'[m]',
                 # these are the quivers
                 'oceTAU':'oceTAU (N/m^2)',\
                 'Velocity':'Velocity (m/s)',
                 'Velocity_comp':'Velocity Anomaly (m/s)',
                 'wind':'Wind Speed (m/s)',\
                 'mag':'Wind Speed (m/s)',\
                 'wind_mag':'Wind Speed (m/s)',
                 'wind_mag_comp': 'Wind Speed Anomaly (m/s)',
                 'wind_speed_anom':'Wind Speed Anomaly (m/s)'}
    
    # Plot data
    fig = plt.figure()
    if region == 'model_domain':
        fig.set_size_inches((4,2))
    else:
        fig.set_size_inches((3,3.5))
    ax = fig.add_subplot(1,1,1,projection=ccrs.SouthPolarStereo(central_longitude=lon_mid)) #eentral_lon =-100 rotates antarctica so pacific is up
    data_2d_ma = data_2d.where(data_2d != 0)
    cf_em = ax.contourf(lons, lats, data_2d_ma,transform=grid_proj,
                            levels=levs, cmap=cmap, zorder=0, extend = 'both')
    if vname == 'SIarea':
        contour = ax.contour(lons,lats,data_2d_ma, transform=grid_proj,colors='k',zorder=3,linewidths=0.5)
    
    if title != None:
        plt.title(title, fontsize=10)
    if data_x is not None:
        bathy_col = 'k' #change bathymetry contours to gray bc quivers are black. eh i think black is fine.
        q_lon,q_lat = data_x.lon.values, data_x.lat.values

        # smaller scale means longer arrows. smaller regrid = fewer arrows.
        if q_scale == None:
                q_scales = {'T':0.4,'oceTAU':0.4,'wind':50,'U':.4,'mag':.2,'wind_mag':70,
                            'wind_mag_comp': 50,'Velocity':.4,'Velocity_comp':0.04}# wind_ma 80
                q_scale = q_scales[quiv_vname]
        if regrid == None:
            regrid_dict = {'AS_near':12,'AS_far':25,'ASBS':25,'full_model_domain':12,'AS_long':8,'PIB':12}
            regrid = regrid_dict[region]
        # arrow width
        width_dict = {'AS_near':.02,'AS_far':.02,'ASBS':.015,'full_model_domain':.01,'AS_long':.02,'PIB':.02}
        if quiv_vname == 'wind_mag_comp':
            quiv = ax.quiver(q_lon,q_lat, data_x.values, data_y.values,
                      transform = grid_proj, regrid_shape=10,
                      pivot = 'middle',scale=3.5, width=0.01,
                      edgecolor='white',headlength=5,
                      linewidth = 0.2,zorder=3)#lw .15, width .005, width=0.007
        elif quiv_vname == 'Velocity':
            quiv = ax.quiver(q_lon,q_lat, data_x.values, data_y.values,
                      transform = grid_proj, regrid_shape=18,
                      pivot = 'middle',scale=.45, width=0.01,
                      edgecolor='white',headwidth=4,headlength=6,
                      linewidth = 0.2,zorder=3)#lw .15, width .005, width=0.007
        # elif quiv_vname == 'Velocity_comp':
        #     quiv = ax.quiver(q_lon,q_lat, data_x.values, data_y.values,
        #               transform = grid_proj, regrid_shape=18,
        #               pivot = 'middle',scale=.15, width=0.01,
        #               edgecolor='white',headlength=4,
        #               linewidth = 0.2,zorder=3)#lw .15, width .005, width=0.007
        elif quiv_vname == 'wind_speed_anom':
            quiv = ax.quiver(q_lon,q_lat, data_x.values, data_y.values,
                      transform = grid_proj, regrid_shape=18,
                      pivot = 'middle',scale=.15, width=0.01,
                      edgecolor='white',headlength=4,
                      linewidth = 0.2,zorder=3)#lw .15, width .005, width=0.007
        else:
            quiv = ax.quiver(q_lon,q_lat, data_x.values, data_y.values,
                          transform = grid_proj, regrid_shape=regrid,
                          pivot = 'middle',scale=q_scale,
                          width=width_dict[region],
                          headlength=2,headaxislength=1.5,
                          minshaft=.5,
                          edgecolor='white',
                          linewidth = 0.2,zorder=3)#lw .15, width .005, width=0.007
            
        ax.quiverkey(quiv, X=0.025, Y=0.0, U=vec_len, label=str(vec_len)+' m/s', labelpos='S')
    else:
        bathy_col = 'k'
    # Add colorbar
    cb_ax = fig.add_axes([0.22,0.1,0.6,0.045])
    cb_em = fig.colorbar(cf_em, cax=cb_ax, extend='both',orientation = 'horizontal')  
    cb_em.set_label(label = labs_dict[vname], fontsize=8)
    cb_em.ax.tick_params(labelsize=7)
    cb_em.set_ticks([levs[0],levs[-1]])
    cb_em.set_ticklabels([levs[0],levs[-1]])
    
    # Plot bathyemtry and ice locations in all subplots, and format map extent
    # Get bathymetry and ice data used on all plots
    land_ice_ds = get_bathymetry_and_troughs()
    land_ice_ds = land_ice_ds.sel(lat=slice(lat1,lat2),lon=slice(lon1,lon2))
    lons,lats = land_ice_ds.lon, land_ice_ds.lat
    blevs = (500,1000)
    
    # Plot bathymetry and ice
    if vname == 'SIarea' or vname == 'Bathymetry':
        ax.contour(lons,lats,land_ice_ds.bathy,blevs,colors='gray',transform=grid_proj,linewidths=0.2,zorder=1)
    else:
        ax.contour(lons,lats,land_ice_ds.bathy,blevs,colors=bathy_col,transform=grid_proj,linewidths=1,zorder=1)
    # Uncomment to Plot shelf break in thicker line
    # ax.contour(lons,lats,land_ice_ds.bathy,(1000,),colors=bathy_col,transform=grid_proj,linewidths=1,zorder=1)
    
    ax.contourf(lons,lats,land_ice_ds.all_ice,transform=grid_proj,colors=['lightgray']*2,alpha=0.6,zorder=2)
    ax.contourf(lons,lats,land_ice_ds.grounded_ice,transform=grid_proj,cmap='binary_r',zorder=2)
    
    # Set shape of map to match shape of data rather than a rectangle
    rect = mpath.Path([[lon1, lat2], [lon2, lat2],
    [lon2, lat1], [lon1, lat1], [lon1, lat2]]).interpolated(50)
    proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    rect_in_target = proj_to_data.transform_path(rect)
    ax.set_boundary(rect_in_target)
    ax.set_xlim(rect_in_target.vertices[:,0].min(), rect_in_target.vertices[:,0].max())
    ax.set_ylim(rect_in_target.vertices[:,1].min(), rect_in_target.vertices[:,1].max())
    # Uncomment for grid lines
    # gl = ax.gridlines(crs=ccrs.PlateCarree(), linestyle='--', linewidth=0.2,color='gray', 
    #                   xlocs=np.arange(-140,-60,5),
    #                   ylocs=np.arange(-76,-60,1),draw_labels=True,\
    #                   x_inline=False,y_inline=False,rotate_labels=False)
    # gl.top_labels=False
    # gl.right_labels=False
    # gl.xlabel_style = {'size': 8}
    # gl.ylabel_style = {'size': 8}
    fig.subplots_adjust(left=0.1,right=0.95,top=0.85,bottom=0.15,wspace=0.1,hspace=.05)
    
    if save:
        plt.savefig(save, bbox_inches='tight', dpi=500)
        print('Saving figure as',save)
        
    return

def make_side_profile_figs(data_2d, vname, plot_format,T_data = None, T_levs = None, col_titles = None, row_titles = None,\
                           subplot_labels = None, levs = None, cmap = None, xlims = None, ylims = None, fig_name = None):
    """
    Makes a set of simple contour plots with depth on the y axis and lat or lon on the x axis 
    Could be 1 plot or multiple plots (with 3 rows)
    Needs to be one data type (only one colorbar)
    
    
    Parameters
    ----------
    data_2d: 2d xr data array or list of 2d xr data arrays
        data array should have shape (n_depths, n_lats) or (n_depths, n_lons)
    vname: str corresponding to key in units_dict below
        for labeling colorbar
    plot_format: list of 6 floats or ints for formatting plot
        [n_rows,n_cols,width,height,bot_pos,cb_height]
        bot_pos = lower position of subplots, cb_height = height (as a fraction of plot height) for colorbar, 
    T_data (optional): same format as data_2d, or None (default)
        for plotting T as contours on top of data_2d
    title (optional): str
        title of plot
    levs (optional): 1d np array or list
        contains contour levels
        if None, uses min and max values and linearly interpolates 11 values
    cmap (optional): str
        matplotlib cmap or accepted list of colors, i.e. 'viridis' or 'PuOr_r'
    xlims (optional): list of 2 floats or ints
        contains figure limits for x axis (latitudes)
    ylims (optional): list of 2 floats or ints (in increasing order)
        contains figure limits for y axis (depths)
        
    Returns
    -------
    None (displays matplotlib figure)
    
    """
    
    units_dict = {'T':'Pot. Temp (°C)',\
                  'T anomaly':'Pot. Temp anomaly(°C)',\
                  'S':'Salinity (psu)',\
                  'S anomaly':'Salinity anomaly (psu)',\
                  'U':'U (m/s)','U anomaly':'U anomaly (m/s)',\
                  'V':'V (m/s)','V anomaly':'V anomaly (m/s)' }
    
    
    n_rows,n_cols,width,height,bot_pos,cb_x_pos,cb_y_pos,cb_height = plot_format
    
    x_var = data_2d[0].dims[-1]
    if x_var =='lat':
        x_lab = 'Latitude'
    elif x_var == 'lon':
        x_lab = 'Longitude'
    n_runs = len(data_2d)

    if type(levs) != np.ndarray:
            levs = np.linspace(np.min(data_2d),np.max(data_2d),11)
    levs_og = levs

    fig = plt.figure()
    fig.set_size_inches((width,height))

    for i in range (n_runs):
        #reset levels in case they were changed for the previous subplot
        levs = levs_og
        ax = fig.add_subplot(n_rows,n_cols,i+1)

        # Mask data where 0 (model uses 0s instead of nans) and get x/y data
        data_i = data_2d[i]
        data_i_mask = data_i.where(data_i.values != 0) #i.e., (70,416)
        x_data = data_i[x_var].values
        y_data = data_i.depth.values/1000
        

        # Plot data as filled contours
        cf=ax.contourf(x_data,y_data,data_i_mask,levs,cmap=cmap,extend='both')
        
        # if T_Data, Plot T data as contours ontop of filled contour plot
        if T_data is not None:
            
            # plot T contours
            
            if len(T_data) > 0 and n_runs == 1:
                T_i = T_data
                colors = ['#bdbdbd','#969696','#737373','#525252','#252525','k'] #grays
                # colors = ['#feebe2','#fcc5c0','#fa9fb5','#f768a1','#c51b8a','#7a0177'] #pinks
                # colors = ['#fc9272','#fb6a4a','#ef3b2c','#cb181d','#99000d']#reds
                # colors = ['#fee391','#fec44f','#fe9929','#d95f0e','#993404']#oranges
                for j in range(len(T_data)):
                    ax.contour(x_data,y_data,T_data[j],levels = T_levs,colors=colors[j],linewidths=1.5)
            else:
                T_i = T_data[i]
                cp = ax.contour(x_data,y_data,T_i,T_levs,colors='k',linewidths=1)
                plt.clabel(cp, inline=True, fontsize=7)
                
        
        # Format plot
        plt.gca().set_facecolor('darkgray')
        plt.yticks(fontsize=8)
        plt.xticks(fontsize=8)
        if ylims:
            plt.ylim(ylims)
            y1,y2 = ylims
            # ax.set_yticks(np.linspace(y1,y2,4,dtype=float)) #4 total depth labels
        if xlims:
            plt.xlim(xlims)
            x1,x2 = xlims
            ax.set_xticks(np.arange(x1,x2,2)) #lat or lon every 2 deg
        if i == n_runs-2 or i == n_runs - 1:
            ax.set_xlabel(x_lab)

        # Add column title 
        if i < n_cols:
            if col_titles != None:
                plt.title(col_titles[i],fontsize=9)

        # Add row annotation to label data on this row
        if i% n_cols == 0: 

            if row_titles != None:
                plt.annotate(row_titles[i//n_cols],(0.02,0.05),xycoords='axes fraction',fontsize=10)
 
        # if no row and col titles given, set subplot titles to run names:
        if subplot_labels is not None:
            plt.annotate(subplot_labels[i],(0.02,0.05),xycoords='axes fraction',fontsize=11,fontweight='bold')

        # Add colorbar (add 2 for salinity for actual psu and psu diff)
        #(i+1) % n_cols == 0
        #elif i == len(data_2d)-1:
        cb_width = 9/(width*3) #was x4
        cb_ax = fig.add_axes([cb_x_pos,cb_y_pos,cb_width,cb_height]) #[.3, 10](9/(2*10) [.9,2]9/(2*5), 
        cb = fig.colorbar(cf, cax=cb_ax, extend='both',orientation = 'horizontal')
        cb.set_label(units_dict[vname],fontsize=10)
        cb.ax.tick_params(labelsize=9)
        cb.set_ticks(levs[::2])
        # cb.set_ticklabels(levs)
    
    
    if len(data_2d) == 1:
        fig.text(0.03,.7,'Depth (km)',va='center',rotation='vertical')
        plt.subplots_adjust(left=0.2,right=0.98,bottom=bot_pos,top=0.97)
    else:
        fig.text(0.03,(.94-bot_pos)/2+.12,'Depth (km)',va='center',rotation='vertical')
        plt.subplots_adjust(left=0.12,right=0.98,bottom=bot_pos,top=0.94,hspace=0.3,wspace=0.14)
        
    
    if fig_name != None:
        plt.savefig(fig_name,dpi=400)
        print('saving as',fig_name)
    
    return 
