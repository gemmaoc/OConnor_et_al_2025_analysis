#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:29:09 2023

@author: gemma
"""

from matplotlib import pyplot as plt
import numpy as np
import itertools

# Ordered from strongest westerly to strongest eastery.
wind_order_forcings = [2015,1991,1982,1994,1992,1984,2009,2003,1989,2011]   
runs_ic_1995_wind_order = ['run_forc_erai_'+str(yr)+'_rep_ic_1995' for yr in wind_order_forcings]
runs_ic_2001_wind_order = ['run_forc_erai_'+str(yr)+'_rep_ic_2001' for yr in wind_order_forcings]
runs_ic_2008_wind_order = ['run_forc_erai_'+str(yr)+'_rep_ic_2008' for yr in wind_order_forcings]

all_runs_wind_order = runs_ic_1995_wind_order + runs_ic_2001_wind_order + runs_ic_2008_wind_order

# order forcings by warm 4, cool 4, 2015, and other (by CDW volume on shelf)
warm_order_forcings = [1982,1984,1989,1992,1994,2003,2009,1991,2015,2011] #for thesis and OSM, 1991 was neutral, 2011 was cool.
runs_ic_1995_warm_order = ['run_forc_erai_'+str(yr)+'_rep_ic_1995' for yr in warm_order_forcings]
runs_ic_2001_warm_order = ['run_forc_erai_'+str(yr)+'_rep_ic_2001' for yr in warm_order_forcings]
runs_ic_2008_warm_order = ['run_forc_erai_'+str(yr)+'_rep_ic_2008' for yr in warm_order_forcings]

all_runs_warm_order = runs_ic_1995_warm_order + runs_ic_2001_warm_order + runs_ic_2008_warm_order


# Incomplete sets (add new runs to this section and then move once complete)
runs_ic_2015_incomplete = ['run_forc_erai_2015_rep_ic_2015']#run complete, set incomplete
runs_ic_1995_incomplete = [\
                           #'run_forc_erai_1996_rep_ic_1995',\ (incomplete run)
                           #'run_forc_erai_2010_rep_ic_1995'\ (incomplete run)
                          ]

exp_set_dict_wind_order = {'control':['control'],\
                'ic_1995':runs_ic_1995_wind_order,\
                'ic_2001':runs_ic_2001_wind_order,\
                'ic_2008':runs_ic_2008_wind_order,\
                'ic_2015':runs_ic_2015_incomplete\
                }

exp_set_dict_warm_order = {'control':['control'],\
                'ic_1995':runs_ic_1995_warm_order,\
                'ic_2001':runs_ic_2001_warm_order,\
                'ic_2008':runs_ic_2008_warm_order }

# colors
west_colors = plt.cm.Reds_r(np.linspace(0,1,5))[:-1] # Cut off the lightest colors
weak_colors = plt.cm.Greens_r(np.linspace(0,1,5))[:-1]
east_colors = plt.cm.Blues(np.linspace(0,1,3))[1:]
all_10_colors = list(itertools.chain(west_colors,weak_colors,east_colors))

colors_dict = {'control':['darkgray'],\
               'ic_1995':all_10_colors,\
               'ic_2001':all_10_colors,\
               'ic_2008':all_10_colors,\
               'ic_2015':[all_10_colors[0]]}
# for selecting only some colors
#[all_10_colors[0],all_10_colors[2],all_10_colors[5],all_10_colors[-1]]

