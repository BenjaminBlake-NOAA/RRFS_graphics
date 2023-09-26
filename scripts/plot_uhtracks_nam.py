#!/bin/usr/env python

import grib2io
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib
import io
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as image
from matplotlib.gridspec import GridSpec
import numpy as np
import time,os,sys,multiprocessing
import multiprocessing.pool
from scipy import ndimage
from netCDF4 import Dataset
import pyproj
import cartopy
import rrfs_plot_utils

#-------------------------------------------------------#

# Necessary to generate figs when not running an Xserver (e.g. via PBS)
plt.switch_backend('agg')

# Read date/time and domain from command line
ymdh = str(sys.argv[1])
dom = str(sys.argv[2])

ymd = ymdh[0:8]
year = int(ymdh[0:4])
month = int(ymdh[4:6])
day = int(ymdh[6:8])
hour = int(ymdh[8:10])
cyc = str(hour).zfill(2)
print(year, month, day, hour)

# Runlength for NAM Nest and RRFS_A forecasts is 60 hours
runlength = 60

# Forecast init and valid date/time
itime = ymdh
fhrs = np.arange(0,runlength+1,1)
vtime_list = [rrfs_plot_utils.ndate(itime,int(x)) for x in fhrs]

# Define the directory paths to the output files
NAM_DIR = '/lfs/h1/ops/prod/com/nam/v4.2/nam.'+ymd
RRFS_DIR = '/lfs/h2/emc/ptmp/emc.lam/rrfs/v0.6.7/prod/rrfs.'+ymd+'/'+cyc

# Define prod and para strings
prod_str = 'NAM Nest'
para_str = 'RRFS_A'

# Paths to image files
im = image.imread('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/rrfs_graphics/noaa.png')

###################################################
# Read in all variables and calculate differences #
###################################################
t1a = time.perf_counter()

uh25_list_1 = []
uh25_list_2 = []
uh25_list_both = []

for j in range(len(vtime_list)):

    fhr = fhrs[j]
    fhour = str(fhr).zfill(2)
    print(('fhour '+fhour))
    vtime = vtime_list[j]

    # Define the input files
    data1 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.conusnest.hiresf'+fhour+'.tm00.grib2')
    data2 = grib2io.open(RRFS_DIR+'/rrfs.t'+cyc+'z.prslev.f0'+fhour+'.conus_3km.grib2')

    # Updraft helicity
    if (fhr > 0):
        uh25_1 = data1.select(shortName='MXUPHL',level='5000-2000 m above ground')[0].data
        uh25_2 = data2.select(shortName='MXUPHL',level='5000-2000 m above ground')[0].data
    elif (fhr == 0):
        uh25_1 = data1.select(shortName='APCP')[0].data * 0.
        uh25_2 = data2.select(shortName='APCP')[0].data * 0.
        uh25_1_final = np.zeros_like(uh25_1)
        uh25_2_final = np.zeros_like(uh25_2)

    uh25_1_final = np.where(uh25_1_final >= uh25_1, uh25_1_final, uh25_1)
    uh25_2_final = np.where(uh25_2_final >= uh25_2, uh25_2_final, uh25_2)
    uh25_1b = np.where(uh25_1_final > 50, 1, 0)
    uh25_2b = np.where(uh25_2_final > 50, 1, 0)
    uh25_both = uh25_1b + uh25_2b

    uh25_list_1.append(uh25_1_final)
    uh25_list_2.append(uh25_2_final)
    uh25_list_both.append(uh25_both)

t2a = time.perf_counter()
t3a = round(t2a-t1a, 3)
print(("%.3f seconds to read all messages") % t3a)


# Get the lats and lons
msg = data1.select(shortName='HGT', level='500 mb')[0]	# msg is a Grib2Message object
lat,lon,lat_shift,lon_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg)

# colors for difference plots, only need to define once
difcolors = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','white','white','#EEEE00','#EEC900','darkorange','orangered','red']

######################################
#    SET UP FIGURE FOR THE DOMAIN    #
######################################

# Call the domain_latlons_proj function from rrfs_plot_utils
xextent,yextent,offset,extent,myproj = rrfs_plot_utils.domain_latlons_proj(dom)

# Create figure and axes instances
fig = plt.figure(figsize=(9,8))
gs = GridSpec(9,8,wspace=0.0,hspace=0.0)

# Define where Cartopy maps are located
cartopy.config['data_dir'] = '/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python/NaturalEarth'
back_res='50m'
back_img='off'

ax1 = fig.add_subplot(gs[0:4,0:4], projection=myproj)
ax2 = fig.add_subplot(gs[0:4,4:], projection=myproj)
ax3 = fig.add_subplot(gs[5:,1:7], projection=myproj)
ax1.set_extent(extent)
ax2.set_extent(extent)
ax3.set_extent(extent)
axes = [ax1, ax2, ax3]

fline_wd = 0.5  # line width
fline_wd_lakes = 0.35  # line width
falpha = 0.5    # transparency

# natural_earth
lakes=cfeature.NaturalEarthFeature('physical','lakes',back_res,
                  edgecolor='black',facecolor='none',
                  linewidth=fline_wd_lakes,alpha=falpha)
coastline=cfeature.NaturalEarthFeature('physical','coastline',
                  back_res,edgecolor='black',facecolor='none',
                  linewidth=fline_wd,alpha=falpha)
states=cfeature.NaturalEarthFeature('cultural','admin_1_states_provinces',
                  back_res,edgecolor='black',facecolor='none',
                  linewidth=fline_wd,alpha=falpha)
borders=cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                  back_res,edgecolor='black',facecolor='none',
                  linewidth=fline_wd,alpha=falpha)

# All lat lons are earth relative, so setup the associated projection correct for that data
transform = ccrs.PlateCarree()

ax1.add_feature(cartopy.feature.OCEAN, color='white', zorder=0)
ax1.add_feature(cartopy.feature.LAND, color='lightgray', zorder=0, linewidth=0.5, edgecolor='black')
ax2.add_feature(cartopy.feature.OCEAN, color='white', zorder=0)
ax2.add_feature(cartopy.feature.LAND, color='lightgray', zorder=0, linewidth=0.5, edgecolor='black')
ax3.add_feature(cartopy.feature.OCEAN, color='white', zorder=0)
ax3.add_feature(cartopy.feature.LAND, color='lightgray', zorder=0, linewidth=0.5, edgecolor='black')

ax1.add_feature(lakes)
ax1.add_feature(states)
#ax1.add_feature(coastline)
ax2.add_feature(lakes)
ax2.add_feature(states)
#ax2.add_feature(coastline)
ax3.add_feature(lakes)
ax3.add_feature(states)
#ax3.add_feature(coastline)

# Map/figure has been set up here, save axes instances for use again later
keep_ax_lst_1 = ax1.get_children()[:]
keep_ax_lst_2 = ax2.get_children()[:]
keep_ax_lst_3 = ax3.get_children()[:]


#################################
  # Plot Run-Total 2-5 km UH
#################################

units = '$\mathregular{m^{2}}$ $\mathregular{s^{-2}}$'
clevs = [25,50,100,150,200,250,300]
clevsdif = [25,1000]
clevsboth = [1.5,2.5]
colorlist = ['blue','turquoise','#EEEE00','darkorange','firebrick','darkviolet']
cm = matplotlib.colors.ListedColormap(colorlist)
norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()
xmax = int(round(xmax))
ymax = int(round(ymax))

for fhr in fhrs:

  t1 = time.perf_counter()
  print('Working on Run-Total 2-5 km UH for forecast hour '+fhour)

  fhour = str(fhr).zfill(2)
  vtime = vtime_list[fhr]

  uh25_1 = uh25_list_1[fhr]
  uh25_2 = uh25_list_2[fhr]
  uh25_both = uh25_list_both[fhr]

  cs_1 = ax1.contourf(lon,lat,uh25_1,levels=clevs,cmap=cm,transform=transform,vmin=25,extend='max')
  cs_1.cmap.set_under('#E5E5E5',alpha=0.)
  cs_1.cmap.set_over('black')
  if (fhour == '00'):
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8)
    cbar1.set_label(units,fontsize=5)
    cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,prod_str+' Max 2-5 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.contourf(lon,lat,uh25_2,levels=clevs,cmap=cm,transform=transform,vmin=25,extend='max')
  cs_2.cmap.set_under('#E5E5E5',alpha=0.)
  cs_2.cmap.set_over('black')
  if (fhour == '00'):
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8)
    cbar2.set_label(units,fontsize=5)
    cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,para_str+' Max 2-5 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  csdif1 = ax3.contourf(lon,lat,uh25_1,levels=clevsdif,colors='red',transform=transform)
  csdif2 = ax3.contourf(lon,lat,uh25_2,levels=clevsdif,colors='dodgerblue',transform=transform)
  csdif3 = ax3.contourf(lon,lat,uh25_both,levels=clevsboth,colors='indigo',transform=transform)
  ax3.text(.5,1.03,prod_str+' (red), '+para_str+' (blue), Both (purple) \n Max 2-5 km Updraft Helicity > 50 ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save('compareuh25_accum_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Run-Total 2-5 km UH for forecast hour '+fhour) % t3)

