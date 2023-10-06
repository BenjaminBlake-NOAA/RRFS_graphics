#!/bin/usr/env/python

import grib2io
import pyproj
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
import time,os,sys,multiprocessing,datetime
import multiprocessing.pool
from scipy import ndimage
from netCDF4 import Dataset
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

fhours = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]
dtime = datetime.datetime(year,month,day,hour,0)
date_list = [dtime + datetime.timedelta(hours=x) for x in fhours]

# Define the directory paths to the output files
NAM_DIR = '/lfs/h1/ops/prod/com/nam/v4.2/nam.'+ymd
RRFS_DIR = '/lfs/h2/emc/ptmp/emc.lam/rrfs/v0.6.7/prod/rrfs.'+ymd+'/'+cyc

# Paths to image files
im = image.imread('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/rrfs_graphics/noaa.png')

######################################
#    SET UP FIGURE FOR THE DOMAIN    #
######################################

# Call the domain_latlons_proj function from rrfs_plot_utils
xextent,yextent,offset,extent,myproj = rrfs_plot_utils.domain_latlons_proj(dom)

# Create the figure
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
coastlines=cfeature.NaturalEarthFeature('physical','coastline',
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

# high-resolution background images
if back_img=='on':
  img = plt.imread('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python/NaturalEarth/raster_files/NE1_50M_SR_W.tif')
  ax1.imshow(img, origin='upper', transform=transform)
  ax2.imshow(img, origin='upper', transform=transform)
  ax3.imshow(img, origin='upper', transform=transform)

ax1.add_feature(cfeature.LAND, linewidth=0, facecolor='lightgray')
ax1.add_feature(lakes)
ax1.add_feature(states)
ax1.add_feature(coastlines)
ax2.add_feature(cfeature.LAND, linewidth=0, facecolor='lightgray')
ax2.add_feature(lakes)
ax2.add_feature(states)
ax2.add_feature(coastlines)
ax3.add_feature(cfeature.LAND, linewidth=0, facecolor='lightgray')
ax3.add_feature(lakes)
ax3.add_feature(states)
ax3.add_feature(coastlines)

# Map/figure has been set up here, save axes instances for use again later
keep_ax_lst_1 = ax1.get_children()[:]
keep_ax_lst_2 = ax2.get_children()[:]
keep_ax_lst_3 = ax3.get_children()[:]

xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()
xmax = int(round(xmax))
ymax = int(round(ymax))

###################################################

t1dom = time.perf_counter()

for j in range(len(date_list)):

  fhour = str(fhours[j]).zfill(2)
  fhr = int(fhour)
  fhrm1 = fhr - 1
  fhour1 = str(fhrm1).zfill(2)
  print('fhour '+fhour)

  # Forecast valid date/time
  itime = ymdh
  vtime = rrfs_plot_utils.ndate(itime,int(fhr))

# Define the input files
  if dom == 'alaska':
    data1 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.alaskanest.hiresf'+fhour+'.tm00.grib2')
    data1_m1 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.alaskanest.hiresf'+fhour1+'.tm00.grib2')
    data2 = grib2io.open(RRFS_DIR+'/rrfs.t'+cyc+'z.prslev.f0'+fhour+'.ak.grib2')
  elif dom == 'hawaii':
    data1 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.hawaiinest.hiresf'+fhour+'.tm00.grib2')
    data1_m1 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.hawaiinest.hiresf'+fhour1+'.tm00.grib2')
    data2 = grib2io.open(RRFS_DIR+'/rrfs.t'+cyc+'z.prslev.f0'+fhour+'.hi.grib2')
  elif dom == 'puerto_rico':
    data1 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.priconest.hiresf'+fhour+'.tm00.grib2')
    data1_m1 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.priconest.hiresf'+fhour1+'.tm00.grib2')
    data2 = grib2io.open(RRFS_DIR+'/rrfs.t'+cyc+'z.prslev.f0'+fhour+'.pr.grib2')
  else:
    data1 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.conusnest.hiresf'+fhour+'.tm00.grib2')
    data1_m1 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.conusnest.hiresf'+fhour1+'.tm00.grib2')
    data2 = grib2io.open(RRFS_DIR+'/rrfs.t'+cyc+'z.prslev.f0'+fhour+'.conus_3km.grib2')

  if (fhr <= 3):
    qpf = data1.select(shortName='APCP',timeRangeOfStatisticalProcess=fhr)[0].data * 0.0393701
    qpf_1 = qpf
    asnow = data1.select(shortName='WEASD')[1].data / 2.54
    asnow_1 = asnow
  elif (fhr > 3) and (fhr % 3 == 1):
    qpf = data1.select(shortName='APCP',timeRangeOfStatisticalProcess=1)[0].data * 0.0393701
    qpf_1 += qpf
    asnow = data1.select(shortName='WEASD')[1].data / 2.54
    asnow_1 += asnow
  elif (fhr > 3) and (fhr % 3 == 2):
    qpf = data1.select(shortName='APCP',timeRangeOfStatisticalProcess=2)[0].data * 0.0393701
    qpfm1 = data1_m1.select(shortName='APCP',timeRangeOfStatisticalProcess=1)[0].data * 0.0393701
    qpf_1 += (qpf-qpfm1)
    asnow = data1.select(shortName='WEASD')[1].data / 2.54
    asnowm1 = data1_m1.select(shortName='WEASD')[1].data /2.54
    asnow_1 += asnow
  elif (fhr > 3) and (fhr % 3 == 0):
    qpf = data1.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
    qpfm1 = data1_m1.select(shortName='APCP',timeRangeOfStatisticalProcess=2)[0].data * 0.0393701
    qpf_1 += (qpf-qpfm1)
    asnow = data1.select(shortName='WEASD')[1].data / 2.54
    asnowm1 = data1_m1.select(shortName='WEASD')[1].data /2.54
    asnow_1 += asnow
#  qpf_2 = data2.select(shortName='APCP',timeRangeOfStatisticalProcess=fhr)[0].data * 0.0393701
  qpf_2 = data2.select(shortName='APCP')[1].data * 0.0393701
  qpf_dif = qpf_2 - qpf_1
  asnow_2 = data2.select(shortName='ASNOW')[0].data * 39.3701
  asnow_dif = asnow_2 - asnow_1

###################################################

# Get the lats and lons - only need to do this once
  if (fhr == 1):
    msg = data1.select(shortName='HGT', level='500 mb')[0]  # msg is a Grib2Message object
    lat,lon,lat_shift,lon_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg)

#################################
  # Plot Total QPF
#################################
  t1 = time.perf_counter()
  print(('Working on Total QPF for forecast hour '+fhour))
  
  # Clear off old plottables but keep all the map info
  if (fhr > 1):
    cbar1.remove()
    cbar2.remove()
    cbar.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
    rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'in'
  clevs = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,7,10,15,20]
  clevsdif = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
  colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','deepskyblue','cyan','mediumpurple','mediumorchid','darkmagenta','darkred','crimson','orangered','darkorange','goldenrod','gold','yellow']
  difcolors = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','white','white','#EEEE00','#EEC900','darkorange','orangered','red']
  cm = matplotlib.colors.ListedColormap(colorlist)
  cmdif = matplotlib.colors.ListedColormap(difcolors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,qpf_1,transform=transform,cmap=cm,vmin=0.01,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('pink')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'NAM Nest '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,qpf_2,transform=transform,cmap=cm,vmin=0.01,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('pink')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,qpf_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar.set_label(units,fontsize=6)
  cbar.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - NAM Nest '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save('compareqpf_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Total QPF for: forecast hour '+fhour) % t3)

#################################
  # Plot Snowfall
#################################
  t1 = time.perf_counter()
  print(('Working on snowfall for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'in'
  clevs = [0.1,1,2,3,6,9,12,18,24,36,48]
  clevsdif = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  cm = rrfs_plot_utils.ncl_perc_11Lev()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,asnow_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'NAM Nest Snowfall (10:1) ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,asnow_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Snowfall (variable density) ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,asnow_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar.set_label(units,fontsize=6)
  cbar.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - NAM Nest Snowfall ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save('compareasnow_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot snowfall for: '+dom) % t3)


t3dom = round(t2-t1dom, 3)
print(("%.3f seconds to plot qpf and snowfall for: "+dom) % t3dom)
plt.clf()

