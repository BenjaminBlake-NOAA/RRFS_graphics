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

# Read date/time and forecast hour from command line
ymdh = str(sys.argv[1])
ymd = ymdh[0:8]
year = int(ymdh[0:4])
month = int(ymdh[4:6])
day = int(ymdh[6:8])
hour = int(ymdh[8:10])
cyc = str(hour).zfill(2)
print(year, month, day, hour)

ymdh_model = str(sys.argv[2])
ymd_model = ymdh_model[0:8]
year_model = int(ymdh_model[0:4])
month_model = int(ymdh_model[4:6])
day_model = int(ymdh_model[6:8])
hour_model = int(ymdh_model[8:10])
cyc_model = str(hour_model).zfill(2)
print(year_model, month_model, day_model, hour_model)

fhr = int(sys.argv[3])
fhour = str(fhr).zfill(2)
fhourm24 = str(fhr-24).zfill(2)
print('fhour '+fhour)

# Forecast valid date/time
itime = ymd_model
vtime_start = rrfs_plot_utils.ndate(ymdh_model,int(fhr-24))
vtime_start = str(vtime_start[0:8])
vtime_end = ymd

# Define the directory paths to the output files
HRRR_DIR = '/lfs/h1/ops/prod/com/hrrr/v4.1/hrrr.'+ymd_model+'/conus'
NAM_DIR = '/lfs/h1/ops/prod/com/nam/v4.2/nam.'+ymd_model
RRFS_DIR = '/lfs/h2/emc/ptmp/Benjamin.Blake/rrfs/na/prod/rrfs.'+ymd_model+'/'+cyc_model
CCPA_DIR = '/lfs/h1/ops/prod/com/evs/v1.0/stats/cam/atmos.'+ymd+'/namnest/precip/spatial_maps'
StageIV_DIR = '/lfs/h1/ops/prod/com/pcpanl/v4.1/pcpanl.'+ymd

# Specify plotting domains
domains = ['conus','boston_nyc','central','colorado','la_vegas','mid_atlantic','north_central','northeast','northwest','ohio_valley','south_central','southeast','south_florida','sf_bay_area','seattle_portland','southwest','upper_midwest']
#domains = ['conus']

# Paths to image files
im = image.imread('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/rrfs_graphics/noaa.png')

#-------------------------------------------------------#

# Make Python process pools non-daemonic
class NoDaemonProcess(multiprocessing.Process):
  # make 'daemon' attribute always return False
  @property
  def daemon(self):
    return False
  
  @daemon.setter
  def daemon(self, value):
    pass

class NoDaemonContext(type(multiprocessing.get_context())):
  Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
  def __init__(self, *args, **kwargs):
    kwargs['context'] = NoDaemonContext()
    super(MyPool, self).__init__(*args, **kwargs)

#-------------------------------------------------------#

def main():

  # Number of processes must coincide with the number of domains to plot
  pool = MyPool(len(domains))
  pool.map(vars_figure,domains)

#-------------------------------------------------------#

def vars_figure(domain):

  global dom
  dom = domain
  print(('Working on '+dom))

  global lat,lon,lat_shift,lon_shift,fig,axes,ax1,ax2,ax3,ax4,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,xextent,yextent,offset,extent,myproj,transform

# Define the input files
  data1 = grib2io.open(HRRR_DIR+'/hrrr.t'+cyc_model+'z.wrfprsf'+fhour+'.grib2')
  fhour_03 = str(fhr - 21).zfill(2)
  data2_03 = grib2io.open(NAM_DIR+'/nam.t'+cyc_model+'z.conusnest.hiresf'+fhour_03+'.tm00.grib2') 
  fhour_06 = str(fhr - 18).zfill(2)
  data2_06 = grib2io.open(NAM_DIR+'/nam.t'+cyc_model+'z.conusnest.hiresf'+fhour_06+'.tm00.grib2')
  fhour_09 = str(fhr - 15).zfill(2)
  data2_09 = grib2io.open(NAM_DIR+'/nam.t'+cyc_model+'z.conusnest.hiresf'+fhour_09+'.tm00.grib2') 
  fhour_12 = str(fhr - 12).zfill(2)
  data2_12 = grib2io.open(NAM_DIR+'/nam.t'+cyc_model+'z.conusnest.hiresf'+fhour_12+'.tm00.grib2') 
  fhour_15 = str(fhr - 9).zfill(2)
  data2_15 = grib2io.open(NAM_DIR+'/nam.t'+cyc_model+'z.conusnest.hiresf'+fhour_15+'.tm00.grib2') 
  fhour_18 = str(fhr - 6).zfill(2)
  data2_18 = grib2io.open(NAM_DIR+'/nam.t'+cyc_model+'z.conusnest.hiresf'+fhour_18+'.tm00.grib2') 
  fhour_21 = str(fhr - 3).zfill(2)
  data2_21 = grib2io.open(NAM_DIR+'/nam.t'+cyc_model+'z.conusnest.hiresf'+fhour_21+'.tm00.grib2') 
  data2_24 = grib2io.open(NAM_DIR+'/nam.t'+cyc_model+'z.conusnest.hiresf'+fhour+'.tm00.grib2') 
  data3 = grib2io.open(RRFS_DIR+'/rrfs.t'+cyc_model+'z.prslev.f0'+fhour+'.conus.grib2')

  data4 = grib2io.open(StageIV_DIR+'/st4_conus.'+ymdh+'.24h.grb2')
  data5 = Dataset(CCPA_DIR+'/ccpa.t'+cyc+'z.a24h.conus.nc','r')

# Get the lats and lons
  msg = data3.select(shortName='HGT', level='500 mb')[0]  # msg is a Grib2Message object
  lat,lon,lat_shift,lon_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg)
# Stage IV
  msg4 = data4.select(shortName='APCP')[0]	# msg is a Grib2Message object
  lat4,lon4,lat4_shift,lon4_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg4)
# CCPA
  lat5 = data5.variables['lat'][:,:]
  lon5 = data5.variables['lon'][:,:]

###################################################
# Read in all variables and calculate differences #
###################################################
  t1a = time.perf_counter()

  global qpf_1,qpf_2,qpf_3,qpf_4

# Total Precipitation
  qpf_1 = data1.select(shortName='APCP',timeRangeOfStatisticalProcess=fhr)[0].data * 0.0393701
  qpf_2_03 = data2_03.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
  qpf_2_06 = data2_06.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
  qpf_2_09 = data2_09.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
  qpf_2_12 = data2_12.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
  qpf_2_15 = data2_15.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
  qpf_2_18 = data2_18.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
  qpf_2_21 = data2_21.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
  qpf_2_24 = data2_24.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
  qpf_2 = qpf_2_03 + qpf_2_06 + qpf_2_09 + qpf_2_12 + qpf_2_15 + qpf_2_18 + qpf_2_21 + qpf_2_24
  qpf_3 = data3.select(shortName='APCP')[1].data * 0.0393701
# Stage IV
  qpf_4 = data4.select(shortName='APCP')[0].data * 0.0393701
# CCPA
  qpf_5 = data5.variables['APCP_24'][:,:] * 0.0393701

  t2a = time.perf_counter()
  t3a = round(t2a-t1a, 3)
  print(("%.3f seconds to read all messages") % t3a)

#######################################
#    SET UP FIGURE FOR EACH DOMAIN    #
#######################################

# Call the domain_latlons_proj function from rrfs_plot_utils
  xextent,yextent,offset,extent,myproj = rrfs_plot_utils.domain_latlons_proj(dom)

# Create figure and axes instances
  fig = plt.figure(figsize=(8,8))           
  gs = GridSpec(8,8,wspace=0.0,hspace=0.0)

  # Define where Cartopy maps are located
  cartopy.config['data_dir'] = '/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python/NaturalEarth'
  back_res='50m'
  back_img='off'

  ax1 = fig.add_subplot(gs[0:4,0:4], projection=myproj)
  ax2 = fig.add_subplot(gs[0:4,4:], projection=myproj)
  ax3 = fig.add_subplot(gs[4:,0:4], projection=myproj)
  ax4 = fig.add_subplot(gs[4:,4:], projection=myproj)
  ax1.set_extent(extent)
  ax2.set_extent(extent)
  ax3.set_extent(extent)
  ax4.set_extent(extent)
  axes = [ax1, ax2, ax3, ax4]

  fline_wd = 0.5  # line width
  fline_wd_lakes = 0.25  # line width
  falpha = 0.5    # transparency

  # natural_earth
  lakes=cfeature.NaturalEarthFeature('physical','lakes',back_res,
                    edgecolor='black',facecolor='none',
                    linewidth=fline_wd_lakes)
  coastline=cfeature.NaturalEarthFeature('physical','coastline',
                    back_res,edgecolor='black',facecolor='none',
                    linewidth=fline_wd,alpha=falpha)
  states=cfeature.NaturalEarthFeature('cultural','admin_1_states_provinces',
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
     ax4.imshow(img, origin='upper', transform=transform)

  ax1.add_feature(cfeature.LAND, linewidth=0, facecolor='white')
  ax1.add_feature(cfeature.OCEAN, linewidth=0, facecolor='lightgray')
  ax1.add_feature(cfeature.LAKES, edgecolor='black', linewidth=fline_wd_lakes, facecolor='lightgray',zorder=0)
  ax1.add_feature(lakes)
  ax1.add_feature(states)
  ax1.add_feature(coastline)
  ax2.add_feature(cfeature.LAND, linewidth=0, facecolor='white')
  ax2.add_feature(cfeature.OCEAN, linewidth=0, facecolor='lightgray')
  ax2.add_feature(cfeature.LAKES, edgecolor='black', linewidth=fline_wd_lakes, facecolor='lightgray',zorder=0)
  ax2.add_feature(lakes)
  ax2.add_feature(states)
  ax2.add_feature(coastline)
  ax3.add_feature(cfeature.LAND, linewidth=0, facecolor='white')
  ax3.add_feature(cfeature.OCEAN, linewidth=0, facecolor='lightgray')
  ax3.add_feature(cfeature.LAKES, edgecolor='black', linewidth=fline_wd_lakes, facecolor='lightgray',zorder=0)
  ax3.add_feature(lakes)
  ax3.add_feature(states)
  ax3.add_feature(coastline)
  ax4.add_feature(cfeature.LAND, linewidth=0, facecolor='white')
  ax4.add_feature(cfeature.OCEAN, linewidth=0, facecolor='lightgray')
  ax4.add_feature(cfeature.LAKES, edgecolor='black', linewidth=fline_wd_lakes, facecolor='lightgray',zorder=0)
  ax4.add_feature(lakes)
  ax4.add_feature(states)
  ax4.add_feature(coastline)

  # Map/figure has been set up here, save axes instances for use again later
  keep_ax_lst_1 = ax1.get_children()[:]
  keep_ax_lst_2 = ax2.get_children()[:]
  keep_ax_lst_3 = ax3.get_children()[:]
  keep_ax_lst_4 = ax4.get_children()[:]

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

#################################
  # Plot 24-hr QPF
#################################
  datasets = ['StageIV', 'CCPA']
  for pcpanl in datasets:

    t1 = time.perf_counter()
    print(('Working on 24-hr QPF for '+dom))

    if pcpanl=='StageIV':

      units = 'Precipitation (inches)'
      clevs = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,7,10,15,20]
      colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','deepskyblue','cyan','mediumpurple','mediumorchid','darkmagenta','darkred','crimson','orangered','darkorange','goldenrod','gold','yellow']  
      cm = matplotlib.colors.ListedColormap(colorlist)
      norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
   
      cs_1 = ax1.pcolormesh(lon_shift,lat_shift,qpf_1,transform=transform,cmap=cm,vmin=0.01,norm=norm)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('pink')
      cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.85,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
      cbar1.set_label(units,fontsize=6,labelpad=0)
      cbar1.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
      cbar1.ax.xaxis.set_tick_params(pad=0)
      cbar1.ax.tick_params(labelsize=6)
      ax1.text(.5,1.02,'HRRR',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax1.text(.5,0.95,itime+' '+cyc_model+'z cycle (f'+fhourm24+'-f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

      cs_2 = ax2.pcolormesh(lon_shift,lat_shift,qpf_2,transform=transform,cmap=cm,vmin=0.01,norm=norm)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('pink')
      cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.85,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
      cbar2.set_label(units,fontsize=6,labelpad=0)
      cbar2.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
      cbar2.ax.xaxis.set_tick_params(pad=0)
      cbar2.ax.tick_params(labelsize=6)
      ax2.text(.5,1.02,'NAM Nest',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax2.text(.5,0.95,itime+' '+cyc_model+'z cycle (f'+fhourm24+'-f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

      cs_3 = ax3.pcolormesh(lon_shift,lat_shift,qpf_3,transform=transform,cmap=cm,vmin=0.01,norm=norm)
      cs_3.cmap.set_under('white',alpha=0.)
      cs_3.cmap.set_over('pink')
      cbar3 = fig.colorbar(cs_3,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.85,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
      cbar3.set_label(units,fontsize=6,labelpad=0)
      cbar3.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
      cbar3.ax.xaxis.set_tick_params(pad=0)
      cbar3.ax.tick_params(labelsize=6)
      ax3.text(.5,1.02,'RRFS',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax3.text(.5,0.95,itime+' '+cyc_model+'z cycle (f'+fhourm24+'-f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax3.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax3.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

      cs_4 = ax4.pcolormesh(lon4,lat4,qpf_4,transform=transform,cmap=cm,vmin=0.01,norm=norm)
      cs_4.cmap.set_under('white',alpha=0.)
      cs_4.cmap.set_over('pink')
      cbar4 = fig.colorbar(cs_4,ax=ax4,orientation='horizontal',pad=0.01,shrink=0.85,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
      cbar4.set_label(units,fontsize=6,labelpad=0)
      cbar4.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
      cbar4.ax.xaxis.set_tick_params(pad=0)
      cbar4.ax.tick_params(labelsize=6)
      ax4.text(.5,1.02,'Stage IV',horizontalalignment='center',fontsize=6,transform=ax4.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax4.text(.5,0.95,vtime_start+' 12z - '+vtime_end+' 12z',horizontalalignment='center',fontsize=6,transform=ax4.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax4.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    elif pcpanl=='CCPA':

    # Clear off old plottables but keep all the map info
      cbar4.remove()
      rrfs_plot_utils.clear_plotables(ax4,keep_ax_lst_4,fig)

      cs_4 = ax4.pcolormesh(lon5,lat5,qpf_5,transform=transform,cmap=cm,vmin=0.01,norm=norm)
      cs_4.cmap.set_under('white',alpha=0.)
      cs_4.cmap.set_over('pink')
      cbar4 = fig.colorbar(cs_4,ax=ax4,orientation='horizontal',pad=0.01,shrink=0.85,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
      cbar4.set_label(units,fontsize=6,labelpad=0)
      cbar4.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
      cbar4.ax.xaxis.set_tick_params(pad=0)
      cbar4.ax.tick_params(labelsize=6)
      ax4.text(.5,1.02,'CCPA',horizontalalignment='center',fontsize=6,transform=ax4.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax4.text(.5,0.95,vtime_start+' 12z - '+vtime_end+' 12z',horizontalalignment='center',fontsize=6,transform=ax4.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax4.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.convert_and_save('compareqpf_'+dom+'_f'+fhour+'_'+pcpanl)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot 24-hr QPF with '+pcpanl+' for: '+dom) % t3)

######################################################

main()
