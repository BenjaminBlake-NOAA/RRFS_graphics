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
from datetime import datetime
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
vtime_end = ymd

# Define the directory paths to the output files
HRRR_DIR = os.path.join(os.environ['COMhrrr'],'hrrr.'+ymd_model)
NAM_DIR = os.path.join(os.environ['COMnam'],'nam.'+ymd_model)
RRFS_DIR = os.path.join(
    '/','lfs','h2','emc','ptmp',os.environ['USER'],'rrfs','na','prod',
    'rrfs.'+ymd_model, cyc_model
)
URMA_DIR = os.environ['COMurma']
RTMA_DIR = os.environ['COMrtma']

# Specify plotting domains
domains = ['conus','alaska','hawaii','puerto_rico','boston_nyc','central','colorado','la_vegas','mid_atlantic','north_central','northeast','northwest','ohio_valley','south_central','southeast','south_florida','sf_bay_area','seattle_portland','southwest','upper_midwest']

# Paths to image files
user = str(sys.argv[4])
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
  if dom == 'alaska':
      dom1a_string = 'ak.'
      dom1a_string2 = dom
      dom1b_string = 'awp242'
      dom2_string = 'alaska'
      dom3_string = 'ak'
      dom3_gridspacing = '3km'
      dom4a_string = 'akurma'
      dom4b_string = 'akrtma'
  elif dom == 'puerto_rico':
      dom1a_string = ''
      dom1a_string2 = dom
      dom1b_string = 'awp237'
      dom2_string = 'prico'
      dom3_string = 'pr'
      dom3_gridspacing = '2p5km'
      dom4a_string = 'prurma'
      dom4b_string = 'prrtma'
  elif dom == 'hawaii':
      dom1a_string = ''
      dom1a_string2 = dom
      dom1b_string = 'awiphi'
      dom2_string = 'hawaii'
      dom3_string = 'hi'
      dom3_gridspacing = '2p5km'
      dom4a_string = 'hiurma'
      dom4b_string = 'hirtma'
  else:
      dom1a_string = ''
      dom1a_string2 = 'conus'
      dom1b_string = 'awip12'
      dom2_string = 'conus'
      dom3_string = 'conus'
      dom3_gridspacing = '3km'
      dom4a_string = 'urma2p5'
      dom4b_string = 'rtma2p5'

  plot_nodata_text = [False, False, False, False, False, False, False]
  
  fname1a = HRRR_DIR+f'/{dom1a_string2}/hrrr.t'+cyc_model+'z.wrfprsf'+fhour+f'.{dom1a_string}grib2'
  fname1b = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour+'.tm00.grib2'
  fname2 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour+'.tm00.grib2'
  fname3 = RRFS_DIR+'/rrfs.t'+cyc_model+f'z.prslev.{dom3_gridspacing}.f0'+fhour+f'.{dom3_string}.grib2'
  if dom in ['puerto_rico']:
      fname4a = URMA_DIR+f'/prurma.{ymd}/{dom4a_string}.t{cyc}z.2dvaranl_ndfd.grb2_allflds'
      fname4b = RTMA_DIR+f'/prrtma.{ymd}/{dom4b_string}.t{cyc}z.2dvaranl_ndfd.grb2_allflds'
  elif dom in ['hawaii']:
      fname4a = URMA_DIR+f'/hiurma.{ymd}/{dom4a_string}.t{cyc}z.2dvaranl_ndfd.grb2_allflds'
      fname4b = RTMA_DIR+f'/hirtma.{ymd}/{dom4b_string}.t{cyc}z.2dvaranl_ndfd.grb2_allflds'
  elif dom in ['alaska']:
      fname4a = URMA_DIR+f'/akurma.{ymd}/{dom4a_string}.t{cyc}z.2dvaranl_ndfd.grb2'
      fname4b = RTMA_DIR+f'/akrtma.{ymd}/{dom4b_string}.t{cyc}z.2dvaranl_ndfd.grb2'
  else:
      fname4a = URMA_DIR+f'/urma2p5.{ymd}/{dom4a_string}.t{cyc}z.2dvaranl_ndfd.grb2_wexp'
      fname4b = RTMA_DIR+f'/rtma2p5.{ymd}/{dom4b_string}.t{cyc}z.2dvaranl_ndfd.grb2_wexp'

  if os.path.exists(fname1a):
      data1a = grib2io.open(fname1a)
  else:
      plot_nodata_text[0] = True
  if os.path.exists(fname1b):
      data1b = grib2io.open(fname1b)
  else:
      plot_nodata_text[1] = True
  if os.path.exists(fname2):
      data2 = grib2io.open(fname2)
  else:
      plot_nodata_text[2] = True
  if os.path.exists(fname3):
      data3 = grib2io.open(fname3)
  else:
      plot_nodata_text[3] = True
  if os.path.exists(fname4a):
      data4a = grib2io.open(fname4a)
  else:
      print(fname4a)
      plot_nodata_text[4] = True
  if os.path.exists(fname4b):
      data4b = grib2io.open(fname4b)
  else:
      print(fname4b)
      plot_nodata_text[5] = True

# Get the lats and lons
  if not plot_nodata_text[0]:
      msg = data1a.select(shortName='HGT', level='500 mb')[0]
      lat1a,lon1a,lat1a_shift,lon1a_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg)
  if not plot_nodata_text[1]:
      msg = data1b.select(shortName='HGT', level='surface')[0]
      lat1b,lon1b,lat1b_shift,lon1b_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg)
  if not plot_nodata_text[2]:
      msg = data2.select(shortName='HGT', level='500 mb')[0]
      lat2,lon2,lat2_shift,lon2_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg)
  if not plot_nodata_text[3]:
      msg = data3.select(shortName='HGT', level='500 mb')[0]
      lat3,lon3,lat3_shift,lon3_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg)
  if not plot_nodata_text[4]:
      # URMA
      msg = data4a.select(shortName='HGT', level='surface')[0]
      lat4a,lon4a,lat4a_shift,lon4a_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg)
  if not plot_nodata_text[5]:
      # RTMA
      msg = data4b.select(shortName='HGT', level='surface')[0]
      lat4b,lon4b,lat4b_shift,lon4b_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg)

###################################################
# Read in all variables and calculate differences #
###################################################
  t1a = time.perf_counter()

  global wind_1a,wind_1b,wind_2,wind_3,wind_4a,wind_4b

# 10-m Winds
  if not plot_nodata_text[0]:
      uwind_1a = data1a.select(shortName='UGRD',level='10 m above ground')[0].data * 1.94384
      vwind_1a = data1a.select(shortName='VGRD',level='10 m above ground')[0].data * 1.94384
      wind_1a = np.sqrt(uwind_1a**2 + vwind_1a**2)
  if not plot_nodata_text[1]:
      uwind_1b = data1b.select(shortName='UGRD',level='10 m above ground')[0].data * 1.94384
      vwind_1b = data1b.select(shortName='VGRD',level='10 m above ground')[0].data * 1.94384
      wind_1b = np.sqrt(uwind_1b**2 + vwind_1b**2)
  if not plot_nodata_text[2]:
      uwind_2 = data2.select(shortName='UGRD',level='10 m above ground')[0].data * 1.94384
      vwind_2 = data2.select(shortName='VGRD',level='10 m above ground')[0].data * 1.94384
      wind_2 = np.sqrt(uwind_2**2 + vwind_2**2)
  if not plot_nodata_text[3]:
      uwind_3 = data3.select(shortName='UGRD',level='10 m above ground')[0].data * 1.94384
      vwind_3 = data3.select(shortName='VGRD',level='10 m above ground')[0].data * 1.94384
      wind_3 = np.sqrt(uwind_3**2 + vwind_3**2)
  if not plot_nodata_text[4]:
    # URMA
      uwind_4a = data4a.select(shortName='UGRD',level='10 m above ground')[0].data * 1.94384
      vwind_4a = data4a.select(shortName='VGRD',level='10 m above ground')[0].data * 1.94384
      wind_4a = np.sqrt(uwind_4a**2 + vwind_4a**2)
  if not plot_nodata_text[5]:
    # RTMA
      uwind_4b = data4b.select(shortName='UGRD',level='10 m above ground')[0].data * 1.94384
      vwind_4b = data4b.select(shortName='VGRD',level='10 m above ground')[0].data * 1.94384
      wind_4b = np.sqrt(uwind_4b**2 + vwind_4b**2)


  t2a = time.perf_counter()
  t3a = round(t2a-t1a, 3)
  print(("%.3f seconds to read all messages") % t3a)

#######################################
#    SET UP FIGURE FOR EACH DOMAIN    #
#######################################

# Call the domain_latlons_proj function from rrfs_plot_utils
  xextent,yextent,offset,extent,myproj = rrfs_plot_utils.domain_latlons_proj(dom)

#######################################
#  RUN 4-PANEL FOR MODELS 1A AND 1B   #
#######################################

  for use_mod1 in ['a','b']:
    # Select data based on which model 1 is used
      if use_mod1 == 'a':
          if not plot_nodata_text[0]:
              use_wind_1 = wind_1a
              use_uwind_1 = uwind_1a
              use_vwind_1 = vwind_1a
              use_lon1_shift = lon1a_shift
              use_lat1_shift = lat1a_shift
              if dom in ['puerto_rico', 'hawaii']:
                  plot_nodata_text[0] = True
          mod1_name = 'HRRR'
      elif use_mod1 == 'b':
          if not plot_nodata_text[1]:
              use_wind_1 = wind_1b
              use_uwind_1 = uwind_1b
              use_vwind_1 = vwind_1b
              use_lon1_shift = lon1b_shift
              use_lat1_shift = lat1b_shift
          mod1_name = 'NAM'
      else:
          raise ValueError(
              f'Unrecognized option for use_mod1: {use_mod1}'
          )    

    # Create figure and axes instances
      fig = plt.figure(figsize=(8,8))           
      ws, hs = rrfs_plot_utils.get_panel_spacing(dom)
      gs = GridSpec(8,8,wspace=ws,hspace=hs)

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
      # Plot 10-m Winds
    #################################
#      datasets = ['URMA','RTMA']
      datasets = ['URMA']
      for anl in datasets:

        t1 = time.perf_counter()
        print((
            'Working on 10-m Winds for '+dom
            +(' with HRRR' if use_mod1=='a' else (
                ' with Parent NAM' if use_mod1=='b' 
                else ' with ~mystery model~'
            ))
        ))

        # Wind barb density settings
        # Wind barb density settings
        if dom == 'conus':
            skip = 100
            skip_1b = 25
            skip_4 = 100
        elif dom in ['southeast', 'alaska']:
            skip = 40
            skip_1b = 10
            skip_4 = 20
        elif dom in ['hawaii']: 
            skip = 20
            skip_1b = 2
            skip_4 = 20
        elif dom in ['puerto_rico']:
            skip = 20
            skip_1b = 2
            skip_4 = 60
        elif dom in ['colorado', 'la_vegas', 'mid_atlantic', 'south_florida']:
            skip = 18
            skip_1b = 4
            skip_4 = 18
        elif dom in ['boston_nyc']:
            skip = 15
            skip_1b = 4
            skip_4 = 15
        elif dom in ['seattle_portland']:
            skip = 13
            skip_1b = 3
            skip_4 = 13
        elif dom in ['sf_bay_area']:
            skip = 4
            skip_1b = 1
            skip_4 = 4
        else:
            skip = 30
            skip_1b = 7
            skip_4 = 30
        barblength = 3.5

        if anl == 'URMA':

            units = 'kts'
            clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
            colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
            cm = matplotlib.colors.ListedColormap(colorlist)
            norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
           
            ax1.text(.5,1.02,mod1_name,horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
            ax1.text(.5,0.95,itime+' '+cyc_model+'z cycle (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
            if use_mod1 == 'a' and plot_nodata_text[0]:
              cs_1 = ax1.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=5,norm=norm)
              ax1.text(
                  0.5, 0.5, 'Not Available', transform=ax1.transAxes, 
                  fontsize=12, color='black', 
                  horizontalalignment='center', bbox=dict(
                      facecolor='white', alpha=0.8, 
                      boxstyle='round,pad=0.3'
                  )
              )
            elif use_mod1 == 'b' and plot_nodata_text[1]:
              cs_1 = ax1.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=5,norm=norm)
              ax1.text(
                  0.5, 0.5, 'Not Available', transform=ax1.transAxes, 
                  fontsize=12, color='black', 
                  horizontalalignment='center', bbox=dict(
                      facecolor='white', alpha=0.8, 
                      boxstyle='round,pad=0.3'
                  )
              )
            else:
              cs_1 = ax1.pcolormesh(use_lon1_shift,use_lat1_shift,use_wind_1,transform=transform,cmap=cm,norm=norm)
              if use_mod1 == 'a':
                ax1.barbs(use_lon1_shift[::skip,::skip], use_lat1_shift[::skip,::skip], use_uwind_1[::skip,::skip], use_vwind_1[::skip,::skip], length=barblength, linewidth=0.5, color='black', transform=transform)
              else:
                ax1.barbs(use_lon1_shift[::skip_1b,::skip_1b], use_lat1_shift[::skip_1b,::skip_1b], use_uwind_1[::skip_1b,::skip_1b], use_vwind_1[::skip_1b,::skip_1b], length=barblength, linewidth=0.5, color='black', transform=transform)
            cs_1.cmap.set_under('white')
            cs_1.cmap.set_over('black')
            cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='max')
            cbar1.set_label(units,fontsize=6,labelpad=0)
            cbar1.ax.xaxis.set_tick_params(pad=0)
            cbar1.ax.tick_params(labelsize=6)
            ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

            ax2.text(.5,1.02,'NAM Nest',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
            ax2.text(.5,0.95,itime+' '+cyc_model+'z cycle (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
            if plot_nodata_text[2]:
              cs_2 = ax2.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=5,norm=norm)
              ax2.text(
                  0.5, 0.5, 'Not Available', transform=ax2.transAxes, 
                  fontsize=12, color='black', 
                  horizontalalignment='center', bbox=dict(
                      facecolor='white', alpha=0.8, 
                      boxstyle='round,pad=0.3'
                  )
              )
            else:
              cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,wind_2,transform=transform,cmap=cm,vmin=5,norm=norm)
              ax2.barbs(lon2_shift[::skip,::skip], lat2_shift[::skip,::skip], uwind_2[::skip,::skip], vwind_2[::skip,::skip], length=barblength, linewidth=0.5, color='black', transform=transform)
            cs_2.cmap.set_under('white')
            cs_2.cmap.set_over('black')
            cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,extend='max')
            cbar2.set_label(units,fontsize=6,labelpad=0)
            cbar2.ax.xaxis.set_tick_params(pad=0)
            cbar2.ax.tick_params(labelsize=6)
            ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

            ax3.text(.5,1.02,'RRFS',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
            ax3.text(.5,0.95,itime+' '+cyc_model+'z cycle (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
            ax3.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
            if plot_nodata_text[3]:
              cs_3 = ax3.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=5,norm=norm)
              ax3.text(
                  0.5, 0.5, 'Not Available', transform=ax3.transAxes, 
                  fontsize=12, color='black', 
                  horizontalalignment='center', bbox=dict(
                      facecolor='white', alpha=0.8, 
                      boxstyle='round,pad=0.3'
                  )
              )
            else:
              cs_3 = ax3.pcolormesh(lon3_shift,lat3_shift,wind_3,transform=transform,cmap=cm,vmin=5,norm=norm)
              ax3.barbs(lon3_shift[::skip,::skip], lat3_shift[::skip,::skip], uwind_3[::skip,::skip], vwind_3[::skip,::skip], length=barblength, linewidth=0.5, color='black', transform=transform)
            cs_3.cmap.set_under('white')
            cs_3.cmap.set_over('black')
            cbar3 = fig.colorbar(cs_3,ax=ax3,orientation='horizontal',pad=0.01,shrink=1.0,extend='max')
            cbar3.set_label(units,fontsize=6,labelpad=0)
            cbar3.ax.xaxis.set_tick_params(pad=0)
            cbar3.ax.tick_params(labelsize=6)
            ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

            ax4.text(.5,1.02,'URMA',horizontalalignment='center',fontsize=6,transform=ax4.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
            ax4.text(.5,0.95,vtime_end+f' {cyc}z',horizontalalignment='center',fontsize=6,transform=ax4.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
            if plot_nodata_text[4]:
              cs_4 = ax4.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=5,norm=norm)
              ax4.text(
                  0.5, 0.5, 'Not Available', transform=ax4.transAxes, 
                  fontsize=12, color='black', 
                  horizontalalignment='center', bbox=dict(
                      facecolor='white', alpha=0.8, 
                      boxstyle='round,pad=0.3'
                  )
              )
            else:
              cs_4 = ax4.pcolormesh(lon4a_shift,lat4a_shift,wind_4a,transform=transform,cmap=cm,vmin=5,norm=norm)
              ax4.barbs(lon4a_shift[::skip_4,::skip_4], lat4a_shift[::skip_4,::skip_4], uwind_4a[::skip_4,::skip_4], vwind_4a[::skip_4,::skip_4], length=barblength, linewidth=0.5, color='black', transform=transform)
            cs_4.cmap.set_under('white')
            cs_4.cmap.set_over('black')
            cbar4 = fig.colorbar(cs_4,ax=ax4,orientation='horizontal',pad=0.01,shrink=1.0,extend='max')
            cbar4.set_label(units,fontsize=6,labelpad=0)
            cbar4.ax.xaxis.set_tick_params(pad=0)
            cbar4.ax.tick_params(labelsize=6)
            ax4.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
        
        elif anl == 'RTMA':

            # Clear off old plotables but keep all the map info
            cbar4.remove()
            rrfs_plot_utils.clear_plotables(ax4,keep_ax_lst_4,fig)

            ax4.text(.5,1.02,'RTMA',horizontalalignment='center',fontsize=6,transform=ax4.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
            ax4.text(.5,0.95,vtime_end+f' {cyc}z',horizontalalignment='center',fontsize=6,transform=ax4.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
            if plot_nodata_text[5]:
              cs_4 = ax4.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=5,norm=norm)
              ax4.text(
                  0.5, 0.5, 'Not Available', transform=ax4.transAxes, 
                  fontsize=12, color='black', 
                  horizontalalignment='center', bbox=dict(
                      facecolor='white', alpha=0.8, 
                      boxstyle='round,pad=0.3'
                  )
              )
            else:
              cs_4 = ax4.pcolormesh(lon4b_shift,lat4b_shift,wind_4b,transform=transform,cmap=cm,vmin=5,norm=norm)
              ax4.barbs(lon4b_shift[::skip_4,::skip_4], lat4b_shift[::skip_4,::skip_4], uwind_4b[::skip_4,::skip_4], vwind_4b[::skip_4,::skip_4], length=barblength, linewidth=0.5, color='black', transform=transform)
            cs_4.cmap.set_under('white')
            cs_4.cmap.set_over('black')
            cbar4 = fig.colorbar(cs_4,ax=ax4,orientation='horizontal',pad=0.01,shrink=1.0,extend='max')
            cbar4.set_label(units,fontsize=6,labelpad=0)
            cbar4.ax.xaxis.set_tick_params(pad=0)
            cbar4.ax.tick_params(labelsize=6)
            ax4.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
        
        rrfs_plot_utils.convert_and_save(f'comparewind10m{use_mod1}_'+dom+'_f'+fhour+'_'+anl)
        t2 = time.perf_counter()
        t3 = round(t2-t1, 3)
        print(('%.3f seconds to plot 10-m Winds with '+anl+' for: '+dom+f'model{use_mod1}') % t3)

######################################################

main()
