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
vtime_start = rrfs_plot_utils.ndate(ymdh_model,int(fhr-24))
vtime_start = str(vtime_start[0:8])
vtime_end = ymd

# Define the directory paths to the output files
STAGE_DIR = '/lfs/h2/emc/stmp/Benjamin.Blake/rrfs_verif'
PARM_DIR = os.path.join(os.environ['HOMEDIR'],'parm')
COMccpa = os.environ['COMccpa']
DCOMmrms = os.environ['DCOMmrms']

HRRR_DIR = os.path.join(os.environ['COMhrrr'],'hrrr.'+ymd_model)
NAM_DIR = os.path.join(os.environ['COMnam'],'nam.'+ymd_model)
RRFS_DIR = os.path.join(
    '/','lfs','h2','emc','ptmp','Benjamin.Blake','rrfs','na','prod',
    'rrfs.'+ymd_model, cyc_model
)
CCPA_DIR = os.path.join(STAGE_DIR)
MRMS_DIR = os.path.join(STAGE_DIR)

# Set up working directories
if not os.path.exists(os.path.join(CCPA_DIR, 'tmp')):
    if not os.path.exists(CCPA_DIR):
        os.makedirs(CCPA_DIR)
    os.makedirs(os.path.join(CCPA_DIR, 'tmp'))
    os.makedirs(os.path.join(CCPA_DIR, 'logs'))
if not os.path.exists(os.path.join(MRMS_DIR, 'tmp')):
    if not os.path.exists(MRMS_DIR):
        os.makedirs(MRMS_DIR)
    os.makedirs(os.path.join(MRMS_DIR, 'tmp'))
    os.makedirs(os.path.join(MRMS_DIR, 'logs'))

# Specify plotting domains
domains = ['conus','alaska','hawaii','puerto_rico','boston_nyc','central','colorado','la_vegas','mid_atlantic','north_central','northeast','northwest','ohio_valley','south_central','southeast','south_florida','sf_bay_area','seattle_portland','southwest','upper_midwest']

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
  if dom == 'alaska':
      dom1a_string = 'ak.'
      dom1a_string2 = dom
      dom1b_string = 'awp242'
      dom2_string = 'alaska'
      dom3_string = 'ak'
      dom5_string = 'ak'
  elif dom == 'puerto_rico':
      dom1a_string = ''
      dom1a_string2 = dom
      dom1b_string = 'awp237'
      dom2_string = 'prico'
      dom3_string = 'pr'
      dom5_string = 'pr'
  elif dom == 'hawaii':
      dom1a_string = ''
      dom1a_string2 = dom
      dom1b_string = 'awiphi'
      dom2_string = 'hawaii'
      dom3_string = 'hi'
      dom5_string = 'hi'
  else:
      dom1a_string = ''
      dom1a_string2 = 'conus'
      dom1b_string = 'awip12'
      dom2_string = 'conus'
      dom3_string = 'conus'
      dom5_string = 'conus'
  
  if dom in ['alaska', 'puerto_rico', 'hawaii']: 
    use_ccpa = False
  else:
    use_ccpa = True
  
  fhour_03 = str(fhr - 21).zfill(2)
  fhour_06 = str(fhr - 18).zfill(2)
  fhour_09 = str(fhr - 15).zfill(2)
  fhour_12 = str(fhr - 12).zfill(2)
  fhour_15 = str(fhr - 9).zfill(2)
  fhour_18 = str(fhr - 6).zfill(2)
  fhour_21 = str(fhr - 3).zfill(2)
 
  plot_nodata_text = [False, False, False, False, False, False]
  fname1a = HRRR_DIR+f'/{dom1a_string2}/hrrr.t'+cyc_model+'z.wrfprsf'+fhour+f'.{dom1a_string}grib2'
  fname1a_fm24 = HRRR_DIR+f'/{dom1a_string2}/hrrr.t'+cyc_model+'z.wrfprsf'+fhourm24+f'.{dom1a_string}grib2'
  fname1b_03 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour_03+'.tm00.grib2'
  fname1b_06 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour_06+'.tm00.grib2'
  fname1b_09 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour_09+'.tm00.grib2'
  fname1b_12 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour_12+'.tm00.grib2'
  fname1b_15 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour_15+'.tm00.grib2'
  fname1b_18 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour_18+'.tm00.grib2'
  fname1b_21 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour_21+'.tm00.grib2'
  fname1b_24 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour+'.tm00.grib2'
  fname2_03 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour_03+'.tm00.grib2'
  fname2_06 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour_06+'.tm00.grib2'
  fname2_09 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour_09+'.tm00.grib2'
  fname2_12 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour_12+'.tm00.grib2'
  fname2_15 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour_15+'.tm00.grib2'
  fname2_18 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour_18+'.tm00.grib2'
  fname2_21 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour_21+'.tm00.grib2'
  fname2_24 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour+'.tm00.grib2'
  if (dom3_string == 'hi') or (dom3_string == 'pr'):
    fname3 = RRFS_DIR+'/rrfs.t'+cyc_model+'z.prslev.2p5km.f0'+fhour+f'.{dom3_string}.grib2'
    fname3_fm24 = RRFS_DIR+'/rrfs.t'+cyc_model+'z.prslev.2p5km.f0'+fhourm24+f'.{dom3_string}.grib2'
  else:
    fname3 = RRFS_DIR+'/rrfs.t'+cyc_model+'z.prslev.3km.f0'+fhour+f'.{dom3_string}.grib2'
    fname3_fm24 = RRFS_DIR+'/rrfs.t'+cyc_model+'z.prslev.3km.f0'+fhourm24+f'.{dom3_string}.grib2'
  if use_ccpa:
      fname5 = CCPA_DIR+f'/ccpa.{ymd}/ccpa.t'+cyc+f'z.a24h.{dom5_string}.nc'
  else:
      fname5 = MRMS_DIR+f'/mrms.{ymd}/mrms.t'+cyc+f'z.a24h.{dom}.nc'
 
  for fname in [
          fname1a, fname1a_fm24
          ]:
      if not os.path.exists(fname):
          plot_nodata_text[0] = True
          break
  if not plot_nodata_text[0]:
      data1a = grib2io.open(fname1a)
      data1a_fm24 = grib2io.open(fname1a_fm24)
  if dom not in ['puerto_rico', 'hawaii']:
    for fname1b in [
            fname1b_03, fname1b_06, fname1b_09, fname1b_12, 
            fname1b_15, fname1b_18, fname1b_21, fname1b_24
            ]:
        if not os.path.exists(fname1b):
            plot_nodata_text[1] = True
            break
    if not plot_nodata_text[1]:
        data1b_03 = grib2io.open(fname1b_03)
        data1b_06 = grib2io.open(fname1b_06)
        data1b_09 = grib2io.open(fname1b_09)
        data1b_12 = grib2io.open(fname1b_12)
        data1b_15 = grib2io.open(fname1b_15)
        data1b_18 = grib2io.open(fname1b_18)
        data1b_21 = grib2io.open(fname1b_21)
        data1b_24 = grib2io.open(fname1b_24)
  else:
    for fname1b in [fname1b_12, fname1b_24]:
        if not os.path.exists(fname1b):
            plot_nodata_text[1] = True
            break
    if not plot_nodata_text[1]:
        data1b_12 = grib2io.open(fname1b_12)
        data1b_24 = grib2io.open(fname1b_24)
  for fname2 in [
          fname2_03, fname2_06, fname2_09, fname2_12, 
          fname2_15, fname2_18, fname2_21, fname2_24]:
      if not os.path.exists(fname2):
          plot_nodata_text[2] = True
          break
  if not plot_nodata_text[2]:
      data2_03 = grib2io.open(fname2_03)
      data2_06 = grib2io.open(fname2_06)
      data2_09 = grib2io.open(fname2_09)
      data2_12 = grib2io.open(fname2_12)
      data2_15 = grib2io.open(fname2_15)
      data2_18 = grib2io.open(fname2_18)
      data2_21 = grib2io.open(fname2_21)
      data2_24 = grib2io.open(fname2_24)
  for fname in [
          fname3, fname3_fm24
          ]:
      if not os.path.exists(fname):
          plot_nodata_text[3] = True
          print(
              f"WARNING: No RRFS file found for VDATE={cyc}Z {ymd} at F{fhour}: {fname}"
          )
          break
  if not plot_nodata_text[3]:
      print(f"File exists: {fname3}. Attempting to open!")
      try:
        data3 = grib2io.open(fname3)
      except KeyError:
        plot_nodata_text[3] = True
      print(f"File exists: {fname3_fm24}. Attempting to open!")
      try:
        data3_fm24 = grib2io.open(fname3_fm24)
      except KeyError:
        plot_nodata_text[3] = True
  if os.path.exists(fname5):
      data5 = Dataset(fname5,'r')
  else:
      plot_nodata_text[5] = True

# Get the lats and lons
  if not plot_nodata_text[0]:
    msg = data1a.select(shortName='HGT', level='500 mb')[0]  # msg is a Grib2Message object
    lat1a,lon1a,lat1a_shift,lon1a_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg)
  if not plot_nodata_text[1]:
    msg = data1b_24.select(shortName='HGT', level='surface')[0]  # msg is a Grib2Message object
    lat1b,lon1b,lat1b_shift,lon1b_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg)
  if not plot_nodata_text[2]:
    msg = data2_24.select(shortName='HGT', level='500 mb')[0]  # msg is a Grib2Message object
    lat2,lon2,lat2_shift,lon2_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg)
  if not plot_nodata_text[3]:
    msg = data3.select(shortName='HGT', level='500 mb')[0]  # msg is a Grib2Message object
    lat3,lon3,lat3_shift,lon3_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg)
# CCPA/MRMS
  if not plot_nodata_text[5]:
    if dom in ['alaska','hawaii','puerto_rico']:
        lat5 = data5.variables['lat'][:]
        lon5 = data5.variables['lon'][:]
        lon5, lat5 = np.meshgrid(lon5, lat5)
    else:
        lat5 = data5.variables['lat'][:,:]
        lon5 = data5.variables['lon'][:,:]

###################################################
# Read in all variables and calculate differences #
###################################################
  t1a = time.perf_counter()

  global qpf_1a,qpf_1b,qpf_2,qpf_3,qpf_5

# Total Precipitation
  if not plot_nodata_text[0]:
      qpf_1a_f = data1a.select(shortName='APCP',timeRangeOfStatisticalProcess=fhr)[0].data * 0.0393701
      qpf_1a_fm24 = data1a_fm24.select(shortName='APCP',timeRangeOfStatisticalProcess=(fhr-24))[0].data * 0.0393701
      qpf_1a = qpf_1a_f - qpf_1a_fm24
  if dom not in ['puerto_rico', 'hawaii']:
    if not plot_nodata_text[1]:
        qpf_1b_03 = data1b_03.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
        qpf_1b_06 = data1b_06.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
        qpf_1b_09 = data1b_09.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
        qpf_1b_12 = data1b_12.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
        qpf_1b_15 = data1b_15.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
        qpf_1b_18 = data1b_18.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
        qpf_1b_21 = data1b_21.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
        qpf_1b_24 = data1b_24.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
        qpf_1b = qpf_1b_03 + qpf_1b_06 + qpf_1b_09 + qpf_1b_12 + qpf_1b_15 + qpf_1b_18 + qpf_1b_21 + qpf_1b_24
  else:
    try:
      if not plot_nodata_text[1]:
        qpf_1b_12 = data1b_12.select(shortName='APCP',timeRangeOfStatisticalProcess=12)[0].data * 0.0393701
        qpf_1b_24 = data1b_24.select(shortName='APCP',timeRangeOfStatisticalProcess=12)[0].data * 0.0393701
      qpf_1b = qpf_1b_12 + qpf_1b_24
    except:
      plot_nodata_text[1] = True
  if not plot_nodata_text[2]:
      qpf_2_03 = data2_03.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2_06 = data2_06.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2_09 = data2_09.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2_12 = data2_12.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2_15 = data2_15.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2_18 = data2_18.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2_21 = data2_21.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2_24 = data2_24.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2 = qpf_2_03 + qpf_2_06 + qpf_2_09 + qpf_2_12 + qpf_2_15 + qpf_2_18 + qpf_2_21 + qpf_2_24
  if not plot_nodata_text[3]:
      qpf_3_f = data3.select(shortName='APCP')[1].data * 0.0393701
      if fhr > 24:
        qpf_3_fm24 = data3_fm24.select(shortName='APCP')[1].data * 0.0393701
        qpf_3 = qpf_3_f - qpf_3_fm24
      else:
        qpf_3 = qpf_3_f
  if not plot_nodata_text[5]:
    # CCPA/MRMS
    if dom in ['alaska','hawaii','puerto_rico']:
      qpf_5 = data5.variables['MultiSensor_QPE_24H_Pass2_Z0'][:,:] * 0.0393701
    else:
      qpf_5 = data5.variables['APCP_24'][:,:] * 0.0393701

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
          use_qpf1 = qpf_1a
          use_lon1_shift = lon1a_shift
          use_lat1_shift = lat1a_shift
          if dom in ['puerto_rico', 'hawaii']:
              plot_nodata_text[0] = True
        mod1_name = 'HRRR'
      elif use_mod1 == 'b':
        if not plot_nodata_text[1]:
          use_qpf1 = qpf_1b
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
      # Plot 24-hr QPF
    #################################
      datasets = ['CCPA-MRMS']
      for pcpanl in datasets:

        t1 = time.perf_counter()
        print((
            'Working on 24-hr QPF for '+dom
            +(' with HRRR' if use_mod1=='a' else (
                ' with NAM' if use_mod1=='b' 
                else ' with ~mystery model~'
            ))
        ))

        units = 'Precipitation (inches)'
        clevs = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,7,10,15,20]
        colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','deepskyblue','cyan','mediumpurple','mediumorchid','darkmagenta','darkred','crimson','orangered','darkorange','goldenrod','gold','yellow']  
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
       
        ax1.text(.5,1.02,mod1_name,horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax1.text(.5,0.95,itime+' '+cyc_model+'z cycle (f'+fhourm24+'-f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        if use_mod1 == 'a' and plot_nodata_text[0]:
          if fhr > 48 and dom not in ['hawaii', 'puerto_rico']:
              not_avail_text1a = "Not available\nat this forecast hour"
          else:
              not_avail_text1a = "Not Available"
          cs_1 = ax1.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=0.01,norm=norm)
          ax1.text(
              0.5, 0.5, not_avail_text1a, transform=ax1.transAxes, 
              fontsize=12, color='black',
              horizontalalignment='center', bbox=dict(
                  facecolor='white', alpha=0.8, 
                  boxstyle='round,pad=0.3'
              )
          )
        elif use_mod1 == 'b' and plot_nodata_text[1]:
          cs_1 = ax1.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=0.01,norm=norm)
          ax1.text(
              0.5, 0.5, 'Not Available', transform=ax1.transAxes, 
              fontsize=12, color='black',
              horizontalalignment='center', bbox=dict(
                  facecolor='white', alpha=0.8, 
                  boxstyle='round,pad=0.3'
              )
          )
        else:
            cs_1 = ax1.pcolormesh(use_lon1_shift,use_lat1_shift,use_qpf1,transform=transform,cmap=cm,vmin=0.01,norm=norm)
        cs_1.cmap.set_under('white',alpha=0.)
        cs_1.cmap.set_over('pink')
        cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.85,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar1.set_label(units,fontsize=6,labelpad=0)
        cbar1.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar1.ax.xaxis.set_tick_params(pad=0)
        cbar1.ax.tick_params(labelsize=6)
        ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

        ax2.text(.5,1.02,'NAM Nest',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax2.text(.5,0.95,itime+' '+cyc_model+'z cycle (f'+fhourm24+'-f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        if plot_nodata_text[2]:
          if fhr > 60:
              not_avail_text2 = "Not available\nat this forecast hour"
          else:
              not_avail_text2 = "Not Available"
          cs_2 = ax2.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=0.01,norm=norm)
          ax2.text(
              0.5, 0.5, not_avail_text2, transform=ax2.transAxes, 
              fontsize=12, color='black',
              horizontalalignment='center', bbox=dict(
                  facecolor='white', alpha=0.8, 
                  boxstyle='round,pad=0.3'
              )
          )
        else:
            cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,qpf_2,transform=transform,cmap=cm,vmin=0.01,norm=norm)
        cs_2.cmap.set_under('white',alpha=0.)
        cs_2.cmap.set_over('pink')
        cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.85,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar2.set_label(units,fontsize=6,labelpad=0)
        cbar2.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar2.ax.xaxis.set_tick_params(pad=0)
        cbar2.ax.tick_params(labelsize=6)
        ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

        ax3.text(.5,1.02,'RRFS',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax3.text(.5,0.95,itime+' '+cyc_model+'z cycle (f'+fhourm24+'-f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax3.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        if plot_nodata_text[3]:
          cs_3 = ax3.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=0.01,norm=norm)
          ax3.text(
              0.5, 0.5, 'Not Available', transform=ax3.transAxes, 
              fontsize=12, color='black',
              horizontalalignment='center', bbox=dict(
                  facecolor='white', alpha=0.8, 
                  boxstyle='round,pad=0.3'
              )
          )
        else:
            cs_3 = ax3.pcolormesh(lon3_shift,lat3_shift,qpf_3,transform=transform,cmap=cm,vmin=0.01,norm=norm)
        cs_3.cmap.set_under('white',alpha=0.)
        cs_3.cmap.set_over('pink')
        cbar3 = fig.colorbar(cs_3,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.85,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar3.set_label(units,fontsize=6,labelpad=0)
        cbar3.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar3.ax.xaxis.set_tick_params(pad=0)
        cbar3.ax.tick_params(labelsize=6)
        ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

        if dom in ['alaska', 'puerto_rico','hawaii']:
          ax4.text(.5,1.02,'MRMS',horizontalalignment='center',fontsize=6,transform=ax4.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        else:
          ax4.text(.5,1.02,'CCPA',horizontalalignment='center',fontsize=6,transform=ax4.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax4.text(.5,0.95,vtime_start+f' {cyc}z - '+vtime_end+f' {cyc}z',horizontalalignment='center',fontsize=6,transform=ax4.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        if plot_nodata_text[5]:
          cs_4 = ax4.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=0.01,norm=norm)
          ax4.text(
              0.5, 0.5, 'Not Available', transform=ax4.transAxes, 
              fontsize=12, color='black',
              horizontalalignment='center', bbox=dict(
                  facecolor='white', alpha=0.8, 
                  boxstyle='round,pad=0.3'
              )
          )
        else:
            cs_4 = ax4.pcolormesh(lon5,lat5,qpf_5,transform=transform,cmap=cm,vmin=0.01,norm=norm)
        cs_4.cmap.set_under('white',alpha=0.)
        cs_4.cmap.set_over('pink')
        cbar4 = fig.colorbar(cs_4,ax=ax4,orientation='horizontal',pad=0.01,shrink=0.85,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar4.set_label(units,fontsize=6,labelpad=0)
        cbar4.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar4.ax.xaxis.set_tick_params(pad=0)
        cbar4.ax.tick_params(labelsize=6)
        ax4.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)


        rrfs_plot_utils.convert_and_save(f'compareqpf{use_mod1}_'+dom+'_f'+fhour+'_'+pcpanl)
        t2 = time.perf_counter()
        t3 = round(t2-t1, 3)
        print(('%.3f seconds to plot 24-hr QPF with '+pcpanl+' for: '+dom+f'model{use_mod1}') % t3)

######################################################

main()
