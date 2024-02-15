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

ymdhm1 = rrfs_plot_utils.ndate(ymdh,-6)
ymdm1 = ymdhm1[0:8]
if cyc == '00':
  cycm1 = '18'
elif cyc == '06':
  cycm1 = '00'
elif cyc == '12':
  cycm1 = '06'
elif cyc == '18':
  cycm1 = '12'

fhr = int(sys.argv[2])
fhour = str(fhr).zfill(2)
print('fhour '+fhour)

# Forecast valid date/time
itime = ymdh
vtime = rrfs_plot_utils.ndate(itime,int(fhr))
itimem1 = ymdhm1
vtimem1 = rrfs_plot_utils.ndate(itimem1,int(fhr))

# Define the directory paths to the input files, read in ensemble member #
member = str(sys.argv[3])
timelag = str(sys.argv[4])	# yes or no for time-lagged member

# Specify plotting domains
domains = ['conus','boston_nyc','central','colorado','la_vegas','mid_atlantic','north_central','northeast','northwest','ohio_valley','south_central','southeast','south_florida','sf_bay_area','seattle_portland','southwest','upper_midwest']

# Check to see if the member does not exist
# Create placeholder images for relevant forecast hours
if ((member == 'HRRR') and (fhr > 48)) or ((timelag == 'yes') and (fhr > 60)):
  vars = ['slp','2mt','2mdew','10mwind','mucape','850t','500','250wind','refc','vis','zceil','snow','asnow','uh25','maxuvv','qpf']
  for var in vars:
    for dom in domains:
      filename = str(member+'_'+timelag+'_'+var+'_'+dom+'_f'+fhour)
      os.system('cp /lfs/h2/emc/lam/noscrub/Benjamin.Blake/rrfs_graphics/NoOutput.gif '+filename+'.gif') 
  sys.exit()

# Define the input files
if member == 'HRRR':
  if timelag == 'yes':
    DATA_DIR = '/lfs/h1/ops/prod/com/hrrr/v4.1/hrrr.'+ymdm1+'/conus'
    data1 = grib2io.open(DATA_DIR+'/hrrr.t'+cycm1+'z.wrfprsf'+fhour+'.grib2')
    memstr = 'HRRR TL'
  else:
    DATA_DIR = '/lfs/h1/ops/prod/com/hrrr/v4.1/hrrr.'+ymd+'/conus'
    data1 = grib2io.open(DATA_DIR+'/hrrr.t'+cyc+'z.wrfprsf'+fhour+'.grib2')
    memstr = 'HRRR'
# RRFS deterministic forecast
elif member == 'Control':
  if timelag == 'yes':
    DATA_DIR = '/lfs/h2/emc/ptmp/emc.lam/rrfs/na/prod/rrfs.'+ymdm1+'/'+cycm1
    data1 = grib2io.open(DATA_DIR+'/rrfs.t'+cycm1+'z.prslev.f0'+fhour+'.conus_3km.grib2')
    memstr = 'Control TL'
  else:
    DATA_DIR = '/lfs/h2/emc/ptmp/emc.lam/rrfs/na/prod/rrfs.'+ymd+'/'+cyc
    data1 = grib2io.open(DATA_DIR+'/rrfs.t'+cyc+'z.prslev.f0'+fhour+'.conus_3km.grib2')
    memstr = 'Control'
# RRFS ensemble member forecasts
else:
  if timelag == 'yes':
    DATA_DIR = '/lfs/h2/emc/ptmp/emc.lam/rrfs/na/prod/refs.'+ymdm1+'/'+cycm1+'/mem000'+member
    data1 = grib2io.open(DATA_DIR+'/rrfs.t'+cycm1+'z.prslev.f0'+fhour+'.conus_3km.grib2')
    memstr = 'Member '+member+' TL'
  else:
    DATA_DIR = '/lfs/h2/emc/ptmp/emc.lam/rrfs/na/prod/refs.'+ymd+'/'+cyc+'/mem000'+member
    data1 = grib2io.open(DATA_DIR+'/rrfs.t'+cyc+'z.prslev.f0'+fhour+'.conus_3km.grib2')
    memstr = 'Member '+member

# Get the lats and lons
msg = data1.select(shortName='HGT', level='500 mb')[0]	# msg is a Grib2Message object
lat,lon,lat_shift,lon_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg)

# Paths to image files
im = image.imread('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/rrfs_graphics/noaa.png')

# colors for difference plots, only need to define once
difcolors = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','white','white','#EEEE00','#EEC900','darkorange','orangered','red']

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


###################################################
# Read in all variables                           #
###################################################
t1a = time.perf_counter()

# Sea level pressure
if member == 'HRRR':
  slp_1 = data1.select(shortName='MSLMA',level='mean sea level')[0].data * 0.01
else:
  slp_1 = data1.select(shortName='MSLET',level='mean sea level')[0].data * 0.01

# 2-m temperature
tmp2m_1 = data1.select(shortName='TMP',level='2 m above ground')[0].data
tmp2m_1 = (tmp2m_1 - 273.15)*1.8 + 32.0

# 2-m dew point temperature
dew2m_1 = data1.select(shortName='DPT',level='2 m above ground')[0].data
dew2m_1 = (dew2m_1 - 273.15)*1.8 + 32.0

# 10-m wind speed
uwind_1 = data1.select(shortName='UGRD',level='10 m above ground')[0].data * 1.94384
vwind_1 = data1.select(shortName='VGRD',level='10 m above ground')[0].data * 1.94384
wspd10m_1 = np.sqrt(uwind_1**2 + vwind_1**2)

# Most unstable CAPE
mucape_1 = data1.select(shortName='CAPE',level='180-0 mb above ground')[0].data

# 850-mb equivalent potential temperature
t850_1 = data1.select(shortName='TMP',level='850 mb')[0].data
dpt850_1 = data1.select(shortName='DPT',level='850 mb')[0].data
q850_1 = data1.select(shortName='SPFH',level='850 mb')[0].data
tlcl_1 = 56.0 + (1.0/((1.0/(dpt850_1-56.0)) + 0.00125*np.log(t850_1/dpt850_1)))
thetae_1 = t850_1*((1000.0/850.0)**(0.2854*(1.0-(0.28*q850_1))))*np.exp(((3376.0/tlcl_1)-2.54)*q850_1*(1.0+(0.81*q850_1)))

# 850-mb winds
u850_1 = data1.select(shortName='UGRD',level='850 mb')[0].data * 1.94384
v850_1 = data1.select(shortName='VGRD',level='850 mb')[0].data * 1.94384

# 500 mb height, wind, vorticity
z500_1 = data1.select(shortName='HGT',level='500 mb')[0].data * 0.1
z500_1 = ndimage.filters.gaussian_filter(z500_1, 6.89)
vort500_1 = data1.select(shortName='ABSV',level='500 mb')[0].data * 100000
vort500_1 = ndimage.filters.gaussian_filter(vort500_1,1.7225)
vort500_1[vort500_1 > 1000] = 0		# Mask out undefined values on domain edge
u500_1 = data1.select(shortName='UGRD',level='500 mb')[0].data * 1.94384
v500_1 = data1.select(shortName='VGRD',level='500 mb')[0].data * 1.94384

# 250 mb winds
u250_1 = data1.select(shortName='UGRD',level='250 mb')[0].data * 1.94384
v250_1 = data1.select(shortName='VGRD',level='250 mb')[0].data * 1.94384
wspd250_1 = np.sqrt(u250_1**2 + v250_1**2)

# Composite reflectivity
refc_1 = data1.select(shortName='REFC')[0].data

# Visibility
vis_1 = data1.select(shortName='VIS',level='surface')[0].data * 0.000621371

# Cloud Ceiling Height
zceil_1 = data1.select(shortName='HGT',level='cloud ceiling')[0].data * (3.28084/1000)

# Snow depth
snow_1 = data1.select(shortName='SNOD')[0].data * 39.3701

# Snowfall
asnow_1 = data1.select(shortName='ASNOW')[0].data * 39.3701

if (fhr > 0):
# Max/Min Hourly 2-5 km Updraft Helicity
  maxuh25_1 = data1.select(shortName='MXUPHL',level='5000-2000 m above ground')[0].data
  minuh25_1 = data1.select(shortName='MNUPHL',level='5000-2000 m above ground')[0].data
  maxuh25_1[maxuh25_1 < 10] = 0
  minuh25_1[minuh25_1 > -10] = 0
  uh25_1 = maxuh25_1 + minuh25_1

# Max Hourly Updraft Speed
  maxuvv_1 = data1.select(shortName='MAXUVV')[0].data

# Total precipitation
  if member == 'HRRR':
    qpf_1 = data1.select(shortName='APCP',timeRangeOfStatisticalProcess=fhr)[0].data * 0.0393701
  else:
    qpf_1 = data1.select(shortName='APCP')[1].data * 0.0393701

  # For 6-hr time lagged members, subtract first 6 hours of precip
  if timelag == 'yes':
    fhrm6 = fhr - 6
    fhourm6 = str(fhrm6).zfill(2)
    if member == 'HRRR':
      data2 = grib2io.open(DATA_DIR+'/hrrr.t'+cycm1+'z.wrfprsf06.grib2')
      qpf_2 = data2.select(shortName='APCP',timeRangeOfStatisticalProcess=6)[0].data * 0.0393701
      qpf_1 = qpf_1 - qpf_2
    else:
      data2 = grib2io.open(DATA_DIR+'/rrfs.t'+cycm1+'z.prslev.f006.conus_3km.grib2')
      qpf_2 = data2.select(shortName='APCP')[1].data * 0.0393701
      qpf_1 = qpf_1 - qpf_2


t2a = time.perf_counter()
t3a = round(t2a-t1a, 3)
print(("%.3f seconds to read all messages") % t3a)

#-------------------------------------------------------#

def main():

  # Number of processes must coincide with the number of domains to plot
  pool = MyPool(len(domains))
  pool.map(create_figure,domains)

#-------------------------------------------------------#

def create_figure(domain):

  global dom
  dom = domain
  print(('Working on '+dom))

  global fig,axes,ax1,keep_ax_lst_1,xextent,yextent,offset,transform

#######################################
#    SET UP FIGURE FOR EACH DOMAIN    #
#######################################

# Call the domain_latlons_proj function from rrfs_plot_utils
  xextent,yextent,offset,extent,myproj = rrfs_plot_utils.domain_latlons_proj(dom)

# Create figure and axes instances
  fig = plt.figure(figsize=(4,4))
  gs = GridSpec(4,4,wspace=0.0,hspace=0.0)

  # Define where Cartopy maps are located
  cartopy.config['data_dir'] = '/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python/NaturalEarth'
  back_res='50m'
  back_img='off'

  ax1 = fig.add_subplot(gs[0:4,0:4], projection=myproj)
  ax1.set_extent(extent)
  axes = [ax1]

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
    ax1.imshow(img, origin='upper', transform=myproj)

  ax1.add_feature(cfeature.LAND, linewidth=0, facecolor='lightgray')
  ax1.add_feature(lakes)
  ax1.add_feature(states)
  ax1.add_feature(coastlines)

  # Map/figure has been set up here, save axes instances for use again later
  keep_ax_lst_1 = ax1.get_children()[:]

  t1dom = time.perf_counter()
  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

################################
  # Plot SLP
################################
  t1 = time.perf_counter()
  print(('Working on slp for '+dom))

  # Wind barb density settings
  if dom == 'conus' or dom == 'alaska':
    skip = 80
  elif dom == 'southeast':
    skip = 35
  elif dom == 'colorado' or dom == 'la_vegas' or dom =='mid_atlantic' or dom == 'south_florida':
    skip = 12
  elif dom == 'boston_nyc':
    skip = 10
  elif dom == 'seattle_portland':
    skip = 9
  elif dom == 'sf_bay_area':
    skip = 3
  else:
    skip = 20
  barblength = 3.5

  units = 'mb'
  clevs = [976,980,984,988,992,996,1000,1004,1008,1012,1016,1020,1024,1028,1032,1036,1040,1044,1048,1052]
  cm = plt.cm.Spectral_r
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs1_a = ax1.pcolormesh(lon_shift,lat_shift,slp_1,transform=transform,cmap=cm,norm=norm)
  if (member == '4') or (member == '5'):
    cbar1 = fig.colorbar(cs1_a,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.9,extend='both')
    cbar1.set_label(units,fontsize=7)
    cbar1.ax.tick_params(labelsize=6)
  cs1_b = ax1.contour(lon_shift,lat_shift,slp_1,np.arange(940,1060,4),colors='black',linewidths=0.1,transform=transform)
#  plt.clabel(cs1_b,inline=1,fmt='%d',fontsize=5,zorder=12,ax=ax)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_1[::skip,::skip],vwind_1[::skip,::skip],length=barblength,linewidth=0.35,color='black',transform=transform)
  if (member == 'Control') or (member == '1'):
    if timelag == 'yes':
      ax1.text(.5,1.03,'SLP ('+units+') and 10-m Winds (kts) \n initialized: '+itimem1+' valid: '+vtimem1 + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    else:
      ax1.text(.5,1.03,'SLP ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  if (member != 'HRRR'):
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.plt_highs_and_lows(lon_shift,lat_shift,slp_1,xmin,xmax,ymin,ymax,offset,ax1,transform,mode='reflect',window=400)

  rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_slp_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot slp for: '+dom) % t3)

#################################
  # Plot 2-m T
#################################
  t1 = time.perf_counter()
  print(('Working on t2m for '+dom))

  # Clear off old plottables but keep all the map info
  if (member == '4') or (member == '5'):
    cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = '\xb0''F'
  clevs = np.linspace(-16,134,26)
  cm = rrfs_plot_utils.cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,tmp2m_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  if (member == '4') or (member == '5'):
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.9,extend='both')
    cbar1.set_label(units,fontsize=7)
    cbar1.ax.tick_params(labelsize=7)
  if (member == 'Control') or (member == '1'):
    if timelag == 'yes':
      ax1.text(.5,1.03,'2-m Temperature ('+units+') \n initialized: '+itimem1+' valid: '+vtimem1 + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    else:
      ax1.text(.5,1.03,'2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  if (member != 'HRRR'):
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_2mt_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 2mt for: '+dom) % t3)

#################################
  # Plot 2-m Dew Point
#################################
  t1 = time.perf_counter()
  print(('Working on 2mdew for '+dom))

  # Clear off old plottables but keep all the map info
  if (member == '4') or (member == '5'):
    cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = '\xb0''F'
  clevs = np.linspace(-10,80,19)
  cm = rrfs_plot_utils.cmap_q2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,dew2m_1,transform=transform,cmap=cm,norm=norm)
  if (member == '4') or (member == '5'):
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.9,extend='both')
    cbar1.set_label(units,fontsize=7)
    cbar1.ax.tick_params(labelsize=7)
  if (member == 'Control') or (member == '1'):
    if timelag == 'yes':
      ax1.text(.5,1.03,'2-m Dew Point Temperature ('+units+') \n initialized: '+itimem1+' valid: '+vtimem1 + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    else:
      ax1.text(.5,1.03,'2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  if (member != 'HRRR'):
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_2mdew_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 2mdew for: '+dom) % t3)

#################################
  # Plot 10-m WSPD
#################################
  t1 = time.perf_counter()
  print(('Working on 10mwspd for '+dom))

  # Clear off old plottables but keep all the map info
  if (member == '4') or (member == '5'):
    cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'kts'
  clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
  colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,wspd10m_1,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  if (member == '4') or (member == '5'):
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.9,extend='max')
    cbar1.set_label(units,fontsize=7)
    cbar1.ax.tick_params(labelsize=7)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_1[::skip,::skip],vwind_1[::skip,::skip],length=barblength,linewidth=0.35,color='black',transform=transform)
  if (member == 'Control') or (member == '1'):
    if timelag == 'yes':
      ax1.text(.5,1.03,'10-m Winds ('+units+') \n initialized: '+itimem1+' valid: '+vtimem1 + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    else:
      ax1.text(.5,1.03,'10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  if (member != 'HRRR'):
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
  rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_10mwind_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 10mwspd for: '+dom) % t3)

#################################
  # Plot Most Unstable CAPE/CIN
#################################
  t1 = time.perf_counter()
  print(('Working on mucapecin for '+dom))

  # Clear off old plottables but keep all the map info
  if (member == '4') or (member == '5'):
    cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'J/kg'
  clevs = [100,250,500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
  colorlist = ['blue','dodgerblue','cyan','mediumspringgreen','#FAFAD2','#EEEE00','#EEC900','darkorange','crimson','darkred','darkviolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,mucape_1,transform=transform,cmap=cm,vmin=100,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  if (member == '4') or (member == '5'):
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.9,ticks=clevs,extend='max')
    cbar1.set_label(units,fontsize=7)
    cbar1.ax.tick_params(labelsize=6)
  if (member == 'Control') or (member == '1'):
    if timelag == 'yes':
      ax1.text(.5,1.03,'Most Unstable CAPE ('+units+') \n initialized: '+itimem1+' valid: '+vtimem1 + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    else:
      ax1.text(.5,1.03,'Most Unstable CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  if (member != 'HRRR'):
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_mucape_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot mucapecin for: '+dom) % t3)

#################################
  # Plot 850-mb THETAE
#################################
  t1 = time.perf_counter()
  print(('Working on 850 mb Theta-e for '+dom))

  # Clear off old plottables but keep all the map info
  if (member == '4') or (member == '5'):
    cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'K'
  clevs = np.linspace(270,360,31)
  cm = rrfs_plot_utils.cmap_t850()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,thetae_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  if (member == '4') or (member == '5'):
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.9,extend='both')
    cbar1.set_label(units,fontsize=7)
    cbar1.ax.tick_params(labelsize=6)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u850_1[::skip,::skip],v850_1[::skip,::skip],length=barblength,linewidth=0.35,color='black',transform=transform)
  if (member == 'Control') or (member == '1'):
    if timelag == 'yes':
      ax1.text(.5,1.03,'850 mb $\Theta$e ('+units+') and Winds (kts) \n initialized: '+itimem1+' valid: '+vtimem1 + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    else:
      ax1.text(.5,1.03,'850 mb $\Theta$e ('+units+') and Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  if (member != 'HRRR'):
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_850t_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 850 mb Theta-e for: '+dom) % t3)

#################################
  # Plot 500 mb HGT/WIND/VORT
#################################
  t1 = time.perf_counter()
  print(('Working on 500 mb Hgt/Wind/Vort for '+dom))

  # Clear off old plottables but keep all the map info
  if (member == '4') or (member == '5'):
    cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'x10${^5}$ s${^{-1}}$'
  vortlevs = [16,20,24,28,32,36,40]
  colorlist = ['yellow','gold','goldenrod','orange','orangered','red']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(vortlevs, cm.N)

  cs1_a = ax1.pcolormesh(lon_shift,lat_shift,vort500_1,transform=transform,cmap=cm,norm=norm)
  cs1_a.cmap.set_under('white')
  cs1_a.cmap.set_over('darkred')
  if (member == '4') or (member == '5'):
    cbar1 = fig.colorbar(cs1_a,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.9,ticks=vortlevs,extend='both')
    cbar1.set_label(units,fontsize=7)
    cbar1.ax.tick_params(labelsize=7)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u500_1[::skip,::skip],v500_1[::skip,::skip],length=barblength,linewidth=0.35,color='steelblue',transform=transform)
  cs1_b = ax1.contour(lon_shift,lat_shift,z500_1,np.arange(486,600,6),colors='black',linewidths=1,transform=transform)
  plt.clabel(cs1_b,np.arange(486,600,6),inline_spacing=1,fmt='%d',fontsize=5)
  if (member == 'Control') or (member == '1'):
    if timelag == 'yes':
      ax1.text(.5,1.03,'500 mb Heights (dam), Winds (kts), and $\zeta$ ('+units+') \n initialized: '+itimem1+' valid: '+vtimem1 + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    else:
      ax1.text(.5,1.03,'500 mb Heights (dam), Winds (kts), and $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  if (member != 'HRRR'):
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_500_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 500 mb Hgt/Wind/Vort for: '+dom) % t3)

#################################
  # Plot 250 mb WIND
#################################
  t1 = time.perf_counter()
  print(('Working on 250 mb WIND for '+dom))

  # Clear off old plottables but keep all the map info
  if (member == '4') or (member == '5'):
    cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'kts'
  clevs = [50,60,70,80,90,100,110,120,130,140,150]
  colorlist = ['turquoise','deepskyblue','dodgerblue','#1874CD','blue','beige','khaki','peru','brown','crimson']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,wspd250_1,transform=transform,cmap=cm,vmin=50,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('red')
  if (member == '4') or (member == '5'):
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.9,extend='max')
    cbar1.set_label(units,fontsize=7)
    cbar1.ax.tick_params(labelsize=7)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u250_1[::skip,::skip],v250_1[::skip,::skip],length=barblength,linewidth=0.35,color='black',transform=transform)
  if (member == 'Control') or (member == '1'):
    if timelag == 'yes':
      ax1.text(.5,1.03,'250 mb Winds ('+units+') \n initialized: '+itimem1+' valid: '+vtimem1 + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    else:
      ax1.text(.5,1.03,'250 mb Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  if (member != 'HRRR'):
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_250wind_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 250 mb WIND for: '+dom) % t3)

#################################
  # Plot composite reflectivity
#################################
  t1 = time.perf_counter()
  print(('Working on composite reflectivity for '+dom))

  # Clear off old plottables but keep all the map info
  if (member == '4') or (member == '5'):
    cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'dBZ'
  clevs = np.linspace(5,70,14)
  colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  
  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,refc_1,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  if (member == '4') or (member == '5'):
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.9,ticks=clevs,extend='max')
    cbar1.set_label(units,fontsize=7)
    cbar1.ax.tick_params(labelsize=6)
  if (member == 'Control') or (member == '1'):
    if timelag == 'yes':
      ax1.text(.5,1.03,'Composite Reflectivity ('+units+') \n initialized: '+itimem1+' valid: '+vtimem1 + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    else:
      ax1.text(.5,1.03,'Composite Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  if (member != 'HRRR'):
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_refc_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot composite reflectivity for: '+dom) % t3)

#################################
  # Plot Surface Visibility
#################################
  t1 = time.perf_counter()
  print(('Working on Surface Visibility for '+dom))

  # Clear off old plottables but keep all the map info
  if (member == '4') or (member == '5'):
    cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'miles'
  clevs = [0.25,0.5,1,2,3,4,5,10]
  colorlist = ['salmon','goldenrod','#EEEE00','palegreen','darkturquoise','blue','mediumpurple']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,vis_1,transform=transform,cmap=cm,vmax=10,norm=norm)
  cs_1.cmap.set_under('firebrick')
  cs_1.cmap.set_over('white',alpha=0.)
  if (member == '4') or (member == '5'):
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.9,ticks=clevs,extend='min')
    cbar1.set_label(units,fontsize=7)
    cbar1.ax.tick_params(labelsize=6)
  if (member == 'Control') or (member == '1'):
    if timelag == 'yes':
      ax1.text(.5,1.03,'Surface Visibility ('+units+') \n initialized: '+itimem1+' valid: '+vtimem1 + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    else:
      ax1.text(.5,1.03,'Surface Visibility ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  if (member != 'HRRR'):
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_vis_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Surface Visibility for: '+dom) % t3)

#################################
  # Plot Cloud Ceiling Height
#################################
  t1 = time.perf_counter()
  print(('Working on Cloud Ceiling Height for '+dom))

  # Clear off old plottables but keep all the map info
  if (member == '4') or (member == '5'):
    cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'kft'
  clevs = [0,0.1,0.3,0.5,1,5,10,15,20,25,30,35,40]
  colorlist = ['firebrick','tomato','salmon','lightsalmon','goldenrod','khaki','gold','yellow','palegreen','mediumspringgreen','lime','limegreen']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,zceil_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_over('white')
  if (member == '4') or (member == '5'):
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.9,ticks=clevs,extend='max')
    cbar1.set_label(units,fontsize=7)
    cbar1.ax.tick_params(labelsize=6)
  if (member == 'Control') or (member == '1'):
    if timelag == 'yes':
      ax1.text(.5,1.03,'Cloud Ceiling Height ('+units+') \n initialized: '+itimem1+' valid: '+vtimem1 + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    else:
      ax1.text(.5,1.03,'Cloud Ceiling Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  if (member != 'HRRR'):
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_zceil_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Cloud Ceiling Height for: '+dom) % t3)

#################################
  # Plot snow depth
#################################
  t1 = time.perf_counter()
  print(('Working on snow depth for '+dom))

  # Clear off old plottables but keep all the map info
  if (member == '4') or (member == '5'):
    cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'in'
  clevs = [0.5,1,2,3,4,6,8,12,18,24,30,36]
  colorlist = ['#adc4d9','#73bdff','#0f69db','#004da8','#002673','#ffff73','#ffaa00','#e64c00','#e60000','#730000','#e8beff']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,snow_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('#CA7AF5')
  if (member == '4') or (member == '5'):
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels(clevs)
    cbar1.ax.tick_params(labelsize=6)
  if (member == 'Control') or (member == '1'):
    if timelag == 'yes':
      ax1.text(.5,1.03,'RRFS_A Snow Depth ('+units+') \n initialized: '+itimem1+' valid: '+vtimem1 + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    else:
      ax1.text(.5,1.03,'RRFS_A Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  if (member != 'HRRR'):
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_snow_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot snow depth for: '+dom) % t3)

#################################
  # Plot Snowfall
#################################
  t1 = time.perf_counter()
  print(('Working on snowfall for '+dom))

  # Clear off old plottables but keep all the map info
  if (member == '4') or (member == '5'):
    cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'in'
  clevs = [0.5,1,2,3,4,6,8,12,18,24,30,36]
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,asnow_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('#CA7AF5')
  if (member == '4') or (member == '5'):
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels(clevs)
    cbar1.ax.tick_params(labelsize=6)
  if (member == 'Control') or (member == '1'):
    if timelag == 'yes':
      ax1.text(.5,1.03,'RRFS_A Snowfall (variable density) ('+units+') \n initialized: '+itimem1+' valid: '+vtimem1 + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    else:
      ax1.text(.5,1.03,'RRFS_A Snowfall (variable density) ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  if (member != 'HRRR'):
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_asnow_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot snowfall for: '+dom) % t3)

#################################
  # Plot Max/Min Hourly 2-5 km UH
#################################
  if (fhr > 0):
    t1 = time.perf_counter()
    print(('Working on Max/Min Hourly 2-5 km UH for '+dom))

    # Clear off old plottables but keep all the map info
    if (member == '4') or (member == '5'):
      cbar1.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

    units = 'm${^2}$ s$^{-2}$'
    clevs = [-150,-100,-75,-50,-25,-10,0,10,25,50,75,100,150,200,250,300]
    clevsdif = [-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60]
    colorlist = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','#E5E5E5','#E5E5E5','#EEEE00','#EEC900','darkorange','orangered','red','firebrick','mediumvioletred','darkviolet']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,uh25_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_under('darkblue')
    cs_1.cmap.set_over('black')
    if (member == '4') or (member == '5'):
      cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.9,extend='both')
      cbar1.set_label(units,fontsize=7)
      cbar1.ax.tick_params(labelsize=7)
    if (member == 'Control') or (member == '1'):
      if timelag == 'yes':
        ax1.text(.5,1.03,'1-h Max/Min 2-5 km Updraft Helicity ('+units+') \n initialized: '+itimem1+' valid: '+vtimem1 + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      else:
        ax1.text(.5,1.03,'1-h Max/Min 2-5 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    if (member != 'HRRR'):
      ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_uh25_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max/Min Hourly 2-5 km UH for: '+dom) % t3)

#################################
  # Plot Max Hourly Updraft Speed
#################################
    t1 = time.perf_counter()
    print(('Working on Max Hourly Updraft Speed for '+dom))

    # Clear off old plottables but keep all the map info
    if (member == '4') or (member == '5'):
      cbar1.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

    units = 'm s$^{-1}$'
    clevs = [0.5,1,2.5,5,7.5,10,12.5,15,20,25,30,35,40,50,75]
    colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia','mediumpurple']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,maxuvv_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_under('white')
    cs_1.cmap.set_over('black')
    if (member == '4') or (member == '5'):
      cbar1 = fig.colorbar(cs_1,ax=ax1,ticks=clevs,orientation='horizontal',pad=0.01,shrink=0.9,extend='both')
      cbar1.set_label(units,fontsize=7)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=6)
    if (member == 'Control') or (member == '1'):
      if timelag == 'yes':
        ax1.text(.5,1.03,'1-h Max 100-1000 mb Updraft Speed ('+units+') \n initialized: '+itimem1+' valid: '+vtimem1 + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      else:
        ax1.text(.5,1.03,'1-h Max 100-1000 mb Updraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    if (member != 'HRRR'):
      ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_maxuvv_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly Updraft Speed for: '+dom) % t3)

#################################
  # Plot Total QPF
#################################
    t1 = time.perf_counter()
    print(('Working on total qpf for '+dom))

    # Clear off old plottables but keep all the map info
    if (member == '4') or (member == '5'):
      cbar1.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

    units = 'in'
    clevs = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,7,10,15,20]
    colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','deepskyblue','cyan','mediumpurple','mediumorchid','darkmagenta','darkred','crimson','orangered','darkorange','goldenrod','gold','yellow']  
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,qpf_1,transform=transform,cmap=cm,vmin=0.01,norm=norm)
    cs_1.cmap.set_under('white',alpha=0.)
    cs_1.cmap.set_over('pink')
    if (member == '4') or (member == '5'):
      cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.9,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
      cbar1.set_label(units,fontsize=7)
      cbar1.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
      cbar1.ax.tick_params(labelsize=7)
    if (member == 'Control') or (member == '1'):
      if timelag == 'yes':
        ax1.text(.5,1.03,''+fhourm6+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      else:
        ax1.text(.5,1.03,''+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.text(.5,0.95,memstr,horizontalalignment='center',fontsize=7,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    if (member != 'HRRR'):
      ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.convert_and_save(member+'_'+timelag+'_qpf_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot total qpf for: '+dom) % t3)

#################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all variables for: "+dom) % t3dom)
  plt.clf()
 
#################################

main()

