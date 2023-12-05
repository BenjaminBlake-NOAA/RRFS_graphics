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

fhr = int(sys.argv[2])
fhour = str(fhr).zfill(2)
print('fhour '+fhour)

# Forecast valid date/time
itime = ymdh
vtime = rrfs_plot_utils.ndate(itime,int(fhr))

# Define the directory paths to the input files
RRFS_DIR = '/lfs/h2/emc/ptmp/emc.lam/rrfs/v0.7.5/prod/rrfs.'+ymd+'/'+cyc

# Define the input files
data1 = grib2io.open(RRFS_DIR+'/rrfs.t'+cyc+'z.prslev.f0'+fhour+'.grib2')

# Get the lats and lons
msg = data1.select(shortName='HGT', level='500 mb')[0]	# msg is a Grib2Message object
lat, lon = msg.grid(unrotate=False)

# Specify plotting domains
domains=['namerica','caribbean']

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
slp_1 = data1.select(shortName='MSLET',level='mean sea level')[0].data * 0.01

# 2-m temperature
tmp2m_1 = data1.select(shortName='TMP',level='2 m above ground')[0].data
tmp2m_1 = (tmp2m_1 - 273.15)*1.8 + 32.0

# Surface temperature
tmpsfc_1 = data1.select(shortName='TMP',level='surface')[0].data
tmpsfc_1 = (tmpsfc_1 - 273.15)*1.8 + 32.0

# 2-m dew point temperature
dew2m_1 = data1.select(shortName='DPT',level='2 m above ground')[0].data
dew2m_1 = (dew2m_1 - 273.15)*1.8 + 32.0

# 10-m wind speed
uwind_1 = data1.select(shortName='UGRD',level='10 m above ground')[0].data * 1.94384
vwind_1 = data1.select(shortName='VGRD',level='10 m above ground')[0].data * 1.94384
wspd10m_1 = np.sqrt(uwind_1**2 + vwind_1**2)

# Surface wind gust
gust_1 = data1.select(shortName='GUST',level='surface')[0].data * 1.94384

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

# 700-mb omega and relative humidity
omg700_1 = data1.select(shortName='VVEL',level='700 mb')[0].data
rh700_1 = data1.select(shortName='RH',level='700 mb')[0].data

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

# Visibility
vis_1 = data1.select(shortName='VIS',level='surface')[0].data * 0.000621371

# Cloud Base Height
zbase_1 = data1.select(shortName='HGT',level='cloud base')[0].data * (3.28084/1000)

# Cloud Ceiling Height
zceil_1 = data1.select(shortName='HGT',level='cloud ceiling')[0].data * (3.28084/1000)

# Cloud Top Height
ztop_1 = data1.select(shortName='HGT',level='cloud top')[0].data * (3.28084/1000)

# Precipitable water
pw_1 = data1.select(shortName='PWAT',level='entire atmosphere (considered as a single layer)')[0].data * 0.0393701

# Percent of frozen precipitation
pofp_1 = data1.select(shortName='CPOFP')[0].data

# Total precipitation
qpf_1 = data1.select(shortName='APCP')[1].data * 0.0393701

# Snow depth
snow_1 = data1.select(shortName='SNOD')[0].data * 39.3701

# Snowfall
asnow_1 = data1.select(shortName='ASNOW')[0].data * 39.3701

# 1-km reflectivity
ref1km_1 = data1.select(shortName='REFD',level='1000 m above ground')[0].data

# Composite reflectivity
refc_1 = data1.select(shortName='REFC')[0].data

# PBL height
hpbl_1 = data1.select(shortName='HGT',level='planetary boundary layer')[0].data

# Total column integrated liquid (cloud water + rain)
tqw_1 = data1.select(shortName='TCOLW')[0].data
tqr_1 = data1.select(shortName='TCOLR')[0].data
tcolw_1 = tqw_1 + tqr_1

# Total column integrated ice (cloud ice + snow)
tqi_1 = data1.select(shortName='TCOLI')[0].data
tqs_1 = data1.select(shortName='TCOLS')[0].data
tcoli_1 = tqi_1 + tqs_1

# 0-3 km Storm Relative Helicity
hel3km_1 = data1.select(shortName='HLCY',scaledValueOfFirstFixedSurface=3000)[0].data

# 0-1 km Storm Relative Helicity
hel1km_1 = data1.select(shortName='HLCY',scaledValueOfFirstFixedSurface=1000)[0].data

if (fhr > 0):
# Max/Min Hourly 2-5 km Updraft Helicity
  maxuh25_1 = data1.select(shortName='MXUPHL',level='5000-2000 m above ground')[0].data
  minuh25_1 = data1.select(shortName='MNUPHL',level='5000-2000 m above ground')[0].data
  maxuh25_1[maxuh25_1 < 10] = 0
  minuh25_1[minuh25_1 > -10] = 0
  uh25_1 = maxuh25_1 + minuh25_1

# Max Hourly Updraft Speed
  maxuvv_1 = data1.select(shortName='MAXUVV')[0].data

# Max Hourly Downdraft Speed
  maxdvv_1 = data1.select(shortName='MAXDVV')[0].data * -1

# Max Hourly 1-km AGL reflectivity
  maxref1km_1 = data1.select(shortName='MAXREF',level='1000 m above ground')[0].data

# Max Hourly Wind
  maxwind_1 = data1.select(shortName='WIND')[0].data * 1.94384

# Total cloud cover
tcdc_1 = data1.select(shortName='TCDC',level='entire atmosphere (considered as a single layer)')[0].data

# Echo top height
retop_1 = data1.select(shortName='RETOP')[0].data * (3.28084/1000)

# Cloud base pressure
pbase_1 = data1.select(shortName='PRES',level='cloud base')[0].data * 0.01

# Cloud top pressure
ptop_1 = data1.select(shortName='PRES',level='cloud top')[0].data * 0.01

# Precipitation rate
prate_1 = data1.select(shortName='PRATE')[0].data * 3600

# 8-m mass density (near-surface smoke)
smoke_1 = data1.select(shortName='MASSDEN',level='8 m above ground')[0].data * 1000000000

# Total column integrated smoke
colsmoke_1 = data1.select(shortName='COLMD',level='entire atmosphere (considered as a single layer)')[0].data * 1000000

# 8-m mass density (near-surface dust)
dust_1 = data1.select(shortName='MASSDEN',level='8 m above ground')[1].data * 1000000000

# Total column-integrated dust
coldust_1 = data1.select(shortName='COLMD',level='entire atmosphere (considered as a single layer)')[1].data * 1000000


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
  transform = ccrs.RotatedPole(pole_longitude=67.0, pole_latitude=35.0)

  # high-resolution background images
  if back_img=='on':
    img = plt.imread('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python/NaturalEarth/raster_files/NE1_50M_SR_W.tif')
    ax1.imshow(img, origin='upper', transform=myproj)

  ax1.add_feature(cfeature.LAND, linewidth=0, facecolor='lightgray')
#  ax1.add_feature(cfeature.OCEAN,linewidth=0)
  ax1.add_feature(lakes)
  ax1.add_feature(states)
  ax1.add_feature(coastlines)
  ax1.add_feature(borders)

  # Map/figure has been set up here, save axes instances for use again later
  keep_ax_lst_1 = ax1.get_children()[:]

  # Split plots into 13 sets with multiprocessing
  sets = [1,2,3,4,5,6,7,8,9,10,11,12,13]
  pool2 = MyPool(len(sets))
  pool2.map(plot_sets,sets)

################################################################################

def plot_sets(set):

  global fig,axes,ax1,keep_ax_lst_1,xextent,yextent,offset,transform

# Add print to see if dom is being passed in
  print(('plot_sets dom variable '+dom))

  if set == 1:
    plot_set_1()
  elif set == 2:
    plot_set_2()
  elif set == 3:
    plot_set_3()
  elif set == 4:
    plot_set_4()
  elif set == 5:
    plot_set_5()
  elif set == 6:
    plot_set_6()
  elif set == 7:
    plot_set_7()
  elif set == 8:
    plot_set_8()
  elif set == 9:
    plot_set_9()
  elif set == 10:
    plot_set_10()
  elif set == 11:
    plot_set_11()
  elif set == 12:
    plot_set_12()
  elif set == 13:
    plot_set_13()

################################################################################

def plot_set_1():
  global fig,axes,ax1,keep_ax_lst_1,xextent,yextent,offset,transform

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
  x1 = xmin + ((xmax-xmin)*0.03)
  y1 = ymin + ((ymax-ymin)*0.03)

#################################
# Plot SFCT
#################################
  t1 = time.perf_counter()
  print(('Working on tsfc for '+dom))

  units = '\xb0''F'
  if dom == 'namerica':
    clevs = np.linspace(-46,134,31)
  elif dom == 'caribbean':
    clevs = np.linspace(23,104,28)
  cm = rrfs_plot_utils.cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,tmpsfc_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'RRFS_A Surface Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparetsfc_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot tsfc for: '+dom) % t3)

  plt.clf()

################################################################################

def plot_set_2():
  global fig,axes,ax1,keep_ax_lst_1,xextent,yextent,offset,transform

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
  x1 = xmin + ((xmax-xmin)*0.03)
  y1 = ymin + ((ymax-ymin)*0.03)

#################################
  # Plot PBL height
#################################
  t1 = time.perf_counter()
  print(('Working on PBL height for '+dom))

  units = 'm'
  clevs = [50,100,250,500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
  colorlist= ['gray','blue','dodgerblue','cyan','mediumspringgreen','#FAFAD2','#EEEE00','#EEC900','darkorange','crimson','darkred','darkviolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,hpbl_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,ticks=clevs,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=4)
  ax1.text(.5,1.03,'RRFS_A PBL Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparehpbl_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot PBL height for: '+dom) % t3)

  plt.clf()

################################################################################

def plot_set_3():
  global fig,axes,ax1,keep_ax_lst_1,xextent,yextent,offset,transform

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
  x1 = xmin + ((xmax-xmin)*0.03)
  y1 = ymin + ((ymax-ymin)*0.03)


#################################
  # Plot 10-m WSPD
#################################
  t1 = time.perf_counter()
  print(('Working on 10mwspd for '+dom))

  units = 'kts'
  if dom == 'namerica':
    skip = 200
  elif dom == 'caribbean':
    skip = 40
  barblength = 3.5
  clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
  colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,wspd10m_1,levels=clevs,cmap=cm,vmin=5,norm=norm,transform=transform)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.barbs(lon[::skip,::skip],lat[::skip,::skip],uwind_1[::skip,::skip],vwind_1[::skip,::skip],length=barblength,linewidth=0.35,color='black',transform=transform)
  ax1.text(.5,1.03,'RRFS_A 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)
 
  rrfs_plot_utils.convert_and_save_2('compare10mwind_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 10mwspd for: '+dom) % t3)

  plt.clf()

################################################################################

def plot_set_4():
  global fig,axes,ax1,keep_ax_lst_1,xextent,yextent,offset,transform

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
  x1 = xmin + ((xmax-xmin)*0.03)
  y1 = ymin + ((ymax-ymin)*0.03)

#################################
  # Plot surface wind gust
#################################
  t1 = time.perf_counter()
  print(('Working on surface wind gust for '+dom))

  units = 'kts'
  if dom == 'namerica':
    skip = 200
  elif dom == 'caribbean':
    skip = 40
  barblength = 3.5
  clevs = [5,12.5,20,27.5,35,42.5,50,57.5,65,72.5,80]
  colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','red','fuchsia','DarkViolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,gust_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.05,'RRFS_A Surface Wind Gust ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparegust_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot surface wind gust for: '+dom) % t3)

  plt.clf()

################################################################################

def plot_set_5():
  global fig,axes,ax1,keep_ax_lst_1,xextent,yextent,offset,transform

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
  x1 = xmin + ((xmax-xmin)*0.03)
  y1 = ymin + ((ymax-ymin)*0.03)

#################################
  # Plot Max Hourly 10-m Winds
#################################
  if (fhr > 0):
    t1 = time.perf_counter()
    print(('Working on Max Hourly 10-m Wind Speed for '+dom))

    units = 'kts'
    if dom == 'namerica':
      skip = 200
    elif dom == 'caribbean':
      skip = 40
    barblength = 3.5
#    clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
    clevs = [10,20,30,40,50,60,70,80,90,100,110,120]
    colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    cs_1 = ax1.contourf(lon,lat,maxwind_1,levels=clevs,cmap=cm,vmin=5,norm=norm,transform=transform)
    cs_1.cmap.set_under('white',alpha=0.)
    cs_1.cmap.set_over('black')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='max')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'RRFS_A 1-h Max 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

    rrfs_plot_utils.convert_and_save_2('comparemaxwind_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly 10-m Wind Speed for: '+dom) % t3)

  plt.clf()

################################################################################

def plot_set_6():
  global fig,axes,ax1,keep_ax_lst_1,xextent,yextent,offset,transform

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
  x1 = xmin + ((xmax-xmin)*0.03)
  y1 = ymin + ((ymax-ymin)*0.03)

#################################
  # Plot total column liquid
#################################
  t1 = time.perf_counter()
  print(('Working on Total column liquid for '+dom))

  units = 'kg m${^{-2}}$'
  clevs = [0.001,0.005,0.01,0.05,0.1,0.25,0.5,1,2,4,6,10,15,20,25]
  q_color_list = plt.cm.gist_stern_r(np.linspace(0, 1, len(clevs)+1))
  cm = matplotlib.colors.ListedColormap(q_color_list)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,tcolw_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
  cs_1.cmap.set_under('white')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar1.set_label(units,fontsize=6)
#  cbar1.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Total Column Cloud Water + Rain ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparetcolw_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Total column liquid for: '+dom) % t3)

  plt.clf()

################################################################################

def plot_set_7():
  global fig,axes,ax1,keep_ax_lst_1,xextent,yextent,offset,transform

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
  x1 = xmin + ((xmax-xmin)*0.03)
  y1 = ymin + ((ymax-ymin)*0.03)

#################################
  # Plot Surface Visibility
#################################
  t1 = time.perf_counter()
  print(('Working on Surface Visibility for '+dom))

  units = 'miles'
  clevs = [0.25,0.5,1,2,3,4,5,10]
  colorlist = ['salmon','goldenrod','#EEEE00','palegreen','darkturquoise','blue','mediumpurple']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,vis_1,levels=clevs,cmap=cm,vmax=10,norm=norm,transform=transform)
  cs_1.cmap.set_under('firebrick')
  cs_1.cmap.set_over('white',alpha=0.)
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='min')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Surface Visibility ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparevis_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Surface Visibility for: '+dom) % t3)

  plt.clf()

################################################################################

def plot_set_8():
  global fig,axes,ax1,keep_ax_lst_1,xextent,yextent,offset,transform

  t1dom = time.perf_counter()
  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
  x1 = xmin + ((xmax-xmin)*0.03)
  y1 = ymin + ((ymax-ymin)*0.03)

################################
  # Plot SLP
################################
  t1 = time.perf_counter()
  print(('Working on slp for '+dom))

  units = 'mb'
  if dom == 'namerica':
    skip = 200
  elif dom == 'caribbean':
    skip = 40
  barblength = 3.5
  clevs = [940,944,948,952,956,960,964,968,972,976,980,984,988,992,996,1000,1004,1008,1012,1016,1020,1024,1028,1032,1036,1040,1044,1048,1052,1056,1060]
  cm = plt.cm.Spectral_r
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs1_a = ax1.contourf(lon,lat,slp_1,levels=clevs,cmap=cm,norm=norm,transform=transform)  
  cbar1 = fig.colorbar(cs1_a,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  cs1_b = ax1.contour(lon,lat,slp_1,np.arange(940,1060,4),colors='black',linewidths=0.1,transform=transform)
#  plt.clabel(cs1_b,inline=1,fmt='%d',fontsize=5,zorder=12,ax=ax)
  ax1.barbs(lon[::skip,::skip],lat[::skip,::skip],uwind_1[::skip,::skip],vwind_1[::skip,::skip],length=barblength,linewidth=0.35,color='black',transform=transform)
  ax1.text(.5,1.03,'RRFS_A SLP ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.plt_highs_and_lows(lon,lat,slp_1,xmin,xmax,ymin,ymax,offset,ax1,transform,mode='reflect',window=600)

  rrfs_plot_utils.convert_and_save_2('compareslp_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot slp for: '+dom) % t3)

#################################
  # Plot 2-m T
#################################
  t1 = time.perf_counter()
  print(('Working on t2m for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = '\xb0''F'
  if dom == 'namerica':
    clevs = np.linspace(-46,134,31)
  elif dom == 'caribbean':
    clevs = np.linspace(23,104,28)
  cm = rrfs_plot_utils.cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,tmp2m_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'RRFS_A 2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('compare2mt_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 2mt for: '+dom) % t3)

#################################
  # Plot 2-m Dew Point
#################################
  t1 = time.perf_counter()
  print(('Working on 2mdew for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = '\xb0''F'
  if dom == 'namerica':
    clevs = np.linspace(-40,85,26)
  elif dom == 'caribbean':
    clevs = np.linspace(-5,85,19)
  cm = rrfs_plot_utils.cmap_q2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,dew2m_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A 2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('compare2mdew_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 2mdew for: '+dom) % t3)

#################################
  # Plot Most Unstable CAPE/CIN
#################################
  t1 = time.perf_counter()
  print(('Working on mucapecin for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'J/kg'
  clevs = [100,250,500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
  colorlist = ['blue','dodgerblue','cyan','mediumspringgreen','#FAFAD2','#EEEE00','#EEC900','darkorange','crimson','darkred','darkviolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,mucape_1,levels=clevs,cmap=cm,vmin=100,norm=norm,transform=transform)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=4)
  ax1.text(.5,1.03,'RRFS_A Most Unstable CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparemucape_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot mucapecin for: '+dom) % t3)

#################################
  # Plot 850-mb THETAE
#################################
  t1 = time.perf_counter()
  print(('Working on 850 mb Theta-e for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'K'
  if dom == 'namerica':
    skip = 200
  elif dom == 'caribbean':
    skip = 40
  barblength = 3.5
  clevs = np.linspace(240,360,21)
  cm = rrfs_plot_utils.cmap_t850()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,thetae_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar1.set_label(units,fontsize=6)   
  cbar1.ax.tick_params(labelsize=4)
  ax1.barbs(lon[::skip,::skip],lat[::skip,::skip],u850_1[::skip,::skip],v850_1[::skip,::skip],length=barblength,linewidth=0.35,color='black',transform=transform)
  ax1.text(.5,1.03,'RRFS_A 850 mb $\Theta$e ('+units+') and Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('compare850t_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 850 mb Theta-e for: '+dom) % t3)

#################################
  # Plot 700-mb OMEGA and RH
#################################
  t1 = time.perf_counter()
  print(('Working on 700 mb omega and RH for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = '%'
  clevs = [50,60,70,80,90,100]
  clevsw = [-100,-5]
  colors = ['blue']
  cm = plt.cm.BuGn
  cmw = matplotlib.colors.ListedColormap(colors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs1_a = ax1.contourf(lon,lat,rh700_1,levels=clevs,cmap=cm,vmin=50,norm=norm,transform=transform)
  cs1_a.cmap.set_under('white',alpha=0.)
  cbar1 = fig.colorbar(cs1_a,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs)
  cbar1.set_label(units,fontsize=6) 
  cbar1.ax.tick_params(labelsize=6)
  cs1_b = ax1.contourf(lon,lat,omg700_1,levels=clevsw,cmap=cmw,vmax=-5,transform=transform)
  cs1_b.cmap.set_over('white',alpha=0.)
  ax1.text(.5,1.03,'RRFS_A 700 mb $\omega$ (rising motion in blue) and RH ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('compare700_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 700 mb $\omega$ and RH for: '+dom) % t3)

#################################
  # Plot 500 mb HGT/WIND/VORT
#################################
  t1 = time.perf_counter()
  print(('Working on 500 mb Hgt/Wind/Vort for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'x10${^5}$ s${^{-1}}$'
  vortlevs = [16,20,24,28,32,36,40]
  colorlist = ['yellow','gold','goldenrod','orange','orangered','red']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(vortlevs, cm.N)

  cs1_a = ax1.contourf(lon,lat,vort500_1,levels=vortlevs,cmap=cm,norm=norm,transform=transform)
  cs1_a.cmap.set_under('white')
  cs1_a.cmap.set_over('darkred')
  cbar1 = fig.colorbar(cs1_a,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=vortlevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.barbs(lon[::skip,::skip],lat[::skip,::skip],u500_1[::skip,::skip],v500_1[::skip,::skip],length=barblength,linewidth=0.35,color='steelblue',transform=transform)
  cs1_b = ax1.contour(lon,lat,z500_1,np.arange(486,600,6),colors='black',linewidths=1,transform=transform)
  plt.clabel(cs1_b,np.arange(486,600,6),inline_spacing=1,fmt='%d',fontsize=5)
  ax1.text(.5,1.03,'RRFS_A 500 mb Heights (dam), Winds (kts), and $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('compare500_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 500 mb Hgt/Wind/Vort for: '+dom) % t3)

#################################
  # Plot 250 mb WIND
#################################
  t1 = time.perf_counter()
  print(('Working on 250 mb WIND for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'kts'
  clevs = [50,60,70,80,90,100,110,120,130,140,150]
  colorlist = ['turquoise','deepskyblue','dodgerblue','#1874CD','blue','beige','khaki','peru','brown','crimson']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,wspd250_1,levels=clevs,cmap=cm,vmin=50,norm=norm,transform=transform)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('red')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.barbs(lon[::skip,::skip],lat[::skip,::skip],u250_1[::skip,::skip],v250_1[::skip,::skip],length=barblength,linewidth=0.35,color='black',transform=transform)
  ax1.text(.5,1.03,'RRFS_A 250 mb Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('compare250wind_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 250 mb WIND for: '+dom) % t3)

######################################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 8 variables for: "+dom) % t3dom)
  plt.clf()

######################################################

def plot_set_9():
  global fig,axes,ax1,keep_ax_lst_1,xextent,yextent,offset,transform

  t1dom = time.perf_counter()
  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
  x1 = xmin + ((xmax-xmin)*0.03)
  y1 = ymin + ((ymax-ymin)*0.03)

#################################
  # Plot PW
#################################
  t1 = time.perf_counter()
  print(('Working on PW for '+dom))

  units = 'in'
  clevs = [0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5]
  colorlist = ['lightsalmon','khaki','palegreen','cyan','turquoise','cornflowerblue','mediumslateblue','darkorchid','deeppink','hotpink']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,pw_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('hotpink')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'RRFS_A Precipitable Water ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparepw_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot PW for: '+dom) % t3)

#################################
  # Plot % FROZEN PRECIP
#################################
  t1 = time.perf_counter()
  print(('Working on PERCENT FROZEN PRECIP for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = '%'
  clevs = [10,20,30,40,50,60,70,80,90,100]
  colorlist = ['blue','dodgerblue','deepskyblue','mediumspringgreen','khaki','sandybrown','salmon','crimson','maroon']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,pofp_1,levels=clevs,cmap=cm,vmin=10,norm=norm,transform=transform)
  cs_1.cmap.set_under('white',alpha=0.)
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs)
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Percent of Frozen Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparepofp_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot PERCENT FROZEN PRECIP for: '+dom) % t3)

#################################
  # Plot Total QPF
#################################
  if (fhr > 0):
    t1 = time.perf_counter()
    print(('Working on total qpf for '+dom))

    # Clear off old plottables but keep all the map info
    cbar1.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

    units = 'in'
    clevs = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,7,10,15,20]
    colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','deepskyblue','cyan','mediumpurple','mediumorchid','darkmagenta','darkred','crimson','orangered','darkorange','goldenrod','gold','yellow']  
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    cs_1 = ax1.contourf(lon,lat,qpf_1,levels=clevs,cmap=cm,vmin=0.01,norm=norm,transform=transform)
    cs_1.cmap.set_under('white',alpha=0.)
    cs_1.cmap.set_over('pink')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'RRFS_A '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

    rrfs_plot_utils.convert_and_save_2('compareqpf_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot total qpf for: '+dom) % t3)

#################################
  # Plot snow depth
#################################
  t1 = time.perf_counter()
  print(('Working on snow depth for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'in'
  clevs = [0.5,1,2,3,4,6,8,12,18,24,30,36]
  colorlist = ['#adc4d9','#73bdff','#0f69db','#004da8','#002673','#ffff73','#ffaa00','#e64c00','#e60000','#730000','#e8beff']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
 
  cs_1 = ax1.contourf(lon,lat,snow_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('#CA7AF5')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparesnow_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot snow depth for: '+dom) % t3)

#################################
  # Plot snowfall
#################################
  t1 = time.perf_counter()
  print(('Working on snowfall for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'in'
  clevs = [0.5,1,2,3,4,6,8,12,18,24,30,36]
  colorlist = ['#adc4d9','#73bdff','#0f69db','#004da8','#002673','#ffff73','#ffaa00','#e64c00','#e60000','#730000','#e8beff']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
 
  cs_1 = ax1.contourf(lon,lat,asnow_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('#CA7AF5')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Snowfall (variable density) ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('compareasnow_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot snowfall for: '+dom) % t3)

#################################
  # Plot total column ice
#################################
  t1 = time.perf_counter()
  print(('Working on Tcoli for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'kg m${^{-2}}$'
  clevs = [0.001,0.005,0.01,0.05,0.1,0.25,0.5,1,2,4,6,10,15,20,25]
  q_color_list = plt.cm.gist_stern_r(np.linspace(0, 1, len(clevs)+1))
  cm = matplotlib.colors.ListedColormap(q_color_list)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,tcoli_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
  cs_1.cmap.set_under('white')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Total Column Cloud Ice + Snow ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparetcoli_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Tcoli for: '+dom) % t3)

#################################
  # Plot 0-3 km Storm Relative Helicity
#################################
  t1 = time.perf_counter()
  print(('Working on 0-3 km SRH for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'm${^2}$ s$^{-2}$'
  clevs = [50,100,150,200,250,300,400,500,600,700,800]
  colorlist = ['mediumblue','dodgerblue','chartreuse','limegreen','darkgreen','#EEEE00','orange','orangered','firebrick','darkmagenta']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,hel3km_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A 0-3 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparehel3km_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 0-3 km SRH for: '+dom) % t3)

#################################
  # Plot 0-1 km Storm Relative Helicity
#################################
  t1 = time.perf_counter()
  print(('Working on 0-1 km SRH for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  cs_1 = ax1.contourf(lon,lat,hel1km_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A 0-1 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparehel1km_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 0-1 km SRH for: '+dom) % t3)

#################################
  # Plot 1-km reflectivity
#################################
  t1 = time.perf_counter()
  print(('Working on 1-km reflectivity for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'dBZ'
  clevs = np.linspace(5,70,14)
  colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  
  cs_1 = ax1.contourf(lon,lat,ref1km_1,levels=clevs,cmap=cm,vmin=5,norm=norm,transform=transform)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('compareref1km_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 1-km reflectivity for: '+dom) % t3)

#################################
  # Plot composite reflectivity
#################################
  t1 = time.perf_counter()
  print(('Working on composite reflectivity for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'dBZ'
  clevs = np.linspace(5,70,14)
  colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  
  cs_1 = ax1.contourf(lon,lat,refc_1,levels=clevs,cmap=cm,vmin=5,norm=norm,transform=transform)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Composite Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparerefc_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot composite reflectivity for: '+dom) % t3)

######################################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 9 variables for: "+dom) % t3dom)
  plt.clf()

######################################################

def plot_set_10():
  global fig,axes,ax1,keep_ax_lst_1,xextent,yextent,offset,transform

  t1dom = time.perf_counter()
  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
  x1 = xmin + ((xmax-xmin)*0.03)
  y1 = ymin + ((ymax-ymin)*0.03)

#################################
  # Plot Max/Min Hourly 2-5 km UH
#################################
  if (fhr > 0):
    t1 = time.perf_counter()
    print(('Working on Max/Min Hourly 2-5 km UH for '+dom))

    units = 'm${^2}$ s$^{-2}$'
    clevs = [-150,-100,-75,-50,-25,-10,0,10,25,50,75,100,150,200,250,300]
    colorlist = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','#E5E5E5','#E5E5E5','#EEEE00','#EEC900','darkorange','orangered','red','firebrick','mediumvioletred','darkviolet'] 
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    cs_1 = ax1.contourf(lon,lat,uh25_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
    cs_1.cmap.set_under('darkblue')
    cs_1.cmap.set_over('black')
    cbar1 = fig.colorbar(cs_1,ax=ax1,ticks=clevs,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=5)
    ax1.text(.5,1.03,'RRFS_A 1-h Max/Min 2-5 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

    rrfs_plot_utils.convert_and_save_2('compareuh25_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly 2-5 km UH for: '+dom) % t3)

#################################
  # Plot Max Hourly Updraft Speed
#################################
    t1 = time.perf_counter()
    print(('Working on Max Hourly Updraft Speed for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

    units = 'm s$^{-1}$'
    clevs = [0.5,1,2.5,5,7.5,10,12.5,15,20,25,30,35,40,50,75]
    colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia','mediumpurple']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    cs_1 = ax1.contourf(lon,lat,maxuvv_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
    cs_1.cmap.set_under('white')
    cs_1.cmap.set_over('black')
    cbar1 = fig.colorbar(cs_1,ax=ax1,ticks=clevs,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels(clevs)
    cbar1.ax.tick_params(labelsize=5)
    ax1.text(.5,1.03,'RRFS_A 1-h Max 100-1000 mb Updraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

    rrfs_plot_utils.convert_and_save_2('comparemaxuvv_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly Updraft Speed for: '+dom) % t3)

#################################
  # Plot Max Hourly Downdraft Speed
#################################
    t1 = time.perf_counter()
    print(('Working on Max Hourly Downdraft Speed for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

    cs_1 = ax1.contourf(lon,lat,maxdvv_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
    cs_1.cmap.set_under('white')
    cs_1.cmap.set_over('black')
    cbar1 = fig.colorbar(cs_1,ax=ax1,ticks=clevs,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels(clevs)
    cbar1.ax.tick_params(labelsize=5)
    ax1.text(.5,1.03,'RRFS_A 1-h Max 100-1000 mb Downdraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

    rrfs_plot_utils.convert_and_save_2('comparemaxdvv_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly Downdraft Speed for: '+dom) % t3)

#################################
  # Plot Max Hourly 1-km Reflectivity
#################################
    t1 = time.perf_counter()
    print(('Working on Max Hourly 1-km Reflectivity for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

    units='dBz'
    clevs = np.linspace(5,70,14)
    colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    cs_1 = ax1.contourf(lon,lat,maxref1km_1,levels=clevs,cmap=cm,vmin=5,norm=norm,transform=transform)
    cs_1.cmap.set_under('white',alpha=0.)
    cs_1.cmap.set_over('black')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='max')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'RRFS_A 1-h Max 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

    rrfs_plot_utils.convert_and_save_2('comparemaxref1km_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly 1-km Reflectivity for: '+dom) % t3)

#################################
  # Plot Total Cloud Cover
#################################
  t1 = time.perf_counter()
  print(('Working on Total Cloud Cover for '+dom))

  if (fhr > 0):
  # Clear off old plottables but keep all the map info
    cbar1.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = '%'
  clevs = [0,10,20,30,40,50,60,70,80,90,100]
  cm = plt.cm.BuGn
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,tcdc_1,levels=clevs,cmap=cm,norm=norm,transform=transform)
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0)
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Total Cloud Cover ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparetcdc_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Total Cloud Cover for: '+dom) % t3)

#################################
  # Plot Echo Top Height
#################################
  t1 = time.perf_counter()
  print(('Working on Echo Top Height for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'kft'
  clevs = [1,5,10,15,20,25,30,35,40]
  colorlist = ['firebrick','tomato','lightsalmon','goldenrod','#EEEE00','palegreen','mediumspringgreen','limegreen']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,retop_1,levels=clevs,cmap=cm,vmin=1,norm=norm,transform=transform)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('darkgreen')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Echo Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('compareretop_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Echo Top Height for: '+dom) % t3)

#################################
  # Plot Precipitation Rate
#################################
  t1 = time.perf_counter()
  print(('Working on Precipitation Rate for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'mm/hr'
  clevs = [0.001,0.005,0.01,0.05,0.1,0.5,1,2.5,5,10,20,30,50,75,100]
  colorlist = ['chartreuse','limegreen','green','darkgreen','blue','dodgerblue','deepskyblue','cyan','darkred','crimson','orangered','darkorange','goldenrod','gold']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,prate_1,levels=clevs,cmap=cm,vmin=0.001,norm=norm,transform=transform)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('yellow')
  cbar1 = fig.colorbar(cs_1,ax=ax1,ticks=clevs,orientation='horizontal',pad=0.01,shrink=1.0,extend='max')
  cbar1.set_label(units,fontsize=6)
#  cbar1.ax.set_xticklabels([0.001,'',0.01,'',0.05,0.1,0.5,1,2.5,5,10,20,30,50,75,100])
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'RRFS_A Precipitation Rate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('compareprate_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Precipitation Rate for: '+dom) % t3)

######################################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 10 variables for: "+dom) % t3dom)
  plt.clf()

######################################################

def plot_set_11():
  global fig,axes,ax1,keep_ax_lst_1,xextent,yextent,offset,transform

  t1dom = time.perf_counter()
  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
  x1 = xmin + ((xmax-xmin)*0.03)
  y1 = ymin + ((ymax-ymin)*0.03)

#################################
  # Plot Cloud Base Height
#################################
  t1 = time.perf_counter()
  print(('Working on Cloud Base Height for '+dom))

  units = 'kft'
  clevs = [0,0.1,0.5,1,5,10,15,20,25,30,35,40]
#  colorlist = ['firebrick','tomato','salmon','lightsalmon','goldenrod','khaki','gold','yellow','palegreen','mediumspringgreen','lime','limegreen']
  colorlist = ['firebrick','tomato','lightsalmon','goldenrod','khaki','gold','yellow','palegreen','mediumspringgreen','lime','limegreen']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,zbase_1,levels=clevs,cmap=cm,vmin=0,norm=norm,transform=transform)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('darkgreen')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Cloud Base Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparezbase_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Cloud Base Height for: '+dom) % t3)

######################################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 11 variables for: "+dom) % t3dom)
  plt.clf()

######################################################

def plot_set_12():
  global fig,axes,ax1,keep_ax_lst_1,xextent,yextent,offset,transform

  t1dom = time.perf_counter()
  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
  x1 = xmin + ((xmax-xmin)*0.03)
  y1 = ymin + ((ymax-ymin)*0.03)

#################################
  # Plot Cloud Ceiling Height
#################################
  t1 = time.perf_counter()
  print(('Working on Cloud Ceiling Height for '+dom))

  units = 'kft'
  clevs = [0,0.1,0.5,1,5,10,15,20,25,30,35,40]
#  colorlist = ['firebrick','tomato','salmon','lightsalmon','goldenrod','khaki','gold','yellow','palegreen','mediumspringgreen','lime','limegreen']
  colorlist = ['firebrick','tomato','lightsalmon','goldenrod','khaki','gold','yellow','palegreen','mediumspringgreen','lime','limegreen']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,zceil_1,levels=clevs,cmap=cm,vmin=0,norm=norm,transform=transform)
  cs_1.cmap.set_over('white')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Cloud Ceiling Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparezceil_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Cloud Ceiling Height for: '+dom) % t3)

#################################
  # Plot Cloud Top Height
#################################
  t1 = time.perf_counter()
  print(('Working on Cloud Top Height for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'kft'
  clevs = [1,5,10,15,20,25,30,35,40,45,50]
#  colorlist = ['firebrick','tomato','salmon','lightsalmon','goldenrod','yellow','palegreen','mediumspringgreen','lime','limegreen']
  colorlist = ['firebrick','tomato','lightsalmon','goldenrod','khaki','yellow','palegreen','mediumspringgreen','lime','limegreen']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,ztop_1,levels=clevs,cmap=cm,vmin=0,norm=norm,transform=transform)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('darkgreen')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Cloud Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('compareztop_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Cloud Top Height for: '+dom) % t3)

######################################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 12 variables for: "+dom) % t3dom)
  plt.clf()

######################################################

def plot_set_13():
  global fig,axes,ax1,keep_ax_lst_1,xextent,yextent,offset,transform

  t1dom = time.perf_counter()
  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
  x1 = xmin + ((xmax-xmin)*0.03)
  y1 = ymin + ((ymax-ymin)*0.03)

#################################
  # Near-surface smoke
#################################
  t1 = time.perf_counter()
  print(('Working on near-surface smoke for '+dom))

  units = '\u03BC''g/m${^{3}}$'
  clevs = [1,2,4,6,8,12,16,20,25,30,40,60,100,200]
  colorlist = ['paleturquoise','lightskyblue','cornflowerblue','mediumblue','forestgreen','limegreen','lightgreen','khaki','goldenrod','darkorange','orangered','crimson','darkred']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N) 

  cs_1 = ax1.contourf(lon,lat,smoke_1,levels=clevs,cmap=cm,vmin=1,norm=norm,transform=transform)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Near-Surface Smoke ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparesmoke_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot near-surface smoke for: '+dom) % t3)

#################################
  # Plot total column-integrated smoke
#################################
  t1 = time.perf_counter()
  print(('Working on total column-integrated smoke for '+dom))
  
  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'mg/m${^{2}}$'
  clevs = [1,4,7,11,15,20,25,30,40,50,75,150,250,500]
  colorlist = ['paleturquoise','lightskyblue','cornflowerblue','mediumblue','forestgreen','limegreen','lightgreen','khaki','goldenrod','darkorange','orangered','crimson','darkred']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,colsmoke_1,levels=clevs,cmap=cm,vmin=1,norm=norm,transform=transform)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Vertically Integrated Smoke ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparecolsmoke_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot total column-integrated smoke for: '+dom) % t3)

#################################
  # Near-surface dust
#################################
  t1 = time.perf_counter()
  print(('Working on near-surface dust for '+dom))

  units = '\u03BC''g/m${^{3}}$'
  clevs = [1,2,4,6,8,12,16,20,25,30,40,60,100,200]
  colorlist = ['paleturquoise','lightskyblue','cornflowerblue','mediumblue','forestgreen','limegreen','lightgreen','khaki','goldenrod','darkorange','orangered','crimson','darkred']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N) 

  cs_1 = ax1.contourf(lon,lat,dust_1,levels=clevs,cmap=cm,vmin=1,norm=norm,transform=transform)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Near-Surface Dust ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparedust_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot near-surface dust for: '+dom) % t3)

#################################
  # Plot total column-integrated dust
#################################
  t1 = time.perf_counter()
  print(('Working on total column-integrated dust for '+dom))
  
  # Clear off old plottables but keep all the map info
  cbar1.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'mg/m${^{2}}$'
  clevs = [1,4,7,11,15,20,25,30,40,50,75,150,250,500]
  colorlist = ['paleturquoise','lightskyblue','cornflowerblue','mediumblue','forestgreen','limegreen','lightgreen','khaki','goldenrod','darkorange','orangered','crimson','darkred']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.contourf(lon,lat,coldust_1,levels=clevs,cmap=cm,vmin=1,norm=norm,transform=transform)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'RRFS_A Vertically Integrated Dust ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax1.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(x1,xextent,y1,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save_2('comparecoldust_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot total column-integrated dust for: '+dom) % t3)

#################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 13 variables for: "+dom) % t3dom)
  plt.clf()
 
#################################

main()

