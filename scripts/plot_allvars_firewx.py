#!/bin/usr/env python

import grib2io
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
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
fhrm1 = fhr - 1
fhour = str(fhr).zfill(2)
fhour1 = str(fhrm1).zfill(2)
print('fhour '+fhour)

# Forecast valid date/time
itime = ymdh
vtime = rrfs_plot_utils.ndate(itime,int(fhr))

# Define the directory paths to the input files
NAM_DIR = '/lfs/h1/ops/prod/com/nam/v4.2/nam.'+ymd
RRFSFW_DIR = '/lfs/h2/emc/ptmp/emc.lam/para/com/rrfsfw/v1.0.0/rrfsfw.'+ymd+'/'+cyc

# Define the input files
data1 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.firewxnest.hiresf'+fhour+'.tm00.grib2')
data2 = grib2io.open(RRFSFW_DIR+'/rrfsfw.t'+cyc+'z.hires.f0'+fhour+'.grib2')

if (fhr >= 1):
  data1_m1 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.firewxnest.hiresf'+fhour1+'.tm00.grib2')
  data2_m1 = grib2io.open(RRFSFW_DIR+'/rrfsfw.t'+cyc+'z.hires.f0'+fhour1+'.grib2')
  data1_f00 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.firewxnest.hiresf00.tm00.grib2')
  data2_f00 = grib2io.open(RRFSFW_DIR+'/rrfsfw.t'+cyc+'z.hires.f000.grib2')

# Get the lats and lons
msg = data1.select(shortName='HGT', level='500 mb')[0]	# msg is a Grib2Message object
lat,lon,lat_shift,lon_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg)

# Specify plotting domains
domain='firewx'

# Paths to image files
im = image.imread('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/rrfs_graphics/noaa.png')

# colors for difference plots, only need to define once
difcolors = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','white','white','#EEEE00','#EEC900','darkorange','orangered','red']
difcolors3 = ['blue','dodgerblue','turquoise','white','white','#EEEE00','darkorange','red']

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
# Read in all variables and calculate differences #
###################################################
t1a = time.perf_counter()

# Sea level pressure
slp_1 = data1.select(shortName='MSLET',level='mean sea level')[0].data * 0.01
slp_2 = data2.select(shortName='MSLET',level='mean sea level')[0].data * 0.01
slp_dif = slp_2 - slp_1

# 1-h accumulated precipitation
if (fhr > 0):
  qpf1_1 = data1.select(shortName='APCP',timeRangeOfStatisticalProcess=1)[0].data * 0.0393701
  qpf1_2 = data2.select(shortName='APCP',timeRangeOfStatisticalProcess=1)[0].data * 0.0393701
  qpf1_dif = qpf1_2 - qpf1_1

# 2-m temperature
tmp2m_1 = data1.select(shortName='TMP',level='2 m above ground')[0].data
tmp2m_1 = (tmp2m_1 - 273.15)*1.8 + 32.0
tmp2m_2 = data2.select(shortName='TMP',level='2 m above ground')[0].data
tmp2m_2 = (tmp2m_2 - 273.15)*1.8 + 32.0
tmp2m_dif = tmp2m_2 - tmp2m_1

# Surface temperature
tmpsfc_1 = data1.select(shortName='TMP',level='surface')[0].data
tmpsfc_1 = (tmpsfc_1 - 273.15)*1.8 + 32.0
tmpsfc_2 = data2.select(shortName='TMP',level='surface')[0].data
tmpsfc_2 = (tmpsfc_2 - 273.15)*1.8 + 32.0
tmpsfc_dif = tmpsfc_2 - tmpsfc_1

# 2-m dew point temperature
dew2m_1 = data1.select(shortName='DPT',level='2 m above ground')[0].data
dew2m_1 = (dew2m_1 - 273.15)*1.8 + 32.0
dew2m_2 = data2.select(shortName='DPT',level='2 m above ground')[0].data
dew2m_2 = (dew2m_2 - 273.15)*1.8 + 32.0
dew2m_dif = dew2m_2 - dew2m_1

# 2-m relative humidity
rh2m_1 = data1.select(shortName='RH',level='2 m above ground')[0].data
rh2m_2 = data2.select(shortName='RH',level='2 m above ground')[0].data
rh2m_dif = rh2m_2 - rh2m_1

# 10-m wind speed
uwind_1 = data1.select(shortName='UGRD',level='10 m above ground')[0].data * 1.94384
uwind_2 = data2.select(shortName='UGRD',level='10 m above ground')[0].data * 1.94384
vwind_1 = data1.select(shortName='VGRD',level='10 m above ground')[0].data * 1.94384
vwind_2 = data2.select(shortName='VGRD',level='10 m above ground')[0].data * 1.94384
wspd10m_1 = np.sqrt(uwind_1**2 + vwind_1**2)
wspd10m_2 = np.sqrt(uwind_2**2 + vwind_2**2)
wspd10m_dif = wspd10m_2 - wspd10m_1

# Terrain height
terra_1 = data1.select(shortName='HGT',level='surface')[0].data * 3.28084
terra_2 = data2.select(shortName='HGT',level='surface')[0].data * 3.28084
terra_dif = terra_2 - terra_1

# Surface wind gust
gust_1 = data1.select(shortName='GUST',level='surface')[0].data * 1.94384
gust_2 = data2.select(shortName='GUST',level='surface')[0].data * 1.94384
gust_dif = gust_2 - gust_1

# 300 mb Most Unstable CAPE
mucape_1 = data1.select(shortName='CAPE',level='255-0 mb above ground')[0].data
mucape_2 = data2.select(shortName='CAPE',level='255-0 mb above ground')[0].data
mucape_dif = mucape_2 - mucape_1

# 925 mb height and wind
z925_1 = data1.select(shortName='HGT',level='925 mb')[0].data * 0.1
#z925_1 = ndimage.filters.gaussian_filter(z925_1, 6.89)
z925_2 = data2.select(shortName='HGT',level='925 mb')[0].data * 0.1
#z925_2 = ndimage.filters.gaussian_filter(z925_2, 6.89)
z925_dif = z925_2 - z925_1
u925_1 = data1.select(shortName='UGRD',level='925 mb')[0].data * 1.94384
u925_2 = data2.select(shortName='UGRD',level='925 mb')[0].data * 1.94384
v925_1 = data1.select(shortName='VGRD',level='925 mb')[0].data * 1.94384
v925_2 = data2.select(shortName='VGRD',level='925 mb')[0].data * 1.94384
wspd925_1 = np.sqrt(u925_1**2 + v925_1**2)
wspd925_2 = np.sqrt(u925_2**2 + v925_2**2)
wspd925_dif = wspd925_2 - wspd925_1

# 850-mb equivalent potential temperature
t850_1 = data1.select(shortName='TMP',level='850 mb')[0].data
dpt850_1 = data1.select(shortName='DPT',level='850 mb')[0].data
q850_1 = data1.select(shortName='SPFH',level='850 mb')[0].data
tlcl_1 = 56.0 + (1.0/((1.0/(dpt850_1-56.0)) + 0.00125*np.log(t850_1/dpt850_1)))
thetae_1 = t850_1*((1000.0/850.0)**(0.2854*(1.0-(0.28*q850_1))))*np.exp(((3376.0/tlcl_1)-2.54)*q850_1*(1.0+(0.81*q850_1)))
t850_2 = data2.select(shortName='TMP',level='850 mb')[0].data
dpt850_2 = data2.select(shortName='DPT',level='850 mb')[0].data
q850_2 = data2.select(shortName='SPFH',level='850 mb')[0].data
tlcl_2 = 56.0 + (1.0/((1.0/(dpt850_2-56.0)) + 0.00125*np.log(t850_2/dpt850_2)))
thetae_2 = t850_2*((1000.0/850.0)**(0.2854*(1.0-(0.28*q850_2))))*np.exp(((3376.0/tlcl_2)-2.54)*q850_2*(1.0+(0.81*q850_2)))
thetae_dif = thetae_2 - thetae_1

# 850-mb winds
u850_1 = data1.select(shortName='UGRD',level='850 mb')[0].data * 1.94384
u850_2 = data2.select(shortName='UGRD',level='850 mb')[0].data * 1.94384
v850_1 = data1.select(shortName='VGRD',level='850 mb')[0].data * 1.94384
v850_2 = data2.select(shortName='VGRD',level='850 mb')[0].data * 1.94384

# 700-mb omega and relative humidity
omg700_1 = data1.select(shortName='VVEL',level='700 mb')[0].data
omg700_2 = data2.select(shortName='VVEL',level='700 mb')[0].data
rh700_1 = data1.select(shortName='RH',level='700 mb')[0].data
rh700_2 = data2.select(shortName='RH',level='700 mb')[0].data
rh700_dif = rh700_2 - rh700_1

# 500 mb height, wind, vorticity
z500_1 = data1.select(shortName='HGT',level='500 mb')[0].data * 0.1
#z500_1 = ndimage.filters.gaussian_filter(z500_1, 6.89)
z500_2 = data2.select(shortName='HGT',level='500 mb')[0].data * 0.1
#z500_2 = ndimage.filters.gaussian_filter(z500_2, 6.89)
z500_dif = z500_2 - z500_1
vort500_1 = data1.select(shortName='ABSV',level='500 mb')[0].data * 100000
vort500_1 = ndimage.filters.gaussian_filter(vort500_1,1.7225)
vort500_1[vort500_1 > 1000] = 0 # Mask out undefined values on domain edge
vort500_2 = data2.select(shortName='ABSV',level='500 mb')[0].data * 100000
vort500_2 = ndimage.filters.gaussian_filter(vort500_2,1.7225)
vort500_2[vort500_2 > 1000] = 0 # Mask out undefined values on domain edge
u500_1 = data1.select(shortName='UGRD',level='500 mb')[0].data * 1.94384
u500_2 = data2.select(shortName='UGRD',level='500 mb')[0].data * 1.94384
v500_1 = data1.select(shortName='VGRD',level='500 mb')[0].data * 1.94384
v500_2 = data2.select(shortName='VGRD',level='500 mb')[0].data * 1.94384

# 250 mb winds
u250_1 = data1.select(shortName='UGRD',level='250 mb')[0].data * 1.94384
u250_2 = data2.select(shortName='UGRD',level='250 mb')[0].data * 1.94384
v250_1 = data1.select(shortName='VGRD',level='250 mb')[0].data * 1.94384
v250_2 = data2.select(shortName='VGRD',level='250 mb')[0].data * 1.94384
wspd250_1 = np.sqrt(u250_1**2 + v250_1**2)
wspd250_2 = np.sqrt(u250_2**2 + v250_2**2)
wspd250_dif = wspd250_2 - wspd250_1

# Precipitable water
pw_1 = data1.select(shortName='PWAT',level='entire atmosphere (considered as a single layer)')[0].data * 0.0393701
pw_2 = data2.select(shortName='PWAT',level='entire atmosphere (considered as a single layer)')[0].data * 0.0393701
pw_dif = pw_2 - pw_1

# Percent of frozen precipitation
pofp_1 = data1.select(shortName='CPOFP')[0].data
pofp_2 = data2.select(shortName='CPOFP')[0].data
pofp_1[pofp_1 < 0] = 0 # Mask out negative undefined values
pofp_2[pofp_2 < 0] = 0 # Mask out negative undefined values
pofp_dif = pofp_2 - pofp_1

# Snow depth
snow_1 = data1.select(shortName='SNOD')[0].data * 39.3701
snow_2 = data2.select(shortName='SNOD')[0].data * 39.3701
snow_dif = snow_2 - snow_1
if (fhr > 0):   # Do not make snow depth from f00 for forecast hour 0
  snowf00_1 = data1_f00.select(shortName='SNOD')[0].data * 39.3701
  snow0_1 = snow_1 - snowf00_1
  snowf00_2 = data2_f00.select(shortName='SNOD')[0].data * 39.3701
  snow0_2 = snow_2 - snowf00_2
  snow0_dif = snow0_2 - snow0_1

# WEASD
weasd_1 = data1.select(shortName='WEASD')[0].data / 2.54
weasd_2 = data2.select(shortName='WEASD')[0].data / 2.54
weasd_dif = weasd_2 - weasd_1

# 1-h accumulated WEASD
if (fhr > 0):
  weasd_1 = data1.select(shortName='WEASD')[0].data / 2.54
  weasd_2 = data2.select(shortName='WEASD')[0].data / 2.54
  weasdm1_1 = data1_m1.select(shortName='WEASD')[0].data / 2.54
  weasdm1_2 = data2_m1.select(shortName='WEASD')[0].data / 2.54
  weasd1_1 = weasd_1 - weasdm1_1
  weasd1_2 = weasd_2 - weasdm1_2
  weasd1_dif = weasd1_2 - weasd1_1

# PBL height
hpbl_1 = data1.select(shortName='HPBL')[0].data
hpbl_2 = data2.select(shortName='HPBL')[0].data
hpbl_dif = hpbl_2 - hpbl_1

# PBL height based on Richardson Number 
hgtpbl_1 = data1.select(shortName='HGT',level='planetary boundary layer')[0].data
hgtpbl_2 = data2.select(shortName='HGT',level='planetary boundary layer')[0].data
hgtpbl_dif = hgtpbl_2 - hgtpbl_1

# Total column condensate
cond_1 = data1.select(shortName='TCOLC')[0].data
cond_2 = data2.select(shortName='TCOLC')[0].data
cond_dif = cond_2 - cond_1

# 1-km reflectivity
ref1km_1 = data1.select(shortName='REFD',level='1000 m above ground')[0].data
ref1km_2 = data2.select(shortName='REFD',level='1000 m above ground')[0].data
ref1km_1b = np.where(ref1km_1 > 20, 1, 0)
ref1km_2b = np.where(ref1km_2 > 20, 1, 0)
ref1km_both = ref1km_1b + ref1km_2b

# Composite reflectivity
refc_1 = data1.select(shortName='REFC')[0].data
refc_2 = data2.select(shortName='REFC')[0].data
refc_1b = np.where(refc_1 > 20, 1, 0)
refc_2b = np.where(refc_2 > 20, 1, 0)
refc_both = refc_1b + refc_2b

if (fhr > 0):
# Max Hourly 2-5 km Updraft Helicity
  uh25_1 = data1.select(shortName='MXUPHL',level='5000-2000 m above ground')[0].data
  uh25_2 = data2.select(shortName='MXUPHL',level='5000-2000 m above ground')[0].data
  uh25_1[uh25_1 < 10] = 0
  uh25_2[uh25_2 < 10] = 0
  uh25_dif = uh25_2 - uh25_1

# Max Hourly Updraft Speed
  maxuvv_1 = data1.select(shortName='MAXUVV')[0].data
  maxuvv_2 = data2.select(shortName='MAXUVV')[0].data
  maxuvv_dif = maxuvv_2 - maxuvv_1

# Max Hourly Downdraft Speed
  maxdvv_1 = data1.select(shortName='MAXDVV')[0].data * -1
  maxdvv_2 = data2.select(shortName='MAXDVV')[0].data * -1
  maxdvv_dif = maxdvv_2 - maxdvv_1

# Max Hourly 1-km AGL reflectivity
  maxref1km_1 = data1.select(shortName='MAXREF',level='1000 m above ground')[0].data
  maxref1km_2 = data2.select(shortName='MAXREF',level='1000 m above ground')[0].data
  maxref1km_1b = np.where(maxref1km_1 > 20, 1, 0)
  maxref1km_2b = np.where(maxref1km_2 > 20, 1, 0)
  maxref1km_both = maxref1km_1b + maxref1km_2b

# Max Hourly 10-m Wind
  maxuw_1 = data1.select(shortName='MAXUW')[0].data * 1.94384
  maxvw_1 = data1.select(shortName='MAXVW')[0].data * 1.94384
  maxwind_1 = np.sqrt(maxuw_1**2 + maxvw_1**2)
  maxwind_2 = data2.select(shortName='WIND')[0].data * 1.94384
  maxwind_dif = maxwind_2 - maxwind_1

# Min Hourly 2-m RH
  minrh_1 = data1.select(shortName='MINRH')[0].data
  minrh_2 = data2.select(shortName='MINRH')[0].data
  minrh_dif = minrh_2 - minrh_1

# Haines index
hindex_1 = data1.select(shortName='HINDEX')[0].data
hindex_2 = data2.select(shortName='HINDEX')[0].data
hindex_dif = hindex_2 - hindex_1

# Transport wind
utrans_1 = data1.select(shortName='UGRD',level='planetary boundary layer')[0].data * 1.94384
vtrans_1 = data1.select(shortName='VGRD',level='planetary boundary layer')[0].data * 1.94384
utrans_2 = data2.select(shortName='UGRD',level='planetary boundary layer')[0].data * 1.94384
vtrans_2 = data2.select(shortName='VGRD',level='planetary boundary layer')[0].data * 1.94384
trans_1 = np.sqrt(utrans_1**2 + vtrans_1**2)
trans_2 = np.sqrt(utrans_2**2 + vtrans_2**2)
trans_dif = trans_2 - trans_1

# Ventilation rate
vrate_1 = data1.select(shortName='VRATE')[0].data
vrate_2 = data2.select(shortName='VRATE')[0].data
vrate_dif = vrate_2 - vrate_1


t2a = time.perf_counter()
t3a = round(t2a-t1a, 3)
print(("%.3f seconds to read all messages") % t3a)

#-------------------------------------------------------#

def main():

  global dom
  dom = domain
  print(('Working on '+dom))

  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,transform,cen_lat,cen_lon
  fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,transform,cen_lat,cen_lon = create_figure()

  # Split plots into 2 sets with multiprocessing
  sets = [1,2]
  pool = MyPool(len(sets))
  pool.map(plot_sets,sets)

#######################################
#    SET UP FIGURE FOR EACH DOMAIN    #
#######################################

def create_figure():

  # Map corners for each domain
  if dom == 'firewx':
    cen_lat = float(sys.argv[3])
    cen_lon = float(sys.argv[4])
    llcrnrlon = cen_lon - 3.5
    llcrnrlat = cen_lat - 2.5
    urcrnrlon = cen_lon + 3.5
    urcrnrlat = cen_lat + 2.5

  # Create figure and axes instances
  fig = plt.figure(figsize=(9,8))           
  gs = GridSpec(9,8,wspace=0.0,hspace=0.0)

  # Define where Cartopy maps are located
  cartopy.config['data_dir'] = '/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python/NaturalEarth'
  back_res='50m'
  back_img='off'

  # set up the map background with cartopy
  extent = [llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat]
  myproj=ccrs.LambertConformal(central_longitude=cen_lon, central_latitude=cen_lat,
                          false_easting=0.0, false_northing=0.0,
                          secant_latitudes=None, standard_parallels=None,globe=None)
  ax1 = fig.add_subplot(gs[0:4,0:4], projection=myproj)
  ax2 = fig.add_subplot(gs[0:4,4:], projection=myproj)
  ax3 = fig.add_subplot(gs[5:,1:7], projection=myproj)
  ax1.set_extent(extent)
  ax2.set_extent(extent)
  ax3.set_extent(extent)
  axes = [ax1, ax2, ax3]

  fline_wd = 0.6  # line width
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

  # add counties
  reader = shpreader.Reader('/apps/prod/python-modules/3.8.6/intel/19.1.3.304/lib/python3.8/site-packages/cartopy/data/shapefiles/USGS/shp/countyl010g.shp')
  counties = list(reader.geometries())
  COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

  # All lat lons are earth relative, so setup the associated projection correct for that data
  transform = ccrs.PlateCarree()

  # high-resolution background images
  if back_img=='on':
     img = plt.imread('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python/NaturalEarth/raster_files/NE1_50M_SR_W.tif')
     ax1.imshow(img, origin='upper', transform=transform)
     ax2.imshow(img, origin='upper', transform=transform)
     ax3.imshow(img, origin='upper', transform=transform)

  ax1.add_feature(COUNTIES, facecolor='none',edgecolor='gray')
  ax1.add_feature(lakes)
  ax1.add_feature(states)
  ax1.add_feature(coastline)
  ax2.add_feature(COUNTIES, facecolor='none',edgecolor='gray')
  ax2.add_feature(lakes)
  ax2.add_feature(states)
  ax2.add_feature(coastline)
  ax3.add_feature(COUNTIES, facecolor='none',edgecolor='gray')
  ax3.add_feature(lakes)
  ax3.add_feature(states)
  ax3.add_feature(coastline)

  # Map/figure has been set up here, save axes instances for use again later
  keep_ax_lst_1 = ax1.get_children()[:]
  keep_ax_lst_2 = ax2.get_children()[:]
  keep_ax_lst_3 = ax3.get_children()[:]

  return fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,transform,cen_lat,cen_lon

################################################################################

def plot_sets(set):
  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,transform,cen_lat,cen_lon

# Add print to see if dom is being passed in
  print(('plot_sets dom variable '+dom))

  if set == 1:
    plot_set_1()
  elif set == 2:
    plot_set_2()

################################################################################

def plot_set_1():
  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,transform,cen_lat,cen_lon

  t1dom = time.perf_counter()
  cenlat = str(cen_lat)
  cenlon = str(cen_lon)
  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
  xextent = xmin + ((xmax-xmin)*0.15)
  yextent = ymin + ((ymax-ymin)*0.15)
  offset=0.25

################################
  # Plot SLP
################################
  t1 = time.perf_counter()
  print(('Working on slp for '+dom))

  # Wind barb density settings
  skip = 30
  barblength = 4

  units = 'mb'
  clevs = [976,980,984,988,992,996,1000,1004,1008,1012,1016,1020,1024,1028,1032,1036,1040,1044,1048,1052]
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = plt.cm.Spectral_r
  cmdif = matplotlib.colors.ListedColormap(difcolors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs1_a = ax1.pcolormesh(lon_shift,lat_shift,slp_1,transform=transform,cmap=cm,norm=norm)  
  cbar1 = fig.colorbar(cs1_a,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  cs1_b = ax1.contour(lon_shift,lat_shift,slp_1,np.arange(940,1060,4),colors='black',linewidths=0.1,transform=transform)
  plt.clabel(cs1_b,np.arange(940,1060,4),inline=1,fmt='%d',fontsize=6)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_1[::skip,::skip],vwind_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)

  rrfs_plot_utils.plt_highs_and_lows(lon_shift,lat_shift,slp_1,xmin,xmax,ymin,ymax,offset,ax1,transform,mode='reflect',window=400)

  ax1.text(.5,1.03,'NAMFW SLP ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs2_a = ax2.pcolormesh(lon_shift,lat_shift,slp_2,transform=transform,cmap=cm,norm=norm)  
  cbar2 = fig.colorbar(cs2_a,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  cs2_b = ax2.contour(lon_shift,lat_shift,slp_2,np.arange(940,1060,4),colors='black',linewidths=0.1,transform=transform)
  plt.clabel(cs2_b,np.arange(940,1060,4),inline=1,fmt='%d',fontsize=6)
  ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_2[::skip,::skip],vwind_2[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)

  rrfs_plot_utils.plt_highs_and_lows(lon_shift,lat_shift,slp_2,xmin,xmax,ymin,ymax,offset,ax2,transform,mode='reflect',window=400)

  ax2.text(.5,1.03,'RRFSFW SLP ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,slp_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=5)
  ax3.text(.5,1.03,'RRFSFW - NAMFW SLP ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('compareslp_'+dom+'_f'+fhour+'.png')
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
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '\xb0''F'
  clevs = np.linspace(-16,134,26)
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = rrfs_plot_utils.cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,tmp2m_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'NAMFW 2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,tmp2m_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('white')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFSFW 2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))       
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,tmp2m_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW 2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2')) 
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('compare2mt_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 2mt for: '+dom) % t3)

#################################
  # Plot SFCT
#################################
  t1 = time.perf_counter()
  print(('Working on tsfc for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '\xb0''F'
  clevs = np.linspace(-16,134,26)
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = rrfs_plot_utils.cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,tmpsfc_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'NAMFW Surface Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,tmpsfc_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('white')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFSFW Surface Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))       
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,tmpsfc_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW Surface Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2')) 
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('comparetsfc_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot tsfc for: '+dom) % t3)

#################################
  # Plot 2-m Dew Point
#################################
  t1 = time.perf_counter()
  print(('Working on 2mdew for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '\xb0''F'
  clevs = np.linspace(-10,80,19)
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = rrfs_plot_utils.cmap_q2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,dew2m_1,transform=transform,cmap=cm,norm=norm)
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'NAMFW 2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,dew2m_2,transform=transform,cmap=cm,norm=norm)
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFSFW 2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,dew2m_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW 2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('compare2mdew_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 2mdew for: '+dom) % t3)

#################################
  # Plot 2-m Relative Humidity
#################################
  t1 = time.perf_counter()
  print(('Working on 2m RH for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '%'
  clevs = [50,60,70,80,90,100]
  clevsdif = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]
  cm = plt.cm.BuGn
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,rh2m_1,transform=transform,cmap=cm,vmin=50,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs)
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'NAMFW 2-m RH ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,rh2m_2,transform=transform,cmap=cm,vmin=50,norm=norm)
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs)
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFSFW 2-m RH ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,rh2m_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW 2-m RH ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('compare2mrh_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 2m RH for: '+dom) % t3)

#################################
  # Plot 10-m WSPD
#################################
  t1 = time.perf_counter()
  print(('Working on 10mwspd for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  # Wind barb density settings
  skip = 30
  barblength = 4

  units = 'kts'
  clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,wspd10m_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_1[::skip,::skip],vwind_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax1.text(.5,1.03,'NAMFW 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
    
  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,wspd10m_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('black')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_2[::skip,::skip],vwind_2[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax2.text(.5,1.03,'RRFSFW 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,wspd10m_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label('kts',fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW 10-m Wind Speed (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))       
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('compare10mwind_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 10mwspd for: '+dom) % t3)

#################################
  # Plot Terrain with 10-m WSPD
#################################
  t1 = time.perf_counter()
  print(('Working on Terrain for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)
  
  units = 'ft'
  clevs = [1,250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000,4250,4500,4750,5000,5250,5500,5750,6000,6250,6500,6750,7000,7250,7500,7750,8000,8250,8500,8750,9000,9250,9500,9750,10000]
  clevsdif = [-300,-250,-200,-150,-100,-50,0,50,100,150,200,250,300]
  cm = rrfs_plot_utils.cmap_terra()
  cmdif = matplotlib.colors.ListedColormap(difcolors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,terra_1,transform=transform,cmap=cm,vmin=1,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('ghostwhite')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_1[::skip,::skip],vwind_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax1.text(.5,1.03,'NAMFW Terrain Height ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,terra_2,transform=transform,cmap=cm,vmin=1,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('ghostwhite')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_2[::skip,::skip],vwind_2[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax2.text(.5,1.03,'RRFSFW Terrain Height ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,terra_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=5)
  ax3.text(.5,1.03,'RRFSFW - NAMFW Terrain Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('compareterra_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Terrain for: '+dom) % t3)

#################################
  # Plot surface wind gust
#################################
  t1 = time.perf_counter()
  print(('Working on surface wind gust for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'kts'
  clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,gust_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.05,'NAMFW Surface Wind Gust ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,gust_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('black')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.05,'RRFSFW Surface Wind Gust ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,gust_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW Surface Wind Gust ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('comparegust_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot surface wind gust for: '+dom) % t3)

#################################
  # Plot Most Unstable CAPE/CIN
#################################
  t1 = time.perf_counter()
  print(('Working on mucapecin for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'J/kg'
  clevs = [100,250,500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
  clevs2 = [-2000,-500,-250,-100,-25]
  clevsdif = [-2000,-1500,-1000,-500,-250,-100,0,100,250,500,1000,1500,2000]
  colorlist = ['blue','dodgerblue','cyan','mediumspringgreen','#FAFAD2','#EEEE00','#EEC900','darkorange','crimson','darkred','darkviolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,mucape_1,transform=transform,cmap=cm,vmin=100,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=4)
#  cs_1b = ax1.contourf(lon_shift,lat_shift,mucin_1,clevs2,colors='none',hatches=['**','++','////','..'],transform=transform)
  ax1.text(.5,1.05,'NAMFW Most Unstable CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,mucape_2,transform=transform,cmap=cm,vmin=100,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=4)
#  cs_2b = ax2.contourf(lon_shift,lat_shift,mucin_2,clevs2,colors='none',hatches=['**','++','////','..'],transform=transform)
  ax2.text(.5,1.05,'RRFSFW Most Unstable CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,mucape_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevsdif,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=4)
  ax3.text(.5,1.03,'RRFSFW - NAMFW Most Unstable CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('comparemucape_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot mucapecin for: '+dom) % t3)

#################################
  # Plot 925-mb Height and Wind
#################################
  t1 = time.perf_counter()
  print(('Working on 925 mb Hgt/Wind for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'kts'
  clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
  clevsdif = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,wspd925_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u925_1[::skip,::skip],v925_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
#  cs1_b = ax1.contour(lon_shift,lat_shift,z925_1,np.arange(60,90,3),colors='black',linewidths=1,transform=transform)
#  plt.clabel(cs1_b,np.arange(60,90,3),inline_spacing=1,fmt='%d',fontsize=5)
  ax1.text(.5,1.03,'NAMFW 925 mb Heights ('+units+') and Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,wspd925_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('black')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u925_2[::skip,::skip],v925_2[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  cs2_b = ax2.contour(lon_shift,lat_shift,z925_2,np.arange(60,90,3),colors='black',linewidths=1,transform=transform)
#  plt.clabel(cs2_b,np.arange(60,90,3),inline_spacing=1,fmt='%d',fontsize=5)
  ax2.text(.5,1.03,'RRFSFW 925 mb Heights ('+units+') and Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,z925_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label('dam',fontsize=6) 
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW 925 mb Heights (dam) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('compare925_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 925 mb Hgt/Wind for: '+dom) % t3)

#################################
  # Plot 850-mb THETAE
#################################
  t1 = time.perf_counter()
  print(('Working on 850 mb Theta-e for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'K'
  clevs = np.linspace(270,360,31)
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = rrfs_plot_utils.cmap_t850()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,thetae_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=[270,276,282,288,294,300,306,312,318,324,330,336,342,348,354,360],extend='both')
  cbar1.set_label(units,fontsize=6)   
  cbar1.ax.tick_params(labelsize=4)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u850_1[::skip,::skip],v850_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax1.text(.5,1.03,'NAMFW 850 mb $\Theta$e ('+units+') and Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,thetae_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('white')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,ticks=[270,276,282,288,294,300,306,312,318,324,330,336,342,348,354,360],extend='both')
  cbar2.set_label(units,fontsize=6)   
  cbar2.ax.tick_params(labelsize=4)
  ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u850_2[::skip,::skip],v850_2[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax2.text(.5,1.03,'RRFSFW 850 mb $\Theta$e ('+units+') and Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
    
  cs = ax3.pcolormesh(lon_shift,lat_shift,thetae_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)   
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW 850 mb $\Theta$e ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('compare850t_'+dom+'_f'+fhour+'.png')
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
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '%'
  clevs = [50,60,70,80,90,100]
  clevsw = [-100,-5]
  clevsdif = [-30,-25,-20,-15,-10,-5,-0,5,10,15,20,25,30]
  colors = ['blue']
  cm = plt.cm.BuGn
  cmw = matplotlib.colors.ListedColormap(colors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normw = matplotlib.colors.BoundaryNorm(clevsw, cmw.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs1_a = ax1.pcolormesh(lon_shift,lat_shift,rh700_1,transform=transform,cmap=cm,vmin=50,norm=norm)
  cs1_a.cmap.set_under('white',alpha=0.)
  cbar1 = fig.colorbar(cs1_a,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs)
  cbar1.set_label(units,fontsize=6) 
  cbar1.ax.tick_params(labelsize=6)
  cs1_b = ax1.pcolormesh(lon_shift,lat_shift,omg700_1,transform=transform,cmap=cmw,vmax=-5,norm=normw)
  cs1_b.cmap.set_over('white',alpha=0.)
  ax1.text(.5,1.03,'NAMFW 700 mb $\omega$ (rising motion in blue) and RH ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs2_a = ax2.pcolormesh(lon_shift,lat_shift,rh700_2,transform=transform,cmap=cm,vmin=50,norm=norm)
  cs2_a.cmap.set_under('white',alpha=0.)
  cbar2 = fig.colorbar(cs2_a,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs)
  cbar2.set_label(units,fontsize=6) 
  cbar2.ax.tick_params(labelsize=6)
  cs2_b = ax2.pcolormesh(lon_shift,lat_shift,omg700_2,transform=transform,cmap=cmw,vmax=-5,norm=normw)
  cs2_b.cmap.set_over('white',alpha=0.)
  ax2.text(.5,1.03,'RRFSFW 700 mb $\omega$ (rising motion in blue) and RH ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,rh700_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW 700 mb RH ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('compare700_'+dom+'_f'+fhour+'.png')
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
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'x10${^5}$ s${^{-1}}$'
  vortlevs = [16,20,24,28,32,36,40]
  clevsdif = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  colorlist = ['yellow','gold','goldenrod','orange','orangered','red']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(vortlevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs1_a = ax1.pcolormesh(lon_shift,lat_shift,vort500_1,transform=transform,cmap=cm,norm=norm)
  cs1_a.cmap.set_under('white')
  cs1_a.cmap.set_over('darkred')
  cbar1 = fig.colorbar(cs1_a,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=vortlevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u500_1[::skip,::skip],v500_1[::skip,::skip],length=barblength,linewidth=0.5,color='steelblue',transform=transform)
  cs1_b = ax1.contour(lon_shift,lat_shift,z500_1,np.arange(486,600,3),colors='black',linewidths=1,transform=transform)
  plt.clabel(cs1_b,np.arange(486,600,3),inline_spacing=1,fmt='%d',fontsize=5)
  ax1.text(.5,1.03,'NAMFW 500 mb Heights (dam), Winds (kts), and $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs2_a = ax2.pcolormesh(lon_shift,lat_shift,vort500_2,transform=transform,cmap=cm,norm=norm)
  cs2_a.cmap.set_under('white')
  cs2_a.cmap.set_over('darkred')
  cbar2 = fig.colorbar(cs2_a,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=vortlevs,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u500_2[::skip,::skip],v500_2[::skip,::skip],length=barblength,linewidth=0.5,color='steelblue',transform=transform)
  cs2_b = ax2.contour(lon_shift,lat_shift,z500_2,np.arange(486,600,3),colors='black',linewidths=1,transform=transform)
  plt.clabel(cs2_b,np.arange(486,600,3),inline_spacing=1,fmt='%d',fontsize=5)
  ax2.text(.5,1.03,'RRFSFW 500 mb Heights (dam), Winds (kts), and $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,z500_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label('dam',fontsize=6) 
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW 500 mb Heights (dam) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('compare500_'+dom+'_f'+fhour+'.png')
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
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'kts'
  clevs = [50,60,70,80,90,100,110,120,130,140,150]
  clevsdif = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]
  colorlist = ['turquoise','deepskyblue','dodgerblue','#1874CD','blue','beige','khaki','peru','brown','crimson']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,wspd250_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('red')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u250_1[::skip,::skip],v250_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax1.text(.5,1.03,'NAMFW 250 mb Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,wspd250_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('red')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u250_2[::skip,::skip],v250_2[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax2.text(.5,1.03,'RRFSFW 250 mb Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,wspd250_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6) 
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW 250 mb Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('compare250wind_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 250 mb WIND for: '+dom) % t3)

#################################
  # Plot PW
#################################
  t1 = time.perf_counter()
  print(('Working on PW for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'in'
  clevs = [0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25]
  clevsdif = [-1.25,-1,-.75,-.5,-.25,-.1,0.,.1,.25,.50,.75,1,1.25]
  colorlist = ['lightsalmon','khaki','palegreen','cyan','turquoise','cornflowerblue','mediumslateblue','darkorchid','deeppink']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,pw_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('hotpink')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'NAMFW Precipitable Water ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,pw_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('hotpink')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFSFW Precipitable Water ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,pw_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevsdif,extend='both')
  cbar3.set_label(units,fontsize=6) 
  cbar3.ax.tick_params(labelsize=4)
  ax3.text(.5,1.03,'RRFSFW - NAMFW Precipitable Water ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('comparepw_'+dom+'_f'+fhour+'.png')
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
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '%'
  clevs = [10,20,30,40,50,60,70,80,90,100]
  clevsdif = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]
  colorlist = ['blue','dodgerblue','deepskyblue','mediumspringgreen','khaki','sandybrown','salmon','crimson','maroon']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,pofp_1,transform=transform,cmap=cm,vmin=10,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs)
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'NAMFW Percent of Frozen Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,pofp_2,transform=transform,cmap=cm,vmin=10,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs)
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFSFW Percent of Frozen Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,pofp_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevsdif,extend='both')
  cbar3.set_label(units,fontsize=6) 
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW Percent of Frozen Precipitaion ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('comparepofp_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot PERCENT FROZEN PRECIP for: '+dom) % t3)

#################################
  # Plot Total QPF
#################################
#  t1 = time.perf_counter()
#  print(('Working on qpf for '+dom))

  # Clear off old plottables but keep all the map info
#  cbar1.remove()
#  cbar2.remove()
#  cbar3.remove()
#  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
#  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
#  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

#  units = 'in'
#  clevs = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,7,10,15,20]
#  clevsdif = [-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5]
#  colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','deepskyblue','cyan','mediumpurple','mediumorchid','darkmagenta','darkred','crimson','orangered','darkorange','goldenrod','gold','yellow']  
#  cm = matplotlib.colors.ListedColormap(colorlist)
#  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
#  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)
   
#  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,qpf_1,transform=transform,cmap=cm,vmin=0.01,norm=norm)
#  cs_1.cmap.set_under('white',alpha=0.)
#  cs_1.cmap.set_over('pink')
#  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.1,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
#  cbar1.set_label(units,fontsize=6)
#  cbar1.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
#  cbar1.ax.tick_params(labelsize=6)
#  ax1.text(.5,1.03,'NAMFW '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
#  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

#  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,qpf_2,transform=transform,cmap=cm,vmin=0.01,norm=norm)
#  cs_2.cmap.set_under('white',alpha=0.)
#  cs_2.cmap.set_over('pink')
#  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.1,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
#  cbar2.set_label(units,fontsize=6)
#  cbar2.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
#  cbar2.ax.tick_params(labelsize=6)
#  ax2.text(.5,1.03,'RRFSFW '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
#  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

#  cs = ax3.pcolormesh(lon_shift,lat_shift,qpf_dif,transform=transform,cmap=cmdif,norm=normdif)
#  cs.cmap.set_under('darkblue')
#  cs.cmap.set_over('darkred')
#  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
#  cbar3.set_label(units,fontsize=6)
#  cbar3.ax.tick_params(labelsize=6)
#  ax3.text(.5,1.03,'RRFSFW - NAMFW '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
#  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

#  rrfs_plot_utils.compress_and_save('compareqpf_'+dom+'_f'+fhour+'.png')
#  t2 = time.perf_counter()
#  t3 = round(t2-t1, 3)
#  print(('%.3f seconds to plot qpf for: '+dom) % t3)

#################################
  # Plot QPF1
#################################
  if (fhr > 0):
    t1 = time.perf_counter()
    print(('Working on 1-h accumulated precip for '+dom))

    # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
    rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 'in'
    clevs = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,7,10,15,20]
    clevsdif = [-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5]
    colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','deepskyblue','cyan','mediumpurple','mediumorchid','darkmagenta','darkred','crimson','orangered','darkorange','goldenrod','gold','yellow']  
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)
   
    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,qpf1_1,transform=transform,cmap=cm,vmin=0.01,norm=norm)
    cs_1.cmap.set_under('white',alpha=0.)
    cs_1.cmap.set_over('pink')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'NAMFW 1-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,qpf1_2,transform=transform,cmap=cm,vmin=0.01,norm=norm)
    cs_2.cmap.set_under('white',alpha=0.)
    cs_2.cmap.set_over('pink')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFSFW 1-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs = ax3.pcolormesh(lon_shift,lat_shift,qpf1_dif,transform=transform,cmap=cmdif,norm=normdif)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('darkred')
    cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar3.set_label(units,fontsize=6)
    cbar3.ax.tick_params(labelsize=6)
    ax3.text(.5,1.03,'RRFSFW - NAMFW 1-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))         
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.compress_and_save('compareqpf1_'+dom+'_f'+fhour+'.png')
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot 1-h accumulated precip for: '+dom) % t3)

#################################
  # Plot snow depth
#################################
  t1 = time.perf_counter()
  print(('Working on snow depth for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'in'
  clevs = [0.1,1,2,3,6,9,12,18,24,36,48]
  clevsdif = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  cm = rrfs_plot_utils.ncl_perc_11Lev()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N) 
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N) 
 
  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,snow_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'NAMFW Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,snow_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFSFW Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,snow_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))         
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('comparesnow_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot snow depth for: '+dom) % t3)

#################################
  # Plot snow depth from f00
#################################
  if (fhr > 0):
    t1 = time.perf_counter()
    print(('Working on snow depth from f00 for '+dom))

    # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
    rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 'in'
    clevs = [-6,-4,-3,-2,-1,-0.5,0,0.5,1,2,3,4,6]
    clevsdif = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
    colorlist = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','white','white','#EEEE00','#EEC900','darkorange','orangered','red']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,snow0_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_under('darkblue')
    cs_1.cmap.set_over('darkred')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels(clevs)
    cbar1.ax.tick_params(labelsize=5)
    ax1.text(.5,1.03,'NAMFW Snow Depth from f00 ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,snow0_2,transform=transform,cmap=cm,norm=norm)
    cs_2.cmap.set_under('darkblue')
    cs_2.cmap.set_over('darkred')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.set_xticklabels(clevs)
    cbar2.ax.tick_params(labelsize=5)
    ax2.text(.5,1.03,'RRFSFW Snow Depth from f00 ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs = ax3.pcolormesh(lon_shift,lat_shift,snow0_dif,transform=transform,cmap=cmdif,norm=normdif)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('darkred')
    cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar3.set_label(units,fontsize=6)
    cbar3.ax.tick_params(labelsize=6)
    ax3.text(.5,1.03,'RRFSFW - NAMFW Snow Depth from f00 ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.compress_and_save('comparesnow0_'+dom+'_f'+fhour+'.png')
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot snow depth from f00 for: '+dom) % t3)

#################################
  # Plot snowfall
#################################
  t1 = time.perf_counter()
  print(('Working on WEASD for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'in'
  clevs = [0.1,1,2,3,6,9,12,18,24,36,48]
  clevsdif = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  cm = rrfs_plot_utils.ncl_perc_11Lev()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N) 
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N) 
 
  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,weasd_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'NAMFW Snowfall ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,weasd_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFSFW Snowfall ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,weasd_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW Snowfall ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))         
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('compareweasd_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot snowfall for: '+dom) % t3)

#################################
  # Plot 1-h WEASD
#################################
  if (fhr > 0):
    t1 = time.perf_counter()
    print(('Working on 1-h WEASD for '+dom))

    # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
    rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 'in'
    clevs = [0.1,0.25,0.5,0.75,1,1.5,2,3,4,5,6]
    clevsdif = [-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5]
    cm = rrfs_plot_utils.ncl_perc_11Lev()
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,weasd1_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_under('white')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels(clevs)
    cbar1.ax.tick_params(labelsize=5)
    ax1.text(.5,1.03,'NAMFW 1-h Snowfall ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,weasd1_2,transform=transform,cmap=cm,norm=norm)
    cs_2.cmap.set_under('white')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.set_xticklabels(clevs)
    cbar2.ax.tick_params(labelsize=5)
    ax2.text(.5,1.03,'RRFSFW 1-h Snowfall ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs = ax3.pcolormesh(lon_shift,lat_shift,weasd1_dif,transform=transform,cmap=cmdif,norm=normdif)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('darkred')
    cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar3.set_label(units,fontsize=6)
    cbar3.ax.tick_params(labelsize=6)
    ax3.text(.5,1.03,'RRFSFW - NAMFW 1-h Snowfall ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.compress_and_save('compareweasd1_'+dom+'_f'+fhour+'.png')
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot 1-h WEASD for: '+dom) % t3)

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 1 variables for: "+dom) % t3dom)
  plt.clf()


################################################################################

def plot_set_2():
  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,transform,cen_lat,cen_lon

  t1dom = time.perf_counter()
  cenlat = str(cen_lat)
  cenlon = str(cen_lon)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
  xextent = xmin + ((xmax-xmin)*0.15)
  yextent = ymin + ((ymax-ymin)*0.15)

#################################
  # Plot PBL height
#################################
  t1 = time.perf_counter()
  print(('Working on PBL height for '+dom))

  units = 'm'
  clevs = [50,100,250,500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
  clevsdif = [-1800,-1500,-1200,-900,-600,-300,0,300,600,900,1200,1500,1800]
  colorlist= ['gray','blue','dodgerblue','cyan','mediumspringgreen','#FAFAD2','#EEEE00','#EEC900','darkorange','crimson','darkred','darkviolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  cmdif = matplotlib.colors.ListedColormap(difcolors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,hpbl_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,ticks=clevs,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=4)
  ax1.text(.5,1.03,'NAMFW PBL Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,hpbl_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('black')
  cbar2 = fig.colorbar(cs_2,ax=ax2,ticks=clevs,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=4)
  ax2.text(.5,1.03,'RRFSFW PBL Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,hpbl_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW PBL Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('comparehpbl_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot PBL height for: '+dom) % t3)

#################################
  # Plot PBL height based on Richardson number
#################################
  t1 = time.perf_counter()
  print(('Working on PBL height based on Richardson number for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'm'
  clevs = [50,100,250,500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
  clevsdif = [-1800,-1500,-1200,-900,-600,-300,0,300,600,900,1200,1500,1800]
  colorlist= ['gray','blue','dodgerblue','cyan','mediumspringgreen','#FAFAD2','#EEEE00','#EEC900','darkorange','crimson','darkred','darkviolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,hgtpbl_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,ticks=clevs,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=4)
  ax1.text(.5,1.03,'NAMFW PBL Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,hgtpbl_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('black')
  cbar2 = fig.colorbar(cs_2,ax=ax2,ticks=clevs,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=4)
  ax2.text(.5,1.03,'RRFSFW PBL Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,hgtpbl_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW PBL Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('comparehgtpbl_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot PBL height for: '+dom) % t3)

#################################
  # Plot total column condensate
#################################
  t1 = time.perf_counter()
  print(('Working on Total condensate for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'kg m${^{-2}}$'
  clevs = [0.001,0.005,0.01,0.05,0.1,0.25,0.5,1,2,4,6,10,15,20,25]
  clevsdif = [-6,-4,-2,-1,-0.5,-0.25,0,0.25,0.5,1,2,4,6]
  q_color_list = plt.cm.gist_stern_r(np.linspace(0, 1, len(clevs)+1))
  cm = matplotlib.colors.ListedColormap(q_color_list)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,cond_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'NAMFW Total Column Condensate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,cond_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFSFW Total Column Condensate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,cond_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW Total Column Condensate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('comparecond_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Total condensate for: '+dom) % t3)

#################################
  # Plot 1-km reflectivity
#################################
  t1 = time.perf_counter()
  print(('Working on 1-km reflectivity for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'dBZ'
  clevs = np.linspace(5,70,14)
  clevsdif = [20,1000]
  clevsboth = [1.5,2.5]
  colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  
  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,ref1km_1,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'NAMFW 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,ref1km_2,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFSFW 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  csdif = ax3.contourf(lon_shift,lat_shift,ref1km_1,clevsdif,colors='red',transform=transform)
  csdif2 = ax3.contourf(lon_shift,lat_shift,ref1km_2,clevsdif,colors='dodgerblue',transform=transform)
  csdif3 = ax3.contourf(lon_shift,lat_shift,ref1km_both,clevsboth,colors='indigo',transform=transform)
  ax3.text(.5,1.03,'NAMFW (red), RRFSFW (blue), Both (purple) \n 1-km Reflectivity > 20 ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  plt.savefig('compareref1km_'+dom+'_f'+fhour+'.png', format='png', bbox_inches='tight', dpi=300)
#  rrfs_plot_utils.compress_and_save('compareref1km_'+dom+'_f'+fhour+'.png')
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
  cbar2.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'dBZ'
  clevs = np.linspace(5,70,14)
  clevsdif = [20,1000]
  clevsboth = [1.5,2.5]
  colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  
  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,refc_1,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'NAMFW Composite Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,refc_2,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFSFW Composite Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  csdif = ax3.contourf(lon_shift,lat_shift,refc_1,clevsdif,colors='red',transform=transform)
  csdif2 = ax3.contourf(lon_shift,lat_shift,refc_2,clevsdif,colors='dodgerblue',transform=transform)
  csdif3 = ax3.contourf(lon_shift,lat_shift,refc_both,clevsboth,colors='indigo',transform=transform)
  ax3.text(.5,1.03,'NAMFW (red), RRFSFW (blue), Both (purple) \n Composite Reflectivity > 20 ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  plt.savefig('comparerefc_'+dom+'_f'+fhour+'.png', format='png', bbox_inches='tight', dpi=300)
#  rrfs_plot_utils.compress_and_save('comparerefc_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot composite reflectivity for: '+dom) % t3)

#################################
  # Plot Max Hourly 2-5 km UH
#################################
  if (fhr > 0):
    t1 = time.perf_counter()
    print(('Working on Max Hourly 2-5 km UH for '+dom))

  # Clear off old plottables but keep all the map info    
    cbar1.remove()
    cbar2.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)    
    rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 'm${^2}$ s$^{-2}$'
    clevs = [10,25,50,75,100,150,200,250,300]
    clevsdif = [-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60]
    colorlist = ['white','skyblue','mediumblue','green','orchid','firebrick','#EEC900','DarkViolet']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,uh25_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_over('black')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'NAMFW 1-h Max 2-5 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,uh25_2,transform=transform,cmap=cm,norm=norm)
    cs_2.cmap.set_over('black')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFSFW 1-h Max 2-5 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs = ax3.pcolormesh(lon_shift,lat_shift,uh25_dif,transform=transform,cmap=cmdif,norm=normdif)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('darkred')
    cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar3.set_label(units,fontsize=6)
    cbar3.ax.tick_params(labelsize=6)
    ax3.text(.5,1.03,'RRFSFW - NAMFW 1-h Max 2-5 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.compress_and_save('compareuh25_'+dom+'_f'+fhour+'.png')
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
    cbar2.remove()
    cbar3.remove()  
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)    
    rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 'm s$^{-1}$'
    clevs = [0.5,1,2.5,5,7.5,10,12.5,15,20,25,30,35,40,50,75]
    clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
    colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia','mediumpurple']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,maxuvv_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_under('white')
    cs_1.cmap.set_over('black')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels(clevs)
    cbar1.ax.tick_params(labelsize=5)
    ax1.text(.5,1.03,'NAMFW 1-h Max 100-1000 mb Updraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,maxuvv_2,transform=transform,cmap=cm,norm=norm)
    cs_2.cmap.set_under('white')
    cs_2.cmap.set_over('black')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='both')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.set_xticklabels(clevs)
    cbar2.ax.tick_params(labelsize=5)
    ax2.text(.5,1.03,'RRFSFW 1-h Max 100-1000 mb Updraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs = ax3.pcolormesh(lon_shift,lat_shift,maxuvv_dif,transform=transform,cmap=cmdif,norm=normdif)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('darkred')
    cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar3.set_label(units,fontsize=6)
    cbar3.ax.tick_params(labelsize=6)
    ax3.text(.5,1.03,'RRFSFW - NAMFW 1-h Max 100-1000 mb Updraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.compress_and_save('comparemaxuvv_'+dom+'_f'+fhour+'.png')
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
    cbar2.remove()
    cbar3.remove()  
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)    
    rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 'm s$^{-1}$'
    clevs = [0.5,1,2.5,5,7.5,10,12.5,15,20,25,30,35,40,50,75]
    clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
    colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia','mediumpurple']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,maxdvv_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_under('white')
    cs_1.cmap.set_over('black')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels(clevs)
    cbar1.ax.tick_params(labelsize=5)
    ax1.text(.5,1.03,'NAMFW 1-h Max 100-1000 mb Downdraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,maxdvv_2,transform=transform,cmap=cm,norm=norm)
    cs_2.cmap.set_under('white')
    cs_2.cmap.set_over('black')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='both')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.set_xticklabels(clevs)
    cbar2.ax.tick_params(labelsize=5)
    ax2.text(.5,1.03,'RRFSFW 1-h Max 100-1000 mb Downdraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs = ax3.pcolormesh(lon_shift,lat_shift,maxdvv_dif,transform=transform,cmap=cmdif,norm=normdif)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('darkred')
    cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar3.set_label(units,fontsize=6)
    cbar3.ax.tick_params(labelsize=6)
    ax3.text(.5,1.03,'RRFSFW - NAMFW 1-h Max 100-1000 mb Downdraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.compress_and_save('comparemaxdvv_'+dom+'_f'+fhour+'.png')
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
    cbar2.remove()
    cbar3.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
    rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

    units='dBz'
    clevs = np.linspace(5,70,14)
    clevsdif = [20,1000]
    colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,maxref1km_1,transform=transform,cmap=cm,vmin=5,norm=norm)
    cs_1.cmap.set_under('white',alpha=0.)
    cs_1.cmap.set_over('black')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='max')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'NAMFW 1-h Max 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,maxref1km_2,transform=transform,cmap=cm,vmin=5,norm=norm)
    cs_2.cmap.set_under('white',alpha=0.)
    cs_2.cmap.set_over('black')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,extend='max')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFSFW 1-h Max 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    csdif = ax3.contourf(lon_shift,lat_shift,maxref1km_1,clevsdif,colors='red',transform=transform)
    csdif2 = ax3.contourf(lon_shift,lat_shift,maxref1km_2,clevsdif,colors='dodgerblue',transform=transform)
    csdif3 = ax3.contourf(lon_shift,lat_shift,maxref1km_both,clevsboth,colors='indigo',transform=transform)
    ax3.text(.5,1.03,'NAMFW (red), RRFSFW (blue), Both (purple) \n 1-h Max 1-km Reflectivity > 20 ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.compress_and_save('comparemaxref1km_'+dom+'_f'+fhour+'.png')
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly 1-km Reflectivity for: '+dom) % t3)

#################################
  # Plot Max Hourly 10-m Winds
#################################
    t1 = time.perf_counter()
    print(('Working on Max Hourly 10-m Wind Speed for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
    rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 'kts'
    clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
    clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
    colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,maxwind_1,transform=transform,cmap=cm,vmin=5,norm=norm)
    cs_1.cmap.set_under('white',alpha=0.)
    cs_1.cmap.set_over('black')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'NAMFW 1-h Max 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,maxwind_2,transform=transform,cmap=cm,vmin=5,norm=norm)
    cs_2.cmap.set_under('white',alpha=0.)
    cs_2.cmap.set_over('black')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFSFW 1-h Max 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs = ax3.pcolormesh(lon_shift,lat_shift,maxwind_dif,transform=transform,cmap=cmdif,norm=normdif)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('darkred')
    cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar3.set_label(units,fontsize=6)
    cbar3.ax.tick_params(labelsize=6)
    ax3.text(.5,1.03,'RRFSFW - NAMFW 1-h Max 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.compress_and_save('comparemaxwind_'+dom+'_f'+fhour+'.png')
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly 10-m Wind Speed for: '+dom) % t3)

#################################
  # Plot Min Hourly 2-m RH
#################################
    t1 = time.perf_counter()
    print(('Working on Min Hourly 2m RH for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
    rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  # Wind barb density settings
    skip = 30
    barblength = 4

    units = '%'
    clevs = [50,60,70,80,90,100]
    clevsdif = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]
    cm = plt.cm.BuGn
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,minrh_1,transform=transform,cmap=cm,vmin=50,norm=norm)
    cs_1.cmap.set_under('white',alpha=0.)
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs)
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=6)
    ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_1[::skip,::skip],vwind_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
    ax1.text(.5,1.03,'NAMFW 1-h Min 2-m RH ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,minrh_2,transform=transform,cmap=cm,vmin=50,norm=norm)
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs)
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=6)
    ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_2[::skip,::skip],vwind_2[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
    ax2.text(.5,1.03,'RRFSFW 1-h Min 2-m RH ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs = ax3.pcolormesh(lon_shift,lat_shift,minrh_dif,transform=transform,cmap=cmdif,norm=normdif)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('darkred')
    cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar3.set_label(units,fontsize=6)
    cbar3.ax.tick_params(labelsize=6)
    ax3.text(.5,1.03,'RRFSFW - NAMFW 1-h Min 2-m RH ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.compress_and_save('compareminrh_'+dom+'_f'+fhour+'.png')
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Min Hourly 2m RH for: '+dom) % t3)

#################################
  # Plot Haines Index
#################################
  t1 = time.perf_counter()
  print(('Working on Haines Index for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  if (fhr > 0):
    cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = ''
  clevs = [1.5,2.5,3.5,4.5,5.5,6.5]
  clevsdif = [-4,-3,-2,-1,0,1,2,3,4]
  colorlist = ['dodgerblue','limegreen','#EEEE00','darkorange','crimson']
  cm = matplotlib.colors.ListedColormap(colorlist)
  cmdif = matplotlib.colors.ListedColormap(difcolors3)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,hindex_1,transform=transform,cmap=cm,norm=norm)
  cbar1 = fig.colorbar(cs_1,ax=ax1,ticks=[2,3,4,5,6],orientation='horizontal',pad=0.01,shrink=0.8)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'NAMFW Haines Index \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,hindex_2,transform=transform,cmap=cm,norm=norm)
  cbar2 = fig.colorbar(cs_2,ax=ax2,ticks=[2,3,4,5,6],orientation='horizontal',pad=0.01,shrink=0.8)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFSFW Haines Index \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,hindex_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW Haines Index \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('comparehindex_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Haines Index for: '+dom) % t3)

#################################
  # Plot transport wind
#################################
  t1 = time.perf_counter()
  print(('Working on transport wind for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'kts'
  skip = 30
  barblength = 4
  clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
  clevsdif = [-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18]
  colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  cmdif = matplotlib.colors.ListedColormap(difcolors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,trans_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],utrans_1[::skip,::skip],vtrans_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax1.text(.5,1.03,'NAMFW Transport Wind ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,trans_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('black')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],utrans_2[::skip,::skip],vtrans_2[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax2.text(.5,1.03,'RRFSFW Transport Wind ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,trans_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW Transport Wind ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('comparetrans_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot transport wind for: '+dom) % t3)

#################################
  # Plot ventilation rate
#################################
  t1 = time.perf_counter()
  print(('Working on ventilarion rate for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
  rrfs_plot_utils.clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'm${^2}$ s${^{-1}}$'
  clevs = [0,1000,2500,5000,7500,10000,12500,15000,20000,25000,30000,35000,40000]
  clevsdif = [-2000,-1500,-1000,-500,-250,-100,0,100,250,500,1000,1500,2000]
  colorlist = ['white','blue','dodgerblue','cyan','mediumspringgreen','#FAFAD2','#EEEE00','#EEC900','darkorange','crimson','darkred','darkviolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,vrate_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=4)
  ax1.text(.5,1.03,'NAMFW Ventilation Rate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,vrate_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=4)
  ax2.text(.5,1.03,'RRFSFW Ventilation Rate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,vrate_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = fig.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFSFW - NAMFW Ventilation Rate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+') \n Lat/Lon of Center: '+cenlat+'\xb0'', '+cenlon+'\xb0',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.compress_and_save('comparevrate_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot ventilation rate for: '+dom) % t3)


######################################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 2 variables for: "+dom) % t3dom)
  plt.clf()

######################################################


main()

