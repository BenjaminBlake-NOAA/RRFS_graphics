#!/bin/env python

import grib2io
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib
#matplotlib.use('Agg')
import io
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as image
from matplotlib.gridspec import GridSpec
import numpy as np
import time,os,sys,multiprocessing
import multiprocessing.pool
import ncepy
from scipy import ndimage
from netCDF4 import Dataset
import pyproj
import cartopy

#--------------Set some classes------------------------#
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


#--------------Define some functions ------------------#

def clear_plotables(ax,keep_ax_lst,fig):
  #### - step to clear off old plottables but leave the map info - ####
  if len(keep_ax_lst) == 0 :
    print("clear_plotables WARNING keep_ax_lst has length 0. Clearing ALL plottables including map info!")
  cur_ax_children = ax.get_children()[:]
  if len(cur_ax_children) > 0:
    for a in cur_ax_children:
      if a not in keep_ax_lst:
       # if the artist isn't part of the initial set up, remove it
        a.remove()

def compress_and_save(filename):
  #### - compress and save the image - ####
#  ram = io.StringIO()
  ram = io.BytesIO()
  plt.savefig(ram, format='png', bbox_inches='tight', dpi=300)
#  plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
  ram.seek(0)
  im = Image.open(ram)
  im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
  im2.save(filename, format='PNG')

def cmap_t2m():
 # Create colormap for 2-m temperature
 # Modified version of the ncl_t2m colormap from Jacob's ncepy code
    r=np.array([255,128,0,  70, 51, 0,  255,0, 0,  51, 255,255,255,255,255,171,128,128,36,162,255])
    g=np.array([0,  0,  0,  70, 102,162,255,92,128,185,255,214,153,102,0,  0,  0,  68, 36,162,255])
    b=np.array([255,128,128,255,255,255,255,0, 0,  102,0,  112,0,  0,  0,  56, 0,  68, 36,162,255])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=float(i)/(float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_t2m_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_T2M_COLTBL',colorDict)
    return cmap_t2m_coltbl


def cmap_t850():
 # Create colormap for 850-mb equivalent potential temperature
    r=np.array([255,128,0,  70, 51, 0,  0,  0, 51, 255,255,255,255,255,171,128,128,96,201])
    g=np.array([0,  0,  0,  70, 102,162,225,92,153,255,214,153,102,0,  0,  0,  68, 96,201])
    b=np.array([255,128,128,255,255,255,162,0, 102,0,  112,0,  0,  0,  56, 0,  68, 96,201])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=float(i)/(float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_t850_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_T850_COLTBL',colorDict)
    return cmap_t850_coltbl


def cmap_terra():
 # Create colormap for terrain height
 # Emerald green to light green to tan to gold to dark red to brown to light brown to white
    r=np.array([0,  152,212,188,127,119,186])
    g=np.array([128,201,208,148,34, 83, 186])
    b=np.array([64, 152,140,0,  34, 64, 186])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=float(i)/(float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_terra_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_TERRA_COLTBL',colorDict)
    cmap_terra_coltbl.set_over(color='#E0EEE0')
    return cmap_terra_coltbl


def extrema(mat,mode='wrap',window=100):
    # find the indices of local extrema (max only) in the input array.
    mx = ndimage.filters.maximum_filter(mat,size=window,mode=mode)
    # (mat == mx) true if pixel is equal to the local max
    return np.nonzero(mat == mx)

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
fhrm2 = fhr - 2
fhrm6 = fhr - 6
fhrm24 = fhr - 24
fhour = str(fhr).zfill(2)
fhour1 = str(fhrm1).zfill(2)
fhour2 = str(fhrm2).zfill(2)
fhour6 = str(fhrm6).zfill(2)
fhour24 = str(fhrm24).zfill(2)
print('fhour '+fhour)

# Define the input files
data1 = grib2io.open('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python.rrfs/data/fv3lamda/fv3lamda.'+ymd+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f0'+fhour+'.grib2')
data2 = grib2io.open('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python.rrfs/data/rrfs_a/rrfs_a.'+ymd+'/'+cyc+'/rrfs.t'+cyc+'z.prslev.f0'+fhour+'.conus_3km.grib2')

if (fhr > 2):
  data1_m1 = grib2io.open('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python.rrfs/data/fv3lamda/fv3lamda.'+ymd+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f0'+fhour1+'.grib2')
  data2_m1 = grib2io.open('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python.rrfs/data/rrfs_a/rrfs_a.'+ymd+'/'+cyc+'/rrfs.t'+cyc+'z.prslev.f0'+fhour1+'.conus_3km.grib2')
  data1_m2 = grib2io.open('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python.rrfs/data/fv3lamda/fv3lamda.'+ymd+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f0'+fhour2+'.grib2')
  data2_m2 = grib2io.open('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python.rrfs/data/rrfs_a/rrfs_a.'+ymd+'/'+cyc+'/rrfs.t'+cyc+'z.prslev.f0'+fhour2+'.conus_3km.grib2')

if (fhr >= 6):
  data1_m6 = grib2io.open('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python.rrfs/data/fv3lamda/fv3lamda.'+ymd+'/'+cyc+'/fv3lam.t'+cyc+'z.conus.f0'+fhour6+'.grib2')
  data2_m6 = grib2io.open('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python.rrfs/data/rrfs_a/rrfs_a.'+ymd+'/'+cyc+'/rrfs.t'+cyc+'z.prslev.f0'+fhour6+'.conus_3km.grib2')


# Get the lats and lons
msg = data1[1][0]
lats = []
lons = []
lats_shift = []
lons_shift = []

# Unshifted grid for contours and wind barbs
lat, lon = msg.latlons()
lats.append(lat)
lons.append(lon)

# Shift grid for pcolormesh
lat1 = msg.latitudeFirstGridpoint
lon1 = msg.longitudeFirstGridpoint
nx = msg.nx
ny = msg.ny
dx = msg.gridlengthXDirection
dy = msg.gridlengthYDirection
pj = pyproj.Proj(msg.projparams)
llcrnrx, llcrnry = pj(lon1,lat1)
llcrnrx = llcrnrx - (dx/2.)
llcrnry = llcrnry - (dy/2.)
x = llcrnrx + dx*np.arange(nx)
y = llcrnry + dy*np.arange(ny)
x,y = np.meshgrid(x,y)
lon, lat = pj(x, y, inverse=True)
lats_shift.append(lat)
lons_shift.append(lon)

# Unshifted lat/lon arrays grabbed directly using latlons() method
lat = lats[0]
lon = lons[0]

# Shifted lat/lon arrays for pcolormesh
lat_shift = lats_shift[0]
lon_shift = lons_shift[0]

# Forecast valid date/time
itime = ymdh
vtime = ncepy.ndate(itime,int(fhr))

# Specify plotting domains
domains = ['conus','boston_nyc','central','colorado','la_vegas','mid_atlantic','north_central','northeast','northwest','ohio_valley','south_central','southeast','sf_bay_area','seattle_portland','southwest','upper_midwest']
#domains=['conus']

###################################################
# Read in all variables and calculate differences #
###################################################
t1a = time.perf_counter()

# Sea level pressure
slp_1 = data1.select(shortName='PRMSL',level='mean sea level')[0].data() * 0.01
slp_2 = data2.select(shortName='PRMSL',level='mean sea level')[0].data() * 0.01
slp_dif = slp_2 - slp_1

# 2-m temperature
tmp2m_1 = data1.select(shortName='TMP',level='2 m above ground')[0].data()
tmp2m_1 = (tmp2m_1 - 273.15)*1.8 + 32.0
tmp2m_2 = data2.select(shortName='TMP',level='2 m above ground')[0].data()
tmp2m_2 = (tmp2m_2 - 273.15)*1.8 + 32.0
tmp2m_dif = tmp2m_2 - tmp2m_1

# Surface temperature
tmpsfc_1 = data1.select(shortName='TMP',level='surface')[0].data()
tmpsfc_1 = (tmpsfc_1 - 273.15)*1.8 + 32.0
tmpsfc_2 = data2.select(shortName='TMP',level='surface')[0].data()
tmpsfc_2 = (tmpsfc_2 - 273.15)*1.8 + 32.0
tmpsfc_dif = tmpsfc_2 - tmpsfc_1

# 2-m dew point temperature
dew2m_1 = data1.select(shortName='DPT',level='2 m above ground')[0].data()
dew2m_1 = (dew2m_1 - 273.15)*1.8 + 32.0
dew2m_2 = data2.select(shortName='DPT',level='2 m above ground')[0].data()
dew2m_2 = (dew2m_2 - 273.15)*1.8 + 32.0
dew2m_dif = dew2m_2 - dew2m_1

# 10-m wind speed
uwind_1 = data1.select(shortName='UGRD',level='10 m above ground')[0].data() * 1.94384
uwind_2 = data2.select(shortName='UGRD',level='10 m above ground')[0].data() * 1.94384
vwind_1 = data1.select(shortName='VGRD',level='10 m above ground')[0].data() * 1.94384
vwind_2 = data2.select(shortName='VGRD',level='10 m above ground')[0].data() * 1.94384
wspd10m_1 = np.sqrt(uwind_1**2 + vwind_1**2)
wspd10m_2 = np.sqrt(uwind_2**2 + vwind_2**2)
wspd10m_dif = wspd10m_2 - wspd10m_1

# Terrain height
terra_1 = data1.select(shortName='HGT',level='surface')[0].data() * 3.28084
terra_2 = data2.select(shortName='HGT',level='surface')[0].data() * 3.28084
terra_dif = terra_2 - terra_1

# Surface wind gust
gust_1 = data1.select(shortName='GUST',level='surface')[0].data() * 1.94384
gust_2 = data2.select(shortName='GUST',level='surface')[0].data() * 1.94384
gust_dif = gust_2 - gust_1

# Most unstable CAPE
mucape_1 = data1.select(shortName='CAPE',level='180-0 mb above ground')[0].data()
mucape_2 = data2.select(shortName='CAPE',level='180-0 mb above ground')[0].data()
mucape_dif = mucape_2 - mucape_1

# Most Unstable CIN
#mucin_1 = data1.select(shortName='CIN',level='180-0 mb above ground')[0].data()
#mucin_2 = data2.select(shortName='CIN',level='180-0 mb above ground')[0].data()
#mucin_dif = mucin_2 - mucin_1

# Surface-based CAPE
cape_1 = data1.select(shortName='CAPE',level='surface')[0].data()
cape_2 = data2.select(shortName='CAPE',level='surface')[0].data()
cape_dif = cape_2 - cape_1

# Surface-based CIN
#sfcin_1 = data1.select(shortName='CIN',level='surface')[0].data()
#sfcin_2 = data2.select(shortName='CIN',level='surface')[0].data()
#sfcin_dif = sfcin_2 - sfcin_1

# Mixed Layer CAPE
mlcape_1 = data1.select(shortName='CAPE',level='90-0 mb above ground')[0].data()
mlcape_2 = data2.select(shortName='CAPE',level='90-0 mb above ground')[0].data()
mlcape_dif = mlcape_2 - mlcape_1

# Mixed Layer CIN
#mlcin_1 = data1.select(shortName='CIN',level='90-0 mb above ground')[0].data()
#mlcin_2 = data2.select(shortName='CIN',level='90-0 mb above ground')[0].data()
#mlcin_dif = mlcin_2 - mlcin_1

# 850-mb equivalent potential temperature
t850_1 = data1.select(shortName='TMP',level='850 mb')[0].data()
dpt850_1 = data1.select(shortName='DPT',level='850 mb')[0].data()
q850_1 = data1.select(shortName='SPFH',level='850 mb')[0].data()
tlcl_1 = 56.0 + (1.0/((1.0/(dpt850_1-56.0)) + 0.00125*np.log(t850_1/dpt850_1)))
thetae_1 = t850_1*((1000.0/850.0)**(0.2854*(1.0-(0.28*q850_1))))*np.exp(((3376.0/tlcl_1)-2.54)*q850_1*(1.0+(0.81*q850_1)))
t850_2 = data2.select(shortName='TMP',level='850 mb')[0].data()
dpt850_2 = data2.select(shortName='DPT',level='850 mb')[0].data()
q850_2 = data2.select(shortName='SPFH',level='850 mb')[0].data()
tlcl_2 = 56.0 + (1.0/((1.0/(dpt850_2-56.0)) + 0.00125*np.log(t850_2/dpt850_2)))
thetae_2 = t850_2*((1000.0/850.0)**(0.2854*(1.0-(0.28*q850_2))))*np.exp(((3376.0/tlcl_2)-2.54)*q850_2*(1.0+(0.81*q850_2)))
thetae_dif = thetae_2 - thetae_1

# 850-mb winds
u850_1 = data1.select(shortName='UGRD',level='850 mb')[0].data() * 1.94384
u850_2 = data2.select(shortName='UGRD',level='850 mb')[0].data() * 1.94384
v850_1 = data1.select(shortName='VGRD',level='850 mb')[0].data() * 1.94384
v850_2 = data2.select(shortName='VGRD',level='850 mb')[0].data() * 1.94384

# 700-mb omega and relative humidity
omg700_1 = data1.select(name='VVEL',level='700 mb')[0].data()
omg700_2 = data2.select(name='VVEL',level='700 mb')[0].data()
rh700_1 = data1.select(shortName='RH',level='700 mb')[0].data()
rh700_2 = data2.select(shortName='RH',level='700 mb')[0].data()
rh700_dif = rh700_2 - rh700_1

# 500 mb height, wind, vorticity
z500_1 = data1.select(shortName='HGT',level='500 mb')[0].data() * 0.1
z500_1 = ndimage.filters.gaussian_filter(z500_1, 6.89)
z500_2 = data2.select(shortName='HGT',level='500 mb')[0].data() * 0.1
z500_2 = ndimage.filters.gaussian_filter(z500_2, 6.89)
z500_dif = z500_2 - z500_1
vort500_1 = data1.select(shortName='ABSV',level='500 mb')[0].data() * 100000
vort500_1 = ndimage.filters.gaussian_filter(vort500_1,1.7225)
vort500_1[vort500_1 > 1000] = 0 # Mask out undefined values on domain edge
vort500_2 = data2.select(shortName='ABSV',level='500 mb')[0].data() * 100000
vort500_2 = ndimage.filters.gaussian_filter(vort500_2,1.7225)
vort500_2[vort500_2 > 1000] = 0 # Mask out undefined values on domain edge
u500_1 = data1.select(shortName='UGRD',level='500 mb')[0].data() * 1.94384
u500_2 = data2.select(shortName='UGRD',level='500 mb')[0].data() * 1.94384
v500_1 = data1.select(shortName='VGRD',level='500 mb')[0].data() * 1.94384
v500_2 = data2.select(shortName='VGRD',level='500 mb')[0].data() * 1.94384

# 250 mb winds
u250_1 = data1.select(shortName='UGRD',level='250 mb')[0].data() * 1.94384
u250_2 = data2.select(shortName='UGRD',level='250 mb')[0].data() * 1.94384
v250_1 = data1.select(shortName='VGRD',level='250 mb')[0].data() * 1.94384
v250_2 = data2.select(shortName='VGRD',level='250 mb')[0].data() * 1.94384
wspd250_1 = np.sqrt(u250_1**2 + v250_1**2)
wspd250_2 = np.sqrt(u250_2**2 + v250_2**2)
wspd250_dif = wspd250_2 - wspd250_1

# Visibility
vis_1 = data1.select(shortName='VIS',level='cloud top')[0].data() * 0.000621371
vis_2 = data2.select(shortName='VIS',level='surface')[0].data() * 0.000621371
vis_dif = vis_2 - vis_1

# Cloud Base Height
zbase_1 = data1.select(shortName='HGT',level='cloud base')[0].data() * (3.28084/1000)
zbase_2 = data2.select(shortName='HGT',level='cloud base')[0].data() * (3.28084/1000)
zbase_dif = zbase_2 - zbase_1

# Cloud Ceiling Height
zceil_1 = data1.select(shortName='HGT',level='cloud ceiling')[0].data() * (3.28084/1000)
zceil_2 = data2.select(shortName='HGT',level='cloud ceiling')[0].data() * (3.28084/1000)
zceil_dif = zceil_2 - zceil_1

# Cloud Top Height
ztop_1 = data1.select(shortName='HGT',level='cloud top')[0].data() * (3.28084/1000)
ztop_2 = data2.select(shortName='HGT',level='cloud top')[0].data() * (3.28084/1000)
ztop_dif = ztop_2 - ztop_1

# Precipitable water
pw_1 = data1.select(shortName='PWAT',level='entire atmosphere (considered as a single layer)')[0].data() * 0.0393701
pw_2 = data2.select(shortName='PWAT',level='entire atmosphere (considered as a single layer)')[0].data() * 0.0393701
pw_dif = pw_2 - pw_1

# Percent of frozen precipitation
pofp_1 = data1.select(shortName='CPOFP')[0].data()
pofp_2 = data2.select(shortName='CPOFP')[0].data()
pofp_dif = pofp_2 - pofp_1

# Total Precipitation is missing, but you can add here
qpf_1 = data1.select(shortName='APCP',timeRangeOfStatisticalProcess=fhr)[0].data() * 0.0393701
qpf_2 = data2.select(shortName='APCP',timeRangeOfStatisticalProcess=fhr)[0].data() * 0.0393701
qpf_dif = qpf_2 - qpf_1

# 3-hr precipitation
if (fhr > 2) and (fhr % 3 == 0):  # Do not make 3-hr plots for forecast hours 1 and 2
  qpfm2_1 = data1_m2.select(shortName='APCP',timeRangeOfStatisticalProcess=1)[0].data() * 0.0393701
  qpfm1_1 = data1_m1.select(shortName='APCP',timeRangeOfStatisticalProcess=1)[0].data() * 0.0393701
  qpfm0_1 = data1.select(shortName='APCP',timeRangeOfStatisticalProcess=1)[0].data() * 0.0393701
  qpfm2_2 = data2_m2.select(shortName='APCP',timeRangeOfStatisticalProcess=1)[0].data() * 0.0393701
  qpfm1_2 = data2_m1.select(shortName='APCP',timeRangeOfStatisticalProcess=1)[0].data() * 0.0393701
  qpfm0_2 = data2.select(shortName='APCP',timeRangeOfStatisticalProcess=1)[0].data() * 0.0393701
  qpf3_1 = qpfm2_1 + qpfm1_1 + qpfm0_1
  qpf3_2 = qpfm2_2 + qpfm1_2 + qpfm0_2
  qpf3_dif = qpf3_2 - qpf3_1

# Snow depth
snow_1 = data1.select(shortName='SNOD')[0].data() * 39.3701
snow_2 = data2.select(shortName='SNOD')[0].data() * 39.3701
snow_dif = snow_2 - snow_1
if (fhr >=6):   # Do not make 6-hr plots for forecast hours less than 6
  snowm6_1 = data1_m6.select(shortName='SNOD')[0].data() * 39.3701
  snow6_1 = snow_1 - snowm6_1
  snowm6_2 = data2_m6.select(shortName='SNOD')[0].data() * 39.3701
  snow6_2 = snow_2 - snowm6_2
  snow6_dif = snow6_2 - snow6_1

# Soil Temperature
tsoil_0_10_1 = (data1.select(shortName='TSOIL',scaledValueOfFirstFixedSurface=0)[0].data() - 273.15)*1.8 + 32.0
tsoil_0_10_2 = (data2.select(shortName='TSOIL',scaledValueOfFirstFixedSurface=0)[0].data() - 273.15)*1.8 + 32.0
tsoil_0_10_dif = tsoil_0_10_2 - tsoil_0_10_1

tsoil_10_40_1 = (data1.select(shortName='TSOIL',scaledValueOfFirstFixedSurface=10)[0].data() - 273.15)*1.8 + 32.0
tsoil_10_40_2 = (data2.select(shortName='TSOIL',scaledValueOfFirstFixedSurface=1)[0].data() - 273.15)*1.8 + 32.0
tsoil_10_40_dif = tsoil_10_40_2 - tsoil_10_40_1

tsoil_40_100_1 = (data1.select(shortName='TSOIL',scaledValueOfFirstFixedSurface=40)[0].data() - 273.15)*1.8 + 32.0
tsoil_40_100_2 = (data2.select(shortName='TSOIL',scaledValueOfFirstFixedSurface=4)[0].data() - 273.15)*1.8 + 32.0
tsoil_40_100_dif = tsoil_40_100_2 - tsoil_40_100_1

tsoil_100_200_1 = (data1.select(shortName='TSOIL',scaledValueOfFirstFixedSurface=100)[0].data() - 273.15)*1.8 + 32.0
tsoil_100_200_2 = (data2.select(shortName='TSOIL',scaledValueOfFirstFixedSurface=10)[0].data() - 273.15)*1.8 + 32.0
tsoil_100_200_dif = tsoil_100_200_2 - tsoil_100_200_1

# Soil Moisture
soilw_0_10_1 = data1.select(shortName='SOILW',scaledValueOfFirstFixedSurface=0)[0].data()
soilw_0_10_2 = data2.select(shortName='SOILW',scaledValueOfFirstFixedSurface=0)[0].data()
soilw_0_10_dif = soilw_0_10_2 - soilw_0_10_1

soilw_10_40_1 = data1.select(shortName='SOILW',scaledValueOfFirstFixedSurface=10)[0].data()
soilw_10_40_2 = data2.select(shortName='SOILW',scaledValueOfFirstFixedSurface=1)[0].data()
soilw_10_40_dif = soilw_10_40_2 - soilw_10_40_1

soilw_40_100_1 = data1.select(shortName='SOILW',scaledValueOfFirstFixedSurface=40)[0].data()
soilw_40_100_2 = data2.select(shortName='SOILW',scaledValueOfFirstFixedSurface=4)[0].data()
soilw_40_100_dif = soilw_40_100_2 - soilw_40_100_1

soilw_100_200_1 = data1.select(shortName='SOILW',scaledValueOfFirstFixedSurface=100)[0].data()
soilw_100_200_2 = data2.select(shortName='SOILW',scaledValueOfFirstFixedSurface=10)[0].data()
soilw_100_200_dif = soilw_100_200_2 - soilw_100_200_1

# Hybrid level 1 fields
clwmr_1 = data1.select(shortName='CLMR',level='1 hybrid level')[0].data() * 1000
clwmr_2 = data2.select(shortName='CLMR',level='1 hybrid level')[0].data() * 1000
clwmr_dif = clwmr_2 - clwmr_1

icmr_1 = data1.select(shortName='ICMR',level='1 hybrid level')[0].data() * 1000
icmr_2 = data2.select(shortName='ICMR',level='1 hybrid level')[0].data() * 1000
icmr_dif = icmr_2 - icmr_1

rwmr_1 = data1.select(shortName='RWMR',level='1 hybrid level')[0].data() * 1000
rwmr_2 = data2.select(shortName='RWMR',level='1 hybrid level')[0].data() * 1000
rwmr_dif = rwmr_2 - rwmr_1

snmr_1 = data1.select(shortName='SNMR',level='1 hybrid level')[0].data() * 1000
snmr_2 = data2.select(shortName='SNMR',level='1 hybrid level')[0].data() * 1000
snmr_dif = snmr_2 - snmr_1

refd_1 = data1.select(shortName='REFD',level='1 hybrid level')[0].data()
refd_2 = data2.select(shortName='REFD',level='1 hybrid level')[0].data()

tmphyb_1 = data1.select(shortName='TMP',level='1 hybrid level')[0].data() - 273.15
tmphyb_2 = data2.select(shortName='TMP',level='1 hybrid level')[0].data() - 273.15

# Downward shortwave radiation
swdown_1 = data1.select(shortName='DSWRF')[1].data()
swdown_2 = data2.select(shortName='DSWRF')[1].data()
swdown_dif = swdown_2 - swdown_1

# Upward shortwave radiation
swup_1 = data1.select(shortName='USWRF')[1].data()
swup_2 = data2.select(shortName='USWRF')[1].data()
swup_dif = swup_2 - swup_1

# Downward longwave radiation
lwdown_1 = data1.select(shortName='DLWRF')[1].data()
lwdown_2 = data2.select(shortName='DLWRF')[1].data()
lwdown_dif = lwdown_2 - lwdown_1

# Upward longwave radiation
lwup_1 = data1.select(shortName='ULWRF')[1].data()
lwup_2 = data2.select(shortName='ULWRF')[1].data()
lwup_dif = lwup_2 - lwup_1

# Ground heat flux
gdhfx_1 = data1.select(shortName='GFLUX')[1].data()
gdhfx_2 = data2.select(shortName='GFLUX')[1].data()
gdhfx_dif = gdhfx_2 - gdhfx_1

# Latent heat flux
lhfx_1 = data1.select(shortName='LHTFL')[1].data()
lhfx_2 = data2.select(shortName='LHTFL')[1].data()
lhfx_dif = lhfx_2 - lhfx_1

# Sensible heat flux
snhfx_1 = data1.select(shortName='SHTFL')[1].data()
snhfx_2 = data2.select(shortName='SHTFL')[1].data()
snhfx_dif = snhfx_2 - snhfx_1

# PBL height
hpbl_1 = data1.select(shortName='HGT',level='planetary boundary layer')[0].data()
hpbl_2 = data2.select(shortName='HGT',level='planetary boundary layer')[0].data()
hpbl_dif = hpbl_2 - hpbl_1

# Total column condensate
cond_1 = data1.select(shortName='TCOLC')[0].data()
cond_2 = data2.select(shortName='TCOLC')[0].data()
cond_dif = cond_2 - cond_1

# Total column integrated liquid (cloud water + rain)
tqw_1 = data1.select(shortName='TCOLW')[0].data()
tqw_2 = data2.select(shortName='TCOLW')[0].data()
tqr_1 = data1.select(shortName='TCOLR')[0].data()
tqr_2 = data2.select(shortName='TCOLR')[0].data()
tcolw_1 = tqw_1 + tqr_1
tcolw_2 = tqw_2 + tqr_2
tcolw_dif = tcolw_2 - tcolw_1

# Total column integrated ice (cloud ice + snow)
tqi_1 = data1.select(shortName='TCOLI')[0].data()
tqi_2 = data2.select(shortName='TCOLI')[0].data()
tqs_1 = data1.select(shortName='TCOLS')[0].data()
tqs_2 = data2.select(shortName='TCOLS')[0].data()
tcoli_1 = tqi_1 + tqs_1
tcoli_2 = tqi_2 + tqs_2
tcoli_dif = tcoli_2 - tcoli_1

# Soil type - Integer (0-16) - only plot for f00
sotyp_1 = data1.select(shortName='SOTYP')[0].data()
sotyp_2 = data2.select(shortName='SOTYP')[0].data()
sotyp_dif = sotyp_2 - sotyp_1

# Vegetation Type - Integer (0-19) - only plot for f00
vgtyp_1 = data1.select(shortName='VGTYP')[0].data()
vgtyp_2 = data2.select(shortName='VGTYP')[0].data()
vgtyp_dif = vgtyp_2 - vgtyp_1

# Vegetation Fraction
veg_1 = data1.select(shortName='VEG')[0].data()
veg_2 = data2.select(shortName='VEG')[0].data()
veg_dif = veg_2 - veg_1

# 0-3 km Storm Relative Helicity
hel3km_1 = data1.select(shortName='HLCY',scaledValueOfFirstFixedSurface=3000)[0].data()
hel3km_2 = data2.select(shortName='HLCY',scaledValueOfFirstFixedSurface=3000)[0].data()
hel3km_dif = hel3km_2 - hel3km_1

# 0-1 km Storm Relative Helicity
hel1km_1 = data1.select(shortName='HLCY',scaledValueOfFirstFixedSurface=1000)[0].data()
hel1km_2 = data2.select(shortName='HLCY',scaledValueOfFirstFixedSurface=1000)[0].data()
hel1km_dif = hel1km_2 - hel1km_1

# 1-km reflectivity
ref1km_1 = data1.select(shortName='REFD',level='1000 m above ground')[0].data()
ref1km_2 = data2.select(shortName='REFD',level='1000 m above ground')[0].data()
ref1km_1b = np.where(ref1km_1 > 20, 1, 0)
ref1km_2b = np.where(ref1km_2 > 20, 1, 0)
ref1km_both = ref1km_1b + ref1km_2b

# Composite reflectivity
refc_1 = data1.select(shortName='REFC')[0].data()
refc_2 = data2.select(shortName='REFC')[0].data()
refc_1b = np.where(refc_1 > 20, 1, 0)
refc_2b = np.where(refc_2 > 20, 1, 0)
refc_both = refc_1b + refc_2b

if (fhr > 0):
# Max/Min Hourly 2-5 km Updraft Helicity
  maxuh25_1 = data1.select(shortName='MXUPHL',level='5000-2000 m above ground')[0].data()
  maxuh25_2 = data2.select(shortName='MXUPHL',level='5000-2000 m above ground')[0].data()
  minuh25_1 = data1.select(shortName='MNUPHL',level='5000-2000 m above ground')[0].data()
  minuh25_2 = data2.select(shortName='MNUPHL',level='5000-2000 m above ground')[0].data()
  maxuh25_1[maxuh25_1 < 10] = 0
  maxuh25_2[maxuh25_2 < 10] = 0
  minuh25_1[minuh25_1 > -10] = 0
  minuh25_2[minuh25_2 > -10] = 0
  uh25_1 = maxuh25_1 + minuh25_1
  uh25_2 = maxuh25_2 + minuh25_2
  uh25_dif = uh25_2 - uh25_1

# Max/Min Hourly 0-3 km Updraft Helicity
  maxuh03_1 = data1.select(shortName='MXUPHL',level='3000-0 m above ground')[0].data()
  maxuh03_2 = data2.select(shortName='MXUPHL',level='3000-0 m above ground')[0].data()
  minuh03_1 = data1.select(shortName='MNUPHL',level='3000-0 m above ground')[0].data()
  minuh03_2 = data2.select(shortName='MNUPHL',level='3000-0 m above ground')[0].data()
  maxuh03_1[maxuh03_1 < 10] = 0
  maxuh03_2[maxuh03_2 < 10] = 0
  minuh03_1[minuh03_1 > -10] = 0
  minuh03_2[minuh03_2 > -10] = 0
  uh03_1 = maxuh03_1 + minuh03_1
  uh03_2 = maxuh03_2 + minuh03_2
  uh03_dif = uh03_2 - uh03_1

# Max Hourly Updraft Speed
  maxuvv_1 = data1.select(shortName='MAXUVV')[0].data() 
  maxuvv_2 = data2.select(shortName='MAXUVV')[0].data()
  maxuvv_dif = maxuvv_2 - maxuvv_1

# Max Hourly Downdraft Speed
  maxdvv_1 = data1.select(shortName='MAXDVV')[0].data() * -1
  maxdvv_2 = data2.select(shortName='MAXDVV')[0].data() * -1
  maxdvv_dif = maxdvv_2 - maxdvv_1

# Max Hourly 1-km AGL reflectivity
  maxref1km_1 = data1.select(shortName='MAXREF',level='1000 m above ground')[0].data()
  maxref1km_2 = data2.select(shortName='MAXREF',level='1000 m above ground')[0].data()

# Max Hourly 10-m Wind
  maxwind_1 = data1.select(shortName='WIND')[0].data() * 1.94384
  maxwind_2 = data2.select(shortName='WIND')[0].data() * 1.94384
  maxwind_dif = maxwind_2 - maxwind_1

# Haines index
hindex_1 = data1.select(shortName='HINDEX')[0].data()
hindex_2 = data2.select(shortName='HINDEX')[0].data()
hindex_dif = hindex_2 - hindex_1

# Transport wind
utrans_1 = data1.select(shortName='UGRD',level='planetary boundary layer')[0].data() * 1.94384
vtrans_1 = data1.select(shortName='VGRD',level='planetary boundary layer')[0].data() * 1.94384
utrans_2 = data2.select(shortName='UGRD',level='planetary boundary layer')[0].data() * 1.94384
vtrans_2 = data2.select(shortName='VGRD',level='planetary boundary layer')[0].data() * 1.94384
trans_1 = np.sqrt(utrans_1**2 + vtrans_1**2)
trans_2 = np.sqrt(utrans_2**2 + vtrans_2**2)
trans_dif = trans_2 - trans_1

# Total cloud cover
tcdc_1 = data1.select(shortName='TCDC')[0].data()
tcdc_2 = data2.select(shortName='TCDC')[0].data()
tcdc_dif = tcdc_2 - tcdc_1

# Echo top height
retop_1 = data1.select(shortName='RETOP')[0].data() * (3.28084/1000)
retop_2 = data2.select(shortName='RETOP')[0].data() * (3.28084/1000)
retop_dif = retop_2 - retop_1

# Precipitation rate - will need to modify because of PRATEMAX!!!
prate_1 = data1.select(shortName='PRATE')[0].data() * 3600
prate_2 = data2.select(shortName='PRATE')[0].data() * 3600
prate_dif = prate_2 - prate_1

# Cloud base pressure
pbase_1 = data1.select(shortName='PRES',level='cloud base')[0].data() * 0.01
pbase_2 = data2.select(shortName='PRES',level='cloud base')[0].data() * 0.01
pbase_dif = pbase_2 - pbase_1

# Cloud top pressure
ptop_1 = data1.select(shortName='PRES',level='cloud top')[0].data() * 0.01
ptop_2 = data2.select(shortName='PRES',level='cloud top')[0].data() * 0.01
ptop_dif = ptop_2 - ptop_1




t2a = time.perf_counter()
t3a = round(t2a-t1a, 3)
print(("%.3f seconds to read all messages") % t3a)

# colors for difference plots, only need to define once
difcolors = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','white','white','#EEEE00','#EEC900','darkorange','orangered','red']
difcolors2 = ['white']
difcolors3 = ['blue','dodgerblue','turquoise','white','white','#EEEE00','darkorange','red']

########################################
#    START PLOTTING FOR EACH DOMAIN    #
########################################

def main():

  # Number of processes must coincide with the number of domains to plot
#  pool = multiprocessing.Pool(len(domains))
  pool = MyPool(len(domains))
  pool.map(plot_all,domains)

def plot_all(domain):

  global dom
  dom = domain
  print(('Working on '+dom))

  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,x,y,xextent,yextent,im,par,transform
  fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,x,y,xextent,yextent,im,par,transform = create_figure()

  # Split plots into 3 sets with multiprocessing
  sets = [1,2,3]
#  sets = [1]
#  pool2 = multiprocessing.Pool(len(sets))
  pool2 = MyPool(len(sets))
  pool2.map(plot_sets,sets)


def create_figure():

  # Map corners for each domain
  if dom == 'conus':
    llcrnrlon = -125.5
    llcrnrlat = 20.0 
    urcrnrlon = -63.5
    urcrnrlat = 51.0
    cen_lat = 35.4
    cen_lon = -97.6
    xextent=-2200000
    yextent=-675000
  elif dom == 'northeast':
    llcrnrlon = -80.0
    llcrnrlat = 40.0
    urcrnrlon = -66.5
    urcrnrlat = 48.0
    cen_lat = 44.0
    cen_lon = -76.0
    xextent=-175000
    yextent=-282791
  elif dom == 'mid_atlantic':
    llcrnrlon = -82.0
    llcrnrlat = 36.5
    urcrnrlon = -73.0
    urcrnrlat = 42.5
    cen_lat = 36.5
    cen_lon = -79.0
    xextent=-123114
    yextent=125850
  elif dom == 'southeast':
    llcrnrlon = -92.0
    llcrnrlat = 24.0
    urcrnrlon = -75.0
    urcrnrlat = 37.0
    cen_lat = 30.5
    cen_lon = -89.0
    xextent=-12438
    yextent=-448648
  elif dom == 'ohio_valley':
    llcrnrlon = -91.5
    llcrnrlat = 34.5
    urcrnrlon = -80.0
    urcrnrlat = 43.0
    cen_lat = 38.75
    cen_lon = -88.0
    xextent=-131129
    yextent=-299910
  elif dom == 'upper_midwest':
    llcrnrlon = -97.5
    llcrnrlat = 40.0
    urcrnrlon = -82.0
    urcrnrlat = 49.5
    cen_lat = 44.75
    cen_lon = -92.0
    xextent=-230258
    yextent=-316762
  elif dom == 'north_central':
    llcrnrlon = -111.5
    llcrnrlat = 39.0
    urcrnrlon = -94.0
    urcrnrlat = 49.5
    cen_lat = 44.25
    cen_lon = -103.0
    xextent=-490381
    yextent=-336700
  elif dom == 'central':
    llcrnrlon = -103.5
    llcrnrlat = 32.0
    urcrnrlon = -89.0
    urcrnrlat = 42.0
    cen_lat = 37.0
    cen_lon = -99.0
    xextent=-220257
    yextent=-337668
  elif dom == 'south_central':
    llcrnrlon = -109.0
    llcrnrlat = 25.0
    urcrnrlon = -88.5
    urcrnrlat = 37.5
    cen_lat = 31.25
    cen_lon = -101.0
    xextent=-529631
    yextent=-407090
  elif dom == 'northwest':
    llcrnrlon = -125.0
    llcrnrlat = 40.0
    urcrnrlon = -110.0
    urcrnrlat = 50.0
    cen_lat = 45.0
    cen_lon = -116.0
    xextent=-540000
    yextent=-333623
  elif dom == 'southwest':
    llcrnrlon = -125.0
    llcrnrlat = 31.0
    urcrnrlon = -108.5
    urcrnrlat = 42.5
    cen_lat = 36.75
    cen_lon = -116.0
    xextent=-593059
    yextent=-377213
  elif dom == 'colorado':
    llcrnrlon = -110.0
    llcrnrlat = 35.0
    urcrnrlon = -101.0
    urcrnrlat = 42.0
    cen_lat = 38.5
    cen_lon = -106.0
    xextent=-224751
    yextent=-238851
  elif dom == 'boston_nyc':
    llcrnrlon = -75.5
    llcrnrlat = 40.0
    urcrnrlon = -69.5
    urcrnrlat = 43.0
    cen_lat = 41.5
    cen_lon = -76.0
    xextent=112182
    yextent=-99031
  elif dom == 'seattle_portland':
    llcrnrlon = -125.0
    llcrnrlat = 44.5
    urcrnrlon = -119.0
    urcrnrlat = 49.5
    cen_lat = 47.0
    cen_lon = -121.0
    xextent=-227169
    yextent=-200000
  elif dom == 'sf_bay_area':
    llcrnrlon = -123.5
    llcrnrlat = 37.25
    urcrnrlon = -121.0
    urcrnrlat = 38.5
    cen_lat = 48.25
    cen_lon = -121.0
    xextent=-185364
    yextent=-1193027
  elif dom == 'la_vegas':
    llcrnrlon = -121.0
    llcrnrlat = 32.0
    urcrnrlon = -114.0
    urcrnrlat = 37.0
    cen_lat = 34.5
    cen_lon = -114.0
    xextent=-540000
    yextent=-173241

  # create figure and axes instances
  fig = plt.figure(figsize=(9,8))
  gs = GridSpec(9,8,wspace=0.0,hspace=0.0)
  im = image.imread('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python.rrfs/noaa.png')
  par = 1

  # Define where Cartopy maps are located
  cartopy.config['data_dir'] = '/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python/NaturalEarth'

  back_res='50m'
  back_img='on'

  # set up the map background with cartopy
  if dom == 'conus':
    extent = [llcrnrlon-1,urcrnrlon-6,llcrnrlat,urcrnrlat+1]
  else:
    extent = [llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat]
  myproj=ccrs.LambertConformal(central_longitude=cen_lon, central_latitude=cen_lat, false_easting=0.0,
                          false_northing=0.0, secant_latitudes=None, standard_parallels=None,
                          globe=None)
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
#  land=cfeature.NaturalEarthFeature('physical','land',back_res,
#                    edgecolor='face',facecolor=cfeature.COLORS['land'],
#                    alpha=falpha)
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

  # high-resolution background images
  if back_img=='on':
     img = plt.imread('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python/NaturalEarth/raster_files/NE1_50M_SR_W.tif')
     ax1.imshow(img, origin='upper', transform=transform)
     ax2.imshow(img, origin='upper', transform=transform)
     ax3.imshow(img, origin='upper', transform=transform)

#  ax.add_feature(land)
  ax1.add_feature(lakes)
  ax1.add_feature(states)
#  ax1.add_feature(borders)
  ax1.add_feature(coastline)
  ax2.add_feature(lakes)
  ax2.add_feature(states)
#  ax2.add_feature(borders)
  ax2.add_feature(coastline)
  ax3.add_feature(lakes)
  ax3.add_feature(states)
#  ax3.add_feature(borders)
  ax3.add_feature(coastline)

  # Map/figure has been set up here, save axes instances for use again later
  keep_ax_lst_1 = ax1.get_children()[:]
  keep_ax_lst_2 = ax2.get_children()[:]
  keep_ax_lst_3 = ax3.get_children()[:]

  return fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,x,y,xextent,yextent,im,par,transform


def plot_sets(set):
# Add print to see if dom is being passed in
  print(('plot_sets dom variable '+dom))

  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,x,y,xextent,yextent,im,par,transform

  if set == 1:
    plot_set_1()
  elif set == 2:
    plot_set_2()
  elif set == 3:
    plot_set_3()

def plot_set_1():
  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,x,y,xextent,yextent,im,par,transform

################################
  # Plot SLP
################################
  t1dom = time.perf_counter()
  t1 = time.perf_counter()
  print(('Working on slp for '+dom))

  # Wind barb density settings
  if dom == 'conus':
    skip = 100
  elif dom == 'southeast':
    skip = 40
  elif dom == 'colorado' or dom == 'la_vegas' or dom =='mid_atlantic':
    skip = 18
  elif dom == 'boston_nyc':
    skip = 15
  elif dom == 'seattle_portland':
    skip = 13
  elif dom == 'sf_bay_area':
    skip = 4
  else:
    skip = 30
  barblength = 3.5

  units = 'mb'
  clevs = [976,980,984,988,992,996,1000,1004,1008,1012,1016,1020,1024,1028,1032,1036,1040,1044,1048,1052]
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = plt.cm.Spectral_r
  cmdif = matplotlib.colors.ListedColormap(difcolors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs1_a = ax1.pcolormesh(lon_shift,lat_shift,slp_1,transform=transform,cmap=cm,norm=norm)  
  cbar1 = plt.colorbar(cs1_a,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  cs1_b = ax1.contour(lon_shift,lat_shift,slp_1,np.arange(940,1060,4),colors='black',linewidths=0.1,transform=transform)
#  plt.clabel(cs1_b,np.arange(940,1060,4),inline=1,fmt='%d',fontsize=6)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_1[::skip,::skip],vwind_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax1.text(.5,1.03,'LAMDA SLP ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs2_a = ax2.pcolormesh(lon_shift,lat_shift,slp_2,transform=transform,cmap=cm,norm=norm)  
  cbar2 = plt.colorbar(cs2_a,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  cs2_b = ax2.contour(lon_shift,lat_shift,slp_2,np.arange(940,1060,4),colors='black',linewidths=0.1,transform=transform)
#  plt.clabel(cs2_b,np.arange(940,1060,4),inline=1,fmt='%d',fontsize=6)
  ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_2[::skip,::skip],vwind_2[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax2.text(.5,1.03,'RRFS_A SLP ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,slp_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=5)
  ax3.text(.5,1.03,'RRFS_A - LAMDA SLP ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compareslp_'+dom+'_f'+fhour+'.png')
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
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '\xb0''F'
  clevs = np.linspace(-16,134,51)
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,tmp2m_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'LAMDA 2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,tmp2m_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFS_A 2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))       
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,tmp2m_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2')) 
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compare2mt_'+dom+'_f'+fhour+'.png')
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
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '\xb0''F'
  clevs = np.linspace(-16,134,51)
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,tmpsfc_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'LAMDA Surface Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,tmpsfc_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFS_A Surface Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,tmpsfc_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Surface Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))       
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparetsfc_'+dom+'_f'+fhour+'.png')
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
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '\xb0''F'
  clevs = np.linspace(-10,80,37)
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = ncepy.cmap_q2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,dew2m_1,transform=transform,cmap=cm,norm=norm)
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA 2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,dew2m_2,transform=transform,cmap=cm,norm=norm)
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A 2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,dew2m_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compare2mdew_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 2mdew for: '+dom) % t3)


#################################
  # Plot 10-m WSPD
#################################
  t1 = time.perf_counter()
  print(('Working on 10mwspd for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  # Wind barb density settings
  if dom == 'conus':
    skip = 80
  elif dom == 'southeast':
    skip = 35
  elif dom == 'colorado' or dom == 'la_vegas' or dom =='mid_atlantic':
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

  units = 'kts'
  clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,wspd10m_1,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_1[::skip,::skip],vwind_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax1.text(.5,1.03,'LAMDA 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
    
  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,wspd10m_2,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_2[::skip,::skip],vwind_2[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax2.text(.5,1.03,'RRFS_A 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,wspd10m_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label('kts',fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 10-m Wind Speed (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))       
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compare10mwind_'+dom+'_f'+fhour+'.png')
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
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)
  
  units = 'ft'
  clevs = [1,250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000,4250,4500,4750,5000,5250,5500,5750,6000,6250,6500,6750,7000,7250,7500,7750,8000,8250,8500,8750,9000,9250,9500,9750,10000]
  clevsdif = [-300,-250,-200,-150,-100,-50,0,50,100,150,200,250,300]
  cm = cmap_terra()
  cmdif = matplotlib.colors.ListedColormap(difcolors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,terra_1,transform=transform,cmap=cm,vmin=1,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('ghostwhite')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_1[::skip,::skip],vwind_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax1.text(.5,1.03,'LAMDA Terrain Height ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,terra_2,transform=transform,cmap=cm,vmin=1,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('ghostwhite')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_2[::skip,::skip],vwind_2[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax2.text(.5,1.03,'RRFS_A Terrain Height ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,terra_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=5)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Terrain Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compareterra_'+dom+'_f'+fhour+'.png')
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
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'kts'
  clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,gust_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.05,'LAMDA Surface Wind Gust ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,gust_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.05,'RRFS_A Surface Wind Gust ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,gust_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Surface Wind Gust ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparegust_'+dom+'_f'+fhour+'.png')
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
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'J/kg'
  clevs = [100,250,500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
  clevs2 = [-2000,-500,-250,-100,-25]
  clevsdif = [-2000,-1500,-1000,-500,-250,-100,0,100,250,500,1000,1500,2000]
  colorlist = ['blue','dodgerblue','cyan','mediumspringgreen','#FAFAD2','#EEEE00','#EEC900','darkorange','crimson','darkred','darkviolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,mucape_1,transform=transform,cmap=cm,vmin=100,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=4)
#  cs_1b = ax1.contourf(lon_shift,lat_shift,mucin_1,clevs2,colors='none',hatches=['**','++','////','..'],transform=transform)
  ax1.text(.5,1.05,'LAMDA Most Unstable CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,mucape_2,transform=transform,cmap=cm,vmin=100,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=4)
#  cs_2b = ax2.contourf(lon_shift,lat_shift,mucin_2,clevs2,colors='none',hatches=['**','++','////','..'],transform=transform)
  ax2.text(.5,1.05,'RRFS_A Most Unstable CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,mucape_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevsdif,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=4)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Most Unstable CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparemucape_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot mucapecin for: '+dom) % t3)

#################################
  # Plot Surface-Based CAPE/CIN
#################################
  t1 = time.perf_counter()
  print(('Working on sfcapecin for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,cape_1,transform=transform,cmap=cm,vmin=100,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=4)
#  cs_1b = ax1.contourf(lon_shift,lat_shift,sfcin_1,clevs2,colors='none',hatches=['**','++','////','..'],transform=transform)
  ax1.text(.5,1.05,'LAMDA Surface-Based CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,cape_2,transform=transform,cmap=cm,vmin=100,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=4)
#  cs_2b = ax2.contourf(lon_shift,lat_shift,sfcin_2,clevs2,colors='none',hatches=['**','++','////','..'],transform=transform)
  ax2.text(.5,1.05,'RRFS_A Surface-Based CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,cape_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevsdif,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=4)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Surface-Based CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparesfcape_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot sfcapecin for: '+dom) % t3)

#################################
  # Plot Mixed Layer CAPE/CIN
#################################
  t1 = time.perf_counter()
  print(('Working on mlcapecin for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,mlcape_1,transform=transform,cmap=cm,vmin=100,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=4)
#  cs_1b = ax1.contourf(lon_shift,lat_shift,mlcin_1,clevs2,colors='none',hatches=['**','++','////','..'],transform=transform)
  ax1.text(.5,1.05,'LAMDA Mixed Layer CAPE ('+units+') \n  initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,mlcape_2,transform=transform,cmap=cm,vmin=100,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=4)
#  cs_2b = ax2.contourf(lon_shift,lat_shift,mlcin_2,clevs2,colors='none',hatches=['**','++','////','..'],transform=transform)
  ax2.text(.5,1.05,'RRFS_A Mixed Layer CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,mlcape_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevsdif,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=4)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Mixed Layer CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparemlcape_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot mlcapecin for: '+dom) % t3)

#################################
  # Plot 850-mb THETAE
#################################
  t1 = time.perf_counter()
  print(('Working on 850 mb Theta-e for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'K'
# Wind barb density settings for 850, 500, and 250 mb plots
  if dom == 'conus':
    skip = 100
  elif dom == 'SE':
    skip = 40
  elif dom == 'CO' or dom == 'LA' or dom =='MA':
    skip = 18
  elif dom == 'BN':
    skip = 15
  elif dom == 'SP':
    skip = 13
  elif dom == 'SF':
    skip = 4
  else:
    skip = 30
  barblength = 4

  clevs = np.linspace(270,360,31)
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = cmap_t850()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,thetae_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=[270,276,282,288,294,300,306,312,318,324,330,336,342,348,354,360],extend='both')
  cbar1.set_label(units,fontsize=6)   
  cbar1.ax.tick_params(labelsize=4)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u850_1[::skip,::skip],v850_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax1.text(.5,1.03,'LAMDA 850 mb $\Theta$e ('+units+') and Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,thetae_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=[270,276,282,288,294,300,306,312,318,324,330,336,342,348,354,360],extend='both')
  cbar2.set_label(units,fontsize=6)   
  cbar2.ax.tick_params(labelsize=4)
  ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u850_2[::skip,::skip],v850_2[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax2.text(.5,1.03,'RRFS_A 850 mb $\Theta$e ('+units+') and Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
    
  cs = ax3.pcolormesh(lon_shift,lat_shift,thetae_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)   
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 850 mb $\Theta$e ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compare850t_'+dom+'_f'+fhour+'.png')
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
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

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

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs1_a = ax1.pcolormesh(lon_shift,lat_shift,rh700_1,transform=transform,cmap=cm,vmin=50,norm=norm)
  cs1_a.cmap.set_under('white',alpha=0.)
  cbar1 = plt.colorbar(cs1_a,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs)
  cbar1.set_label(units,fontsize=6) 
  cbar1.ax.tick_params(labelsize=6)
  cs1_b = ax1.pcolormesh(lon_shift,lat_shift,omg700_1,transform=transform,cmap=cmw,vmax=-5,norm=normw)
  cs1_b.cmap.set_over('white',alpha=0.)
  ax1.text(.5,1.03,'LAMDA 700 mb $\omega$ (rising motion in blue) and RH ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs2_a = ax2.pcolormesh(lon_shift,lat_shift,rh700_2,transform=transform,cmap=cm,vmin=50,norm=norm)
  cs2_a.cmap.set_under('white',alpha=0.)
  cbar2 = plt.colorbar(cs2_a,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs)
  cbar2.set_label(units,fontsize=6) 
  cbar2.ax.tick_params(labelsize=6)
  cs2_b = ax2.pcolormesh(lon_shift,lat_shift,omg700_2,transform=transform,cmap=cmw,vmax=-5,norm=normw)
  cs2_b.cmap.set_over('white',alpha=0.)
  ax2.text(.5,1.03,'RRFS_A 700 mb $\omega$ (rising motion in blue) and RH ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,rh700_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 700 mb RH ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compare700_'+dom+'_f'+fhour+'.png')
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
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'x10${^5}$ s${^{-1}}$'
  vortlevs = [16,20,24,28,32,36,40]
  clevsdif = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  colorlist = ['yellow','gold','goldenrod','orange','orangered','red']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(vortlevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs1_a = ax1.pcolormesh(lon_shift,lat_shift,vort500_1,transform=transform,cmap=cm,norm=norm)
  cs1_a.cmap.set_under('white')
  cs1_a.cmap.set_over('darkred')
  cbar1 = plt.colorbar(cs1_a,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=vortlevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u500_1[::skip,::skip],v500_1[::skip,::skip],length=barblength,linewidth=0.5,color='steelblue',transform=transform)
  cs1_b = ax1.contour(lon_shift,lat_shift,z500_1,np.arange(486,600,6),colors='black',linewidths=1,transform=transform)
  plt.clabel(cs1_b,np.arange(486,600,6),inline_spacing=1,fmt='%d',fontsize=5)
  ax1.text(.5,1.03,'LAMDA 500 mb Heights (dam), Winds (kts), and $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs2_a = ax2.pcolormesh(lon_shift,lat_shift,vort500_2,transform=transform,cmap=cm,norm=norm)
  cs2_a.cmap.set_under('white')
  cs2_a.cmap.set_over('darkred')
  cbar2 = plt.colorbar(cs2_a,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=vortlevs,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u500_2[::skip,::skip],v500_2[::skip,::skip],length=barblength,linewidth=0.5,color='steelblue',transform=transform)
  cs2_b = ax2.contour(lon_shift,lat_shift,z500_2,np.arange(486,600,6),colors='black',linewidths=1,transform=transform)
  plt.clabel(cs2_b,np.arange(486,600,6),inline_spacing=1,fmt='%d',fontsize=5)
  ax2.text(.5,1.03,'RRFS_A 500 mb Heights (dam), Winds (kts), and $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,z500_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6) 
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 500 mb Heights (dam) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compare500_'+dom+'_f'+fhour+'.png')
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
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'kts'
  clevs = [50,60,70,80,90,100,110,120,130,140,150]
  clevsdif = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]
  colorlist = ['turquoise','deepskyblue','dodgerblue','#1874CD','blue','beige','khaki','peru','brown','crimson']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,wspd250_1,transform=transform,cmap=cm,vmin=50,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('red')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u250_1[::skip,::skip],v250_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax1.text(.5,1.03,'LAMDA 250 mb Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,wspd250_2,transform=transform,cmap=cm,vmin=50,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('red')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],u250_2[::skip,::skip],v250_2[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax2.text(.5,1.03,'RRFS_A 250 mb Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,wspd250_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6) 
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 250 mb Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compare250wind_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 250 mb WIND for: '+dom) % t3)

#################################
  # Plot Surface Visibility
#################################
  t1 = time.perf_counter()
  print(('Working on Surface Visibility for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'miles'
  clevs = [0.25,0.5,1,2,3,4,5,10]
  clevsdif = [-15,-12.5,-10,-7.5,-5,-2.5,0.,2.5,5,7.5,10,12.5,15]
  colorlist = ['salmon','goldenrod','#EEEE00','palegreen','darkturquoise','blue','mediumpurple']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,vis_1,transform=transform,cmap=cm,vmax=10,norm=norm)
  cs_1.cmap.set_under('firebrick')
  cs_1.cmap.set_over('white',alpha=0.)
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='min')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Surface Visibility ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,vis_2,transform=transform,cmap=cm,vmax=10,norm=norm)
  cs_2.cmap.set_under('firebrick')
  cs_2.cmap.set_over('white',alpha=0.)
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='min')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Surface Visibility ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,vis_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6) 
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Surface Visibility ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparevis_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Surface Visibility for: '+dom) % t3)

#################################
  # Plot Cloud Base Height
#################################
  t1 = time.perf_counter()
  print(('Working on Cloud Base Height for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'kft'
  clevs = [0,0.1,0.3,0.5,1,5,10,15,20,25,30,35,40]
  clevsdif = [-12,-10,-8,-6,-4,-2,0.,2,4,6,8,10,12]
  colorlist = ['firebrick','tomato','salmon','lightsalmon','goldenrod','khaki','gold','yellow','palegreen','mediumspringgreen','lime','limegreen']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,zbase_1,transform=transform,cmap=cm,vmin=0,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('darkgreen')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Cloud Base Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,zbase_2,transform=transform,cmap=cm,vmin=0,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('darkgreen')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Cloud Base Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,zbase_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevsdif,extend='both')
  cbar3.set_label(units,fontsize=6) 
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Cloud Base Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparezbase_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Cloud Base Height for: '+dom) % t3)

#################################
  # Plot Cloud Ceiling Height
#################################
  t1 = time.perf_counter()
  print(('Working on Cloud Ceiling Height for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'kft'
  clevs = [0,0.1,0.3,0.5,1,5,10,15,20,25,30,35,40]
  clevsdif = [-12,-10,-8,-6,-4,-2,0.,2,4,6,8,10,12]
  colorlist = ['firebrick','tomato','salmon','lightsalmon','goldenrod','khaki','gold','yellow','palegreen','mediumspringgreen','lime','limegreen']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,zceil_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Cloud Ceiling Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,zceil_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_over('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Cloud Ceiling Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,zceil_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevsdif,extend='both')
  cbar3.set_label(units,fontsize=6) 
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Cloud Ceiling Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparezceil_'+dom+'_f'+fhour+'.png')
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
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'kft'
  clevs = [1,5,10,15,20,25,30,35,40,45,50]
  clevsdif = [-12,-10,-8,-6,-4,-2,0.,2,4,6,8,10,12]
  colorlist = ['firebrick','tomato','salmon','lightsalmon','goldenrod','yellow','palegreen','mediumspringgreen','lime','limegreen']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,ztop_1,transform=transform,cmap=cm,vmin=0,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('darkgreen')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Cloud Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,ztop_2,transform=transform,cmap=cm,vmin=0,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('darkgreen')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Cloud Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,ztop_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevsdif,extend='both')
  cbar3.set_label(units,fontsize=6) 
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Cloud Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compareztop_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Cloud Top Height for: '+dom) % t3)

#################################
  # Plot PW
#################################
  t1 = time.perf_counter()
  print(('Working on PW for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'in'
  clevs = [0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25]
  clevsdif = [-1.25,-1,-.75,-.5,-.25,-.1,0.,.1,.25,.50,.75,1,1.25]
  colorlist = ['lightsalmon','khaki','palegreen','cyan','turquoise','cornflowerblue','mediumslateblue','darkorchid','deeppink']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,pw_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('hotpink')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'LAMDA Precipitable Water ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,pw_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('hotpink')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFS_A Precipitable Water ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,pw_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevsdif,extend='both')
  cbar3.set_label(units,fontsize=6) 
  cbar3.ax.tick_params(labelsize=4)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Precipitable Water ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparepw_'+dom+'_f'+fhour+'.png')
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
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '%'
  clevs = [10,20,30,40,50,60,70,80,90,100]
  clevsdif = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]
  colorlist = ['blue','dodgerblue','deepskyblue','mediumspringgreen','khaki','sandybrown','salmon','crimson','maroon']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,pofp_1,transform=transform,cmap=cm,vmin=10,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs)
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Percent of Frozen Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,pofp_2,transform=transform,cmap=cm,vmin=10,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs)
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Percent of Frozen Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,pofp_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevsdif,extend='both')
  cbar3.set_label(units,fontsize=6) 
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Percent of Frozen Precipitaion ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparepofp_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot PERCENT FROZEN PRECIP for: '+dom) % t3)

#################################
  # Plot QPF
#################################
  t1 = time.perf_counter()
  print(('Working on qpf3 for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'in'
  clevs = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,7,10,15,20]
  clevsdif = [-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5]
  colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','deepskyblue','cyan','mediumpurple','mediumorchid','darkmagenta','darkred','crimson','orangered','darkorange','goldenrod','gold','yellow']  
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)
   
  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,qpf_1,transform=transform,cmap=cm,vmin=0.01,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('pink')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.1,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,qpf_2,transform=transform,cmap=cm,vmin=0.01,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('pink')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.1,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,qpf_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compareqpf_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot qpf for: '+dom) % t3)

#################################
  # Plot QPF3
#################################
  if (fhr % 3 == 0) and (fhr > 0):
    t1 = time.perf_counter()
    print(('Working on qpf3 for '+dom))

    # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 'in'
    clevs = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,7,10,15,20]
    clevsdif = [-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5]
    colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','deepskyblue','cyan','mediumpurple','mediumorchid','darkmagenta','darkred','crimson','orangered','darkorange','goldenrod','gold','yellow']  
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)
   
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,qpf3_1,transform=transform,cmap=cm,vmin=0.01,norm=norm)
    cs_1.cmap.set_under('white',alpha=0.)
    cs_1.cmap.set_over('pink')
    cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.1,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'LAMDA 3-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,qpf3_2,transform=transform,cmap=cm,vmin=0.01,norm=norm)
    cs_2.cmap.set_under('white',alpha=0.)
    cs_2.cmap.set_over('pink')
    cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.1,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFS_A 3-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs = ax3.pcolormesh(lon_shift,lat_shift,qpf3_dif,transform=transform,cmap=cmdif,norm=normdif)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('darkred')
    cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar3.set_label(units,fontsize=6)
    cbar3.ax.tick_params(labelsize=6)
    ax3.text(.5,1.03,'RRFS_A - LAMDA 3-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))         
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    compress_and_save('compareqpf3_'+dom+'_f'+fhour+'.png')
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot qpf3 for: '+dom) % t3)

#################################
  # Plot snow depth
#################################
  t1 = time.perf_counter()
  print(('Working on snow depth for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'in'
  clevs = [0.1,1,2,3,6,9,12,18,24,36,48]
  clevsdif = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  cm = ncepy.ncl_perc_11Lev()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N) 
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N) 
 
  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,snow_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,snow_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,snow_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))         
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparesnow_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot snow depth for: '+dom) % t3)

#################################
  # Plot 6-hr change in snow depth
#################################
  if (fhr % 3 == 0) and (fhr >= 6):
    t1 = time.perf_counter()
    print(('Working on 6-hr change in snow depth for '+dom))

    # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 'in'
    clevs = [-6,-4,-3,-2,-1,-0.5,0,0.5,1,2,3,4,6]
    clevsdif = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
    colorlist = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','white','white','#EEEE00','#EEC900','darkorange','orangered','red']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,snow6_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_under('darkblue')
    cs_1.cmap.set_over('darkred')
    cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels(clevs)
    cbar1.ax.tick_params(labelsize=5)
    ax1.text(.5,1.03,'LAMDA 6-hr Change in Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,snow6_2,transform=transform,cmap=cm,norm=norm)
    cs_2.cmap.set_under('darkblue')
    cs_2.cmap.set_over('darkred')
    cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.set_xticklabels(clevs)
    cbar2.ax.tick_params(labelsize=5)
    ax2.text(.5,1.03,'RRFS_A 6-hr Change in Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs = ax3.pcolormesh(lon_shift,lat_shift,snow6_dif,transform=transform,cmap=cmdif,norm=normdif)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('darkred')
    cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar3.set_label(units,fontsize=6)
    cbar3.ax.tick_params(labelsize=6)
    ax3.text(.5,1.03,'RRFS_A - LAMDA 6-hr Change in Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    compress_and_save('comparesnow6_'+dom+'_f'+fhour+'.png')
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot snow depth for: '+dom) % t3)


  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 1 variables for: "+dom) % t3dom)
  plt.clf()


################################################################################

def plot_set_2():
  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,x,y,xextent,yextent,im,par,transform

#################################
  # Plot 0-10cm soil temperature
#################################
  t1 = time.perf_counter()
  print(('Working on 0-10cm soil temperature for '+dom))

  # Clear off old plottables but keep all the map info
#  cbar1.remove()
#  cbar2.remove()
#  cbar3.remove()
#  clear_plotables(ax1,keep_ax_lst_1,fig)
#  clear_plotables(ax2,keep_ax_lst_2,fig)
#  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '\xb0''F'
  clevs = np.linspace(-36,104,36)
  clevsdif = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  cm = cmap_t2m()
  cmdif = matplotlib.colors.ListedColormap(difcolors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

#      mysoilt = maskoceans(lon_shift,lat_shift,tsoil_0_10_1,inlands=True,resolution='l')
  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,tsoil_0_10_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA 0-10 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

#      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_0_10_2,inlands=True,resolution='l')
  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,tsoil_0_10_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A 0-10 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

#      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_0_10_dif,inlands=True,resolution='l')
  cs = ax3.pcolormesh(lon_shift,lat_shift,tsoil_0_10_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 0-10 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparetsoil_0_10_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 0-10 cm soil temperature for: '+dom) % t3)

#################################
  # Plot 10-40cm soil temperature
#################################
  t1 = time.perf_counter()
  print(('Working on 10-40 cm soil temperature for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

#      mysoilt = maskoceans(lon_shift,lat_shift,tsoil_10_40_1,inlands=True,resolution='l')
  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,tsoil_10_40_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA 10-40 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

#      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_10_40_2,inlands=True,resolution='l')
  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,tsoil_10_40_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A 10-40 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

#      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_10_40_dif,inlands=True,resolution='l')
  cs = ax3.pcolormesh(lon_shift,lat_shift,tsoil_10_40_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 10-40 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparetsoil_10_40_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 10-40 cm soil temperature for: '+dom) % t3)

#################################
  # Plot 40-100cm soil temperature
#################################
  t1 = time.perf_counter()
  print(('Working on 40-100 cm soil temperature for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

#      mysoilt = maskoceans(lon_shift,lat_shift,tsoil_40_100_1,inlands=True,resolution='l')
  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,tsoil_40_100_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA 40-100 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

#      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_40_100_2,inlands=True,resolution='l')
  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,tsoil_40_100_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A 40-100 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

#      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_40_100_dif,inlands=True,resolution='l')
  cs = ax3.pcolormesh(lon_shift,lat_shift,tsoil_40_100_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 40-100 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparetsoil_40_100_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 40-100 cm soil temperature for: '+dom) % t3)

#################################
  # Plot 100-200 cm soil temperature
#################################
  t1 = time.perf_counter()
  print(('Working on 100-200 cm soil temperature for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

#      mysoilt = maskoceans(lon_shift,lat_shift,tsoil_100_200_1,inlands=True,resolution='l')
  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,tsoil_100_200_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA 1-2 m Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

#      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_100_200_2,inlands=True,resolution='l')
  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,tsoil_100_200_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A 1-2 m Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

#      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_100_200_dif,inlands=True,resolution='l')
  cs = ax3.pcolormesh(lon_shift,lat_shift,tsoil_100_200_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 1-2 m Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparetsoil_100_200_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 100-200 cm soil temperature for: '+dom) % t3)

#################################
  # Plot 0-10 cm Soil Moisture Content
#################################
  t1 = time.perf_counter()
  print(('Working on 0-10 cm soil moisture for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = ''
  clevs = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
  clevsdif = [-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0,0.01,0.02,0.03,0.04,0.05,0.06]
  colorlist = ['crimson','darkorange','darkgoldenrod','#EEC900','chartreuse','limegreen','green','#1C86EE','deepskyblue']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,soilw_0_10_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('darkred')
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA 0-10 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,soilw_0_10_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('darkred')
  cs_2.cmap.set_over('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A 0-10 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,soilw_0_10_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 0-10 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparesoilw_0_10_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 0-10 cm soil moisture content for: '+dom) % t3)

#################################
  # Plot 10-40 cm Soil Moisture Content
#################################
  t1 = time.perf_counter()
  print(('Working on 10-40 cm soil moisture for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,soilw_10_40_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('darkred')
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA 10-40 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,soilw_10_40_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('darkred')
  cs_2.cmap.set_over('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A 10-40 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,soilw_10_40_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 10-40 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparesoilw_10_40_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 10-40 cm soil moisture content for: '+dom) % t3)

#################################
  # Plot 40-100 cm Soil Moisture Content
#################################
  t1 = time.perf_counter()
  print(('Working on 40-100 cm soil moisture for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,soilw_40_100_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('darkred')
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA 40-100 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,soilw_40_100_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('darkred')
  cs_2.cmap.set_over('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A 40-100 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,soilw_40_100_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 40-100 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparesoilw_40_100_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 40-100 cm soil moisture content for: '+dom) % t3)

#################################
  # Plot 1-2 m Soil Moisture Content
#################################
  t1 = time.perf_counter()
  print(('Working on 1-2 m soil moisture for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,soilw_100_200_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('darkred')
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA 1-2 m Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,soilw_100_200_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('darkred')
  cs_2.cmap.set_over('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A 1-2 m Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,soilw_100_200_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 1-2 m Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparesoilw_100_200_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 1-2 m soil moisture content for: '+dom) % t3)

#################################
  # Plot lowest model level cloud water
#################################
  t1 = time.perf_counter()
  print(('Working on lowest model level cloud water for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'g/kg'
  clevs = [0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1,2]
  clevsref = [20,1000]
  clevsdif = [-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6]
  colorlist = ['blue','dodgerblue','deepskyblue','mediumspringgreen','khaki','sandybrown','salmon','crimson','maroon']
  colorsref = ['Grey']
  cm = matplotlib.colors.ListedColormap(colorlist)
  cmref = matplotlib.colors.ListedColormap(colorsref)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normref = matplotlib.colors.BoundaryNorm(clevsref, cmref.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  csref_1 = ax1.pcolormesh(lon_shift,lat_shift,refd_1,transform=transform,cmap=cmref,vmin=20,norm=normref)
  csref_1.cmap.set_under('white')
  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,clwmr_1,transform=transform,cmap=cm,vmin=0.01,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('hotpink')
  cstmp_1 = ax1.contour(lon_shift,lat_shift,tmphyb_1,[0],colors='red',linewidths=0.5,transform=transform)
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'LAMDA Lowest Mdl Lvl Cld Water ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  csref_2 = ax2.pcolormesh(lon_shift,lat_shift,refd_2,transform=transform,cmap=cmref,vmin=20,norm=normref)
  csref_2.cmap.set_under('white')
  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,clwmr_2,transform=transform,cmap=cm,vmin=0.01,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('hotpink')
  cstmp_2 = ax2.contour(lon_shift,lat_shift,tmphyb_2,[0],colors='red',linewidths=0.5,transform=transform)
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFS_A Lowest Mdl Lvl Cld Water ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,clwmr_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=5)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Lowest Mdl Lvl Cld Water ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compareclwmr_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot lowest model level cloud water for: '+dom) % t3)

#################################
  # Plot lowest model level cloud ice
#################################
  t1 = time.perf_counter()
  print(('Working on lowest model level cloud ice for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  csref_1 = ax1.pcolormesh(lon_shift,lat_shift,refd_1,transform=transform,cmap=cmref,vmin=20,norm=normref)
  csref_1.cmap.set_under('white')
  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,icmr_1,transform=transform,cmap=cm,vmin=0.01,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('hotpink')
  cstmp_1 = ax1.contour(lon_shift,lat_shift,tmphyb_1,[0],colors='red',linewidths=0.5,transform=transform)
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=4)
  ax1.text(.5,1.03,'LAMDA Lowest Mdl Lvl Cld Ice ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  csref_2 = ax2.pcolormesh(lon_shift,lat_shift,refd_2,transform=transform,cmap=cmref,vmin=20,norm=normref)
  csref_2.cmap.set_under('white')
  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,icmr_2,transform=transform,cmap=cm,vmin=0.01,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('hotpink')
  cstmp_2 = ax2.contour(lon_shift,lat_shift,tmphyb_2,[0],colors='red',linewidths=0.5,transform=transform)
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=4)
  ax2.text(.5,1.03,'RRFS_A Lowest Mdl Lvl Cld Ice ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,icmr_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Lowest Mdl Lvl Cld Ice ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compareicmr_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot lowest model level cloud ice for: '+dom) % t3)

#################################
  # Plot lowest model level rain
#################################
  t1 = time.perf_counter()
  print(('Working on lowest model level rain for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  csref_1 = ax1.pcolormesh(lon_shift,lat_shift,refd_1,transform=transform,cmap=cmref,vmin=20,norm=normref)
  csref_1.cmap.set_under('white')
  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,rwmr_1,transform=transform,cmap=cm,vmin=0.01,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('hotpink')
  cstmp_1 = ax1.contour(lon_shift,lat_shift,tmphyb_1,[0],colors='red',linewidths=0.5,transform=transform)
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'LAMDA Lowest Mdl Lvl Rain ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  csref_2 = ax2.pcolormesh(lon_shift,lat_shift,refd_2,transform=transform,cmap=cmref,vmin=20,norm=normref)
  csref_2.cmap.set_under('white')
  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,rwmr_2,transform=transform,cmap=cm,vmin=0.01,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('hotpink')
  cstmp_2 = ax2.contour(lon_shift,lat_shift,tmphyb_2,[0],colors='red',linewidths=0.5,transform=transform)
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFS_A Lowest Mdl Lvl Rain ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,rwmr_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Lowest Mdl Lvl Rain ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparerwmr_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot lowest model level rain for: '+dom) % t3)

#################################
  # Plot lowest model level snow
#################################
  t1 = time.perf_counter()
  print(('Working on lowest model level snow for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  csref_1 = ax1.pcolormesh(lon_shift,lat_shift,refd_1,transform=transform,cmap=cmref,vmin=20,norm=normref)
  csref_1.cmap.set_under('white')
  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,snmr_1,transform=transform,cmap=cm,vmin=0.01,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('hotpink')
  cstmp_1 = ax1.contour(lon_shift,lat_shift,tmphyb_1,[0],colors='red',linewidths=0.5,transform=transform)
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'LAMDA Lowest Mdl Lvl Snow ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  csref_2 = ax2.pcolormesh(lon_shift,lat_shift,refd_2,transform=transform,cmap=cmref,vmin=20,norm=normref)
  csref_2.cmap.set_under('white')
  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,snmr_2,transform=transform,cmap=cm,vmin=0.01,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('hotpink')
  cstmp_2 = ax2.contour(lon_shift,lat_shift,tmphyb_2,[0],colors='red',linewidths=0.5,transform=transform)
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFS_A Lowest Mdl Lvl Snow ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,snmr_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Lowest Mdl Lvl Snow ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparesnmr_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot lowest model level snow for: '+dom) % t3)

#################################
  # Plot downward shortwave
#################################
  t1dom = time.perf_counter()
  t1 = time.perf_counter()
  print(('Working on downward shortwave for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'W m${^{-2}}$'
  clevs = np.arange(0,1025,25)
  clevsdif = [-300,-250,-200,-150,-100,-50,0,50,100,150,200,250,300]
  cm = plt.get_cmap(name='Spectral_r')
  cmdif = matplotlib.colors.ListedColormap(difcolors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,swdown_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Surface Downward Shortwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,swdown_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Surface Downward Shortwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,swdown_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Surface Downward Shortwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compareswdown_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot downward shortwave for: '+dom) % t3)

#################################
  # Plot upward shortwave
#################################
  t1 = time.perf_counter()
  print(('Working on upward shortwave for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'W m${^{-2}}$'
  clevs = np.arange(0,525,25)
  clevsdif = [-150,-125,-100,-75,-50,-25,0,25,50,75,100,125,150]
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,swup_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Surface Upward Shortwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,swup_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Surface Upward Shortwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,swup_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Surface Upward Shortwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compareswup_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot upward shortwave for: '+dom) % t3)

#################################
  # Plot downward longwave
#################################
  t1 = time.perf_counter()
  print(('Working on downward longwave for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'W m${^{-2}}$'
  clevs = np.arange(0,525,25)
  clevsdif = [-150,-125,-100,-75,-50,-25,0,25,50,75,100,125,150]
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,lwdown_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Surface Downward Longwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,lwdown_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Surface Downward Longwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,lwdown_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Surface Downward Longwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparelwdown_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot downward longwave for: '+dom) % t3)

#################################
  # Plot upward longwave
#################################
  t1 = time.perf_counter()
  print(('Working on upward longwave for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'W m${^{-2}}$'
  clevs = np.arange(0,525,25)
  clevsdif = [-150,-125,-100,-75,-50,-25,0,25,50,75,100,125,150]
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,lwup_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Surface Upward Longwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,lwup_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Surface Upward Longwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,lwup_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Surface Upward Longwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparelwup_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot upward longwave for: '+dom) % t3)

#################################
  # Plot ground heat flux
#################################
  t1 = time.perf_counter()
  print(('Working on ground heat flux for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'W m${^{-2}}$'
  clevs = [-300,-200,-100,-75,-50,-25,-10,0,10,25,50,75,100,200,300]
  clevsdif = [-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60]
  cm = ncepy.ncl_grnd_hflux()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,gdhfx_1,transform=transform,cmap=cm,norm=norm)
  cbar1 = plt.colorbar(cs_1,ax=ax1,ticks=clevs,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=4.5)
  ax1.text(.5,1.03,'LAMDA Ground Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,gdhfx_2,transform=transform,cmap=cm,norm=norm)
  cbar2 = plt.colorbar(cs_2,ax=ax2,ticks=clevs,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=4.5)
  ax2.text(.5,1.03,'RRFS_A Ground Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,gdhfx_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Ground Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparegdhfx_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot ground heat flux for: '+dom) % t3)

#################################
  # Plot latent heat flux
#################################
  t1 = time.perf_counter()
  print(('Working on latent heat flux for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'W m${^{-2}}$'
  clevs = [-2000,-1500,-1000,-750,-500,-300,-200,-100,-75,-50,-25,0,25,50,75,100,200,300,500,750,1000,1500,2000]
  clevsdif = [-150,-125,-100,-75,-50,-25,0,25,50,75,100,125,150]
  cm = ncepy.ncl_grnd_hflux()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,lhfx_1,transform=transform,cmap=cm,norm=norm)
  cbar1 = plt.colorbar(cs_1,ax=ax1,ticks=[-2000,-500,-100,-50,0,50,100,500,1000,2000],orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'LAMDA Latent Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,lhfx_2,transform=transform,cmap=cm,norm=norm)
  cbar2 = plt.colorbar(cs_2,ax=ax2,ticks=[-2000,-500,-100,-50,0,50,100,500,1000,2000],orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFS_A Latent Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,lhfx_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Latent Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparelhfx_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot latent heat flux for: '+dom) % t3)

#################################
  # Plot sensible heat flux
#################################
  t1 = time.perf_counter()
  print(('Working on sensible heat flux for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'W m${^{-2}}$'
  clevs = [-2000,-1500,-1000,-750,-500,-300,-200,-100,-75,-50,-25,0,25,50,75,100,200,300,500,750,1000,1500,2000]
  clevsdif = [-150,-125,-100,-75,-50,-25,0,25,50,75,100,125,150]
  cm = ncepy.ncl_grnd_hflux()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,snhfx_1,transform=transform,cmap=cm,norm=norm)
  cbar1 = plt.colorbar(cs_1,ax=ax1,ticks=[-2000,-500,-100,-50,0,50,100,500,1000,2000],orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'LAMDA Sensible Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,snhfx_2,transform=transform,cmap=cm,norm=norm)
  cbar2 = plt.colorbar(cs_2,ax=ax2,ticks=[-2000,-500,-100,-50,0,50,100,500,1000,2000],orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFS_A Sensible Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,snhfx_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Sensible Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparesnhfx_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot sensible heat flux for: '+dom) % t3)

#################################
  # Plot PBL height
#################################
  t1 = time.perf_counter()
  print(('Working on PBL height for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'm'
  clevs = [50,100,250,500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
  clevsdif = [-1800,-1500,-1200,-900,-600,-300,0,300,600,900,1200,1500,1800]
  colorlist= ['gray','blue','dodgerblue','cyan','mediumspringgreen','#FAFAD2','#EEEE00','#EEC900','darkorange','crimson','darkred','darkviolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,hpbl_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = plt.colorbar(cs_1,ax=ax1,ticks=clevs,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=4)
  ax1.text(.5,1.03,'LAMDA PBL Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,hpbl_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('black')
  cbar2 = plt.colorbar(cs_2,ax=ax2,ticks=clevs,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=4)
  ax2.text(.5,1.03,'RRFS_A PBL Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,hpbl_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA PBL Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparehpbl_'+dom+'_f'+fhour+'.png')
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
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'kg m${^{-2}}$'
  clevs = [0.001,0.005,0.01,0.05,0.1,0.25,0.5,1,2,4,6,10,15,20,25]
  clevsdif = [-6,-4,-2,-1,-0.5,-0.25,0,0.25,0.5,1,2,4,6]
  q_color_list = plt.cm.gist_stern_r(np.linspace(0, 1, len(clevs)+1))
  cm = matplotlib.colors.ListedColormap(q_color_list)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,cond_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Total Column Condensate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,cond_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Total Column Condensate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,cond_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Total Column Condensate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparecond_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Total condensate for: '+dom) % t3)

#################################
  # Plot total column liquid
#################################
  t1 = time.perf_counter()
  print(('Working on Total column liquid for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,tcolw_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Total Column Cloud Water + Rain ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,tcolw_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Total Column Cloud Water + Rain ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,cond_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Total Column Cloud Water + Rain ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparetcolw_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Total column liquid for: '+dom) % t3)

#################################
  # Plot total column ice
#################################
  t1 = time.perf_counter()
  print(('Working on Tcoli for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,tcoli_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Total Column Cloud Ice + Snow ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,tcoli_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Total Column Cloud Ice + Snow ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,cond_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Total Column Cloud Ice + Snow ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparetcoli_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Tcoli for: '+dom) % t3)

#################################
  # Plot soil type
#################################
#  if (fhr == 0):
  t1 = time.perf_counter()
  print('Working on soil type for '+dom)

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'Integer(0-16)'
  clevs = [-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5]
  clevsdif = [-0.1,0.1]
  colorlist = ['#00CDCD','saddlebrown','khaki','gray','#3D9140','palegreen','firebrick','lightcoral','darkorchid','plum','blue','lightskyblue','#CDAD00','yellow','#FF4500','lightsalmon','#CD1076']
  cm = matplotlib.colors.ListedColormap(colorlist)
  cmdif = matplotlib.colors.ListedColormap(difcolors2)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,sotyp_1,transform=transform,cmap=cm,norm=norm)
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'LAMDA Soil Type \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,sotyp_2,transform=transform,cmap=cm,norm=norm)
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFS_A Soil Type \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,sotyp_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkred')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,ticks=[0],extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Soil Type ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparesotyp_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot soil type for: '+dom) % t3)

#################################
  # Plot vegetation type
#################################
#  if (fhr == 0):
  t1 = time.perf_counter()
  print('Working on vegetation type for '+dom)

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'Integer(0-19)'
  clevs = [-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5]
  clevsdif = [-0.1,0.1]
  colorlist = ['#00CDCD','saddlebrown','khaki','gray','#3D9140','palegreen','firebrick','lightcoral','darkorchid','plum','blue','lightskyblue','#CDAD00','yellow','#FF4500','lightsalmon','#CD1076','mediumspringgreen','white','black']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,vgtyp_1,transform=transform,cmap=cm,norm=norm)
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'LAMDA Vegetation Type \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,vgtyp_2,transform=transform,cmap=cm,norm=norm)
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFS_A Vegetation Type \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,vgtyp_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkred')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,ticks=[0],extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Vegetation Type ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparevgtyp_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot vegetation type for: '+dom) % t3)

#################################
  # Plot vegetation fraction
#################################
  t1 = time.perf_counter()
  print(('Working on vegetation fraction for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '%'
  clevs = [10,20,30,40,50,60,70,80,90,100]
  clevsdif = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]
  cm = ncepy.cmap_q2m()
  cmdif = matplotlib.colors.ListedColormap(difcolors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,veg_1,transform=transform,cmap=cm,vmax=100,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white',alpha=0.)
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='min')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Vegetation Fraction ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,veg_2,transform=transform,cmap=cm,vmax=100,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('white',alpha=0.)
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='min')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Vegetation Fraction ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,veg_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Vegetation Fraction ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compareveg_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot vegetation fraction for: '+dom) % t3)

#################################
  # Plot 0-3 km Storm Relative Helicity
#################################
  t1 = time.perf_counter()
  print(('Working on 0-3 km SRH for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'm${^2}$ s$^{-2}$'
  clevs = [50,100,150,200,250,300,400,500,600,700,800]
  clevsdif = [-120,-100,-80,-60,-40,-20,0,20,40,60,80,100,120]
  colorlist = ['mediumblue','dodgerblue','chartreuse','limegreen','darkgreen','#EEEE00','orange','orangered','firebrick','darkmagenta']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,hel3km_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA 0-3 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,hel3km_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('black')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A 0-3 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,hel3km_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 0-3 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparehel3km_'+dom+'_f'+fhour+'.png')
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
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,hel1km_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('black')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA 0-1 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,hel1km_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('black')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A 0-1 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,hel1km_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA 0-1 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparehel1km_'+dom+'_f'+fhour+'.png')
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
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'dBZ'
  clevs = np.linspace(5,70,14)
  clevsdif = [20,1000]
  clevsboth = [1.5,2.5]
  colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  
  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,ref1km_1,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,ref1km_2,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  csdif = ax3.contourf(lon_shift,lat_shift,ref1km_1,clevsdif,colors='red',transform=transform)
  csdif2 = ax3.contourf(lon_shift,lat_shift,ref1km_2,clevsdif,colors='dodgerblue',transform=transform)
  csdif3 = ax3.contourf(lon_shift,lat_shift,ref1km_both,clevsboth,colors='indigo',transform=transform)
  ax3.text(.5,1.03,'LAMDA (red), RRFS_A (blue), Both (purple) \n 1-km Reflectivity > 20 ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  plt.savefig('compareref1km_'+dom+'_f'+fhour+'.png', format='png', bbox_inches='tight', dpi=300)
#  compress_and_save('compareref1km_'+dom+'_f'+fhour+'.png')
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
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'dBZ'
  clevs = np.linspace(5,70,14)
  clevsdif = [20,1000]
  clevsboth = [1.5,2.5]
  colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  
  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,refc_1,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Composite Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,refc_2,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Composite Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  csdif = ax3.contourf(lon_shift,lat_shift,refc_1,clevsdif,colors='red',transform=transform)
  csdif2 = ax3.contourf(lon_shift,lat_shift,refc_2,clevsdif,colors='dodgerblue',transform=transform)
  csdif3 = ax3.contourf(lon_shift,lat_shift,refc_both,clevsboth,colors='indigo',transform=transform)
  ax3.text(.5,1.03,'LAMDA (red), RRFS_A (blue), Both (purple) \n Composite Reflectivity > 20 ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  plt.savefig('comparerefc_'+dom+'_f'+fhour+'.png', format='png', bbox_inches='tight', dpi=300)
#  compress_and_save('comparerefc_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot composite reflectivity for: '+dom) % t3)


######################################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 2 variables for: "+dom) % t3dom)
  plt.clf()

######################################################

def plot_set_3():
  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,x,y,xextent,yextent,im,par,transform

#################################
  # Plot Max/Min Hourly 2-5 km UH
#################################
  t1dom = time.perf_counter()
  if (fhr > 0):
    t1 = time.perf_counter()
    print(('Working on Max/Min Hourly 2-5 km UH for '+dom))

    units = 'm${^2}$ s$^{-2}$'
    clevs = [-150,-100,-75,-50,-25,-10,0,10,25,50,75,100,150,200,250,300]
    clevsdif = [-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60]
#    colorlist = ['white','skyblue','mediumblue','green','orchid','firebrick','#EEC900','DarkViolet']
    colorlist = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','#E5E5E5','#E5E5E5','#EEEE00','#EEC900','darkorange','orangered','red','firebrick','mediumvioletred','darkviolet']
    cm = matplotlib.colors.ListedColormap(colorlist)
    cmdif = matplotlib.colors.ListedColormap(difcolors)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,uh25_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_under('darkblue')
    cs_1.cmap.set_over('black')
    cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'LAMDA 1-h Max/Min 2-5 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,uh25_2,transform=transform,cmap=cm,norm=norm)
    cs_2.cmap.set_under('darkblue')
    cs_2.cmap.set_over('black')
    cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFS_A 1-h Max/Min 2-5 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs = ax3.pcolormesh(lon_shift,lat_shift,uh25_dif,transform=transform,cmap=cmdif,norm=normdif)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('darkred')
    cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar3.set_label(units,fontsize=6)
    cbar3.ax.tick_params(labelsize=6)
    ax3.text(.5,1.03,'RRFS_A - LAMDA 1-h Max/Min 2-5 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    compress_and_save('compareuh25_'+dom+'_f'+fhour+'.png')
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly 2-5 km UH for: '+dom) % t3)

#################################
  # Plot Max/Min Hourly 0-3 km UH
#################################
    t1 = time.perf_counter()    
    print(('Working on Max/Min Hourly 0-3 km UH for '+dom))

  # Clear off old plottables but keep all the map info    
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()  
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)    
    clear_plotables(ax3,keep_ax_lst_3,fig)

    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,uh03_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_under('darkblue')
    cs_1.cmap.set_over('black')
    cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'LAMDA 1-h Max/Min 0-3 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,uh03_2,transform=transform,cmap=cm,norm=norm)
    cs_2.cmap.set_under('darkblue')
    cs_2.cmap.set_over('black')
    cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFS_A 1-h Max/Min 0-3 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs = ax3.pcolormesh(lon_shift,lat_shift,uh03_dif,transform=transform,cmap=cmdif,norm=normdif)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('darkred')
    cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar3.set_label(units,fontsize=6)
    cbar3.ax.tick_params(labelsize=6)
    ax3.text(.5,1.03,'RRFS_A - LAMDA 1-h Max/Min 0-3 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    compress_and_save('compareuh03_'+dom+'_f'+fhour+'.png')
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly 0-3 km UH for: '+dom) % t3)

#################################
  # Plot Max Hourly Updraft Speed
#################################
    t1 = time.perf_counter()    
    print(('Working on Max Hourly Updraft Speed for '+dom))

  # Clear off old plottables but keep all the map info    
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()  
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)    
    clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 'm s$^{-1}$'
    clevs = [0.5,1,2.5,5,7.5,10,12.5,15,20,25,30,35,40,50,75]
    clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
    colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia','mediumpurple']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,maxuvv_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_under('white')
    cs_1.cmap.set_over('black')
    cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels(clevs)
    cbar1.ax.tick_params(labelsize=5)
    ax1.text(.5,1.03,'LAMDA 1-h Max 100-1000 mb Updraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,maxuvv_2,transform=transform,cmap=cm,norm=norm)
    cs_2.cmap.set_under('white')
    cs_2.cmap.set_over('black')
    cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.set_xticklabels(clevs)
    cbar2.ax.tick_params(labelsize=5)
    ax2.text(.5,1.03,'RRFS_A 1-h Max 100-1000 mb Updraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs = ax3.pcolormesh(lon_shift,lat_shift,maxuvv_dif,transform=transform,cmap=cmdif,norm=normdif)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('darkred')
    cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar3.set_label(units,fontsize=6)
    cbar3.ax.tick_params(labelsize=6)
    ax3.text(.5,1.03,'RRFS_A - LAMDA 1-h Max 100-1000 mb Updraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    compress_and_save('comparemaxuvv_'+dom+'_f'+fhour+'.png')
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
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)    
    clear_plotables(ax3,keep_ax_lst_3,fig)

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
    cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels(clevs)
    cbar1.ax.tick_params(labelsize=5)
    ax1.text(.5,1.03,'LAMDA 1-h Max 100-1000 mb Downdraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,maxdvv_2,transform=transform,cmap=cm,norm=norm)
    cs_2.cmap.set_under('white')
    cs_2.cmap.set_over('black')
    cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.set_xticklabels(clevs)
    cbar2.ax.tick_params(labelsize=5)
    ax2.text(.5,1.03,'RRFS_A 1-h Max 100-1000 mb Downdraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs = ax3.pcolormesh(lon_shift,lat_shift,maxdvv_dif,transform=transform,cmap=cmdif,norm=normdif)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('darkred')
    cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar3.set_label(units,fontsize=6)
    cbar3.ax.tick_params(labelsize=6)
    ax3.text(.5,1.03,'RRFS_A - LAMDA 1-h Max 100-1000 mb Downdraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    compress_and_save('comparemaxdvv_'+dom+'_f'+fhour+'.png')
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
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

    units='dBz'
    clevs = np.linspace(5,70,14)
    clevsdif = [20,1000]
    colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,maxref1km_1,transform=transform,cmap=cm,vmin=5,norm=norm)
    cs_1.cmap.set_under('white',alpha=0.)
    cs_1.cmap.set_over('black')
    cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'LAMDA 1-h Max 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,maxref1km_2,transform=transform,cmap=cm,vmin=5,norm=norm)
    cs_2.cmap.set_under('white',alpha=0.)
    cs_2.cmap.set_over('black')
    cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFS_A 1-h Max 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    csdif = ax3.contourf(lon_shift,lat_shift,maxref1km_1,clevsdif,colors='red',transform=transform)
    csdif2 = ax3.contourf(lon_shift,lat_shift,maxref1km_2,clevsdif,colors='dodgerblue',transform=transform)
    ax3.text(.5,1.03,'LAMDA (red) and RRFS_A (blue) 1-h Max 1-km Reflectivity > 20 ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    compress_and_save('comparemaxref1km_'+dom+'_f'+fhour+'.png')
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
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 'kts'
    clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
    clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
    colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,maxwind_1,transform=transform,cmap=cm,vmin=5,norm=norm)
    cs_1.cmap.set_under('white',alpha=0.)
    cs_1.cmap.set_over('black')
    cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'LAMDA 1-h Max 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon_shift,lat_shift,maxwind_2,transform=transform,cmap=cm,vmin=5,norm=norm)
    cs_2.cmap.set_under('white',alpha=0.)
    cs_2.cmap.set_over('black')
    cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFS_A 1-h Max 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs = ax3.pcolormesh(lon_shift,lat_shift,maxwind_dif,transform=transform,cmap=cmdif,norm=normdif)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('darkred')
    cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar3.set_label(units,fontsize=6)
    cbar3.ax.tick_params(labelsize=6)
    ax3.text(.5,1.03,'RRFS_A - LAMDA 1-h Max 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    compress_and_save('comparemaxwind_'+dom+'_f'+fhour+'.png')
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly 10-m Wind Speed for: '+dom) % t3)

#################################
  # Plot Haines Index
#################################
  t1dom = time.perf_counter()
  t1 = time.perf_counter()
  print(('Working on Haines Index for '+dom))

  # Clear off old plottables but keep all the map info
  if (fhr > 0):
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

  units = ''
  clevs = [1.5,2.5,3.5,4.5,5.5,6.5]
  clevsdif = [-4,-3,-2,-1,0,1,2,3,4]
  colorlist = ['dodgerblue','limegreen','#EEEE00','darkorange','crimson']
  cm = matplotlib.colors.ListedColormap(colorlist)
  cmdif = matplotlib.colors.ListedColormap(difcolors3)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,hindex_1,transform=transform,cmap=cm,norm=norm)
  cbar1 = plt.colorbar(cs_1,ax=ax1,ticks=[2,3,4,5,6],orientation='horizontal',pad=0.01,shrink=0.8)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Haines Index \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,hindex_2,transform=transform,cmap=cm,norm=norm)
  cbar2 = plt.colorbar(cs_2,ax=ax2,ticks=[2,3,4,5,6],orientation='horizontal',pad=0.01,shrink=0.8)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Haines Index \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,hindex_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Haines Index \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparehindex_'+dom+'_f'+fhour+'.png')
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
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'kts'
  if dom == 'conus':
    skip = 80
  elif dom == 'southeast':
    skip = 35
  elif dom == 'colorado' or dom == 'la_vegas' or dom == 'mid_atlantic':
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
  clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
  clevsdif = [-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18]
  colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  cmdif = matplotlib.colors.ListedColormap(difcolors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,trans_1,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],utrans_1[::skip,::skip],vtrans_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax1.text(.5,1.03,'LAMDA Transport Wind ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,trans_2,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],utrans_2[::skip,::skip],vtrans_2[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax2.text(.5,1.03,'RRFS_A Transport Wind ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,trans_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Transport Wind ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparetrans_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot transport wind for: '+dom) % t3)

#################################
  # Plot Total Cloud Cover
#################################
  t1 = time.perf_counter()
  print(('Working on Total Cloud Cover for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '%'
  clevs = [0,10,20,30,40,50,60,70,80,90,100]
  clevsdif = [-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60]
  cm = plt.cm.BuGn
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,tcdc_1,transform=transform,cmap=cm,norm=norm)
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8)
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Total Cloud Cover ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,tcdc_2,transform=transform,cmap=cm,norm=norm)
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8)
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Total Cloud Cover ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,tcdc_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Total Cloud Cover ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparetcdc_'+dom+'_f'+fhour+'.png')
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
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'kft'
  clevs = [1,5,10,15,20,25,30,35,40]
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  colorlist = ['firebrick','tomato','lightsalmon','goldenrod','#EEEE00','palegreen','mediumspringgreen','limegreen']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,retop_1,transform=transform,cmap=cm,vmin=1,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('darkgreen')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Echo Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,retop_2,transform=transform,cmap=cm,vmin=1,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('darkgreen')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Echo Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,retop_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Echo Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compareretop_'+dom+'_f'+fhour+'.png')
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
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'mm/hr'
  clevs = [0.01,0.05,0.1,0.5,1,2.5,5,7.5,10,15,20,30,50,75,100]
  clevsdif = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
  colorlist = ['chartreuse','limegreen','green','darkgreen','blue','dodgerblue','deepskyblue','cyan','darkred','crimson','orangered','darkorange','goldenrod','gold']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,prate_1,transform=transform,cmap=cm,vmin=0.01,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('yellow')
  cbar1 = plt.colorbar(cs_1,ax=ax1,ticks=clevs,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'LAMDA Precipitation Rate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,prate_2,transform=transform,cmap=cm,vmin=0.01,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('yellow')
  cbar2 = plt.colorbar(cs_2,ax=ax2,ticks=clevs,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFS_A Precipitation Rate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,prate_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=5)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Precipitation Rate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compareprate_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Precipitation Rate for: '+dom) % t3)

#################################
  # Plot Cloud Base Pressure
#################################
  t1 = time.perf_counter()
  print(('Working on Cloud Base Pressure for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'mb'
  clevs = [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
  clevsdif = [-300,-250,-200,-150,-100,-50,0,50,100,150,200,250,300]
  hex=['#F00000','#F03800','#F55200','#F57200','#FA8900','#FFA200','#FFC800','#FFEE00','#BFFF00','#8CFF00','#11FF00','#05FF7E','#05F7FF','#05B8FF','#0088FF','#0055FF','#002BFF','#3700FF','#6E00FF','#A600FF','#E400F5']
  hex=hex[::-1]
  cm = matplotlib.colors.ListedColormap(hex)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,pbase_1,transform=transform,cmap=cm,vmin=50,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('red')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'LAMDA Pressure at Cloud Base ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,pbase_2,transform=transform,cmap=cm,vmin=50,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('red')
  cbar2 = plt.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS_A Pressure at Cloud Base ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,pbase_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  cbar3 = plt.colorbar(cs,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
  cbar3.set_label(units,fontsize=6)
  cbar3.ax.tick_params(labelsize=6)
  ax3.text(.5,1.03,'RRFS_A - LAMDA Pressure at Cloud Base ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('comparepbase_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Cloud Base Pressure for: '+dom) % t3)

#################################
  # Plot Cloud Top Pressure
#################################
  t1 = time.perf_counter()
  print(('Working on Cloud Top Pressure for '+dom))

  # Clear off old plottables but keep all the map info
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,ptop_1,transform=transform,cmap=cm,vmin=50,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('red')
  ax1.text(.5,1.03,'LAMDA Pressure at Cloud Top ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon_shift,lat_shift,ptop_2,transform=transform,cmap=cm,vmin=50,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('red')
  ax2.text(.5,1.03,'RRFS_A Pressure at Cloud Top ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs = ax3.pcolormesh(lon_shift,lat_shift,ptop_dif,transform=transform,cmap=cmdif,norm=normdif)
  cs.cmap.set_under('darkblue')
  cs.cmap.set_over('darkred')
  ax3.text(.5,1.03,'RRFS_A - LAMDA Pressure at Cloud Top ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save('compareptop_'+dom+'_f'+fhour+'.png')
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Cloud Top Pressure for: '+dom) % t3)

######################################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 3 variables for: "+dom) % t3dom)
  plt.clf()

######################################################


main()

