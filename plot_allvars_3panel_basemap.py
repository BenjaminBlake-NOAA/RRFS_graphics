import pygrib
import matplotlib
matplotlib.use('Agg')
import io
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.gridspec import GridSpec
try:
    from mpl_toolkits.basemap import Basemap, maskoceans
except:
    import mpl_toolkits
    mpl_toolkits.__path__.append('/gpfs/dell2/emc/modeling/noscrub/gwv/py/lib/python/basemap-1.2.1-py3.6-linux-x86_64.egg/mpl_toolkits/')
    from mpl_toolkits.basemap import Basemap, maskoceans
import numpy as np
import time,os,sys,multiprocessing
import multiprocessing.pool
import ncepy
from scipy import ndimage
import pyproj

#--------------Set some classes------------------------#
# Make Python process pools non-daemonic
class NoDaemonProcess(multiprocessing.Process):
  # make 'daemon' attribute always return False
  def _get_daemon(self):
    return False
  def _set_daemon(self, value):
    pass
  daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
  Process = NoDaemonProcess


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
  plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)

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
        xNorm=np.float(i)/(np.float(np.size(r))-1.0)
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
        xNorm=np.float(i)/(np.float(np.size(r))-1.0)
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
        xNorm=np.float(i)/(np.float(np.size(r))-1.0)
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
# plt.switch_backend('agg')

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
DATA_DIR_1 = str(sys.argv[3])
DATA_DIR_2 = str(sys.argv[4])
data1 = pygrib.open(DATA_DIR_1+'/fv3lam.t'+cyc+'z.conus.f'+fhour+'.grib2')
data2 = pygrib.open(DATA_DIR_2+'/fv3lam.t'+cyc+'z.conus.f'+fhour+'.grib2')

if (fhr > 2):
  data1_m1 = pygrib.open(DATA_DIR_1+'/fv3lam.t'+cyc+'z.conus.f'+fhour1+'.grib2')
  data2_m1 = pygrib.open(DATA_DIR_2+'/fv3lam.t'+cyc+'z.conus.f'+fhour1+'.grib2')
  data1_m2 = pygrib.open(DATA_DIR_1+'/fv3lam.t'+cyc+'z.conus.f'+fhour2+'.grib2')
  data2_m2 = pygrib.open(DATA_DIR_2+'/fv3lam.t'+cyc+'z.conus.f'+fhour2+'.grib2')
if (fhr >= 6):
  data1_m6 = pygrib.open(DATA_DIR_1+'/fv3lam.t'+cyc+'z.conus.f'+fhour6+'.grib2')
  data2_m6 = pygrib.open(DATA_DIR_2+'/fv3lam.t'+cyc+'z.conus.f'+fhour6+'.grib2')
if (fhr >= 24):
  data1_m24 = pygrib.open(DATA_DIR_1+'/fv3lam.t'+cyc+'z.conus.f'+fhour24+'.grib2')
  data2_m24 = pygrib.open(DATA_DIR_2+'/fv3lam.t'+cyc+'z.conus.f'+fhour24+'.grib2')

# Get the lats and lons
grids = [data1, data2]
lats = []
lons = []
lats_shift = []
lons_shift = []

for data in grids:
    # Unshifted grid for contours and wind barbs
    lat, lon = data[1].latlons()
    lats.append(lat)
    lons.append(lon)

    # Shift grid for pcolormesh
    lat1 = data[1]['latitudeOfFirstGridPointInDegrees']
    lon1 = data[1]['longitudeOfFirstGridPointInDegrees']
    try:
        nx = data[1]['Nx']
        ny = data[1]['Ny']
    except:
        nx = data[1]['Ni']
        ny = data[1]['Nj']
    dx = data[1]['DxInMetres']
    dy = data[1]['DyInMetres']
    pj = pyproj.Proj(data[1].projparams)
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
lat2 = lats[1]
lon2 = lons[1]

# Shifted lat/lon arrays for pcolormesh
lat_shift = lats_shift[0]
lon_shift = lons_shift[0]
lat2_shift = lats_shift[1]
lon2_shift = lons_shift[1]

Lat0 = data1[1]['LaDInDegrees']
Lon0 = data1[1]['LoVInDegrees']
#Lon0 = 262.5
print(Lat0)
print(Lon0)

# Forecast valid date/time
itime = ymdh
vtime = ncepy.ndate(itime,int(fhr))

# Specify plotting domains
#domains = ['conus','BN','CE','CO','LA','MA','NC','NE','NW','OV','SC','SE','SF','SP','SW','UM']
domains=['conus','BN','LA','SF']

###################################################
# Read in all variables and calculate differences #
###################################################
t1a = time.clock()

# Sea level pressure
slp_1 = data1.select(name='Pressure reduced to MSL')[0].values * 0.01
slpsmooth1 = ndimage.filters.gaussian_filter(slp_1, 13.78)
slp_2 = data2.select(name='Pressure reduced to MSL')[0].values * 0.01
slpsmooth2 = ndimage.filters.gaussian_filter(slp_2, 13.78)
slp_dif = slp_2 - slp_1

# 2-m temperature
tmp2m_1 = data1.select(name='2 metre temperature')[0].values
tmp2m_1 = (tmp2m_1 - 273.15)*1.8 + 32.0
tmp2m_2 = data2.select(name='2 metre temperature')[0].values
tmp2m_2 = (tmp2m_2 - 273.15)*1.8 + 32.0
tmp2m_dif = tmp2m_2 - tmp2m_1

# Surface temperature
tmpsfc_1 = data1.select(name='Temperature',typeOfLevel='surface')[0].values
tmpsfc_1 = (tmpsfc_1 - 273.15)*1.8 + 32.0
tmpsfc_2 = data2.select(name='Temperature',typeOfLevel='surface')[0].values
tmpsfc_2 = (tmpsfc_2 - 273.15)*1.8 + 32.0
tmpsfc_dif = tmpsfc_2 - tmpsfc_1

# 2-m dew point temperature
dew2m_1 = data1.select(name='2 metre dewpoint temperature')[0].values
dew2m_1 = (dew2m_1 - 273.15)*1.8 + 32.0
dew2m_2 = data2.select(name='2 metre dewpoint temperature')[0].values
dew2m_2 = (dew2m_2 - 273.15)*1.8 + 32.0
dew2m_dif = dew2m_2 - dew2m_1

# 2-m relative humidity
#rh2m_1 = data1.select(name='Relative humidity',level=2)[0].values
#rh2m_2 = data2.select(name='Relative humidity',level=2)[0].values
#rh2m_dif = rh2m_2 - rh2m_1

# 10-m wind speed
uwind_1 = data1.select(name='10 metre U wind component')[0].values * 1.94384
uwind_2 = data2.select(name='10 metre U wind component')[0].values * 1.94384
vwind_1 = data1.select(name='10 metre V wind component')[0].values * 1.94384
vwind_2 = data2.select(name='10 metre V wind component')[0].values * 1.94384
# Rotate winds from grid relative to Earth relative
#uwind_1, vwind_1 = ncepy.rotate_wind(Lat0,Lon0,lon,uwind_1,vwind_1,'lcc',inverse=False)
#uwind_2, vwind_2 = ncepy.rotate_wind(Lat0,Lon0,lon2,uwind_2,vwind_2,'lcc',inverse=False)
wspd10m_1 = np.sqrt(uwind_1**2 + vwind_1**2)
wspd10m_2 = np.sqrt(uwind_2**2 + vwind_2**2)
wspd10m_dif = wspd10m_2 - wspd10m_1

# Terrain height
terra_1 = data1.select(name='Orography')[0].values * 3.28084
terra_2 = data2.select(name='Orography')[0].values * 3.28084
terra_dif = terra_2 - terra_1

# Surface wind gust
gust_1 = data1.select(name='Wind speed (gust)')[0].values * 1.94384
gust_2 = data2.select(name='Wind speed (gust)')[0].values * 1.94384
gust_dif = gust_2 - gust_1

# Fosberg Index
#emc_1 = np.zeros_like(tmp2m_1)
#emc_1 = np.where(rh2m_1.any < 10, 0.03229 + (0.281073*rh2m_1) - (0.000578*rh2m_1*tmp2m_1), emc_1)
#emc_1 = np.where(10 <= rh2m_1.any <= 50, 2.22749 + (0.160107*rh2m_1) - (0.01478*tmp2m_1), emc_1)
#emc_1 = np.where(rh2m_1.any > 50, 21.0606 + (0.005565*(rh2m_1**2)) - (0.00035*rh2m_1*tmp2m_1) - (0.483199*rh2m_1), emc_1)
#mdc_1 = 1 - 2*(emc_1/30) + 1.5*((emc_1/30)**2) - 0.5*((emc_1/30)**3)
#ffwi_1 = (mdc_1 * np.sqrt(1 + wspd10m_1**2)) / 0.3002

#emc_2 = np.zeros_like(tmp2m_2)
#emc_2 = np.where(rh2m_2.any < 10, 0.03229 + (0.281073*rh2m_2) - (0.000578*rh2m_2*tmp2m_2), emc_2)
#emc_2 = np.where(10 <= rh2m_2.any <= 50, 2.22749 + (0.160107*rh2m_2) - (0.01478*tmp2m_2), emc_2)
#emc_2 = np.where(rh2m_2.any > 50, 21.0606 + (0.005565*(rh2m_2**2)) - (0.00035*rh2m_2*tmp2m_2) - (0.483199*rh2m_2), emc_2)
#mdc_2 = 1 - 2*(emc_2/30) + 1.5*((emc_2/30)**2) - 0.5*((emc_2/30)**3)
#ffwi_2 = (mdc_2 * np.sqrt(1 + wspd10m_2**2)) / 0.3002

#ffwi_dif = ffwi_2 - ffwi_1

# Most unstable CAPE
mucape_1 = data1.select(name='Convective available potential energy',topLevel=18000)[0].values
mucape_2 = data2.select(name='Convective available potential energy',topLevel=18000)[0].values
mucape_dif = mucape_2 - mucape_1

# Most Unstable CIN
mucin_1 = data1.select(name='Convective inhibition',topLevel=18000)[0].values
mucin_2 = data2.select(name='Convective inhibition',topLevel=18000)[0].values
mucin_dif = mucin_2 - mucin_1

# Surface-based CAPE
cape_1 = data1.select(name='Convective available potential energy',typeOfLevel='surface')[0].values
cape_2 = data2.select(name='Convective available potential energy',typeOfLevel='surface')[0].values
cape_dif = cape_2 - cape_1

# Surface-based CIN
sfcin_1 = data1.select(name='Convective inhibition',typeOfLevel='surface')[0].values
sfcin_2 = data2.select(name='Convective inhibition',typeOfLevel='surface')[0].values
sfcin_dif = sfcin_2 - sfcin_1

# Mixed Layer CAPE
mlcape_1 = data1.select(name='Convective available potential energy',topLevel=9000)[0].values
mlcape_2 = data2.select(name='Convective available potential energy',topLevel=9000)[0].values
mlcape_dif = mlcape_2 - mlcape_1

# Mixed Layer CIN
mlcin_1 = data1.select(name='Convective inhibition',topLevel=9000)[0].values
mlcin_2 = data2.select(name='Convective inhibition',topLevel=9000)[0].values
mlcin_dif = mlcin_2 - mlcin_1

# 850-mb equivalent potential temperature
t850_1 = data1.select(name='Temperature',level=850)[0].values
dpt850_1 = data1.select(name='Dew point temperature',level=850)[0].values
q850_1 = data1.select(name='Specific humidity',level=850)[0].values
tlcl_1 = 56.0 + (1.0/((1.0/(dpt850_1-56.0)) + 0.00125*np.log(t850_1/dpt850_1)))
thetae_1 = t850_1*((1000.0/850.0)**(0.2854*(1.0-(0.28*q850_1))))*np.exp(((3376.0/tlcl_1)-2.54)*q850_1*(1.0+(0.81*q850_1)))
t850_2 = data2.select(name='Temperature',level=850)[0].values
dpt850_2 = data2.select(name='Dew point temperature',level=850)[0].values
q850_2 = data2.select(name='Specific humidity',level=850)[0].values
tlcl_2 = 56.0 + (1.0/((1.0/(dpt850_2-56.0)) + 0.00125*np.log(t850_2/dpt850_2)))
thetae_2 = t850_2*((1000.0/850.0)**(0.2854*(1.0-(0.28*q850_2))))*np.exp(((3376.0/tlcl_2)-2.54)*q850_2*(1.0+(0.81*q850_2)))
thetae_dif = thetae_2 - thetae_1

# 850-mb winds
u850_1 = data1.select(name='U component of wind',level=850)[0].values * 1.94384
u850_2 = data2.select(name='U component of wind',level=850)[0].values * 1.94384
v850_1 = data1.select(name='V component of wind',level=850)[0].values * 1.94384
v850_2 = data2.select(name='V component of wind',level=850)[0].values * 1.94384
# Rotate winds from grid relative to Earth relative
#u850_1, v850_1 = ncepy.rotate_wind(Lat0,Lon0,lon,u850_1,v850_1,'lcc',inverse=False)
#u850_2, v850_2 = ncepy.rotate_wind(Lat0,Lon0,lon2,u850_2,v850_2,'lcc',inverse=False)

# 700-mb omega and relative humidity
omg700_1 = data1.select(name='Vertical velocity',level=700)[0].values
omg700_2 = data2.select(name='Vertical velocity',level=700)[0].values
rh700_1 = data1.select(name='Relative humidity',level=700)[0].values
rh700_2 = data2.select(name='Relative humidity',level=700)[0].values
rh700_dif = rh700_2 - rh700_1

# 500 mb height, wind, vorticity
z500_1 = data1.select(name='Geopotential Height',level=500)[0].values * 0.1
z500_1 = ndimage.filters.gaussian_filter(z500_1, 6.89)
z500_2 = data2.select(name='Geopotential Height',level=500)[0].values * 0.1
z500_2 = ndimage.filters.gaussian_filter(z500_2, 6.89)
z500_dif = z500_2 - z500_1
vort500_1 = data1.select(name='Absolute vorticity',level=500)[0].values * 100000
vort500_1 = ndimage.filters.gaussian_filter(vort500_1,1.7225)
vort500_1[vort500_1 > 1000] = 0	# Mask out undefined values on domain edge
vort500_2 = data2.select(name='Absolute vorticity',level=500)[0].values * 100000
vort500_2 = ndimage.filters.gaussian_filter(vort500_2,1.7225)
vort500_2[vort500_2 > 1000] = 0	# Mask out undefined values on domain edge
u500_1 = data1.select(name='U component of wind',level=500)[0].values * 1.94384
u500_2 = data2.select(name='U component of wind',level=500)[0].values * 1.94384
v500_1 = data1.select(name='V component of wind',level=500)[0].values * 1.94384
v500_2 = data2.select(name='V component of wind',level=500)[0].values * 1.94384
# Rotate winds from grid relative to Earth relative
#u500_1, v500_1 = ncepy.rotate_wind(Lat0,Lon0,lon,u500_1,v500_1,'lcc',inverse=False)
#u500_2, v500_2 = ncepy.rotate_wind(Lat0,Lon0,lon2,u500_2,v500_2,'lcc',inverse=False)

# 250 mb winds
u250_1 = data1.select(name='U component of wind',level=250)[0].values * 1.94384
u250_2 = data2.select(name='U component of wind',level=250)[0].values * 1.94384
v250_1 = data1.select(name='V component of wind',level=250)[0].values * 1.94384
v250_2 = data2.select(name='V component of wind',level=250)[0].values * 1.94384
# Rotate winds from grid relative to Earth relative
#u250_1, v250_1 = ncepy.rotate_wind(Lat0,Lon0,lon,u250_1,v250_1,'lcc',inverse=False)
#u250_2, v250_2 = ncepy.rotate_wind(Lat0,Lon0,lon2,u250_2,v250_2,'lcc',inverse=False)
wspd250_1 = np.sqrt(u250_1**2 + v250_1**2)
wspd250_2 = np.sqrt(u250_2**2 + v250_2**2)
wspd250_dif = wspd250_2 - wspd250_1

# Visibility
visemc_1 = data1.select(name='Visibility',typeOfLevel='surface')[0].values * 0.000621371
visemc_2 = data2.select(name='Visibility',typeOfLevel='surface')[0].values * 0.000621371
visemc_dif = visemc_2 - visemc_1

visgsd_1 = data1.select(name='Visibility',typeOfLevel='cloudTop')[0].values * 0.000621371
visgsd_2 = data2.select(name='Visibility',typeOfLevel='cloudTop')[0].values * 0.000621371
visgsd_dif = visgsd_2 - visgsd_1

# Cloud Base Height
zbase_1 = data1.select(name='Geopotential Height',typeOfLevel='cloudBase')[0].values * (3.28084/1000)
zbase_2 = data2.select(name='Geopotential Height',typeOfLevel='cloudBase')[0].values * (3.28084/1000)
zbase_dif = zbase_2 - zbase_1

# Cloud Ceiling Height
zceil_1 = data1.select(name='Geopotential Height',nameOfFirstFixedSurface='215')[0].values * (3.28084/1000)
zceil_2 = data2.select(name='Geopotential Height',nameOfFirstFixedSurface='215')[0].values * (3.28084/1000)
zceil_dif = zceil_2 - zceil_1

# Cloud Top Height
ztop_1 = data1.select(name='Geopotential Height',typeOfLevel='cloudTop')[0].values * (3.28084/1000)
ztop_2 = data2.select(name='Geopotential Height',typeOfLevel='cloudTop')[0].values * (3.28084/1000)
ztop_dif = ztop_2 - ztop_1

# Precipitable water
pw_1 = data1.select(name='Precipitable water',level=0)[0].values * 0.0393701
pw_2 = data2.select(name='Precipitable water',level=0)[0].values * 0.0393701
pw_dif = pw_2 - pw_1

# Percent of frozen precipitation
pofp_1 = data1.select(name='Percent frozen precipitation')[0].values
pofp_2 = data2.select(name='Percent frozen precipitation')[0].values
pofp_dif = pofp_2 - pofp_1

# Total precipitation
qpf_1 = data1.select(name='Total Precipitation',lengthOfTimeRange=fhr)[0].values * 0.0393701
qpf_2 = data2.select(name='Total Precipitation',lengthOfTimeRange=fhr)[0].values * 0.0393701
qpf_dif = qpf_2 - qpf_1

# 3-hr precipitation
if (fhr > 2):  # Do not make 3-hr plots for forecast hours 1 and 2
  qpfm2_1 = data1_m2.select(name='Total Precipitation',lengthOfTimeRange=1)[0].values * 0.0393701
  qpfm1_1 = data1_m1.select(name='Total Precipitation',lengthOfTimeRange=1)[0].values * 0.0393701
  qpfm0_1 = data1.select(name='Total Precipitation',lengthOfTimeRange=1)[0].values * 0.0393701
  qpf3_1 = qpfm2_1 + qpfm1_1 + qpfm0_1
  qpfm2_2 = data2_m2.select(name='Total Precipitation',lengthOfTimeRange=1)[0].values * 0.0393701
  qpfm1_2 = data2_m1.select(name='Total Precipitation',lengthOfTimeRange=1)[0].values * 0.0393701
  qpfm0_2 = data2.select(name='Total Precipitation',lengthOfTimeRange=1)[0].values * 0.0393701
  qpf3_2 = qpfm2_2 + qpfm1_2 + qpfm0_2
  qpf3_dif = qpf3_2 - qpf3_1

# Snow depth
snow_1 = data1.select(name='Snow depth')[0].values * 39.3701
snow_2 = data2.select(name='Snow depth')[0].values * 39.3701
snow_dif = snow_2 - snow_1
if (fhr >=6):	# Do not make 6-hr plots for forecast hours less than 6
  snowm6_1 = data1_m6.select(name='Snow depth')[0].values * 39.3701
  snow6_1 = snow_1 - snowm6_1 
  snowm6_2 = data2_m6.select(name='Snow depth')[0].values * 39.3701
  snow6_2 = snow_2 - snowm6_2
  snow6_dif = snow6_2 - snow6_1

# 1-km reflectivity
ref1km_1 = data1.select(name='Derived radar reflectivity',level=1000)[0].values
ref1km_2 = data2.select(name='Derived radar reflectivity',level=1000)[0].values

# Composite reflectivity
refc_1 = data1.select(name='Maximum/Composite radar reflectivity')[0].values 
refc_2 = data2.select(name='Maximum/Composite radar reflectivity')[0].values 

#Hybrid level 1 fields
clwmr_1 = data1.select(name='Cloud mixing ratio',typeOfLevel='hybrid',level=1)[0].values * 1000
clwmr_2 = data2.select(name='Cloud mixing ratio',typeOfLevel='hybrid',level=1)[0].values * 1000
clwmr_dif = clwmr_2 - clwmr_1

icmr_1 = data1.select(name='Ice water mixing ratio',typeOfLevel='hybrid',level=1)[0].values * 1000
icmr_2 = data2.select(name='Ice water mixing ratio',typeOfLevel='hybrid',level=1)[0].values * 1000
icmr_dif = icmr_2 - icmr_1

rwmr_1 = data1.select(name='Rain mixing ratio',typeOfLevel='hybrid',level=1)[0].values * 1000
rwmr_2 = data2.select(name='Rain mixing ratio',typeOfLevel='hybrid',level=1)[0].values * 1000
rwmr_dif = rwmr_2 - rwmr_1

snmr_1 = data1.select(name='Snow mixing ratio',typeOfLevel='hybrid',level=1)[0].values * 1000
snmr_2 = data2.select(name='Snow mixing ratio',typeOfLevel='hybrid',level=1)[0].values * 1000
snmr_dif = snmr_2 - snmr_1

grle_1 = data1.select(name='Graupel (snow pellets)',typeOfLevel='hybrid',level=1)[0].values * 1000
grle_2 = data2.select(name='Graupel (snow pellets)',typeOfLevel='hybrid',level=1)[0].values * 1000
grle_dif = grle_2 - grle_1

refd_1 = data1.select(name='Derived radar reflectivity',typeOfLevel='hybrid',level=1)[0].values
refd_2 = data2.select(name='Derived radar reflectivity',typeOfLevel='hybrid',level=1)[0].values

tmphyb_1 = data1.select(name='Temperature',typeOfLevel='hybrid',level=1)[0].values - 273.15
tmphyb_2 = data2.select(name='Temperature',typeOfLevel='hybrid',level=1)[0].values - 273.15

# Soil type - Integer (0-16) - only plot for f00
sotyp_1 = data1.select(name='Soil type')[0].values
sotyp_2 = data2.select(name='Soil type')[0].values
sotyp_dif = sotyp_2 - sotyp_1

# Vegetation Type - Integer (0-19) - only plot for f00
vgtyp_1 = data1.select(name='Vegetation Type')[0].values
vgtyp_2 = data2.select(name='Vegetation Type')[0].values
vgtyp_dif = vgtyp_2 - vgtyp_1

# Vegetation Fraction
veg_1 = data1.select(name='Vegetation')[0].values
veg_2 = data2.select(name='Vegetation')[0].values
veg_dif = veg_2 - veg_1

# Soil Temperature
tsoil_0_10_1 = (data1.select(name='Soil Temperature',scaledValueOfFirstFixedSurface=0)[0].values - 273.15)*1.8 + 32.0
tsoil_0_10_2 = (data2.select(name='Soil Temperature',scaledValueOfFirstFixedSurface=0)[0].values - 273.15)*1.8 + 32.0
tsoil_0_10_dif = tsoil_0_10_2 - tsoil_0_10_1

tsoil_10_40_1 = (data1.select(name='Soil Temperature',scaledValueOfFirstFixedSurface=10)[0].values - 273.15)*1.8 + 32.0
tsoil_10_40_2 = (data2.select(name='Soil Temperature',scaledValueOfFirstFixedSurface=10)[0].values - 273.15)*1.8 + 32.0
tsoil_10_40_dif = tsoil_10_40_2 - tsoil_10_40_1

tsoil_40_100_1 = (data1.select(name='Soil Temperature',scaledValueOfFirstFixedSurface=40)[0].values - 273.15)*1.8 + 32.0
tsoil_40_100_2 = (data2.select(name='Soil Temperature',scaledValueOfFirstFixedSurface=40)[0].values - 273.15)*1.8 + 32.0
tsoil_40_100_dif = tsoil_40_100_2 - tsoil_40_100_1

tsoil_100_200_1 = (data1.select(name='Soil Temperature',scaledValueOfFirstFixedSurface=100)[0].values - 273.15)*1.8 + 32.0
tsoil_100_200_2 = (data2.select(name='Soil Temperature',scaledValueOfFirstFixedSurface=100)[0].values - 273.15)*1.8 + 32.0
tsoil_100_200_dif = tsoil_100_200_2 - tsoil_100_200_1

# Soil Moisture
soilw_0_10_1 = data1.select(name='Volumetric soil moisture content',scaledValueOfFirstFixedSurface=0)[0].values
soilw_0_10_2 = data2.select(name='Volumetric soil moisture content',scaledValueOfFirstFixedSurface=0)[0].values
soilw_0_10_dif = soilw_0_10_2 - soilw_0_10_1

soilw_10_40_1 = data1.select(name='Volumetric soil moisture content',scaledValueOfFirstFixedSurface=10)[0].values
soilw_10_40_2 = data2.select(name='Volumetric soil moisture content',scaledValueOfFirstFixedSurface=10)[0].values
soilw_10_40_dif = soilw_10_40_2 - soilw_10_40_1

soilw_40_100_1 = data1.select(name='Volumetric soil moisture content',scaledValueOfFirstFixedSurface=40)[0].values
soilw_40_100_2 = data2.select(name='Volumetric soil moisture content',scaledValueOfFirstFixedSurface=40)[0].values
soilw_40_100_dif = soilw_40_100_2 - soilw_40_100_1

soilw_100_200_1 = data1.select(name='Volumetric soil moisture content',scaledValueOfFirstFixedSurface=100)[0].values
soilw_100_200_2 = data2.select(name='Volumetric soil moisture content',scaledValueOfFirstFixedSurface=100)[0].values
soilw_100_200_dif = soilw_100_200_2 - soilw_100_200_1

# Downward shortwave radiation
swdown_1 = data1.select(name='Downward short-wave radiation flux',stepType='instant')[0].values
swdown_2 = data2.select(name='Downward short-wave radiation flux',stepType='instant')[0].values
swdown_dif = swdown_2 - swdown_1

# Upward shortwave radiation
swup_1 = data1.select(name='Upward short-wave radiation flux',stepType='instant')[0].values
swup_2 = data2.select(name='Upward short-wave radiation flux',stepType='instant')[0].values
swup_dif = swup_2 - swup_1

# Downward longwave radiation
lwdown_1 = data1.select(name='Downward long-wave radiation flux',stepType='instant')[0].values
lwdown_2 = data2.select(name='Downward long-wave radiation flux',stepType='instant')[0].values
lwdown_dif = lwdown_2 - lwdown_1

# Upward longwave radiation
lwup_1 = data1.select(name='Upward long-wave radiation flux',stepType='instant',typeOfLevel='surface')[0].values
lwup_2 = data2.select(name='Upward long-wave radiation flux',stepType='instant',typeOfLevel='surface')[0].values
lwup_dif = lwup_2 - lwup_1

# Ground heat flux
gdhfx_1 = data1.select(name='Ground heat flux',stepType='instant',typeOfLevel='surface')[0].values
gdhfx_2 = data2.select(name='Ground heat flux',stepType='instant',typeOfLevel='surface')[0].values
gdhfx_dif = gdhfx_2 - gdhfx_1

# Latent heat flux
lhfx_1 = data1.select(name='Latent heat net flux',stepType='instant',typeOfLevel='surface')[0].values
lhfx_2 = data2.select(name='Latent heat net flux',stepType='instant',typeOfLevel='surface')[0].values
lhfx_dif = lhfx_2 - lhfx_1

# Sensible heat flux
snhfx_1 = data1.select(name='Sensible heat net flux',stepType='instant',typeOfLevel='surface')[0].values
snhfx_2 = data2.select(name='Sensible heat net flux',stepType='instant',typeOfLevel='surface')[0].values
snhfx_dif = snhfx_2 - snhfx_1

# PBL height
hpbl_1 = data1.select(name='Planetary boundary layer height')[0].values
hpbl_2 = data2.select(name='Planetary boundary layer height')[0].values
hpbl_dif = hpbl_2 - hpbl_1

# Total column condensate
cond_1 = data1.select(name='Total column-integrated condensate',stepType='instant')[0].values
cond_2 = data2.select(name='Total column-integrated condensate',stepType='instant')[0].values
cond_dif = cond_2 - cond_1

# Total column integrated liquid (cloud water + rain)
tqw_1 = data1.select(name='Total column-integrated cloud water',stepType='instant')[0].values
tqw_2 = data2.select(name='Total column-integrated cloud water',stepType='instant')[0].values
tqr_1 = data1.select(name='Total column integrated rain',stepType='instant')[0].values
tqr_2 = data2.select(name='Total column integrated rain',stepType='instant')[0].values
tcolw_1 = tqw_1 + tqr_1
tcolw_2 = tqw_2 + tqr_2
tcolw_dif = tcolw_2 - tcolw_1

# Total column integrated ice (cloud ice + snow)
tqi_1 = data1.select(name='Total column-integrated cloud ice',stepType='instant')[0].values
tqi_2 = data2.select(name='Total column-integrated cloud ice',stepType='instant')[0].values
tqs_1 = data1.select(name='Total column integrated snow',stepType='instant')[0].values
tqs_2 = data2.select(name='Total column integrated snow',stepType='instant')[0].values
tcoli_1 = tqi_1 + tqs_1
tcoli_2 = tqi_2 + tqs_2
tcoli_dif = tcoli_2 - tcoli_1

# 0-3 km Storm Relative Helicity
hel3km_1 = data1.select(name='Storm relative helicity',topLevel=3000,bottomLevel=0)[0].values
hel3km_2 = data2.select(name='Storm relative helicity',topLevel=3000,bottomLevel=0)[0].values
hel3km_dif = hel3km_2 - hel3km_1

# 0-1 km Storm Relative Helicity
hel1km_1 = data1.select(name='Storm relative helicity',topLevel=1000,bottomLevel=0)[0].values
hel1km_2 = data2.select(name='Storm relative helicity',topLevel=1000,bottomLevel=0)[0].values
hel1km_dif = hel1km_2 - hel1km_1

if (fhr > 0):
# Max/Min Hourly 2-5 km Updraft Helicity
  maxuh25_1 = data1.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
  maxuh25_2 = data2.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
  minuh25_1 = data1.select(stepType='min',parameterName="200",topLevel=5000,bottomLevel=2000)[0].values
  minuh25_2 = data2.select(stepType='min',parameterName="200",topLevel=5000,bottomLevel=2000)[0].values
  maxuh25_1[maxuh25_1 < 10] = 0
  maxuh25_2[maxuh25_2 < 10] = 0
  minuh25_1[minuh25_1 > -10] = 0
  minuh25_2[minuh25_2 > -10] = 0
  uh25_1 = maxuh25_1 + minuh25_1
  uh25_2 = maxuh25_2 + minuh25_2
  uh25_dif = uh25_2 - uh25_1

# Max/Min Hourly 0-3 km Updraft Helicity
  maxuh03_1 = data1.select(stepType='max',parameterName="199",topLevel=3000,bottomLevel=0)[0].values
  maxuh03_2 = data2.select(stepType='max',parameterName="199",topLevel=3000,bottomLevel=0)[0].values
  minuh03_1 = data1.select(stepType='min',parameterName="200",topLevel=3000,bottomLevel=0)[0].values
  minuh03_2 = data2.select(stepType='min',parameterName="200",topLevel=3000,bottomLevel=0)[0].values
  maxuh03_1[maxuh03_1 < 10] = 0
  maxuh03_2[maxuh03_2 < 10] = 0
  minuh03_1[minuh03_1 > -10] = 0
  minuh03_2[minuh03_2 > -10] = 0
  uh03_1 = maxuh03_1 + minuh03_1
  uh03_2 = maxuh03_2 + minuh03_2
  uh03_dif = uh03_2 - uh03_1

# Max Hourly Updraft Speed
  maxuvv_1 = data1.select(stepType='max',parameterName="220",typeOfLevel="isobaricLayer",topLevel=100,bottomLevel=1000)[0].values
  maxuvv_2 = data2.select(stepType='max',parameterName="220",typeOfLevel="isobaricLayer",topLevel=100,bottomLevel=1000)[0].values
  maxuvv_dif = maxuvv_2 - maxuvv_1

# Max Hourly Downdraft Speed
  maxdvv_1 = data1.select(stepType='max',parameterName="221",typeOfLevel="isobaricLayer",topLevel=100,bottomLevel=1000)[0].values * -1
  maxdvv_2 = data2.select(stepType='max',parameterName="221",typeOfLevel="isobaricLayer",topLevel=100,bottomLevel=1000)[0].values * -1
  maxdvv_dif = maxdvv_2 - maxdvv_1

# Max Hourly 1-km AGL reflectivity
  maxref1km_1 = data1.select(parameterName="198",typeOfLevel="heightAboveGround",level=1000)[0].values
  maxref1km_2 = data2.select(parameterName="198",typeOfLevel="heightAboveGround",level=1000)[0].values

# Max Hourly -10C reflectivity
  maxref10C_1 = data1.select(parameterName="198",typeOfLevel="isothermal",level=263)[0].values
  maxref10C_2 = data2.select(parameterName="198",typeOfLevel="isothermal",level=263)[0].values

# Max Hourly Wind
  maxwind_1 = data1.select(name='10 metre wind speed',stepType='max')[0].values * 1.94384
  maxwind_2 = data2.select(name='10 metre wind speed',stepType='max')[0].values * 1.94384
  maxwind_dif = maxwind_2 - maxwind_1

# Max 0-1 km Vertical Vorticity
  relv01_1 = data1.select(name='Vorticity (relative)',stepType='max',typeOfLevel='heightAboveGroundLayer',topLevel=1000,bottomLevel=0)[0].values
  relv01_2 = data2.select(name='Vorticity (relative)',stepType='max',typeOfLevel='heightAboveGroundLayer',topLevel=1000,bottomLevel=0)[0].values
  relv01_dif = relv01_2 - relv01_1

# Max 0-2 km Vertical Vorticity
  relv02_1 = data1.select(name='Vorticity (relative)',stepType='max',typeOfLevel='heightAboveGroundLayer',topLevel=2000,bottomLevel=0)[0].values
  relv02_2 = data2.select(name='Vorticity (relative)',stepType='max',typeOfLevel='heightAboveGroundLayer',topLevel=2000,bottomLevel=0)[0].values
  relv02_dif = relv02_2 - relv02_1

# Max Hybrid Level 1 Vertical Vorticity
  relvhyb_1 = data1.select(name='Vorticity (relative)',stepType='max',typeOfLevel='hybrid',level=1)[0].values
  relvhyb_2 = data2.select(name='Vorticity (relative)',stepType='max',typeOfLevel='heightAboveGroundLayer',topLevel=2000,bottomLevel=0)[0].values
  relvhyb_dif = relvhyb_2 - relvhyb_1

# Haines index
hindex_1 = data1.select(parameterName="2",typeOfLevel='surface')[0].values
hindex_2 = data2.select(parameterName="2",typeOfLevel='surface')[0].values
hindex_dif = hindex_2 - hindex_1

# Transport wind
utrans_1 = data1.select(name='U component of wind',nameOfFirstFixedSurface="220")[0].values * 1.94384
vtrans_1 = data1.select(name='V component of wind',nameOfFirstFixedSurface="220")[0].values * 1.94384
utrans_2 = data2.select(name='U component of wind',nameOfFirstFixedSurface="220")[0].values * 1.94384
vtrans_2 = data2.select(name='V component of wind',nameOfFirstFixedSurface="220")[0].values * 1.94384
# Rotate winds from grid relative to Earth relative
#utrans_1, vtrans_1 = ncepy.rotate_wind(Lat0,Lon0,lon,utrans_1,vtrans_1,'lcc',inverse=False)
#utrans_2, vtrans_2 = ncepy.rotate_wind(Lat0,Lon0,lon2,utrans_2,vtrans_2,'lcc',inverse=False)
trans_1 = np.sqrt(utrans_1**2 + vtrans_1**2)
trans_2 = np.sqrt(utrans_2**2 + vtrans_2**2)
trans_dif = trans_2 - trans_1

# Total cloud cover
tcdc_1 = data1.select(name='Total Cloud Cover')[0].values
tcdc_2 = data2.select(name='Total Cloud Cover',typeOfLevel='unknown')[0].values
tcdc_dif = tcdc_2 - tcdc_1

# Echo top height
retop_1 = data1.select(parameterName="197",stepType='instant',nameOfFirstFixedSurface='200')[0].values * (3.28084/1000)
retop_2 = data2.select(parameterName="197",stepType='instant',nameOfFirstFixedSurface='200')[0].values * (3.28084/1000)
retop_dif = retop_2 - retop_1

# Cloud base pressure
pbase_1 = data1.select(name='Pressure',typeOfLevel='cloudBase')[0].values * 0.01
pbase_2 = data2.select(name='Pressure',typeOfLevel='cloudBase')[0].values * 0.01
pbase_dif = pbase_2 - pbase_1

# Cloud top pressure
ptop_1 = data1.select(name='Pressure',typeOfLevel='cloudTop')[0].values * 0.01
ptop_2 = data2.select(name='Pressure',typeOfLevel='cloudTop')[0].values * 0.01
ptop_dif = ptop_2 - ptop_1

# Precipitation rate
prate_1 = data1.select(name='Precipitation rate')[0].values * 3600
prate_2 = data2.select(name='Precipitation rate')[0].values * 3600
prate_dif = prate_2 - prate_1


t2a = time.clock()
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

  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,m,x,y,x2,y2,x_shift,y_shift,x2_shift,y2_shift,xscale,yscale,im,par
  fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,m,x,y,x2,y2,x_shift,y_shift,x2_shift,y2_shift,xscale,yscale,im,par = create_figure()

  # Split plots into 2 sets with multiprocessing
  sets = [1,2,3]
#  sets = [1]
  pool2 = multiprocessing.Pool(len(sets))
  pool2.map(plot_sets,sets)

def create_figure():

  # create figure and axes instances
  fig = plt.figure()
  gs = GridSpec(9,9,wspace=0.0,hspace=0.0)
  ax1 = fig.add_subplot(gs[0:4,0:4])
  ax2 = fig.add_subplot(gs[0:4,5:])
  ax3 = fig.add_subplot(gs[5:,1:8])
  axes = [ax1, ax2, ax3]
  im = image.imread('noaa.png')
  par = 1

  # Map corners for each domain
  if dom == 'conus':
    llcrnrlon = -120.5
    llcrnrlat = 21.0 
    urcrnrlon = -64.5
    urcrnrlat = 49.0
    lat_0 = 35.4
    lon_0 = -97.6
    xscale=0.15
    yscale=0.2
  elif dom == 'BN':
    llcrnrlon = -75.75
    llcrnrlat = 40.0
    urcrnrlon = -69.5
    urcrnrlat = 43.0
    lat_0 = 41.0
    lon_0 = -74.6
    xscale=0.14
    yscale=0.19
  elif dom == 'CE':
    llcrnrlon = -103.0
    llcrnrlat = 32.5
    urcrnrlon = -88.5
    urcrnrlat = 41.5
    lat_0 = 35.0
    lon_0 = -97.0
    xscale=0.15
    yscale=0.18
  elif dom == 'CO':
    llcrnrlon = -110.5
    llcrnrlat = 35.0
    urcrnrlon = -100.5
    urcrnrlat = 42.0
    lat_0 = 38.0
    lon_0 = -105.0
    xscale=0.17
    yscale=0.18
  elif dom == 'LA':
    llcrnrlon = -121.0
    llcrnrlat = 32.0
    urcrnrlon = -114.0
    urcrnrlat = 37.0
    lat_0 = 34.0
    lon_0 = -114.0
    xscale=0.16
    yscale=0.18
  elif dom == 'MA':
    llcrnrlon = -82.0
    llcrnrlat = 36.5
    urcrnrlon = -73.5
    urcrnrlat = 42.0
    lat_0 = 37.5
    lon_0 = -80.0
    xscale=0.18
    yscale=0.18
  elif dom == 'NC':
    llcrnrlon = -111.0
    llcrnrlat = 39.0
    urcrnrlon = -93.5
    urcrnrlat = 49.0
    lat_0 = 44.5
    lon_0 = -102.0
    xscale=0.16
    yscale=0.18
  elif dom == 'NE':
    llcrnrlon = -80.0     
    llcrnrlat = 40.5
    urcrnrlon = -66.0
    urcrnrlat = 47.5
    lat_0 = 42.0
    lon_0 = -80.0
    xscale=0.16
    yscale=0.18
  elif dom == 'NW':
    llcrnrlon = -125.5     
    llcrnrlat = 40.5
    urcrnrlon = -109.0
    urcrnrlat = 49.5
    lat_0 = 44.0
    lon_0 = -116.0
    xscale=0.15
    yscale=0.18
  elif dom == 'OV':
    llcrnrlon = -91.5 
    llcrnrlat = 34.75
    urcrnrlon = -80.0
    urcrnrlat = 43.0
    lat_0 = 38.0
    lon_0 = -87.0          
    xscale=0.18
    yscale=0.17
  elif dom == 'SC':
    llcrnrlon = -108.0 
    llcrnrlat = 25.0
    urcrnrlon = -88.0
    urcrnrlat = 37.0
    lat_0 = 32.0
    lon_0 = -98.0      
    xscale=0.14
    yscale=0.18
  elif dom == 'SE':
    llcrnrlon = -91.5 
    llcrnrlat = 24.0
    urcrnrlon = -74.0
    urcrnrlat = 36.5
    lat_0 = 34.0
    lon_0 = -85.0
    xscale=0.17
    yscale=0.18
  elif dom == 'SF':
    llcrnrlon = -123.25 
    llcrnrlat = 37.25
    urcrnrlon = -121.25
    urcrnrlat = 38.5
    lat_0 = 37.5
    lon_0 = -121.0
    xscale=0.16
    yscale=0.19
  elif dom == 'SP':
    llcrnrlon = -125.0
    llcrnrlat = 45.0
    urcrnrlon = -119.5
    urcrnrlat = 49.2
    lat_0 = 46.0
    lon_0 = -120.0
    xscale=0.19
    yscale=0.18
  elif dom == 'SW':
    llcrnrlon = -125.0 
    llcrnrlat = 30.0
    urcrnrlon = -108.0
    urcrnrlat = 42.5
    lat_0 = 37.0
    lon_0 = -113.0
    xscale=0.17
    yscale=0.18
  elif dom == 'UM':
    llcrnrlon = -96.75 
    llcrnrlat = 39.75
    urcrnrlon = -81.0
    urcrnrlat = 49.0
    lat_0 = 44.0
    lon_0 = -91.5
    xscale=0.18
    yscale=0.18

  # Create basemap instance and set the dimensions
  for ax in axes:
    if dom == 'BN' or dom == 'LA' or dom == 'SF' or dom == 'SP':
      m = Basemap(ax=ax,projection='gnom',lat_0=lat_0,lon_0=lon_0,\
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                  resolution='h')
    elif dom == 'conus':
      m = Basemap(ax=ax,projection='gnom',lat_0=lat_0,lon_0=lon_0,\
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                  resolution='i')
    else:
      m = Basemap(ax=ax,projection='gnom',lat_0=lat_0,lon_0=lon_0,\
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                  resolution='l')
    m.fillcontinents(color='LightGrey',zorder=0)
    m.drawcoastlines(linewidth=0.75)
    m.drawstates(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
##  parallels = np.arange(0.,90.,10.)
##  map.drawparallels(parallels,labels=[1,0,0,0],fontsize=6)
##  meridians = np.arange(180.,360.,10.)
##  map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=6)
    x,y = m(lon,lat)
    x2,y2 = m(lon2,lat2)

    x_shift,y_shift   = m(lon_shift,lat_shift)
    x2_shift,y2_shift = m(lon2_shift,lat2_shift) 
 
  # Map/figure has been set up here, save axes instances for use again later
    if par == 1:
      keep_ax_lst_1 = ax.get_children()[:]
    elif par == 2:
      keep_ax_lst_2 = ax.get_children()[:]
    elif par == 3:
      keep_ax_lst_3 = ax.get_children()[:]

    par += 1
  par = 1

  return fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,m,x,y,x2,y2,x_shift,y_shift,x2_shift,y2_shift,xscale,yscale,im,par


def plot_sets(set):
# Add print to see if dom is being passed in
  print(('plot_sets dom variable '+dom))

  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,m,x,y,x2,y2,x_shift,y_shift,x2_shift,y2_shift,xscale,yscale,im,par

  if set == 1:
    plot_set_1()
  elif set == 2:
    plot_set_2()
  elif set == 3:
    plot_set_3()

def plot_set_1():
  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,m,x,y,x2,y2,x_shift,y_shift,x2_shift,y2_shift,xscale,yscale,im,par

################################
  # Plot SLP
################################
  t1dom = time.clock()
  t1 = time.clock()
  print(('Working on slp for '+dom))

  units = 'mb'
  clevs = [976,980,984,988,992,996,1000,1004,1008,1012,1016,1020,1024,1028,1032,1036,1040,1044,1048,1052]
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = plt.cm.Spectral_r
  cmdif = matplotlib.colors.ListedColormap(difcolors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1_a = m.pcolormesh(x_shift,y_shift,slpsmooth1,cmap=cm,norm=norm,ax=ax)  
      cbar1 = m.colorbar(cs1_a,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=5)
      cs1_b = m.contour(x,y,slpsmooth1,np.arange(940,1060,4),colors='black',linewidths=1.25,ax=ax)
      plt.clabel(cs1_b,inline=1,fmt='%d',fontsize=6,zorder=12,ax=ax)
      ax.text(.5,1.03,'FV3LAM SLP ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)
  # plot highs and lows - window parameter controls the number of highs and lows detected
      ncepy.plt_highs_and_lows(m,slp_1,lon,lat,mode='reflect',window=500)

    elif par == 2:
      cs2_a = m.pcolormesh(x_shift,y_shift,slpsmooth2,cmap=cm,norm=norm,ax=ax)  
      cbar2 = m.colorbar(cs2_a,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=5)
      cs2_b = m.contour(x2,y2,slpsmooth2,np.arange(940,1060,4),colors='black',linewidths=1.25,ax=ax)
      plt.clabel(cs2_b,inline=1,fmt='%d',fontsize=6,zorder=12,ax=ax)
      ax.text(.5,1.03,'FV3LAM-X SLP ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)
  # plot highs and lows - window parameter controls the number of highs and lows detected
      ncepy.plt_highs_and_lows(m,slp_2,lon,lat,mode='reflect',window=500)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,slp_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM SLP ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareslp_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot slp for: '+dom) % t3)

#################################
  # Plot 2-m T
#################################
  t1 = time.clock()
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
  clevsdif = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  cm = cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,tmp2m_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM 2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,tmp2m_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM-X 2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))       
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,tmp2m_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2')) 
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compare2mt_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 2mt for: '+dom) % t3)

#################################
# Plot SFCT
#################################
  t1 = time.clock()
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
  clevsdif = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  cm = cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,tmpsfc_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM Surface Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,tmpsfc_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM-X Surface Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,tmpsfc_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Surface Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))       
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparetsfc_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot tsfc for: '+dom) % t3)

#################################
  # Plot 2-m Dew Point
#################################
  t1 = time.clock()
  print(('Working on 2mdew for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = '\xb0''F'
  clevs = np.linspace(-5,80,35)
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = ncepy.cmap_q2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,dew2m_1,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM 2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,dew2m_2,cmap=cm,norm=norm,ax=ax)
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X 2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,dew2m_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compare2mdew_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 2mdew for: '+dom) % t3)

#################################
  # Plot 2-m Relative Humidity
#################################
#  t1 = time.clock()
#  print(('Working on 2-m Relative Humidity for '+dom))

  # Clear off old plottables but keep all the map info
#  cbar1.remove()
#  cbar2.remove()
#  cbar3.remove()
#  clear_plotables(ax1,keep_ax_lst_1,fig)
#  clear_plotables(ax2,keep_ax_lst_2,fig)
#  clear_plotables(ax3,keep_ax_lst_3,fig)

#  units = '%'
#  clevs = [0,10,20,30,40,50,60,70,80,90,100]
#  clevsdif = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]
#  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
#  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

#  for ax in axes:
#    xmin, xmax = ax.get_xlim()
#    ymin, ymax = ax.get_ylim()
#    xmax = int(round(xmax))
#    ymax = int(round(ymax))

#    if par == 1:
#      cs_1 = m.pcolormesh(x_shift,y_shift,rh2m_1,cmap=cm,norm=norm,ax=ax)
#      cbar1 = m.colorbar(cs_1,ax=ax,ticks=clevs,location='bottom',pad=0.05)
#      cbar1.set_label(units,fontsize=6)
#      cbar1.ax.tick_params(labelsize=6)
#      ax.text(.5,1.03,'FV3LAM 2-m Relative Humidity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
#      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

#    elif par == 2:
#      cs_2 = m.pcolormesh(x2_shift,y2_shift,rh2m_2,cmap=cm,norm=norm,ax=ax)
#      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05)
#      cbar2.set_label(units,fontsize=6)
#      cbar2.ax.tick_params(labelsize=6)
#      ax.text(.5,1.03,'FV3LAM-X 2-m Relative Humidity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
#      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

#    elif par == 3:
#      cs = m.pcolormesh(x2_shift,y2_shift,rh2m_dif,cmap=cmdif,norm=normdif,ax=ax)
#      cs.cmap.set_under('darkblue')
#      cs.cmap.set_over('darkred')
#      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
#      cbar3.set_label(units,fontsize=6)
#      cbar3.ax.tick_params(labelsize=6)
#      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 2-m Relative Humidity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
#      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

#    par += 1
#  par = 1

#  compress_and_save('compare2mrh_'+dom+'_f'+fhour+'.png')
#  t2 = time.clock()
#  t3 = round(t2-t1, 3)
#  print(('%.3f seconds to plot 2-m Relative Humidity for: '+dom) % t3)

#################################
  # Plot Fosberg Index
#################################
#  t1 = time.clock()
#  print(('Working on Filip Forsberg Index for '+dom))

  # Clear off old plottables but keep all the map info
#  cbar1.remove()
#  cbar2.remove()
#  cbar3.remove()
#  clear_plotables(ax1,keep_ax_lst_1,fig)
#  clear_plotables(ax2,keep_ax_lst_2,fig)
#  clear_plotables(ax3,keep_ax_lst_3,fig)

#  units = ''
#  clevs = [20,25,30,35,40,45,50,55,60,65,70]
#  clevsdif = [-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18]
#  colorlist = ['dodgerblue','darkturquoise','cyan','mediumspringgreen','#FFF68F','#EEEE00','#EEC900','darkorange','crimson','darkred']
#  cm = matplotlib.colors.ListedColormap(colorlist)
#  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
#  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

#  for ax in axes:
#    xmin, xmax = ax.get_xlim()
#    ymin, ymax = ax.get_ylim()
#    xmax = int(round(xmax))
#    ymax = int(round(ymax))

#    if par == 1:
#      cs_1 = m.pcolormesh(x_shift,y_shift,ffwi_1,cmap=cm,vmin=20,norm=norm,ax=ax)
#      cs_1.cmap.set_under('white',alpha=0.)
#      cs_1.cmap.set_over('darkviolet')
#      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='max')
#      cbar1.set_label(units,fontsize=6)
#      cbar1.ax.tick_params(labelsize=6)
#      ax.text(.5,1.03,'FV3LAM Fosberg Index \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
#      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

#    elif par == 2:
#      cs_2 = m.pcolormesh(x2_shift,y2_shift,ffwi_2,cmap=cm,vmin=20,norm=norm,ax=ax)
#      cs_1.cmap.set_under('white',alpha=0.)
#      cs_2.cmap.set_over('darkviolet')
#      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='max')
#      cbar2.set_label(units,fontsize=6)
#      cbar2.ax.tick_params(labelsize=6)
#      ax.text(.5,1.03,'FV3LAM-X Fosberg Index \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
#      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

#    elif par == 3:
#      cs = m.pcolormesh(x2_shift,y2_shift,ffwi_dif,cmap=cmdif,norm=normdif,ax=ax)
#      cs.cmap.set_under('darkblue')
#      cs.cmap.set_over('darkred')
#      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
#      cbar3.set_label(units,fontsize=6)
#      cbar3.ax.tick_params(labelsize=6)
#      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Fosberg Index \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
#      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

#    par += 1
#  par = 1

#  compress_and_save('compareffwi_'+dom+'_f'+fhour+'.png')
#  t2 = time.clock()
#  t3 = round(t2-t1, 3)
#  print(('%.3f seconds to plot Filip Forsberg Index for: '+dom) % t3)

#################################
  # Plot 10-m WSPD
#################################
  t1 = time.clock()
  print(('Working on 10mwspd for '+dom))

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
  elif dom == 'SE':
    skip = 35
  elif dom == 'CO' or dom == 'LA' or dom == 'MA':
    skip = 12
  elif dom == 'BN':
    skip = 10
  elif dom == 'SP':
    skip = 9
  elif dom == 'SF':
    skip = 3
  else:
    skip = 20
  barblength = 4

  clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
  clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  # Rotate winds to gnomonic projection
  urot_1, vrot_1 = m.rotate_vector(uwind_1,vwind_1,lon,lat)
  urot_2, vrot_2 = m.rotate_vector(uwind_2,vwind_2,lon2,lat2)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,wspd10m_1,cmap=cm,vmin=5,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('black')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],urot_1[::skip,::skip],vrot_1[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.03,'FV3LAM 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)
    
    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,wspd10m_2,cmap=cm,vmin=5,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('black')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],urot_2[::skip,::skip],vrot_2[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.03,'FV3LAM-X 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,wspd10m_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label('kts',fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 10-m Wind Speed (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))       
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compare10mwind_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 10mwspd for: '+dom) % t3)

#################################
  # Plot Terrain with 10-m WSPD
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,terra_1,cmap=cm,vmin=1,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('ghostwhite')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],urot_1[::skip,::skip],vrot_1[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.03,'FV3LAM Terrain Height ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,terra_2,cmap=cm,vmin=1,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('ghostwhite')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],urot_2[::skip,::skip],vrot_2[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.03,'FV3LAM-X Terrain Height ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,terra_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Terrain Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareterra_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Terrain for: '+dom) % t3)

#################################
  # Plot surface wind gust
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,gust_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('black')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.05,'FV3LAM Surface Wind Gust ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,gust_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('black')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.05,'FV3LAM-X Surface Wind Gust ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,gust_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Surface Wind Gust ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparegust_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot surface wind gust for: '+dom) % t3)

#################################
  # Plot Most Unstable CAPE/CIN
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,mucape_1,cmap=cm,vmin=100,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('black')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=4)
      cs_1b = m.contourf(x,y,mucin_1,clevs2,colors='none',hatches=['**','++','////','..'],ax=ax)
      ax.text(.5,1.05,'FV3LAM MUCAPE (shaded) and MUCIN (hatched) ('+units+') \n <-500 (*), -500<-250 (+), -250<-100 (/), -100<-25 (.) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,mucape_2,cmap=cm,vmin=100,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('black')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=4)
      cs_2b = m.contourf(x2,y2,mucin_2,clevs2,colors='none',hatches=['**','++','////','..'],ax=ax)
      ax.text(.5,1.05,'FV3LAM-X MUCAPE (shaded) and MUCIN (hatched) ('+units+') \n <-500 (*), -500<-250 (+), -250<-100 (/), -100<-25 (.) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,mucape_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,ticks=clevsdif,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=4)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Most Unstable CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparemucape_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot mucapecin for: '+dom) % t3)

#################################
  # Plot Surface-Based CAPE/CIN
#################################
  t1 = time.clock()
  print(('Working on sfcapecin for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,cape_1,cmap=cm,vmin=100,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('black')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=4)
      cs_1b = m.contourf(x,y,sfcin_1,clevs2,colors='none',hatches=['**','++','////','..'],ax=ax)
      ax.text(.5,1.05,'FV3LAM SFCAPE (shaded) and SFCIN (hatched) ('+units+') \n <-500 (*), -500<-250 (+), -250<-100 (/), -100<-25 (.) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,cape_2,cmap=cm,vmin=100,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('black')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=4)
      cs_2b = m.contourf(x2,y2,sfcin_2,clevs2,colors='none',hatches=['**','++','////','..'],ax=ax)
      ax.text(.5,1.05,'FV3LAM-X SFCAPE (shaded) and SFCIN (hatched) ('+units+') \n <-500 (*), -500<-250 (+), -250<-100 (/), -100<-25 (.) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,cape_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,ticks=clevsdif,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=4)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Surface-Based CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparesfcape_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot sfcapecin for: '+dom) % t3)

#################################
  # Plot Mixed Layer CAPE/CIN
#################################
  t1 = time.clock()
  print(('Working on mlcapecin for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,mlcape_1,cmap=cm,vmin=100,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('black')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=4)
      cs_1b = m.contourf(x,y,mlcin_1,clevs2,colors='none',hatches=['**','++','////','..'],ax=ax)
      ax.text(.5,1.05,'FV3LAM MLCAPE (shaded) and MLCIN (hatched) ('+units+') \n <-500 (*), -500<-250 (+), -250<-100 (/), -100<-25 (.) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,mlcape_2,cmap=cm,vmin=100,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('black')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=4)
      cs_2b = m.contourf(x2,y2,mlcin_2,clevs2,colors='none',hatches=['**','++','////','..'],ax=ax)
      ax.text(.5,1.05,'FV3LAM-X MLCAPE (shaded) and MLCIN (hatched) ('+units+') \n <-500 (*), -500<-250 (+), -250<-100 (/), -100<-25 (.) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,mlcape_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,ticks=clevsdif,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=4)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Mixed Layer CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparemlcape_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot mlcapecin for: '+dom) % t3)

#################################
  # Plot 850-mb THETAE
#################################
  t1 = time.clock()
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

  # Rotate winds to gnomonic projection
  urot_1, vrot_1 = m.rotate_vector(u850_1,v850_1,lon,lat)
  urot_2, vrot_2 = m.rotate_vector(u850_2,v850_2,lon2,lat2)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,thetae_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[270,276,282,288,294,300,306,312,318,324,330,336,342,348,354,360],extend='both')
      cbar1.set_label(units,fontsize=6)   
      cbar1.ax.tick_params(labelsize=4)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],urot_1[::skip,::skip],vrot_1[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.03,'FV3LAM 850 mb $\Theta$e ('+units+') and Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,thetae_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=[270,276,282,288,294,300,306,312,318,324,330,336,342,348,354,360],extend='both')
      cbar2.set_label(units,fontsize=6)   
      cbar2.ax.tick_params(labelsize=4)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],urot_2[::skip,::skip],vrot_2[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.03,'FV3LAM-X 850 mb $\Theta$e ('+units+') and Winds (kts) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)
    
    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,thetae_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)   
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 850 mb $\Theta$e ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compare850t_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 850 mb Theta-e for: '+dom) % t3)

#################################
  # Plot 700-mb OMEGA and RH
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1_a = m.pcolormesh(x_shift,y_shift,rh700_1,cmap=cm,vmin=50,norm=norm,ax=ax)
      cs1_a.cmap.set_under('white',alpha=0.)
      cbar1 = m.colorbar(cs1_a,ax=ax,location='bottom',pad=0.05,ticks=clevs)
      cbar1.set_label(units,fontsize=6) 
      cbar1.ax.tick_params(labelsize=6)
      cs1_b = m.pcolormesh(x_shift,y_shift,omg700_1,cmap=cmw,vmax=-5,norm=normw,ax=ax)
      cs1_b.cmap.set_over('white',alpha=0.)
      ax.text(.5,1.03,'FV3LAM 700 mb $\omega$ (rising motion in blue) and RH ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs2_a = m.pcolormesh(x2_shift,y2_shift,rh700_2,cmap=cm,vmin=50,norm=norm,ax=ax)
      cs2_a.cmap.set_under('white',alpha=0.)
      cbar2 = m.colorbar(cs2_a,ax=ax,location='bottom',pad=0.05,ticks=clevs)
      cbar2.set_label(units,fontsize=6) 
      cbar2.ax.tick_params(labelsize=6)
      cs2_b = m.pcolormesh(x2_shift,y2_shift,omg700_2,cmap=cmw,vmax=-5,norm=normw,ax=ax)
      cs2_b.cmap.set_over('white',alpha=0.)
      ax.text(.5,1.03,'FV3LAM-X 700 mb $\omega$ (rising motion in blue) and RH ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,rh700_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 700 mb RH ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compare700_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 700 mb $\omega$ and RH for: '+dom) % t3)

#################################
  # Plot 500 mb HGT/WIND/VORT
#################################
  t1 = time.clock()
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

  # Rotate winds to gnomonic projection
  urot_1, vrot_1 = m.rotate_vector(u500_1,v500_1,lon,lat)
  urot_2, vrot_2 = m.rotate_vector(u500_2,v500_2,lon2,lat2)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs1_a = m.pcolormesh(x_shift,y_shift,vort500_1,cmap=cm,norm=norm,ax=ax)
      cs1_a.cmap.set_under('white')
      cs1_a.cmap.set_over('darkred')
      cbar1 = m.colorbar(cs1_a,ax=ax,location='bottom',pad=0.05,ticks=vortlevs,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)

      # plot vorticity maxima as black X's
      local_max = extrema(vort500_1,mode='wrap',window=100)
      xhighs = lon[local_max]
      yhighs = lat[local_max]
      highvals = vort500_1[local_max]
      xyplotted = []
      # don't plot if there is already a X within dmin meters
      yoffset = 0.022*(m.ymax - m.ymin)
      dmin = yoffset
      for x,y,p in zip(xhighs, yhighs, highvals):
        if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin and p > 35:
          dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
          if not dist or min(dist) > dmin:
            ax.text(x,y,'x',fontsize=6,fontweight='bold',\
                    ha='center',va='center',color='black',zorder=10)
            xyplotted.append((x,y))

      m.barbs(lon[::skip,::skip],lat[::skip,::skip],urot_1[::skip,::skip],vrot_1[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='steelblue',ax=ax)
      x,y = m(lon,lat)	# need to redefine to avoid index error
      cs1_b = m.contour(x,y,z500_1,np.arange(486,600,6),colors='black',linewidths=1,ax=ax)
      plt.clabel(cs1_b,inline_spacing=1,fmt='%d',fontsize=6,dorder=12,ax=ax)
      ax.text(.5,1.03,'FV3LAM 500 mb Heights (dam), Winds (kts), and $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs2_a = m.pcolormesh(x2_shift,y2_shift,vort500_2,cmap=cm,norm=norm,ax=ax)
      cs2_a.cmap.set_under('white')
      cs2_a.cmap.set_over('darkred')
      cbar2 = m.colorbar(cs2_a,ax=ax,location='bottom',pad=0.05,ticks=vortlevs,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)

      # plot vorticity maxima as black X's
      local_max = extrema(vort500_2,mode='wrap',window=100)
      xhighs = lon[local_max]
      yhighs = lat[local_max]
      highvals = vort500_2[local_max]
      xyplotted = []
      # don't plot if there is already a X within dmin meters
      yoffset = 0.022*(m.ymax - m.ymin)
      dmin = yoffset
      for x,y,p in zip(xhighs, yhighs, highvals):
        if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin and p > 35:
          dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
          if not dist or min(dist) > dmin:
            ax.text(x,y,'x',fontsize=6,fontweight='bold',\
                    ha='center',va='center',color='black',zorder=10)
            xyplotted.append((x,y))

      m.barbs(lon[::skip,::skip],lat[::skip,::skip],urot_2[::skip,::skip],vrot_2[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='steelblue',ax=ax)
#      x,y = m(lon,lat)	# need to redefine to avoid index error
      cs2_b = m.contour(x2,y2,z500_2,np.arange(486,600,6),colors='black',linewidths=1,ax=ax)
      plt.clabel(cs2_b,inline_spacing=1,fmt='%d',fontsize=6,dorder=12,ax=ax)
      ax.text(.5,1.03,'FV3LAM-X 500 mb Heights (dam), Winds (kts), and $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,z500_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6) 
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 500 mb Heights (dam) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compare500_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 500 mb Hgt/Wind/Vort for: '+dom) % t3)

#################################
  # Plot 250 mb WIND
#################################
  t1 = time.clock()
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

  # Rotate winds to gnomonic projection
  urot_1, vrot_1 = m.rotate_vector(u250_1,v250_1,lon,lat)
  urot_2, vrot_2 = m.rotate_vector(u250_2,v250_2,lon2,lat2)

  x,y = m(lon,lat)	# need to redefine to avoid index error

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,wspd250_1,cmap=cm,vmin=50,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('red')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],urot_1[::skip,::skip],vrot_1[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.03,'FV3LAM 250 mb Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,wspd250_2,cmap=cm,vmin=50,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('red')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],urot_2[::skip,::skip],vrot_2[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.03,'FV3LAM-X 250 mb Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,wspd250_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6) 
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 250 mb Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1
   
  compress_and_save('compare250wind_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 250 mb WIND for: '+dom) % t3)

#################################
  # Plot EMC Visibility
#################################
  t1dom = time.clock()
  t1 = time.clock()
  print(('Working on EMC Surface Visibility for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'miles'
  clevs = [0.25,0.5,1,2,3,4,5,10]
  clevsdif = [-15,-12.5,-10,-7.5,-5,-2.5,0,2.5,5,7.5,10,12.5,15]
  colorlist = ['salmon','goldenrod','#EEEE00','palegreen','darkturquoise','blue','mediumpurple']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,visemc_1,cmap=cm,vmax=10,norm=norm,ax=ax)
      cs_1.cmap.set_under('firebrick')
      cs_1.cmap.set_over('white',alpha=0.)
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='min')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM EMC Surface Visibility ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,visemc_2,cmap=cm,vmax=10,norm=norm,ax=ax)
      cs_2.cmap.set_under('firebrick')
      cs_2.cmap.set_over('white',alpha=0.)
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='min')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X EMC Surface Visibility ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,visemc_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6) 
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM EMC Surface Visibility ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparevis_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot EMC Surface Visibility for: '+dom) % t3)


#################################
  # Plot GSD Surface Visibility
#################################
  t1 = time.clock()
  print(('Working on GSD Surface Visibility for '+dom))

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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,visgsd_1,cmap=cm,vmax=10,norm=norm,ax=ax)
      cs_1.cmap.set_under('firebrick')
      cs_2.cmap.set_over('white',alpha=0.)
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='min')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM GSD Surface Visibility ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,visgsd_2,cmap=cm,vmax=10,norm=norm,ax=ax)
      cs_2.cmap.set_under('firebrick')
      cs_2.cmap.set_over('white',alpha=0.)
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='min')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X GSD Surface Visibility ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,visgsd_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6) 
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM GSD Surface Visibility ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparevis2_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot GSD Surface Visibility for: '+dom) % t3)

#################################
  # Plot Cloud Base Height
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,zbase_1,cmap=cm,vmin=0,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('darkgreen')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Cloud Base Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,zbase_2,cmap=cm,vmin=0,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('darkgreen')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Cloud Base Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,zbase_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,ticks=clevsdif,extend='both')
      cbar3.set_label(units,fontsize=6) 
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Cloud Base Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparezbase_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Cloud Base Height for: '+dom) % t3)

#################################
  # Plot Cloud Ceiling Height
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,zceil_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Cloud Ceiling Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,zceil_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Cloud Ceiling Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,zceil_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,ticks=clevsdif,extend='both')
      cbar3.set_label(units,fontsize=6) 
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Cloud Ceiling Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparezceil_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Cloud Ceiling Height for: '+dom) % t3)

#################################
  # Plot Cloud Top Height
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,ztop_1,cmap=cm,vmin=0,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('darkgreen')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Cloud Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,ztop_2,cmap=cm,vmin=0,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('darkgreen')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Cloud Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,ztop_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,ticks=clevsdif,extend='both')
      cbar3.set_label(units,fontsize=6) 
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Cloud Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareztop_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Cloud Top Height for: '+dom) % t3)

#################################
  # Plot PW
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,pw_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('hotpink')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM Precipitable Water ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,pw_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('hotpink')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM-X Precipitable Water ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,pw_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,ticks=clevsdif,extend='both')
      cbar3.set_label(units,fontsize=6) 
      cbar3.ax.tick_params(labelsize=4)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Precipitable Water ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparepw_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot PW for: '+dom) % t3)

#################################
  # Plot % FROZEN PRECIP
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,pofp_1,cmap=cm,vmin=10,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs)
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Percent of Frozen Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,pofp_2,cmap=cm,vmin=10,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs)
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Percent of Frozen Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,pofp_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,ticks=clevsdif,extend='both')
      cbar3.set_label(units,fontsize=6) 
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Percent of Frozen Precipitaion ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparepofp_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot PERCENT FROZEN PRECIP for: '+dom) % t3)


#################################
  # Plot Total QPF
#################################
  if (fhr > 0):
    t1 = time.clock()
    print(('Working on total qpf for '+dom))

    # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 'in'
    clevs = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,7,10,15,20]
    clevsdif = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
    colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','deepskyblue','cyan','mediumpurple','mediumorchid','darkmagenta','darkred','crimson','orangered','darkorange','goldenrod','gold','yellow']  
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,qpf_1,cmap=cm,vmin=0.01,norm=norm,ax=ax)
        cs_1.cmap.set_under('white',alpha=0.)
        cs_1.cmap.set_over('pink')
        cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar1.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,qpf_2,cmap=cm,vmin=0.01,norm=norm,ax=ax)
        cs_2.cmap.set_under('white',alpha=0.)
        cs_2.cmap.set_over('pink')
        cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar2.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        cs = m.pcolormesh(x2_shift,y2_shift,qpf_dif,cmap=cmdif,norm=normdif,ax=ax)
        cs.cmap.set_under('darkblue')
        cs.cmap.set_over('darkred')
        cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar3.set_label(units,fontsize=6)
        cbar3.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X - FV3LAM '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))         
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('compareqpf_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot total qpf for: '+dom) % t3)

#################################
  # Plot QPF3
#################################
  if (fhr % 3 == 0) and (fhr > 0):
    t1 = time.clock()
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
   
    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,qpf3_1,cmap=cm,vmin=0.01,norm=norm,ax=ax)
        cs_1.cmap.set_under('white',alpha=0.)
        cs_1.cmap.set_over('pink')
        cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar1.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM 3-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,qpf3_2,cmap=cm,vmin=0.01,norm=norm,ax=ax)
        cs_2.cmap.set_under('white',alpha=0.)
        cs_2.cmap.set_over('pink')
        cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar2.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X 3-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        cs = m.pcolormesh(x2_shift,y2_shift,qpf3_dif,cmap=cmdif,norm=normdif,ax=ax)
        cs.cmap.set_under('darkblue')
        cs.cmap.set_over('darkred')
        cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar3.set_label(units,fontsize=6)
        cbar3.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X - FV3LAM 3-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))         
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('compareqpf3_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot qpf3 for: '+dom) % t3)

#################################
  # Plot snow depth
#################################
  t1 = time.clock()
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
 
  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,snow_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,snow_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,snow_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))         
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparesnow_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot snow depth for: '+dom) % t3)

#################################
  # Plot 6-hr change in snow depth
#################################
  if (fhr % 3 == 0) and (fhr >= 6):
    t1 = time.clock()
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

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,snow6_1,cmap=cm,norm=norm,ax=ax)
        cs_1.cmap.set_under('darkblue')
        cs_1.cmap.set_over('darkred')
        cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.set_xticklabels(clevs)
        cbar1.ax.tick_params(labelsize=5)
        ax.text(.5,1.03,'FV3LAM 6-hr Change in Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,snow6_2,cmap=cm,norm=norm,ax=ax)
        cs_2.cmap.set_under('darkblue')
        cs_2.cmap.set_over('darkred')
        cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='both')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.set_xticklabels(clevs)
        cbar2.ax.tick_params(labelsize=5)
        ax.text(.5,1.03,'FV3LAM-X 6-hr Change in Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        cs = m.pcolormesh(x2_shift,y2_shift,snow6_dif,cmap=cmdif,norm=normdif,ax=ax)
        cs.cmap.set_under('darkblue')
        cs.cmap.set_over('darkred')
        cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar3.set_label(units,fontsize=6)
        cbar3.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X - FV3LAM 6-hr Change in Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('comparesnow6_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot snow depth for: '+dom) % t3)


  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 1 variables for: "+dom) % t3dom)
  plt.clf()


################################################################################

def plot_set_2():
  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,m,x,y,x2,y2,x_shift,y_shift,x2_shift,y2_shift,xscale,yscale,im,par

#################################
  # Plot 0-10cm soil temperature
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      mysoilt = maskoceans(lon_shift,lat_shift,tsoil_0_10_1,inlands=True,resolution='l')
      cs_1 = m.pcolormesh(lon_shift,lat_shift,mysoilt,cmap=cm,norm=norm,latlon=True,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM 0-10 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_0_10_2,inlands=True,resolution='l')
      cs_2 = m.pcolormesh(lon2_shift,lat2_shift,mysoilt,cmap=cm,norm=norm,latlon=True,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X 0-10 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_0_10_dif,inlands=True,resolution='l')
      cs = m.pcolormesh(lon2_shift,lat2_shift,mysoilt,cmap=cmdif,norm=normdif,latlon=True,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 0-10 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparetsoil_0_10_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 0-10 cm soil temperature for: '+dom) % t3)

#################################
  # Plot 10-40cm soil temperature
#################################
  t1 = time.clock()
  print(('Working on 10-40 cm soil temperature for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      mysoilt = maskoceans(lon_shift,lat_shift,tsoil_10_40_1,inlands=True,resolution='l')
      cs_1 = m.pcolormesh(lon_shift,lat_shift,mysoilt,cmap=cm,norm=norm,latlon=True,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM 10-40 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_10_40_2,inlands=True,resolution='l')
      cs_2 = m.pcolormesh(lon2_shift,lat2_shift,mysoilt,cmap=cm,norm=norm,latlon=True,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X 10-40 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_10_40_dif,inlands=True,resolution='l')
      cs = m.pcolormesh(lon2_shift,lat2_shift,mysoilt,cmap=cmdif,norm=normdif,latlon=True,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 10-40 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparetsoil_10_40_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 10-40 cm soil temperature for: '+dom) % t3)

#################################
  # Plot 40-100cm soil temperature
#################################
  t1 = time.clock()
  print(('Working on 40-100 cm soil temperature for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      mysoilt = maskoceans(lon_shift,lat_shift,tsoil_40_100_1,inlands=True,resolution='l')
      cs_1 = m.pcolormesh(lon_shift,lat_shift,mysoilt,cmap=cm,norm=norm,latlon=True,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM 40-100 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_40_100_2,inlands=True,resolution='l')
      cs_2 = m.pcolormesh(lon2_shift,lat2_shift,mysoilt,cmap=cm,norm=norm,latlon=True,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X 40-100 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_40_100_dif,inlands=True,resolution='l')
      cs = m.pcolormesh(lon2_shift,lat2_shift,mysoilt,cmap=cmdif,norm=normdif,latlon=True,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 40-100 cm Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparetsoil_40_100_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 40-100 cm soil temperature for: '+dom) % t3)

#################################
  # Plot 100-200 cm soil temperature
#################################
  t1 = time.clock()
  print(('Working on 100-200 cm soil temperature for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      mysoilt = maskoceans(lon_shift,lat_shift,tsoil_100_200_1,inlands=True,resolution='l')
      cs_1 = m.pcolormesh(lon_shift,lat_shift,mysoilt,cmap=cm,norm=norm,latlon=True,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM 1-2 m Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_100_200_2,inlands=True,resolution='l')
      cs_2 = m.pcolormesh(lon2,lat2,mysoilt,cmap=cm,norm=norm,latlon=True,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X 1-2 m Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      mysoilt = maskoceans(lon2_shift,lat2_shift,tsoil_100_200_dif,inlands=True,resolution='l')
      cs = m.pcolormesh(lon2_shift,lat2_shift,mysoilt,cmap=cmdif,norm=normdif,latlon=True,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 1-2 m Soil Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparetsoil_100_200_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 100-200 cm soil temperature for: '+dom) % t3)

#################################
  # Plot 0-10 cm Soil Moisture Content
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,soilw_0_10_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('darkred')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM 0-10 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,soilw_0_10_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('darkred')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X 0-10 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,soilw_0_10_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 0-10 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparesoilw_0_10_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 0-10 cm soil moisture content for: '+dom) % t3)

#################################
  # Plot 10-40 cm Soil Moisture Content
#################################
  t1 = time.clock()
  print(('Working on 10-40 cm soil moisture for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,soilw_10_40_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('darkred')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM 10-40 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,soilw_10_40_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('darkred')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X 10-40 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,soilw_10_40_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 10-40 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparesoilw_10_40_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 10-40 cm soil moisture content for: '+dom) % t3)

#################################
  # Plot 40-100 cm Soil Moisture Content
#################################
  t1 = time.clock()
  print(('Working on 40-100 cm soil moisture for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,soilw_40_100_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('darkred')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM 40-100 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,soilw_40_100_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('darkred')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X 40-100 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,soilw_40_100_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 40-100 cm Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparesoilw_40_100_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 40-100 cm soil moisture content for: '+dom) % t3)

#################################
  # Plot 1-2 m Soil Moisture Content
#################################
  t1 = time.clock()
  print(('Working on 1-2 m soil moisture for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,soilw_100_200_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('darkred')
      cs_1.cmap.set_over('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM 1-2 m Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,soilw_100_200_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('darkred')
      cs_2.cmap.set_over('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X 1-2 m Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,soilw_100_200_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 1-2 m Soil Moisture Content (fraction) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparesoilw_100_200_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 1-2 m soil moisture content for: '+dom) % t3)

#################################
  # Plot lowest model level cloud water
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      csref_1 = m.pcolormesh(x_shift,y_shift,refd_1,cmap=cmref,vmin=20,norm=normref,ax=ax)
      csref_1.cmap.set_under('white')
      cs_1 = m.pcolormesh(x_shift,y_shift,clwmr_1,cmap=cm,vmin=0.01,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('hotpink')
      cstmp_1 = m.contour(x,y,tmphyb_1,[0],colors='red',linewidths=0.5,ax=ax)
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM Lowest Mdl Lvl Cld Water ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      csref_2 = m.pcolormesh(x2_shift,y2_shift,refd_2,cmap=cmref,vmin=20,norm=normref,ax=ax)
      csref_2.cmap.set_under('white')
      cs_2 = m.pcolormesh(x2_shift,y2_shift,clwmr_2,cmap=cm,vmin=0.01,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('hotpink')
      cstmp_2 = m.contour(x2,y2,tmphyb_2,[0],colors='red',linewidths=0.5,ax=ax)
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM-X Lowest Mdl Lvl Cld Water ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,clwmr_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Lowest Mdl Lvl Cld Water ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareclwmr_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot lowest model level cloud water for: '+dom) % t3)

#################################
  # Plot lowest model level cloud ice
#################################
  t1 = time.clock()
  print(('Working on lowest model level cloud ice for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      csref_1 = m.pcolormesh(x_shift,y_shift,refd_1,cmap=cmref,vmin=20,norm=normref,ax=ax)
      csref_1.cmap.set_under('white')
      cs_1 = m.pcolormesh(x_shift,y_shift,icmr_1,cmap=cm,vmin=0.01,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('hotpink')
      cstmp_1 = m.contour(x,y,tmphyb_1,[0],colors='red',linewidths=0.5,ax=ax)
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=4)
      ax.text(.5,1.03,'FV3LAM Lowest Mdl Lvl Cld Ice ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      csref_2 = m.pcolormesh(x2_shift,y2_shift,refd_2,cmap=cmref,vmin=20,norm=normref,ax=ax)
      csref_2.cmap.set_under('white')
      cs_2 = m.pcolormesh(x2_shift,y2_shift,icmr_2,cmap=cm,vmin=0.01,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('hotpink')
      cstmp_2 = m.contour(x2,y2,tmphyb_2,[0],colors='red',linewidths=0.5,ax=ax)
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=4)
      ax.text(.5,1.03,'FV3LAM-X Lowest Mdl Lvl Cld Ice ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,icmr_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Lowest Mdl Lvl Cld Ice ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareicmr_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot lowest model level cloud ice for: '+dom) % t3)

#################################
  # Plot lowest model level rain
#################################
  t1 = time.clock()
  print(('Working on lowest model level rain for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      csref_1 = m.pcolormesh(x_shift,y_shift,refd_1,cmap=cmref,vmin=20,norm=normref,ax=ax)
      csref_1.cmap.set_under('white')
      cs_1 = m.pcolormesh(x_shift,y_shift,rwmr_1,cmap=cm,vmin=0.01,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('hotpink')
      cstmp_1 = m.contour(x,y,tmphyb_1,[0],colors='red',linewidths=0.5,ax=ax)
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM Lowest Mdl Lvl Rain ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      csref_2 = m.pcolormesh(x2_shift,y2_shift,refd_2,cmap=cmref,vmin=20,norm=normref,ax=ax)
      csref_1.cmap.set_under('white')
      cs_2 = m.pcolormesh(x2_shift,y2_shift,rwmr_2,cmap=cm,vmin=0.01,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('hotpink')
      cstmp_2 = m.contour(x2,y2,tmphyb_2,[0],colors='red',linewidths=0.5,ax=ax)
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM-X Lowest Mdl Lvl Rain ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,rwmr_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Lowest Mdl Lvl Rain ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparerwmr_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot lowest model level rain for: '+dom) % t3)

#################################
  # Plot lowest model level snow
#################################
  t1 = time.clock()
  print(('Working on lowest model level snow for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      csref_1 = m.pcolormesh(x_shift,y_shift,refd_1,cmap=cmref,vmin=20,norm=normref,ax=ax)
      csref_1.cmap.set_under('white')
      cs_1 = m.pcolormesh(x_shift,y_shift,snmr_1,cmap=cm,vmin=0.01,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('hotpink')
      cstmp_1 = m.contour(x,y,tmphyb_1,[0],colors='red',linewidths=0.5,ax=ax)
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM Lowest Mdl Lvl Snow ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      csref_2 = m.pcolormesh(x2_shift,y2_shift,refd_2,cmap=cmref,vmin=20,norm=normref,ax=ax)
      csref_2.cmap.set_under('white')
      cs_2 = m.pcolormesh(x2_shift,y2_shift,snmr_2,cmap=cm,vmin=0.01,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('hotpink')
      cstmp_2 = m.contour(x2,y2,tmphyb_2,[0],colors='red',linewidths=0.5,ax=ax)
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs)
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM-X Lowest Mdl Lvl Snow ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,snmr_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Lowest Mdl Lvl Snow ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparesnmr_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot lowest model level snow for: '+dom) % t3)

#################################
  # Plot lowest model level graupel
#################################
  t1 = time.clock()
  print(('Working on lowest model level graupel for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      csref_1 = m.pcolormesh(x_shift,y_shift,refd_1,cmap=cmref,vmin=20,norm=normref,ax=ax)
      csref_1.cmap.set_under('white')
      cs_1 = m.pcolormesh(x_shift,y_shift,grle_1,cmap=cm,vmin=0.01,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('hotpink')
      cstmp_1 = m.contour(x,y,tmphyb_1,[0],colors='red',linewidths=0.5,ax=ax)
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM Lowest Mdl Lvl Graupel ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      csref_2 = m.pcolormesh(x2_shift,y2_shift,refd_2,cmap=cmref,vmin=20,norm=normref,ax=ax)
      csref_2.cmap.set_under('white')
      cs_2 = m.pcolormesh(x2_shift,y2_shift,grle_2,cmap=cm,vmin=0.01,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('hotpink')
      cstmp_2 = m.contour(x2,y2,tmphyb_2,[0],colors='red',linewidths=0.5,ax=ax)
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM-X Lowest Mdl Lvl Graupel ('+units+'), Reflectivity (gray), 0''\xb0''C line (red) \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,grle_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Lowest Mdl Lvl Graupel ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparegrle_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot lowest model level graupel for: '+dom) % t3)

#################################
  # Plot downward shortwave
#################################
  t1dom = time.clock()
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,swdown_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Surface Downward Shortwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,swdown_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05)
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Surface Downward Shortwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,swdown_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Surface Downward Shortwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareswdown_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot downward shortwave for: '+dom) % t3)

#################################
  # Plot upward shortwave
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,swup_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Surface Upward Shortwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,swup_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Surface Upward Shortwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,swup_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Surface Upward Shortwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareswup_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot upward shortwave for: '+dom) % t3)

#################################
  # Plot downward longwave
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,lwdown_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Surface Downward Longwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,lwdown_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Surface Downward Longwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,lwdown_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Surface Downward Longwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparelwdown_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot downward longwave for: '+dom) % t3)

#################################
  # Plot upward longwave
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,lwup_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Surface Upward Longwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,lwup_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Surface Upward Longwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,lwup_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Surface Upward Longwave Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparelwup_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot upward longwave for: '+dom) % t3)

#################################
  # Plot ground heat flux
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,gdhfx_1,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs_1,ax=ax,ticks=clevs,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=4.5)
      ax.text(.5,1.03,'FV3LAM Ground Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,gdhfx_2,cmap=cm,norm=norm,ax=ax)
      cbar2 = m.colorbar(cs_2,ax=ax,ticks=clevs,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=4.5)
      ax.text(.5,1.03,'FV3LAM-X Ground Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,gdhfx_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Ground Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparegdhfx_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot ground heat flux for: '+dom) % t3)

#################################
  # Plot latent heat flux
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,lhfx_1,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs_1,ax=ax,ticks=[-2000,-500,-100,-50,0,50,100,500,1000,2000],location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM Latent Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,lhfx_2,cmap=cm,norm=norm,ax=ax)
      cbar2 = m.colorbar(cs_2,ax=ax,ticks=[-2000,-500,-100,-50,0,50,100,500,1000,2000],location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM-X Latent Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,lhfx_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Latent Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparelhfx_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot latent heat flux for: '+dom) % t3)

#################################
  # Plot sensible heat flux
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,snhfx_1,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs_1,ax=ax,ticks=[-2000,-500,-100,-50,0,50,100,500,1000,2000],location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM Sensible Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,snhfx_2,cmap=cm,norm=norm,ax=ax)
      cbar2 = m.colorbar(cs_2,ax=ax,ticks=[-2000,-500,-100,-50,0,50,100,500,1000,2000],location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM-X Sensible Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,snhfx_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Sensible Heat Flux ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparesnhfx_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot sensible heat flux for: '+dom) % t3)

#################################
  # Plot PBL height
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,hpbl_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('black')
      cbar1 = m.colorbar(cs_1,ax=ax,ticks=clevs,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=4)
      ax.text(.5,1.03,'FV3LAM PBL Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,hpbl_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('black')
      cbar2 = m.colorbar(cs_2,ax=ax,ticks=clevs,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=4)
      ax.text(.5,1.03,'FV3LAM-X PBL Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,hpbl_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM PBL Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparehpbl_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot PBL height for: '+dom) % t3)

#################################
  # Plot total column condensate
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,cond_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Total Column Condensate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,cond_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Total Column Condensate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,cond_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Total Column Condensate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparecond_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Total condensate for: '+dom) % t3)

#################################
  # Plot total column liquid
#################################
  t1 = time.clock()
  print(('Working on Total column liquid for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,tcolw_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Total Column Cloud Water + Rain ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,tcolw_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Total Column Cloud Water + Rain ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,cond_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Total Column Cloud Water + Rain ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparetcolw_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Total column liquid for: '+dom) % t3)

#################################
  # Plot total column ice
#################################
  t1 = time.clock()
  print(('Working on Tcoli for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,tcoli_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Total Column Cloud Ice + Snow ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,tcoli_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels([0.001,0.01,0.1,0.5,2,6,15,25])
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Total Column Cloud Ice + Snow ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,cond_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Total Column Cloud Ice + Snow ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparetcoli_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Tcoli for: '+dom) % t3)

#################################
  # Plot soil type
#################################
#  if (fhr == 0):
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,sotyp_1,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM Soil Type \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,sotyp_2,cmap=cm,norm=norm,ax=ax)
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM-X Soil Type \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,sotyp_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkred')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,ticks=[0],extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Soil Type ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparesotyp_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot soil type for: '+dom) % t3)

#################################
  # Plot vegetation type
#################################
#  if (fhr == 0):
  t1 = time.clock()
  print('Working on vegetation type for '+dom)

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  units = 'Integer(0-19)'
  clevs = [-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5]
  clevsdif = [-0.1,0.1]
  colorlist = ['#00CDCD','saddlebrown','khaki','gray','#3D9140','palegreen','firebrick','lightcoral','darkorchid','plum','blue','lightskyblue','#CDAD00','yellow','#FF4500','lightsalmon','#CD1076']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,vgtyp_1,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM Vegetation Type \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,vgtyp_2,cmap=cm,norm=norm,ax=ax)
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM-X Vegetation Type \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,vgtyp_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkred')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,ticks=[0],extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Vegetation Type ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparevgtyp_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot vegetation type for: '+dom) % t3)

#################################
  # Plot vegetation fraction
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,veg_1,cmap=cm,vmax=100,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('white',alpha=0.)
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='min')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Vegetation Fraction ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,veg_2,cmap=cm,vmax=100,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cs_1.cmap.set_over('white',alpha=0.)
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='min')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Vegetation Fraction ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,veg_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Vegetation Fraction ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareveg_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot vegetation fraction for: '+dom) % t3)

#################################
  # Plot 0-3 km Storm Relative Helicity
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,hel3km_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('black')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM 0-3 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,hel3km_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('black')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X 0-3 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,hel3km_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 0-3 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparehel3km_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 0-3 km SRH for: '+dom) % t3)

#################################
  # Plot 0-1 km Storm Relative Helicity
#################################

  t1 = time.clock()
  print(('Working on 0-1 km SRH for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  cbar3.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,hel1km_1,cmap=cm,norm=norm,ax=ax)
      cs_1.cmap.set_under('white')
      cs_1.cmap.set_over('black')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM 0-1 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,hel1km_2,cmap=cm,norm=norm,ax=ax)
      cs_2.cmap.set_under('white')
      cs_2.cmap.set_over('black')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X 0-1 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,hel1km_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM 0-1 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparehel1km_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 0-1 km SRH for: '+dom) % t3)

#################################
  # Plot 1-km reflectivity
#################################
  t1 = time.clock()
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
  colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  
  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,ref1km_1,cmap=cm,vmin=5,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('black')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,ref1km_2,cmap=cm,vmin=5,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('black')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      csdif = m.contourf(x,y,ref1km_1,clevsdif,colors='red')
      csdif2 = m.contourf(x2,y2,ref1km_2,clevsdif,colors='dodgerblue')
      ax.text(.5,1.03,'FV3LAM (red) and FV3LAM-X (blue) 1-km Reflectivity > 20 ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareref1km_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 1-km reflectivity for: '+dom) % t3)

#################################
  # Plot composite reflectivity
#################################
  t1 = time.clock()
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
  colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  
  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,refc_1,cmap=cm,vmin=5,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('black')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Composite Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,refc_2,cmap=cm,vmin=5,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('black')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Composite Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      csdif = m.contourf(x,y,refc_1,clevsdif,colors='red')
      csdif2 = m.contourf(x2,y2,refc_2,clevsdif,colors='dodgerblue')
      ax.text(.5,1.03,'FV3LAM (red) and FV3LAM-X (blue) Composite Reflectivity > 20 ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparerefc_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot composite reflectivity for: '+dom) % t3)


######################################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 2 variables for: "+dom) % t3dom)
  plt.clf()

######################################################

def plot_set_3():
  global fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,m,x,y,x2,y2,x_shift,y_shift,x2_shift,y2_shift,xscale,yscale,im,par

#################################
  # Plot Max/Min Hourly 2-5 km UH
#################################
  t1dom = time.clock()
  if (fhr > 0):
    t1 = time.clock()
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

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,uh25_1,cmap=cm,norm=norm,ax=ax)
        cs_1.cmap.set_under('darkblue')
        cs_1.cmap.set_over('black')
        cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM 1-h Max/Min 2-5 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,uh25_2,cmap=cm,norm=norm,ax=ax)
        cs_2.cmap.set_under('darkblue')
        cs_2.cmap.set_over('black')
        cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X 1-h Max/Min 2-5 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        cs = m.pcolormesh(x2_shift,y2_shift,uh25_dif,cmap=cmdif,norm=normdif,ax=ax)
        cs.cmap.set_under('darkblue')
        cs.cmap.set_over('darkred')
        cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar3.set_label(units,fontsize=6)
        cbar3.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X - FV3LAM 1-h Max/Min 2-5 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))   
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('compareuh25_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max/Min Hourly 2-5 km UH for: '+dom) % t3)

#################################
  # Plot Max Hourly 0-3 km UH
#################################
    t1 = time.clock()
    print(('Working on Max Hourly 0-3 km UH for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,uh03_1,cmap=cm,norm=norm,ax=ax)
        cs_1.cmap.set_under('darkblue')
        cs_1.cmap.set_over('black')
        cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM 1-h Max/Min 0-3 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,uh03_2,cmap=cm,norm=norm,ax=ax)
        cs_2.cmap.set_under('darkblue')
        cs_2.cmap.set_over('black')
        cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X 1-h Max/Min 0-3 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        cs = m.pcolormesh(x2_shift,y2_shift,uh03_dif,cmap=cmdif,norm=normdif,ax=ax)
        cs.cmap.set_under('darkblue')
        cs.cmap.set_over('darkred')
        cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar3.set_label(units,fontsize=6)
        cbar3.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X - FV3LAM 1-h Max/Min 0-3 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))   
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('compareuh03_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max/Min Hourly 0-3 km UH for: '+dom) % t3)

#################################
  # Plot Max Hourly Updraft Speed
#################################
    t1 = time.clock()
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

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,maxuvv_1,cmap=cm,norm=norm,ax=ax)
        cs_1.cmap.set_under('white')
        cs_1.cmap.set_over('black')
        cbar1 = m.colorbar(cs_1,ax=ax,ticks=clevs,location='bottom',pad=0.05,extend='both')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.set_xticklabels(clevs)
        cbar1.ax.tick_params(labelsize=5)
        ax.text(.5,1.03,'FV3LAM 1-h Max 100-1000 mb Updraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,maxuvv_2,cmap=cm,norm=norm,ax=ax)
        cs_2.cmap.set_under('white')
        cs_2.cmap.set_over('black')
        cbar2 = m.colorbar(cs_2,ax=ax,ticks=clevs,location='bottom',pad=0.05,extend='both')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.set_xticklabels(clevs)
        cbar2.ax.tick_params(labelsize=5)
        ax.text(.5,1.03,'FV3LAM-X 1-h Max 100-1000 mb Updraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        cs = m.pcolormesh(x2_shift,y2_shift,maxuvv_dif,cmap=cmdif,norm=normdif,ax=ax)
        cs.cmap.set_under('darkblue')
        cs.cmap.set_over('darkred')
        cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar3.set_label(units,fontsize=6)
        cbar3.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X - FV3LAM 1-h Max 100-1000 mb Updraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))     
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('comparemaxuvv_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly Updraft Speed for: '+dom) % t3)

#################################
  # Plot Max Hourly Downdraft Speed
#################################
    t1 = time.clock()
    print(('Working on Max Hourly Downdraft Speed for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,maxdvv_1,cmap=cm,norm=norm,ax=ax)
        cs_1.cmap.set_under('white')
        cs_1.cmap.set_over('black')
        cbar1 = m.colorbar(cs_1,ax=ax,ticks=clevs,location='bottom',pad=0.05,extend='both')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.set_xticklabels(clevs)
        cbar1.ax.tick_params(labelsize=5)
        ax.text(.5,1.03,'FV3LAM 1-h Max 100-1000 mb Downdraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,maxdvv_2,cmap=cm,norm=norm,ax=ax)
        cs_2.cmap.set_under('white')
        cs_2.cmap.set_over('black')
        cbar2 = m.colorbar(cs_2,ax=ax,ticks=clevs,location='bottom',pad=0.05,extend='both')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.set_xticklabels(clevs)
        cbar2.ax.tick_params(labelsize=5)
        ax.text(.5,1.03,'FV3LAM-X 1-h Max 100-1000 mb Downdraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        cs = m.pcolormesh(x2_shift,y2_shift,maxdvv_dif,cmap=cmdif,norm=normdif,ax=ax)
        cs.cmap.set_under('darkblue')
        cs.cmap.set_over('darkred')
        cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar3.set_label(units,fontsize=6)
        cbar3.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X - FV3LAM 1-h Max 100-1000 mb Downdraft Speed ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))   
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('comparemaxdvv_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly Downdraft Speed for: '+dom) % t3)

#################################
  # Plot Max Hourly 1-km Reflectivity
#################################
    t1 = time.clock()
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

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,maxref1km_1,cmap=cm,vmin=5,norm=norm,ax=ax)
        cs_1.cmap.set_under('white',alpha=0.)
        cs_1.cmap.set_over('black')
        cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='max')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM 1-h Max 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,maxref1km_2,cmap=cm,vmin=5,norm=norm,ax=ax)
        cs_2.cmap.set_under('white',alpha=0.)
        cs_2.cmap.set_over('black')
        cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='max')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X 1-h Max 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        csdif = m.contourf(x,y,maxref1km_1,clevsdif,colors='red')
        csdif2 = m.contourf(x2,y2,maxref1km_2,clevsdif,colors='dodgerblue')
        ax.text(.5,1.03,'FV3LAM (red) and FV3LAM-X (blue) 1-h Max 1-km Reflectivity > 20 ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('comparemaxref1km_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly 1-km Reflectivity for: '+dom) % t3)

#################################
  # Plot Max Hourly -10C Reflectivity
#################################
    t1 = time.clock()
    print(('Working on Max Hourly 263K Reflectivity for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,maxref10C_1,cmap=cm,vmin=5,norm=norm,ax=ax)
        cs_1.cmap.set_under('white',alpha=0.)
        cs_1.cmap.set_over('black')
        cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='max')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM 1-h Max -10''\xb0''C Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,maxref10C_2,cmap=cm,vmin=5,norm=norm,ax=ax)
        cs_2.cmap.set_under('white',alpha=0.)
        cs_2.cmap.set_over('black')
        cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='max')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X 1-h Max -10''\xb0''C Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        csdif = m.contourf(x,y,maxref10C_1,clevsdif,colors='red')
        csdif2 = m.contourf(x2,y2,maxref10C_2,clevsdif,colors='dodgerblue')
        ax.text(.5,1.03,'FV3LAM (red) and FV3LAM-X (blue) 1-h Max -10''\xb0''C Reflectivity > 20 ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=5,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('comparemaxref10C_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly 263K Reflectivity for: '+dom) % t3)

#################################
  # Plot Max Hourly 10-m Winds
#################################
    t1 = time.clock()
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

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,maxwind_1,cmap=cm,vmin=5,norm=norm,ax=ax)
        cs_1.cmap.set_under('white',alpha=0.)
        cs_1.cmap.set_over('black')
        cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='max')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM 1-h Max 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,maxwind_2,cmap=cm,vmin=5,norm=norm,ax=ax)
        cs_2.cmap.set_under('white',alpha=0.)
        cs_2.cmap.set_over('black')
        cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='max')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X 1-h Max 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        cs = m.pcolormesh(x2_shift,y2_shift,maxwind_dif,cmap=cmdif,norm=normdif,ax=ax)
        cs.cmap.set_under('darkblue')
        cs.cmap.set_over('darkred')
        cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar3.set_label(units,fontsize=6)
        cbar3.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X - FV3LAM 1-h Max 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('comparemaxwind_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hourly 10-m Wind Speed for: '+dom) % t3)

#################################
  # Plot Max 0-1 km Vertical Vorticity
#################################
    t1 = time.clock()
    print(('Working on Max 0-1 km Vorticity for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

    units = 's$^{-1}$'
    clevs = [0.001,0.0025,0.005,0.0075,0.01,0.0125,0.015]
    clevsdif = [-0.006,-0.005,-0.004,-0.003,-0.002,-0.001,0,0.001,0.002,0.003,0.004,0.005,0.006]
    colorlist = ['#EEEE00','#EEC900','darkorange','red','firebrick','DarkViolet']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
    normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,relv01_1,cmap=cm,norm=norm,ax=ax)
        cs_1.cmap.set_under('white')
        cs_1.cmap.set_over('black')
        cbar1 = m.colorbar(cs_1,ax=ax,ticks=clevs,location='bottom',pad=0.05,extend='both')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.set_xticklabels(clevs)
        cbar1.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM 1-h Max 0-1 km Vertical $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,relv01_2,cmap=cm,norm=norm,ax=ax)
        cs_2.cmap.set_under('white')
        cs_2.cmap.set_over('black')
        cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.set_xticklabels(clevs)
        cbar2.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X 1-h Max 0-1 km Vertical $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        cs = m.pcolormesh(x2_shift,y2_shift,relv01_dif,cmap=cmdif,norm=normdif,ax=ax)
        cs.cmap.set_under('darkblue')
        cs.cmap.set_over('darkred')
        cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar3.set_label(units,fontsize=6)
        cbar3.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X - FV3LAM 1-h Max 0-1 km Vertical $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('comparerelv01_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max 0-1 km Vorticity for: '+dom) % t3)

#################################
  # Plot Max 0-2 km Vertical Vorticity
#################################
    t1 = time.clock()
    print(('Working on Max 0-2 km Vorticity for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,relv02_1,cmap=cm,norm=norm,ax=ax)
        cs_1.cmap.set_under('white')
        cs_1.cmap.set_over('black')
        cbar1 = m.colorbar(cs_1,ax=ax,ticks=clevs,location='bottom',pad=0.05,extend='both')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.set_xticklabels(clevs)
        cbar1.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM 1-h Max 0-2 km Vertical $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,relv02_2,cmap=cm,norm=norm,ax=ax)
        cs_2.cmap.set_under('white')
        cs_2.cmap.set_over('black')
        cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.set_xticklabels(clevs)
        cbar2.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X 1-h Max 0-2 km Vertical $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        cs = m.pcolormesh(x2_shift,y2_shift,relv02_dif,cmap=cmdif,norm=normdif,ax=ax)
        cs.cmap.set_under('darkblue')
        cs.cmap.set_over('darkred')
        cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar3.set_label(units,fontsize=6)
        cbar3.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X - FV3LAM 1-h Max 0-2 km Vertical $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('comparerelv02_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max 0-2 km Vorticity for: '+dom) % t3)

#################################
  # Plot Max Hybrid Level 1 Vertical Vorticity
#################################
    t1 = time.clock()
    print(('Working on Max Hybrid Level 1 Vorticity for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    cbar3.remove()
    clear_plotables(ax1,keep_ax_lst_1,fig)
    clear_plotables(ax2,keep_ax_lst_2,fig)
    clear_plotables(ax3,keep_ax_lst_3,fig)

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      if par == 1:
        cs_1 = m.pcolormesh(x_shift,y_shift,relvhyb_1,cmap=cm,norm=norm,ax=ax)
        cs_1.cmap.set_under('white')
        cs_1.cmap.set_over('black')
        cbar1 = m.colorbar(cs_1,ax=ax,ticks=clevs,location='bottom',pad=0.05,extend='both')
        cbar1.set_label(units,fontsize=6)
        cbar1.ax.set_xticklabels(clevs)
        cbar1.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM 1-h Max Lowest Mdl Lvl Vertical $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 2:
        cs_2 = m.pcolormesh(x2_shift,y2_shift,relvhyb_2,cmap=cm,norm=norm,ax=ax)
        cs_2.cmap.set_under('white')
        cs_2.cmap.set_over('black')
        cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar2.set_label(units,fontsize=6)
        cbar2.ax.set_xticklabels(clevs)
        cbar2.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X 1-h Max Lowest Mdl Lvl Vertical $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      elif par == 3:
        cs = m.pcolormesh(x2_shift,y2_shift,relvhyb_dif,cmap=cmdif,norm=normdif,ax=ax)
        cs.cmap.set_under('darkblue')
        cs.cmap.set_over('darkred')
        cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
        cbar3.set_label(units,fontsize=6)
        cbar3.ax.tick_params(labelsize=6)
        ax.text(.5,1.03,'FV3LAM-X - FV3LAM 1-h Max Lowest Mdl Lvl Vertical $\zeta$ ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

      par += 1
    par = 1

    compress_and_save('comparerelvhyb_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Max Hybrid Level 1 Vorticity for: '+dom) % t3)

#################################
  # Plot Haines Index
#################################
  t1dom = time.clock()
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,hindex_1,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs_1,ax=ax,ticks=[2,3,4,5,6],location='bottom',pad=0.05)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Haines Index \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,hindex_2,cmap=cm,norm=norm,ax=ax)
      cbar2 = m.colorbar(cs_2,ax=ax,ticks=[2,3,4,5,6],location='bottom',pad=0.05)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Haines Index \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,hindex_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Haines Index \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparehindex_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Haines Index for: '+dom) % t3)

#################################
  # Plot transport wind
#################################
  t1 = time.clock()
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
  elif dom == 'SE':
    skip = 35
  elif dom == 'CO' or dom == 'LA' or dom == 'MA':
    skip = 12
  elif dom == 'BN':
    skip = 10
  elif dom == 'SP':
    skip = 9
  elif dom == 'SF':
    skip = 3
  else:
    skip = 20
  barblength = 4
  clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
  clevsdif = [-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18]
  colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  cmdif = matplotlib.colors.ListedColormap(difcolors)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
  normdif = matplotlib.colors.BoundaryNorm(clevsdif, cmdif.N)

  # Rotate winds to gnomonic projection
  urot_1, vrot_1 = m.rotate_vector(utrans_1,vtrans_1,lon,lat)
  urot_2, vrot_2 = m.rotate_vector(utrans_2,vtrans_2,lon2,lat2)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,trans_1,cmap=cm,vmin=5,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('black')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      m.barbs(lon[::skip,::skip],lat[::skip,::skip],urot_1[::skip,::skip],vrot_1[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.03,'FV3LAM Transport Wind ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,trans_2,cmap=cm,vmin=5,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('black')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      m.barbs(lon2[::skip,::skip],lat2[::skip,::skip],urot_2[::skip,::skip],vrot_2[::skip,::skip],latlon=True,length=barblength,linewidth=0.5,color='black',ax=ax)
      ax.text(.5,1.03,'FV3LAM-X Transport Wind ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,trans_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Transport Wind ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparetrans_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot transport wind for: '+dom) % t3)

#################################
  # Plot Total Cloud Cover
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,tcdc_1,cmap=cm,norm=norm,ax=ax)
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05)
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Total Cloud Cover ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,tcdc_2,cmap=cm,norm=norm,ax=ax)
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05)
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Total Cloud Cover ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,tcdc_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Total Cloud Cover ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparetcdc_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Total Cloud Cover for: '+dom) % t3)

#################################
  # Plot Echo Top Height
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,retop_1,cmap=cm,vmin=1,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('darkgreen')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Echo Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,retop_2,cmap=cm,vmin=1,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('darkgreen')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Echo Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,retop_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Echo Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareretop_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Echo Top Height for: '+dom) % t3)

#################################
  # Plot Precipitation Rate
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,prate_1,cmap=cm,vmin=0.01,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('yellow')
      cbar1 = m.colorbar(cs_1,ax=ax,ticks=clevs,location='bottom',pad=0.05,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.set_xticklabels(clevs)
      cbar1.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM Precipitation Rate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,prate_2,cmap=cm,vmin=0.01,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('yellow')
      cbar2 = m.colorbar(cs_2,ax=ax,ticks=clevs,location='bottom',pad=0.05,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.set_xticklabels(clevs)
      cbar2.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM-X Precipitation Rate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,prate_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=5)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Precipitation Rate ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareprate_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Precipitation Rate for: '+dom) % t3)

#################################
  # Plot Cloud Base Pressure
#################################
  t1 = time.clock()
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

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,pbase_1,cmap=cm,vmin=50,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('red')
      cbar1 = m.colorbar(cs_1,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar1.set_label(units,fontsize=6)
      cbar1.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM Pressure at Cloud Base ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,pbase_2,cmap=cm,vmin=50,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('red')
      cbar2 = m.colorbar(cs_2,ax=ax,location='bottom',pad=0.05,extend='max')
      cbar2.set_label(units,fontsize=6)
      cbar2.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X Pressure at Cloud Base ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,pbase_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      cbar3 = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar3.set_label(units,fontsize=6)
      cbar3.ax.tick_params(labelsize=6)
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Pressure at Cloud Base ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('comparepbase_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Cloud Base Pressure for: '+dom) % t3)

#################################
  # Plot Cloud Top Pressure
#################################
  t1 = time.clock()
  print(('Working on Cloud Top Pressure for '+dom))

  # Clear off old plottables but keep all the map info
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    if par == 1:
      cs_1 = m.pcolormesh(x_shift,y_shift,ptop_1,cmap=cm,vmin=50,norm=norm,ax=ax)
      cs_1.cmap.set_under('white',alpha=0.)
      cs_1.cmap.set_over('red')
      ax.text(.5,1.03,'FV3LAM Pressure at Cloud Top ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 2:
      cs_2 = m.pcolormesh(x2_shift,y2_shift,ptop_2,cmap=cm,vmin=50,norm=norm,ax=ax)
      cs_2.cmap.set_under('white',alpha=0.)
      cs_2.cmap.set_over('red')
      ax.text(.5,1.03,'FV3LAM-X Pressure at Cloud Top ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    elif par == 3:
      cs = m.pcolormesh(x2_shift,y2_shift,ptop_dif,cmap=cmdif,norm=normdif,ax=ax)
      cs.cmap.set_under('darkblue')
      cs.cmap.set_over('darkred')
      ax.text(.5,1.03,'FV3LAM-X - FV3LAM Pressure at Cloud Top ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('compareptop_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Cloud Top Pressure for: '+dom) % t3)

######################################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 3 variables for: "+dom) % t3dom)
  plt.clf()

######################################################


main()

