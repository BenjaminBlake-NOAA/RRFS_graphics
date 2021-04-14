import pygrib
import matplotlib
matplotlib.use('Agg')
import io
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.gridspec import GridSpec
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

data1 = pygrib.open('/scratch2/NCEPDEV/fv3-cam/James.A.Abeles/hwt/run/CONUS/member1/PRSLEV.GrbF'+fhour)
data2 = pygrib.open('/scratch2/NCEPDEV/fv3-cam/James.A.Abeles/hwt/run/CONUS/member2/PRSLEV.GrbF'+fhour)
data3 = pygrib.open('/scratch2/NCEPDEV/fv3-cam/James.A.Abeles/hwt/run/CONUS/member3/PRSLEV.GrbF'+fhour)
data4 = pygrib.open('/scratch2/NCEPDEV/fv3-cam/James.A.Abeles/hwt/run/CONUS/member4/PRSLEV.GrbF'+fhour)
data5 = pygrib.open('/scratch2/NCEPDEV/fv3-cam/James.A.Abeles/hwt/run/CONUS/member5/PRSLEV.GrbF'+fhour)
data6 = pygrib.open('/scratch2/NCEPDEV/fv3-cam/James.A.Abeles/hwt/run/CONUS/member6/PRSLEV.GrbF'+fhour)
data7 = pygrib.open('/scratch2/NCEPDEV/fv3-cam/James.A.Abeles/hwt/run/CONUS/member7/PRSLEV.GrbF'+fhour)
data8 = pygrib.open('/scratch2/NCEPDEV/fv3-cam/James.A.Abeles/hwt/run/CONUS/member8/PRSLEV.GrbF'+fhour)
data9 = pygrib.open('/scratch2/NCEPDEV/fv3-cam/James.A.Abeles/hwt/run/CONUS/member9/PRSLEV.GrbF'+fhour)


# Get the lats and lons
grids = [data1]
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
    try:
        dx = data[1]['DxInMetres']
        dy = data[1]['DyInMetres']
    except:
        dx = data[1]['DiInMetres']
        dy = data[1]['DjInMetres']
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

# Shifted lat/lon arrays for pcolormesh
lat_shift = lats_shift[0]
lon_shift = lons_shift[0]

Lat0 = data1[1]['LaDInDegrees']
Lon0 = data1[1]['LoVInDegrees']
print(Lat0)
print(Lon0)

# Forecast valid date/time
itime = ymdh
vtime = ncepy.ndate(itime,int(fhr))

# Specify plotting domains
#domains = ['conus','BN','CE','CO','LA','MA','NC','NE','NW','OV','SC','SE','SF','SP','SW','UM']
domains=['storm']

###################################################
# Read in all variables #
###################################################
t1a = time.clock()

# 2-m temperature
tmp2m_1 = data1.select(name='2 metre temperature')[0].values
tmp2m_1 = (tmp2m_1 - 273.15)*1.8 + 32.0
tmp2m_2 = data2.select(name='2 metre temperature')[0].values
tmp2m_2 = (tmp2m_2 - 273.15)*1.8 + 32.0
tmp2m_3 = data3.select(name='2 metre temperature')[0].values
tmp2m_3 = (tmp2m_3 - 273.15)*1.8 + 32.0
tmp2m_4 = data4.select(name='2 metre temperature')[0].values
tmp2m_4 = (tmp2m_4 - 273.15)*1.8 + 32.0
tmp2m_5 = data5.select(name='2 metre temperature')[0].values
tmp2m_5 = (tmp2m_5 - 273.15)*1.8 + 32.0
tmp2m_6 = data6.select(name='2 metre temperature')[0].values
tmp2m_6 = (tmp2m_6 - 273.15)*1.8 + 32.0
tmp2m_7 = data7.select(name='2 metre temperature')[0].values
tmp2m_7 = (tmp2m_7 - 273.15)*1.8 + 32.0
tmp2m_8 = data8.select(name='2 metre temperature')[0].values
tmp2m_8 = (tmp2m_8 - 273.15)*1.8 + 32.0
tmp2m_9 = data9.select(name='2 metre temperature')[0].values
tmp2m_9 = (tmp2m_9 - 273.15)*1.8 + 32.0

# 2-m dew point temperature
dew2m_1 = data1.select(name='2 metre dewpoint temperature')[0].values
dew2m_1 = (dew2m_1 - 273.15)*1.8 + 32.0
dew2m_2 = data2.select(name='2 metre dewpoint temperature')[0].values
dew2m_2 = (dew2m_2 - 273.15)*1.8 + 32.0
dew2m_3 = data3.select(name='2 metre dewpoint temperature')[0].values
dew2m_3 = (dew2m_3 - 273.15)*1.8 + 32.0
dew2m_4 = data4.select(name='2 metre dewpoint temperature')[0].values
dew2m_4 = (dew2m_4 - 273.15)*1.8 + 32.0
dew2m_5 = data5.select(name='2 metre dewpoint temperature')[0].values
dew2m_5 = (dew2m_5 - 273.15)*1.8 + 32.0
dew2m_6 = data6.select(name='2 metre dewpoint temperature')[0].values
dew2m_6 = (dew2m_6 - 273.15)*1.8 + 32.0
dew2m_7 = data7.select(name='2 metre dewpoint temperature')[0].values
dew2m_7 = (dew2m_7 - 273.15)*1.8 + 32.0
dew2m_8 = data8.select(name='2 metre dewpoint temperature')[0].values
dew2m_8 = (dew2m_8 - 273.15)*1.8 + 32.0
dew2m_9 = data9.select(name='2 metre dewpoint temperature')[0].values
dew2m_9 = (dew2m_9 - 273.15)*1.8 + 32.0

# Composite reflectivity
refc_1 = data1.select(name='Maximum/Composite radar reflectivity')[0].values
refc_2 = data2.select(name='Maximum/Composite radar reflectivity')[0].values
refc_3 = data3.select(name='Maximum/Composite radar reflectivity')[0].values
refc_4 = data4.select(name='Maximum/Composite radar reflectivity')[0].values
refc_5 = data5.select(name='Maximum/Composite radar reflectivity')[0].values
refc_6 = data6.select(name='Maximum/Composite radar reflectivity')[0].values
refc_7 = data7.select(name='Maximum/Composite radar reflectivity')[0].values
refc_8 = data8.select(name='Maximum/Composite radar reflectivity')[0].values
refc_9 = data9.select(name='Maximum/Composite radar reflectivity')[0].values

# Max/Min Hourly 2-5 km Updraft Helicity
maxuh25_1 = data1.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
maxuh25_2 = data2.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
maxuh25_3 = data3.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
maxuh25_4 = data4.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
maxuh25_5 = data5.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
maxuh25_6 = data6.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
maxuh25_7 = data7.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
maxuh25_8 = data8.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
maxuh25_9 = data9.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
minuh25_1 = data1.select(stepType='min',parameterName="200",topLevel=5000,bottomLevel=2000)[0].values
minuh25_2 = data2.select(stepType='min',parameterName="200",topLevel=5000,bottomLevel=2000)[0].values
minuh25_3 = data3.select(stepType='min',parameterName="200",topLevel=5000,bottomLevel=2000)[0].values
minuh25_4 = data4.select(stepType='min',parameterName="200",topLevel=5000,bottomLevel=2000)[0].values
minuh25_5 = data5.select(stepType='min',parameterName="200",topLevel=5000,bottomLevel=2000)[0].values
minuh25_6 = data6.select(stepType='min',parameterName="200",topLevel=5000,bottomLevel=2000)[0].values
minuh25_7 = data7.select(stepType='min',parameterName="200",topLevel=5000,bottomLevel=2000)[0].values
minuh25_8 = data8.select(stepType='min',parameterName="200",topLevel=5000,bottomLevel=2000)[0].values
minuh25_9 = data9.select(stepType='min',parameterName="200",topLevel=5000,bottomLevel=2000)[0].values

maxuh25_1[maxuh25_1 < 10] = 0
maxuh25_2[maxuh25_2 < 10] = 0
maxuh25_3[maxuh25_3 < 10] = 0
maxuh25_4[maxuh25_4 < 10] = 0
maxuh25_5[maxuh25_5 < 10] = 0
maxuh25_6[maxuh25_6 < 10] = 0
maxuh25_7[maxuh25_7 < 10] = 0
maxuh25_8[maxuh25_8 < 10] = 0
maxuh25_9[maxuh25_9 < 10] = 0
minuh25_1[minuh25_1 > -10] = 0
minuh25_2[minuh25_2 > -10] = 0
minuh25_3[minuh25_3 > -10] = 0
minuh25_4[minuh25_4 > -10] = 0
minuh25_5[minuh25_5 > -10] = 0
minuh25_6[minuh25_6 > -10] = 0
minuh25_7[minuh25_7 > -10] = 0
minuh25_8[minuh25_8 > -10] = 0
minuh25_9[minuh25_9 > -10] = 0

uh25_1 = maxuh25_1 + minuh25_1
uh25_2 = maxuh25_2 + minuh25_2
uh25_3 = maxuh25_3 + minuh25_3
uh25_4 = maxuh25_4 + minuh25_4
uh25_5 = maxuh25_5 + minuh25_5
uh25_6 = maxuh25_6 + minuh25_6
uh25_7 = maxuh25_7 + minuh25_7
uh25_8 = maxuh25_8 + minuh25_8
uh25_9 = maxuh25_9 + minuh25_9

# Most Unstable CAPE
mucape_1 = data1.select(name='Convective available potential energy',topLevel=18000)[0].values
mucape_2 = data2.select(name='Convective available potential energy',topLevel=18000)[0].values
mucape_3 = data3.select(name='Convective available potential energy',topLevel=18000)[0].values
mucape_4 = data4.select(name='Convective available potential energy',topLevel=18000)[0].values
mucape_5 = data5.select(name='Convective available potential energy',topLevel=18000)[0].values
mucape_6 = data6.select(name='Convective available potential energy',topLevel=18000)[0].values
mucape_7 = data7.select(name='Convective available potential energy',topLevel=18000)[0].values
mucape_8 = data8.select(name='Convective available potential energy',topLevel=18000)[0].values
mucape_9 = data9.select(name='Convective available potential energy',topLevel=18000)[0].values

# Most Unstable CIN
mucin_1 = data1.select(name='Convective inhibition',topLevel=18000)[0].values
mucin_2 = data2.select(name='Convective inhibition',topLevel=18000)[0].values
mucin_3 = data3.select(name='Convective inhibition',topLevel=18000)[0].values
mucin_4 = data4.select(name='Convective inhibition',topLevel=18000)[0].values
mucin_5 = data5.select(name='Convective inhibition',topLevel=18000)[0].values
mucin_6 = data6.select(name='Convective inhibition',topLevel=18000)[0].values
mucin_7 = data7.select(name='Convective inhibition',topLevel=18000)[0].values
mucin_8 = data8.select(name='Convective inhibition',topLevel=18000)[0].values
mucin_9 = data9.select(name='Convective inhibition',topLevel=18000)[0].values

# Max Hourly Upward Vertical Velocity
maxuvv_1 = data1.select(stepType='max',parameterName="220",typeOfLevel="isobaricLayer",topLevel=100,bottomLevel=1000)[0].values
maxuvv_2 = data2.select(stepType='max',parameterName="220",typeOfLevel="isobaricLayer",topLevel=100,bottomLevel=1000)[0].values
maxuvv_3 = data3.select(stepType='max',parameterName="220",typeOfLevel="isobaricLayer",topLevel=100,bottomLevel=1000)[0].values
maxuvv_4 = data4.select(stepType='max',parameterName="220",typeOfLevel="isobaricLayer",topLevel=100,bottomLevel=1000)[0].values
maxuvv_5 = data5.select(stepType='max',parameterName="220",typeOfLevel="isobaricLayer",topLevel=100,bottomLevel=1000)[0].values
maxuvv_6 = data6.select(stepType='max',parameterName="220",typeOfLevel="isobaricLayer",topLevel=100,bottomLevel=1000)[0].values
maxuvv_7 = data7.select(stepType='max',parameterName="220",typeOfLevel="isobaricLayer",topLevel=100,bottomLevel=1000)[0].values
maxuvv_8 = data8.select(stepType='max',parameterName="220",typeOfLevel="isobaricLayer",topLevel=100,bottomLevel=1000)[0].values
maxuvv_9 = data9.select(stepType='max',parameterName="220",typeOfLevel="isobaricLayer",topLevel=100,bottomLevel=1000)[0].values


t2a = time.clock()
t3a = round(t2a-t1a, 3)
print(("%.3f seconds to read all messages") % t3a)

########################################
#    START PLOTTING FOR EACH DOMAIN    #
########################################

def main():

  # Number of processes must coincide with the number of domains to plot
  pool = MyPool(len(domains))
  pool.map(plot_all,domains)

def plot_all(domain):

  global dom
  dom = domain
  print(('Working on '+dom))

  # Call function to create figure and axes instances
  global fig,axes,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,keep_ax_lst_4,keep_ax_lst_5,keep_ax_lst_6,keep_ax_lst_7,keep_ax_lst_8,keep_ax_lst_9,m,x,y,x_shift,y_shift,xscale,yscale,im,par
  fig,axes,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,keep_ax_lst_4,keep_ax_lst_5,keep_ax_lst_6,keep_ax_lst_7,keep_ax_lst_8,keep_ax_lst_9,m,x,y,x_shift,y_shift,xscale,yscale,im,par = create_figure()
 
  # Call function to plot all variables
  plot_allvars()


def create_figure():

  # create figure and axes instances
  fig = plt.figure()
  gs = GridSpec(14,12,wspace=0.0,hspace=0.0)
  ax1 = fig.add_subplot(gs[0:4,0:4])
  ax2 = fig.add_subplot(gs[0:4,4:8])
  ax3 = fig.add_subplot(gs[0:4,8:])
  ax4 = fig.add_subplot(gs[5:9,0:4])
  ax5 = fig.add_subplot(gs[5:9,4:8])
  ax6 = fig.add_subplot(gs[5:9,8:])
  ax7 = fig.add_subplot(gs[10:,0:4])
  ax8 = fig.add_subplot(gs[10:,4:8])
  ax9 = fig.add_subplot(gs[10:,8:])
  axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
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
  elif dom == 'storm':
    llcrnrlon = -82.0
    llcrnrlat = 36.5
    urcrnrlon = -73.5
    urcrnrlat = 42.0
    lat_0 = 37.5
    lon_0 = -80.0
    xscale=0.18
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
                  resolution='l')
    else:
      m = Basemap(ax=ax,projection='gnom',lat_0=lat_0,lon_0=lon_0,\
                  llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,\
                  llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,\
                  resolution='l')
    m.fillcontinents(color='LightGrey',zorder=0)
    m.drawcoastlines(linewidth=0.25)
    m.drawstates(linewidth=0.25)
    m.drawcountries(linewidth=0.25)
##  parallels = np.arange(0.,90.,10.)
##  map.drawparallels(parallels,labels=[1,0,0,0],fontsize=6)
##  meridians = np.arange(180.,360.,10.)
##  map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=6)
    x,y = m(lon,lat)
    x_shift,y_shift   = m(lon_shift,lat_shift)

  # Map/figure has been set up here, save axes instances for use again later
    if par == 1:
      keep_ax_lst_1 = ax.get_children()[:]
    elif par == 2:
      keep_ax_lst_2 = ax.get_children()[:]
    elif par == 3:
      keep_ax_lst_3 = ax.get_children()[:]
    elif par == 4:
      keep_ax_lst_4 = ax.get_children()[:]
    elif par == 5:
      keep_ax_lst_5 = ax.get_children()[:]
    elif par == 6:
      keep_ax_lst_6 = ax.get_children()[:]
    elif par == 7:
      keep_ax_lst_7 = ax.get_children()[:]
    elif par == 8:
      keep_ax_lst_8 = ax.get_children()[:]
    elif par == 9:
      keep_ax_lst_9 = ax.get_children()[:]

    par += 1
  par = 1

  return fig,axes,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,keep_ax_lst_4,keep_ax_lst_5,keep_ax_lst_6,keep_ax_lst_7,keep_ax_lst_8,keep_ax_lst_9,m,x,y,x_shift,y_shift,xscale,yscale,im,par


def plot_allvars():
# Add print to see if dom is being passed in correctly
  print(('Running plot_allvars over domain: '+dom))

  global fig,axes,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,keep_ax_lst_4,keep_ax_lst_5,keep_ax_lst_6,keep_ax_lst_7,keep_ax_lst_8,keep_ax_lst_9,m,x,y,x_shift,y_shift,xscale,yscale,im,par


#################################
  # Plot 2-m temperature
#################################
  t1 = time.clock()
  print(('Working on 2-m temperature for '+dom))

  units = '\xb0''F'
  clevs = np.linspace(-16,134,51)
  cm = cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cbars = []
  var = [tmp2m_1,tmp2m_2,tmp2m_3,tmp2m_4,tmp2m_5,tmp2m_6,tmp2m_7,tmp2m_8,tmp2m_9]

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    cs = m.pcolormesh(x_shift,y_shift,var[par-1],cmap=cm,norm=norm,ax=ax)
    cs.cmap.set_under('white')
    cs.cmap.set_over('white')

    if par == 7 or par == 8 or par == 9:
      cbar = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,ticks=[-16,-4,8,20,32,44,56,68,80,92,104,116,128],extend='both')
      cbar.ax.tick_params(labelsize=4)
      cbar.set_label(units,fontsize=4)
      cbars.append(cbar)

    ax.text(.5,1.03,'Member '+str(par)+' 2-m Temperature ('+units+') \n init: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=4,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('2mt_members_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 2-m temperature for: '+dom) % t3)

#################################
  # Plot 2-m Dew Point
#################################
  t1 = time.clock()
  print(('Working on 2-m Dew Point for '+dom))

  # Clear off old plottables but keep all the map info
  cbars[0].remove()
  cbars[1].remove()
  cbars[2].remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)
  clear_plotables(ax4,keep_ax_lst_4,fig)
  clear_plotables(ax5,keep_ax_lst_5,fig)
  clear_plotables(ax6,keep_ax_lst_6,fig)
  clear_plotables(ax7,keep_ax_lst_7,fig)
  clear_plotables(ax8,keep_ax_lst_8,fig)
  clear_plotables(ax9,keep_ax_lst_9,fig)

  units = '\xb0''F'
  clevs = np.linspace(-5,80,35)
  cm = ncepy.cmap_q2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cbars = []
  var = [dew2m_1,dew2m_2,dew2m_3,dew2m_4,dew2m_5,dew2m_6,dew2m_7,dew2m_8,dew2m_9]

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    cs = m.pcolormesh(x_shift,y_shift,var[par-1],cmap=cm,norm=norm,ax=ax)

    if par == 7 or par == 8 or par == 9:
      cbar = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar.ax.tick_params(labelsize=4)
      cbar.set_label(units,fontsize=4)
      cbars.append(cbar)

    ax.text(.5,1.03,'Member '+str(par)+' 2-m Dew Point ('+units+') \n init: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=4,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('2mdew_members_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 2-m Dew Point for: '+dom) % t3)

#################################
  # Plot composite reflectivity
#################################
  t1 = time.clock()
  print(('Working on composite reflectivity for '+dom))

  # Clear off old plottables but keep all the map info
  cbars[0].remove()
  cbars[1].remove()
  cbars[2].remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)
  clear_plotables(ax4,keep_ax_lst_4,fig)
  clear_plotables(ax5,keep_ax_lst_5,fig)
  clear_plotables(ax6,keep_ax_lst_6,fig)
  clear_plotables(ax7,keep_ax_lst_7,fig)
  clear_plotables(ax8,keep_ax_lst_8,fig)
  clear_plotables(ax9,keep_ax_lst_9,fig)

  units = 'dBZ'
  clevs = np.linspace(5,70,14)
  colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cbars = []
  var = [refc_1,refc_2,refc_3,refc_4,refc_5,refc_6,refc_7,refc_8,refc_9]

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    cs = m.pcolormesh(x_shift,y_shift,var[par-1],cmap=cm,vmin=5,norm=norm,ax=ax)
    cs.cmap.set_under('white',alpha=0.)
    cs.cmap.set_over('black')

    if par == 7 or par == 8 or par == 9:
      cbar = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,ticks=clevs,extend='max')
      cbar.ax.tick_params(labelsize=4)
      cbar.set_label(units,fontsize=4)
      cbars.append(cbar)

    ax.text(.5,1.03,'Member '+str(par)+' Composite Reflectivity ('+units+') \n init: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=4,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('refc_members_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot composite reflectivity for: '+dom) % t3)

#################################
  # Plot Max/Min Hourly 2-5 km UH
#################################
  t1 = time.clock()
  print(('Working on Max/Min Hourly 2-5 km UH for '+dom))

  # Clear off old plottables but keep all the map info
  cbars[0].remove()
  cbars[1].remove()
  cbars[2].remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)
  clear_plotables(ax4,keep_ax_lst_4,fig)
  clear_plotables(ax5,keep_ax_lst_5,fig)
  clear_plotables(ax6,keep_ax_lst_6,fig)
  clear_plotables(ax7,keep_ax_lst_7,fig)
  clear_plotables(ax8,keep_ax_lst_8,fig)
  clear_plotables(ax9,keep_ax_lst_9,fig)

  units = 'm${^2}$ s$^{-2}$'
  clevs = [-150,-100,-75,-50,-25,-10,0,10,25,50,75,100,150,200,250,300]
#  colorlist = ['white','skyblue','mediumblue','green','orchid','firebrick','#EEC900','DarkViolet']
  colorlist = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','#E5E5E5','#E5E5E5','#EEEE00','#EEC900','darkorange','orangered','red','firebrick','mediumvioletred','darkviolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cbars = []
  var = [uh25_1,uh25_2,uh25_3,uh25_4,uh25_5,uh25_6,uh25_7,uh25_8,uh25_9]

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    cs = m.pcolormesh(x_shift,y_shift,var[par-1],cmap=cm,norm=norm,ax=ax)
    cs.cmap.set_under('darkblue')
    cs.cmap.set_over('black')

    if par == 7 or par == 8 or par == 9:
      cbar = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar.ax.tick_params(labelsize=4)
      cbar.set_label(units,fontsize=4)
      cbars.append(cbar)

    ax.text(.5,1.03,'Member '+str(par)+' 1-h Max/Min 2-5 km Updraft Helicity ('+units+') \n init: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=4,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('uh25_members_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Max/Min Hourly 2-5 km UH for: '+dom) % t3)

#################################
  # Plot Most Unstable CAPE/CIN
#################################
  t1 = time.clock()
  print(('Working on mucapecin for '+dom))

  # Clear off old plottables but keep all the map info
  cbars[0].remove()
  cbars[1].remove()
  cbars[2].remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)
  clear_plotables(ax4,keep_ax_lst_4,fig)
  clear_plotables(ax5,keep_ax_lst_5,fig)
  clear_plotables(ax6,keep_ax_lst_6,fig)
  clear_plotables(ax7,keep_ax_lst_7,fig)
  clear_plotables(ax8,keep_ax_lst_8,fig)
  clear_plotables(ax9,keep_ax_lst_9,fig)

  units = 'J/kg'
  clevs = [100,250,500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
  clevs2 = [-2000,-500,-250,-100,-25]
  colorlist = ['blue','dodgerblue','cyan','mediumspringgreen','#FAFAD2','#EEEE00','#EEC900','darkorange','crimson','darkred','darkviolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cbars = []
  var1 = [mucape_1,mucape_2,mucape_3,mucape_4,mucape_5,mucape_6,mucape_7,mucape_8,mucape_9]
  var2 = [mucin_1,mucin_2,mucin_3,mucin_4,mucin_5,mucin_6,mucin_7,mucin_8,mucin_9]

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    cs = m.pcolormesh(x_shift,y_shift,var1[par-1],cmap=cm,vmin=100,norm=norm,ax=ax)
    cs.cmap.set_under('white',alpha=0.)
    cs.cmap.set_over('black')
    cs_b = m.contourf(x,y,var2[par-1],clevs2,colors='none',hatches=['**','++','////','..'],ax=ax)

    if par == 7 or par == 8 or par == 9:
      cbar = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar.ax.tick_params(labelsize=4)
      cbar.set_label(units,fontsize=4)
      cbars.append(cbar)

    ax.text(.5,1.03,'Member '+str(par)+' MUCAPE (shaded) and MUCIN (hatched) ('+units+') \n <-500 (*), -500<-250 (+), -250<-100 (/), -100<-25 (.) \n init: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=4,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('mucape_members_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot mucapecin for: '+dom) % t3)

#################################
  # Plot Max Hourly Upward Vertical Velocity
#################################
  t1 = time.clock()
  print(('Working on maxuvv for '+dom))

  # Clear off old plottables but keep all the map info
  cbars[0].remove()
  cbars[1].remove()
  cbars[2].remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)
  clear_plotables(ax2,keep_ax_lst_2,fig)
  clear_plotables(ax3,keep_ax_lst_3,fig)
  clear_plotables(ax4,keep_ax_lst_4,fig)
  clear_plotables(ax5,keep_ax_lst_5,fig)
  clear_plotables(ax6,keep_ax_lst_6,fig)
  clear_plotables(ax7,keep_ax_lst_7,fig)
  clear_plotables(ax8,keep_ax_lst_8,fig)
  clear_plotables(ax9,keep_ax_lst_9,fig)

  units = 'm s$^{-1}$'
  clevs = [0.5,1,2.5,5,7.5,10,12.5,15,20,25,30,35,40,50,75]
  colorlist = ['turquoise','dodgerblue','mediumblue','lime','limegreen','green','#EEEE00','#EEC900','darkorange','red','firebrick','darkred','fuchsia','mediumpurple']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cbars = []
  var = [maxuvv_1,maxuvv_2,maxuvv_3,maxuvv_4,maxuvv_5,maxuvv_6,maxuvv_7,maxuvv_8,maxuvv_9]

  for ax in axes:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    cs = m.pcolormesh(x_shift,y_shift,var[par-1],cmap=cm,norm=norm,ax=ax)
    cs.cmap.set_under('white')
    cs.cmap.set_over('black')
    cbar.ax.tick_params(labelsize=4)

    if par == 7 or par == 8 or par == 9:
      cbar = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='both')
      cbar.ax.tick_params(labelsize=4)
      cbar.set_label(units,fontsize=4)
      cbars.append(cbars)
    
    ax.text(.5,1.03,'Member '+str(par)+' 1-h Max 100-1000 mb UVV ('+units+') \n init: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=4,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)

    par += 1
  par = 1

  compress_and_save('maxuvv_members_'+dom+'_f'+fhour+'.png')
  t2 = time.clock()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot maxuvv for: '+dom) % t3)



  plt.clf()

################################################################################

main()

