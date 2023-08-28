#!/bin/usr/env python

import numpy as np
import sys
import grib2io
import pyproj
import cartopy.crs as ccrs
import io
from PIL import Image
from scipy.ndimage.filters import minimum_filter, maximum_filter
import dateutil.relativedelta, dateutil.parser
from subprocess import call
import matplotlib.pyplot as plt
from matplotlib import colors

####################################
#  Time and date
####################################

def ndate(cdate,hours):
   if not isinstance(cdate, str):
     if isinstance(cdate, int):
       cdate=str(cdate)
     else:
       sys.exit('NDATE: Error - input cdate must be string or integer.  Exit!')
   if not isinstance(hours, int):
     if isinstance(hours, str):
       hours=int(hours)
     else:
       sys.exit('NDATE: Error - input delta hour must be a string or integer.  Exit!')
    
   indate=cdate.strip()
   hh=indate[8:10]
   yyyy=indate[0:4]
   mm=indate[4:6]
   dd=indate[6:8]
   #set date/time field
   parseme=(yyyy+' '+mm+' '+dd+' '+hh)
   datetime_cdate=dateutil.parser.parse(parseme)
   valid=datetime_cdate+dateutil.relativedelta.relativedelta(hours=+hours)
   vyyyy=str(valid.year)
   vm=str(valid.month).zfill(2)
   vd=str(valid.day).zfill(2)
   vh=str(valid.hour).zfill(2)
   return vyyyy+vm+vd+vh

####################################
#  Functions
####################################

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
  ram = io.BytesIO()
  plt.savefig(ram, format='png', bbox_inches='tight', dpi=300)
  ram.seek(0)
  im = Image.open(ram)
  im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
  im2.save(filename, format='PNG')

def extrema(mat,mode='wrap',window=10):
  # From: http://matplotlib.org/basemap/users/examples.html
  
  """find the indices of local extrema (min and max)
  in the input array."""
  mn = minimum_filter(mat, size=window, mode=mode)
  mx = maximum_filter(mat, size=window, mode=mode)
  # (mat == mx) true if pixel is equal to the local max  # (mat == mn) true if pixel is equal to the local in
  # Return the indices of the maxima, minima
  return np.nonzero(mat == mn), np.nonzero(mat == mx)

def plt_highs_and_lows(x,y,mat,xmin,xmax,ymin,ymax,offset,ax,transform,mode='wrap',window=10):
  # From: http://matplotlib.org/basemap/users/examples.html
  if isinstance(window,int) == False:
    raise TypeError("The window argument to plt_highs_and_lows must be an integer.")
  local_min, local_max = extrema(mat,mode,window)
  xlows = x[local_min]; xhighs = x[local_max]
  ylows = y[local_min]; yhighs = y[local_max]
  lowvals = mat[local_min]; highvals = mat[local_max]
  # plot lows as red L's, with min pressure value underneath.
  xyplotted = []
  # don't plot if there is already a L or H within dmin meters.
#  yoffset = 0.022*(ymax-ymin)
  yoffset = offset
  dmin = yoffset
  for x,y,p in zip(xlows, ylows, lowvals):
#    if x < xmax and x > xmin and y < ymax and y > ymin:
#        dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
#        if not dist or min(dist) > dmin:
            ax.text(x,y,'L',fontsize=14,fontweight='bold',
                    ha='center',va='center',color='r',zorder=4,clip_on=True,
                    transform=transform)
            ax.text(x,y-yoffset,repr(int(p)),fontsize=6,zorder=4,
                    ha='center',va='top',color='r',
                    bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)),clip_on=True,
                    transform=transform)
            xyplotted.append((x,y))
  # plot highs as blue H's, with max pressure value underneath.
  xyplotted = []
  for x,y,p in zip(xhighs, yhighs, highvals):
#    if x < xmax and x > xmin and y < ymax and y > ymin:
#        dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
#        if not dist or min(dist) > dmin:
            ax.text(x,y,'H',fontsize=14,fontweight='bold',
                    ha='center',va='center',color='b',zorder=4,clip_on=True,
                    transform=transform)
            ax.text(x,y-yoffset,repr(int(p)),fontsize=6,
                    ha='center',va='top',color='b',zorder=4,
                    bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)),clip_on=True,
                    transform=transform)
            xyplotted.append((x,y))

def get_latlons_pcolormesh(msg):
# Get shifted lats and lons for plotting with pcolormesh
  lats = []
  lons = []
  lats_shift = []
  lons_shift = []

# Unshifted grid for contours and wind barbs
  lat, lon = msg.grid()
  lats.append(lat)
  lons.append(lon)

# Unshifted grid for contours and wind barbs
  lat, lon = msg.grid()
  lats.append(lat)
  lons.append(lon)

# Shift grid for pcolormesh
  lat1 = msg.latitudeFirstGridpoint
  lon1 = msg.longitudeFirstGridpoint
  nx = msg.nx
  ny = msg.ny
  dx = msg.gridlengthXDirection
  dy = msg.gridlengthYDirection
  pj = pyproj.Proj(msg.projParameters)
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

# Fix for Alaska
  if (np.min(lon_shift) < 0) and (np.max(lon_shift) > 0):
    lon_shift = np.where(lon_shift>0,lon_shift-360,lon_shift)

  return lat, lon, lat_shift, lon_shift


####################################
#  Color shading / Color bars
####################################

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
    cmap_t2m_coltbl = colors.LinearSegmentedColormap('CMAP_T2M_COLTBL',colorDict)
    return cmap_t2m_coltbl

def cmap_q2m():
  # Create colormap for dew point temperature
    r=np.array([255,179,96,128,0, 0,  51, 0,  0,  0,  133,51, 70, 0,  128,128,180])
    g=np.array([255,179,96,128,92,128,153,155,155,255,162,102,70, 0,  0,  0,  0])
    b=np.array([255,179,96,0,  0, 0,  102,155,255,255,255,255,255,128,255,128,128])
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
    cmap_q2m_coltbl = colors.LinearSegmentedColormap('CMAP_Q2M_COLTBL',colorDict)
    cmap_q2m_coltbl.set_over(color='deeppink')
    return cmap_q2m_coltbl

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
    cmap_t850_coltbl = colors.LinearSegmentedColormap('CMAP_T850_COLTBL',colorDict)
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
    cmap_terra_coltbl = colors.LinearSegmentedColormap('CMAP_TERRA_COLTBL',colorDict)
    cmap_terra_coltbl.set_over(color='#E0EEE0')
    return cmap_terra_coltbl

def ncl_perc_11Lev():
  # Create colormap for snowfall
    r=np.array([202,89,139,96,26,145,217,254,252,215,150])
    g=np.array([202,141,239,207,152,207,239,224,141,48,0])
    b=np.array([200,252,217,145,80,96,139,139,89,39,100])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    blue = []
    green = []
    for i in range(len(xsize)):
        xNorm=np.float(i)/(np.float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    my_coltbl = colors.LinearSegmentedColormap('NCL_PERC_11LEV_COLTBL',colorDict)
    return my_coltbl

def ncl_grnd_hflux():
  # Create colormap for ground heat flux
    r=np.array([0,8,16,24,32,40,48,85,133,181,230,253,253,253,253,253,253,253,253,253,253,
253])
    g=np.array([253,222,189,157,125,93,60,85,133,181,230,230,181,133,85,60,93,125,157,189,
224,253])
    b=np.array([253,253,253,253,253,253,253,253,253,253,253,230,181,133,85,48,40,32,24,16,
8,0])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    blue = []
    green = []
    for i in range(len(xsize)):
        xNorm=np.float(i)/(np.float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    my_coltbl = colors.LinearSegmentedColormap('NCL_GRND_HFLUX_COLTBL',colorDict)
    return my_coltbl


####################################
#  Domain settings
####################################

def domain_latlons_proj(dom):
#  These are the available pre-defined domains:
#  
#  namerica, alaska, hawaii, puerto rico,
#  conus, northeast, mid_atlantic, southeast, ohio_valley,
#  upper_midwest, north_central, central, south_central,
#  northwest, southwest, colorado, boston_nyc,
#  seattle_portland, sf_bay_area, la_vegas

# Latitudes and longitudes
  if dom == 'namerica': 
    llcrnrlon = -160.0
    llcrnrlat = 15.0
    urcrnrlon = -55.0
    urcrnrlat = 65.0
    cen_lat = 35.4
    cen_lon = -105.0
    xextent = -3700000
    yextent = -2500000
    offset = 1
  elif dom == 'alaska':
    llcrnrlon = -167.5
    llcrnrlat = 50.5
    urcrnrlon = -135.8
    urcrnrlat = 72.5
    cen_lat = 60.0
    cen_lon = -150.0
    lat_ts = 60.0
    xextent=-850000
    yextent=-600000
    offset=1
  elif dom == 'hawaii':
    llcrnrlon = -162.3
    llcrnrlat = 16.2
    urcrnrlon = -153.1
    urcrnrlat = 24.3
    cen_lat = 20.4
    cen_lon = -157.6
    xextent=-325000
    yextent=-285000
    offset=0.25
  elif dom == 'puerto_rico':
    llcrnrlon = -76.5
    llcrnrlat = 13.3
    urcrnrlon = -61.0
    urcrnrlat = 22.7
    cen_lat = 18.4
    cen_lon = -66.6
    xextent=-925000
    yextent=-375000
    offset=0.25
  elif dom == 'conus':
    llcrnrlon = -125.5
    llcrnrlat = 20.0
    urcrnrlon = -63.5
    urcrnrlat = 51.0
    cen_lat = 35.4
    cen_lon = -97.6
    xextent=-2200000
    yextent=-675000
    offset=1
  elif dom == 'northeast':
    llcrnrlon = -80.0
    llcrnrlat = 40.0
    urcrnrlon = -66.5
    urcrnrlat = 48.0
    cen_lat = 44.0
    cen_lon = -76.0
    xextent=-175000
    yextent=-282791
    offset=0.25
  elif dom == 'mid_atlantic':
    llcrnrlon = -82.0
    llcrnrlat = 36.5
    urcrnrlon = -73.0
    urcrnrlat = 42.5
    cen_lat = 36.5
    cen_lon = -79.0
    xextent=-123114
    yextent=125850
    offset=0.25
  elif dom == 'southeast':
    llcrnrlon = -92.0
    llcrnrlat = 24.0
    urcrnrlon = -75.0
    urcrnrlat = 37.0
    cen_lat = 30.5
    cen_lon = -89.0
    xextent=-12438
    yextent=-448648
    offset=0.25
  elif dom == 'ohio_valley':
    llcrnrlon = -91.5
    llcrnrlat = 34.5
    urcrnrlon = -80.0
    urcrnrlat = 43.0
    cen_lat = 38.75
    cen_lon = -88.0
    xextent=-131129
    yextent=-299910
    offset=0.25
  elif dom == 'upper_midwest':
    llcrnrlon = -97.5
    llcrnrlat = 40.0
    urcrnrlon = -82.0
    urcrnrlat = 49.5
    cen_lat = 44.75
    cen_lon = -92.0
    xextent=-230258
    yextent=-316762
    offset=0.25
  elif dom == 'north_central':
    llcrnrlon = -111.5
    llcrnrlat = 39.0
    urcrnrlon = -94.0
    urcrnrlat = 49.5
    cen_lat = 44.25
    cen_lon = -103.0
    xextent=-490381
    yextent=-336700
    offset=0.25
  elif dom == 'central':
    llcrnrlon = -103.5
    llcrnrlat = 32.0
    urcrnrlon = -89.0
    urcrnrlat = 42.0
    cen_lat = 37.0
    cen_lon = -99.0
    xextent=-220257
    yextent=-337668
    offset=0.25
  elif dom == 'south_central':
    llcrnrlon = -109.0
    llcrnrlat = 25.0
    urcrnrlon = -88.5
    urcrnrlat = 37.5
    cen_lat = 31.25
    cen_lon = -101.0
    xextent=-529631
    yextent=-407090
    offset=0.25
  elif dom == 'northwest':
    llcrnrlon = -125.0
    llcrnrlat = 40.0
    urcrnrlon = -110.0
    urcrnrlat = 50.0
    cen_lat = 45.0
    cen_lon = -116.0
    xextent=-540000
    yextent=-333623
    offset=0.25
  elif dom == 'southwest':
    llcrnrlon = -125.0
    llcrnrlat = 31.0
    urcrnrlon = -108.5
    urcrnrlat = 42.5
    cen_lat = 36.75
    cen_lon = -116.0
    xextent=-593059
    yextent=-377213
    offset=0.25
  elif dom == 'colorado':
    llcrnrlon = -110.0
    llcrnrlat = 35.0
    urcrnrlon = -101.0
    urcrnrlat = 42.0
    cen_lat = 38.5
    cen_lon = -106.0
    xextent=-224751
    yextent=-238851
    offset=0.25
  elif dom == 'boston_nyc':
    llcrnrlon = -75.5
    llcrnrlat = 40.0
    urcrnrlon = -69.5
    urcrnrlat = 43.0
    cen_lat = 41.5
    cen_lon = -76.0
    xextent=112182
    yextent=-99031
    offset=0.25
  elif dom == 'seattle_portland':
    llcrnrlon = -125.0
    llcrnrlat = 44.5
    urcrnrlon = -119.0
    urcrnrlat = 49.5
    cen_lat = 47.0
    cen_lon = -121.0
    xextent=-227169
    yextent=-200000
    offset=0.25
  elif dom == 'sf_bay_area':
    llcrnrlon = -123.5
    llcrnrlat = 37.25
    urcrnrlon = -121.0
    urcrnrlat = 38.5
    cen_lat = 48.25
    cen_lon = -121.0
    xextent=-185364
    yextent=-1193027
    offset=0.25
  elif dom == 'la_vegas':
    llcrnrlon = -121.0
    llcrnrlat = 32.0
    urcrnrlon = -114.0
    urcrnrlat = 37.0
    cen_lat = 34.5
    cen_lon = -114.0
    xextent=-540000
    yextent=-173241
    offset=0.25

# Projection settings
  if dom == 'namerica':
    extent = [-176.,0.,0.5,45.]
    myproj = ccrs.Orthographic(central_longitude=-114, central_latitude=54.0, globe=None)
  elif dom == 'alaska':
    extent = [llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat]
    myproj = ccrs.Stereographic(central_longitude=cen_lon, central_latitude=cen_lat,
         true_scale_latitude=None,false_easting=0.0,false_northing=0.0,globe=None)
  elif dom == 'conus':
    extent = [llcrnrlon-1, urcrnrlon-6, llcrnrlat, urcrnrlat+1]
    myproj=ccrs.LambertConformal(central_longitude=cen_lon, central_latitude=cen_lat,
         false_easting=0.0, false_northing=0.0, secant_latitudes=None,
         standard_parallels=None, globe=None)
  else:
    extent = [llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat]
    myproj=ccrs.LambertConformal(central_longitude=cen_lon, central_latitude=cen_lat,
         false_easting=0.0, false_northing=0.0, secant_latitudes=None,
         standard_parallels=None, globe=None)

  return xextent, yextent, offset, extent, myproj
