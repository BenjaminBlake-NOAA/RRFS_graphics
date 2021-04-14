import pygrib
import matplotlib
matplotlib.use('Agg')
import io
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap
import numpy as np
import time,os,sys,multiprocessing,itertools
import ncepy
import csv
from scipy import ndimage
import pyproj

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


#-------------------------------------------------------#

# Necessary to generate figs when not running an Xserver (e.g. via PBS)
# plt.switch_backend('agg')

# Read date/time and forecast hour from command line
cycle = str(sys.argv[1])
ymd = cycle[0:8]
year = int(cycle[0:4])
month = int(cycle[4:6])
day = int(cycle[6:8])
hour = int(cycle[8:10])
cyc = str(hour).zfill(2)
print(year, month, day, hour)

# Define the paths to the input files
DATA_DIR = '/scratch2/NCEPDEV/fv3-cam/Rajendra.Panda/'+str(ymd)

# Specify runlength for plots
runlength = 36

# Forecast init and valid date/time
itime = cycle
fhrs = np.arange(0,runlength+1,1)
vtime_list = [ncepy.ndate(itime,int(x)) for x in fhrs]


###################################################
# Read in all variables and calculate differences #
###################################################
t1a = time.clock()

uh25_list_1 = []
uh25_list_2 = []
uh25_list_3 = []
uh25_list_4 = []
uh25_list_5 = []
uh25_list_6 = []
uh25_list_7 = []
uh25_list_8 = []
uh25_list_9 = []

for j in range(len(vtime_list)):

    fhr = fhrs[j]
    fhour = str(fhr).zfill(2)
    print('fhour '+fhour)
    vtime = vtime_list[j]


    # Define the output files
    data1 = pygrib.open(DATA_DIR+'/1/PRSLEV.GrbF'+fhour)
    data2 = pygrib.open(DATA_DIR+'/2/PRSLEV.GrbF'+fhour)
    data3 = pygrib.open(DATA_DIR+'/3/PRSLEV.GrbF'+fhour)
    data4 = pygrib.open(DATA_DIR+'/4/PRSLEV.GrbF'+fhour)
    data5 = pygrib.open(DATA_DIR+'/5/PRSLEV.GrbF'+fhour)
    data6 = pygrib.open(DATA_DIR+'/6/PRSLEV.GrbF'+fhour)
    data7 = pygrib.open(DATA_DIR+'/7/PRSLEV.GrbF'+fhour)
    data8 = pygrib.open(DATA_DIR+'/8/PRSLEV.GrbF'+fhour)
    data9 = pygrib.open(DATA_DIR+'/9/PRSLEV.GrbF'+fhour)
    

    # Updraft helicity
    if (fhr > 0):
        uh25_1 = data1.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
        uh25_2 = data2.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
        uh25_3 = data3.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
        uh25_4 = data4.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
        uh25_5 = data5.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
        uh25_6 = data6.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
        uh25_7 = data7.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
        uh25_8 = data8.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
        uh25_9 = data9.select(stepType='max',parameterName="199",topLevel=5000,bottomLevel=2000)[0].values
        uh25_1[uh25_1 < 10] = 0
        uh25_2[uh25_2 < 10] = 0
        uh25_3[uh25_3 < 10] = 0
        uh25_4[uh25_4 < 10] = 0
        uh25_5[uh25_5 < 10] = 0
        uh25_6[uh25_6 < 10] = 0
        uh25_7[uh25_7 < 10] = 0
        uh25_8[uh25_8 < 10] = 0
        uh25_9[uh25_9 < 10] = 0
    else:
        uh25_1 = data1.select(name='Total Precipitation',lengthOfTimeRange=fhr)[0].values * 0.
        try:
            uh25_2 = data2.select(name='Total Precipitation',lengthOfTimeRange=fhr)[0].values * 0.
            uh25_3 = data3.select(name='Total Precipitation',lengthOfTimeRange=fhr)[0].values * 0.
            uh25_4 = data4.select(name='Total Precipitation',lengthOfTimeRange=fhr)[0].values * 0.
            uh25_5 = data5.select(name='Total Precipitation',lengthOfTimeRange=fhr)[0].values * 0.
            uh25_6 = data6.select(name='Total Precipitation',lengthOfTimeRange=fhr)[0].values * 0.
            uh25_7 = data7.select(name='Total Precipitation',lengthOfTimeRange=fhr)[0].values * 0.
            uh25_8 = data8.select(name='Total Precipitation',lengthOfTimeRange=fhr)[0].values * 0.
            uh25_9 = data9.select(name='Total Precipitation',lengthOfTimeRange=fhr)[0].values * 0.
        except:
            uh25_2 = data2.select(parameterName='Total precipitation',lengthOfTimeRange=fhr)[0].values * 0.
            uh25_3 = data3.select(parameterName='Total precipitation',lengthOfTimeRange=fhr)[0].values * 0.
            uh25_4 = data4.select(parameterName='Total precipitation',lengthOfTimeRange=fhr)[0].values * 0.
            uh25_5 = data5.select(parameterName='Total precipitation',lengthOfTimeRange=fhr)[0].values * 0.
            uh25_6 = data6.select(parameterName='Total precipitation',lengthOfTimeRange=fhr)[0].values * 0.
            uh25_7 = data7.select(parameterName='Total precipitation',lengthOfTimeRange=fhr)[0].values * 0.
            uh25_8 = data8.select(parameterName='Total precipitation',lengthOfTimeRange=fhr)[0].values * 0.
            uh25_9 = data9.select(parameterName='Total precipitation',lengthOfTimeRange=fhr)[0].values * 0.

    uh25_list_1.append(uh25_1)
    uh25_list_2.append(uh25_2)
    uh25_list_3.append(uh25_3)
    uh25_list_4.append(uh25_4)
    uh25_list_5.append(uh25_5)
    uh25_list_6.append(uh25_6)
    uh25_list_7.append(uh25_7)
    uh25_list_8.append(uh25_8)
    uh25_list_9.append(uh25_9)


    if vtime != vtime_list[-1]:
        data1.close() 
        data2.close() 
        data3.close() 
        data4.close() 
        data5.close() 
        data6.close() 
        data7.close() 
        data8.close() 
        data9.close() 


# Get the lats and lons
grids = [data1]
lats_shift = []
lons_shift = []

for data in grids:
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

# Shifted lat/lon arrays for pcolormesh 
lat1_shift = lats_shift[0]
lon1_shift = lons_shift[0]

# Close grib files 
data1.close()

t2a = time.clock()
t3a = round(t2a-t1a, 3)
print(("%.3f seconds to read all messages") % t3a)


# Specify plotting domains
domain = str(sys.argv[2])


########################################
#    START PLOTTING FOR EACH DOMAIN    #
########################################

def main():

  # Number of processes must coincide with the number of domains to plot
# pool = multiprocessing.Pool(len(fhrs[0:2]))
# pool.map(plot_all,fhrs[0:2])
  print(plots)
  pool = multiprocessing.Pool(len(plots))
  pool.map(plot_all,plots)

def plot_all(plot):

  thing = np.asarray(plot)
  dom = thing[0]
  fhr = int(thing[1])

  fhour = str(fhr).zfill(2)
  vtime = vtime_list[fhr]

  t1dom = time.clock()
  print('Working on '+dom+' for fhr '+fhour)

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
  im = image.imread('/scratch2/NCEPDEV/fv3-cam/Benjamin.Blake/python.rrfs/noaa.png')
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
    llcrnrlon = -93.0
    llcrnrlat = 27.0
    urcrnrlon = -79.0
    urcrnrlat = 37.0
    lat_0 = 33.0
    lon_0 = -86.0
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
    # Don't need unshifted arrays in this script so using alternate names
    x,y = m(lon1_shift,lat1_shift)
  
    par += 1
  par = 1


#################################
  # Plot Run-Total 2-5 km UH
#################################
  if (fhr >= 0):
    t1 = time.clock()
    print('Working on Run-Total 2-5 km UH for '+dom)

    uh25_1 = np.amax(np.array(uh25_list_1[0:fhr+1]),axis=0)
    uh25_2 = np.amax(np.array(uh25_list_2[0:fhr+1]),axis=0)
    uh25_3 = np.amax(np.array(uh25_list_3[0:fhr+1]),axis=0)
    uh25_4 = np.amax(np.array(uh25_list_4[0:fhr+1]),axis=0)
    uh25_5 = np.amax(np.array(uh25_list_5[0:fhr+1]),axis=0)
    uh25_6 = np.amax(np.array(uh25_list_6[0:fhr+1]),axis=0)
    uh25_7 = np.amax(np.array(uh25_list_7[0:fhr+1]),axis=0)
    uh25_8 = np.amax(np.array(uh25_list_8[0:fhr+1]),axis=0)
    uh25_9 = np.amax(np.array(uh25_list_9[0:fhr+1]),axis=0)

    var = [uh25_1,uh25_2,uh25_3,uh25_4,uh25_5,uh25_6,uh25_7,uh25_8,uh25_9]

    units = '$\mathregular{m^{2}}$ $\mathregular{s^{-2}}$'
    clevs = [25,50,75,100,150,200,250,300,400]
    clevsdif = [-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60]
    colorlist = ['turquoise','dodgerblue','lime','limegreen','yellow','darkorange','red','firebrick']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    for ax in axes:
      xmin, xmax = ax.get_xlim()
      ymin, ymax = ax.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

      cs = m.pcolormesh(x,y,uh25_1,cmap=cm,vmin=25,norm=norm,ax=ax)
      cs.cmap.set_under('white',alpha=0.)
      cs.cmap.set_over('fuchsia')

      if par == 7 or par == 8 or par == 9:
        cbar = m.colorbar(cs,ax=ax,location='bottom',pad=0.05,extend='max')
        cbar.ax.tick_params(labelsize=4)
        cbar.set_label(units,fontsize=4)

      ax.text(.5,1.03,'Member '+str(par)+' '+fhour+'-h Max 2-5 km Updraft Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=4,transform=ax.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
      ax.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(0,int(round(xmax*xscale)),0,int(round(ymax*yscale))),zorder=4)


      par += 1
    par = 1

    compress_and_save('compareuh25_accum_'+dom+'_f'+fhour+'.png')
    t2 = time.clock()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Run-Total 2-5 km UH for: '+dom) % t3)



######################################################

  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all variables for: "+dom) % t3dom)
  plt.clf()

######################################################

for fhr in fhrs:
    plot_all([domain,fhr])
#    for domain in domains:
#        plot_all([domain,fhr])

