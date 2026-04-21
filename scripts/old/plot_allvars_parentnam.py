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
fhourm24 = str(fhr-24).zfill(2)
print('fhour '+fhour)

# Forecast valid date/time
itime = ymdh
vtime_end = rrfs_plot_utils.ndate(itime,int(fhr))

# Define the directory paths to the output files
user = str(sys.argv[3])
NAM_DIR = os.path.join(os.environ['COMnam'],'nam.'+ymd)
RRFS_DIR = os.path.join(
            '/','lfs','h2','emc','ptmp',user,'rrfs','na','prod',
                'rrfs.'+ymd, cyc
                )

# Specify plotting domains

# Specify plotting domains
domset = str(sys.argv[4])
if domset == 'conus':
  domains = ['conus','boston_nyc','central','colorado','la_vegas','mid_atlantic','north_central','northeast','northwest','ohio_valley','south_central','southeast','south_florida','sf_bay_area','seattle_portland','southwest','upper_midwest']
elif domset == 'oconus':
  domains = ['alaska','hawaii','puerto_rico']

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

  global lat,lon,lat2,lon2,lat_shift,lon_shift,lat2_shift, lon2_shift,fig,axes,ax1,ax2,ax3,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,xextent,yextent,offset,extent,myproj,transform

# Define the input files

  if dom == 'alaska':
    data1 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.awak3d'+fhour+'.tm00.grib2')
    data2 = grib2io.open(RRFS_DIR+'/rrfs.t'+cyc+'z.prslev.3km.f0'+fhour+'.ak.grib2')
  elif dom == 'hawaii':
    data1 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.awiphi'+fhour+'.tm00.grib2')
    data2 = grib2io.open(RRFS_DIR+'/rrfs.t'+cyc+'z.prslev.2p5km.f0'+fhour+'.hi.grib2')
  elif dom == 'puerto_rico':
    data1 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.awp237'+fhour+'.tm00.grib2')
    data2 = grib2io.open(RRFS_DIR+'/rrfs.t'+cyc+'z.prslev.2p5km.f0'+fhour+'.pr.grib2')
  else:
    data1 = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.awip12'+fhour+'.tm00.grib2')
    data2 = grib2io.open(RRFS_DIR+'/rrfs.t'+cyc+'z.prslev.3km.f0'+fhour+'.conus.grib2')

# Get the lats and lons

  msg1 = data1.select(shortName='HGT', level='surface')[0]  # msg is a Grib2Message object
  lat,lon,lat_shift,lon_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg1)
  msg2 = data2.select(shortName='HGT', level='500 mb')[0]  # msg is a Grib2Message object
  lat2,lon2,lat2_shift,lon2_shift = rrfs_plot_utils.get_latlons_pcolormesh(msg2)

###################################################
# Read in all variables and calculate differences #
###################################################
  t1a = time.perf_counter()

  global slp_1,tmp2m_1,tmpsfc_1,dew2m_1,uwind_1,vwind_1,wspd10m_1,gust_1,mucape_1,cape_1,mlcape_1,thetae_1,u850_1,v850_1,z500_1,vort500_1,u500_1,v500_1,u250_1,v250_1,wspd250_1,vis_1,zbase_1,zceil_1,ztop_1,pw_1,qpf_1,snow_1,snow0_1,hpbl_1,hel3km_1,hel1km_1,ref1km_1,refc_1,uh25_1,tcdc_1,retop_1,prate_1,rain1,fr1,pl1,sn1,mix1

  global slp_2,tmp2m_2,tmpsfc_2,dew2m_2,uwind_2,vwind_2,wspd10m_2,gust_2,mucape_2,cape_2,mlcape_2,thetae_2,u850_2,v850_2,z500_2,vort500_2,u500_2,v500_2,u250_2,v250_2,wspd250_2,vis_2,zbase_2,zceil_2,ztop_2,pw_2,qpf_2,snow_2,snow0_2,hpbl_2,hel3km_2,hel1km_2,ref1km_2,refc_2,uh25_2,tcdc_2,retop_2,prate_2,rain2,fr2,pl2,sn2,mix2

# Sea level pressure
  slp_1 = data1.select(shortName='PRMSL',level='mean sea level')[0].data * 0.01
  slp_2 = data2.select(shortName='MSLET',level='mean sea level')[0].data * 0.01

# 2-m temperature
  tmp2m_1 = data1.select(shortName='TMP',level='2 m above ground')[0].data
  tmp2m_1 = (tmp2m_1 - 273.15)*1.8 + 32.0
  tmp2m_2 = data2.select(shortName='TMP',level='2 m above ground')[0].data
  tmp2m_2 = (tmp2m_2 - 273.15)*1.8 + 32.0

# Surface temperature
  tmpsfc_1 = data1.select(shortName='TMP',level='surface')[0].data
  tmpsfc_1 = (tmpsfc_1 - 273.15)*1.8 + 32.0
  tmpsfc_2 = data2.select(shortName='TMP',level='surface')[0].data
  tmpsfc_2 = (tmpsfc_2 - 273.15)*1.8 + 32.0

# 2-m dew point temperature
  if dom != "hawaii" and dom != "puerto_rico":
    dew2m_1 = data1.select(shortName='DPT',level='2 m above ground')[0].data
    dew2m_1 = (dew2m_1 - 273.15)*1.8 + 32.0
    dew2m_2 = data2.select(shortName='DPT',level='2 m above ground')[0].data
    dew2m_2 = (dew2m_2 - 273.15)*1.8 + 32.0
  else:
    print (f"no DPT data is available {dom}")
 
 # 10-m wind speed
  uwind_1 = data1.select(shortName='UGRD',level='10 m above ground')[0].data * 1.94384
  uwind_2 = data2.select(shortName='UGRD',level='10 m above ground')[0].data * 1.94384
  vwind_1 = data1.select(shortName='VGRD',level='10 m above ground')[0].data * 1.94384
  vwind_2 = data2.select(shortName='VGRD',level='10 m above ground')[0].data * 1.94384
  wspd10m_1 = np.sqrt(uwind_1**2 + vwind_1**2)
  wspd10m_2 = np.sqrt(uwind_2**2 + vwind_2**2)
  
  # Surface wind gust
  if dom != "hawaii":
    gust_1 = data1.select(shortName='GUST',level='surface')[0].data * 1.94384
    gust_2 = data2.select(shortName='GUST',level='surface')[0].data * 1.94384
  else:
    print (f"no GUST data is available {dom}")

  # Most unstable CAPE
  mucape_1 = data1.select(shortName='CAPE',level='180-0 mb above ground')[0].data
  mucape_2 = data2.select(shortName='CAPE',level='180-0 mb above ground')[0].data
  
  # Surface-based CAPE
  cape_1 = data1.select(shortName='CAPE',level='surface')[0].data
  cape_2 = data2.select(shortName='CAPE',level='surface')[0].data
  
  # Mixed Layer CAPE
  mlcape_1 = data1.select(shortName='CAPE',level='90-0 mb above ground')[0].data
  mlcape_2 = data2.select(shortName='CAPE',level='90-0 mb above ground')[0].data
  
  # Visibility
  vis_1 = data1.select(shortName='VIS',level='surface')[0].data * 0.000621371
  vis_2 = data2.select(shortName='VIS',level='surface')[0].data * 0.000621371
  
  
  # Cloud Ceiling Height
  if dom != "hawaii" and dom != "puerto_rico":
    zceil_1 = data1.select(shortName='HGT',level='cloud ceiling')[0].data * (3.28084/1000)
    zceil_1[zceil_1 < 0] = 99999
    zceil_2 = data2.select(shortName='HGT',level='cloud ceiling')[0].data * (3.28084/1000)
  else:
    print (f"no HGT data at cload ceiling level is available for {dom}")

  
  # Precipitable water
  pw_1 = data1.select(shortName='PWAT',level='entire atmosphere (considered as a single layer)')[0].data * 0.0393701
  pw_2 = data2.select(shortName='PWAT',level='entire atmosphere (considered as a single layer)')[0].data * 0.0393701
 

  # Accumulated snowfall
  #snow_1 = data1.select(shortName='SNOD')[0].data * 39.3701
  #snow_2 = data2.select(shortName='SNOD')[0].data * 39.3701

  if (fhr > 0): 
    snow_1 = data1.select(shortName='WEASD')[0].data / 2.54
    snow_2 = data2.select(shortName='ASNOW')[0].data * 39.3701
  
  # PBL height
  if dom != "hawaii" and dom != "puerto_rico" and dom != "alaska":
    hpbl_1 = data1.select(shortName='HGT',level='planetary boundary layer')[0].data
    hpbl_2 = data2.select(shortName='HGT',level='planetary boundary layer')[0].data
  else:
    print (f"no PBL height data is available for {dom}")

  # 0-3 km Storm Relative Helicity
  if dom != "hawaii" and dom != "puerto_rico" and dom != "alaska":
    hel3km_1 = data1.select(shortName='HLCY',scaledValueOfFirstFixedSurface=3000)[0].data
    hel3km_2 = data2.select(shortName='HLCY',scaledValueOfFirstFixedSurface=3000)[0].data
  else:
      print (f"no HGT data at cload ceiling level is available for {dom}")

  # 0-1 km Storm Relative Helicity
  if dom != "hawaii" and dom != "puerto_rico" and dom != "alaska":
    hel1km_1 = data1.select(shortName='HLCY',scaledValueOfFirstFixedSurface=1000)[0].data
    hel1km_2 = data2.select(shortName='HLCY',scaledValueOfFirstFixedSurface=1000)[0].data
  else:
    print (f"no Helicity data is available for {dom}")

  # 1-km reflectivity
  if dom != "hawaii" and dom != "puerto_rico" and dom != "alaska":
    ref1km_1 = data1.select(shortName='REFD',level='1000 m above ground')[0].data
    ref1km_2 = data2.select(shortName='REFD',level='1000 m above ground')[0].data
  else:
    print (f"no reflectivity data is available for {dom}")

  # Composite reflectivity
  if dom != "puerto_rico":
    refc_1 = data1.select(shortName='REFC')[0].data
    refc_2 = data2.select(shortName='REFC')[0].data
  else:
    print (f"no RFC data is available for {dom}")
  
  # Total cloud cover
  tcdc_1 = data1.select(shortName='TCDC')[0].data
  tcdc_2 = data2.select(shortName='TCDC',level='entire atmosphere (considered as a single layer)')[0].data
  
  # Echo top height
  if dom != "puerto_rico":
    retop_1 = data1.select(shortName='RETOP')[0].data * (3.28084/1000)
    retop_2 = data2.select(shortName='RETOP')[0].data * (3.28084/1000)
  else:
    print (f"no RETOP data is available for {dom}")

  # Precipitation rate
  prate_1 = data1.select(shortName='PRATE')[0].data * 3600
  prate_2 = data2.select(shortName='PRATE')[0].data * 3600

  # Accumulated precipitation

  if (fhr > 0):
    if (fhr == 1):
      qpf_2 = data2.select(shortName='APCP')[0].data * 0.0393701
    else:
      qpf_2 = data2.select(shortName='APCP')[1].data * 0.0393701

  if (fhr > 0) and (fhr % 12 == 0):
    qpf_1 = 0
    fhr_qpf = fhr
    while fhr_qpf > 0:
      if (fhr_qpf % 12 == 0) and (fhr_qpf <= 84):
        fhour_qpf = str(fhr_qpf).zfill(2)
        if dom == "alaska":
          data1_qpf = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.awak3d'+fhour_qpf+'.tm00.grib2')
        elif dom == "hawaii":
          data1_qpf = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.awiphi'+fhour_qpf+'.tm00.grib2')      
        elif dom == "puerto_rico":
          data1_qpf = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.awp237'+fhour_qpf+'.tm00.grib2')
        else:
          data1_qpf = grib2io.open(NAM_DIR+'/nam.t'+cyc+'z.awip12'+fhour_qpf+'.tm00.grib2')
        
        qpf = data1_qpf.select(shortName='APCP')[0].data * 0.0393701
        qpf_1 += qpf
#        print(f"At hour {fhr_qpf}, the total precipitation is {qpf_1}")
        fhr_qpf -= 12
#        print(f"total precipitation array is {qpf_1}")


  # Precipitation type
  if dom != "puerto_rico":
    rain1 = data1.select(shortName='CRAIN')[0].data
    fr1 = data1.select(shortName='CFRZR')[0].data
    pl1 = data1.select(shortName='CICEP')[0].data
    sn1 = data1.select(shortName='CSNOW')[0].data
  
    rain2 = data2.select(shortName='CRAIN')[0].data
    fr2 = data2.select(shortName='CFRZR')[0].data
    pl2 = data2.select(shortName='CICEP')[0].data
    sn2 = data2.select(shortName='CSNOW')[0].data
  
    types1 = np.zeros(fr1.shape)
    types1[rain1==1]=types1[rain1==1]+1
    types1[fr1==1]=types1[fr1==1]+3
    types1[pl1==1]=types1[pl1==1]+5
    types1[sn1==1]=types1[sn1==1]+7
    rain1=np.copy(refc_1)
    fr1=np.copy(refc_1)
    pl1=np.copy(refc_1)
    sn1=np.copy(refc_1)
    mix1=np.copy(refc_1)
    rain1[types1!=1]=-1 
    fr1[types1!=3]=-1
    pl1[types1!=5]=-1
    sn1[types1!=7]=-1
    mix1[types1==0]=-1
    mix1[types1==1]=-1
    mix1[types1==3]=-1
    mix1[types1==5]=-1
    mix1[types1==7]=-1
  
    types2 = np.zeros(fr2.shape)
    types2[rain2==1]=types2[rain2==1]+1
    types2[fr2==1]=types2[fr2==1]+3
    types2[pl2==1]=types2[pl2==1]+5
    types2[sn2==1]=types2[sn2==1]+7
    rain2=np.copy(refc_2)
    fr2=np.copy(refc_2)
    pl2=np.copy(refc_2)
    sn2=np.copy(refc_2)
    mix2=np.copy(refc_2)
    rain2[types2!=1]=-1
    fr2[types2!=3]=-1
    pl2[types2!=5]=-1
    sn2[types2!=7]=-1
    mix2[types2==0]=-1
    mix2[types2==1]=-1
    mix2[types2==3]=-1
    mix2[types2==5]=-1
    mix2[types2==7]=-1
  else:
    print (f"no precipitation type data is available for {dom}")

  t2a = time.perf_counter()
  t3a = round(t2a-t1a, 3)
  print(("%.3f seconds to read all messages") % t3a)

#######################################
#    SET UP FIGURE FOR EACH DOMAIN    #
#######################################

# Call the domain_latlons_proj function from rrfs_plot_utils
  xextent,yextent,offset,extent,myproj = rrfs_plot_utils.domain_latlons_proj(dom)

# Create figure and axes instances
  fig = plt.figure(figsize=(9,8))  
  ws, hs = rrfs_plot_utils.get_panel_spacing(dom, type='3panel')
  gs = GridSpec(9,8,wspace=ws,hspace=hs)

  # Define where Cartopy maps are located
  cartopy.config['data_dir'] = '/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python/NaturalEarth'
  back_res='50m'
  back_img='off'

  ax1 = fig.add_subplot(gs[0:4,0:4], projection=myproj)
  ax2 = fig.add_subplot(gs[0:4,4:], projection=myproj)
  ax1.set_extent(extent)
  ax2.set_extent(extent)
  axes = [ax1, ax2]

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

  # Map/figure has been set up here, save axes instances for use again later
  keep_ax_lst_1 = ax1.get_children()[:]
  keep_ax_lst_2 = ax2.get_children()[:]

  # Split plots into 2 sets with multiprocessing
  sets = [1,2]
  pool2 = MyPool(len(sets))
  pool2.map(plot_sets,sets)

################################################################################

def plot_sets(set):
# Add print to ensure dom is being passed in
  print(('plot_sets dom variable '+dom))

  if set == 1:
    plot_set_1()
  elif set == 2:
    plot_set_2()

################################################################################

def plot_set_1():

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
  if dom == 'conus':
    skip = 25
    skip2 = 100
  elif dom == 'southeast' or dom == 'alaska':
    skip = 10
    skip2 =40
  elif dom == 'hawaii' or dom == 'puerto_rico':
    skip = 2
    skip2 =20
  elif dom == 'colorado' or dom == 'la_vegas' or dom =='mid_atlantic' or dom == 'south_florida':
    skip = 4
    skip2 =18
  elif dom == 'boston_nyc':
    skip = 4  
    skip2 = 15
  elif dom == 'seattle_portland':
    skip = 3 
    skip2 = 13
  elif dom == 'sf_bay_area':
    skip = 1
    skip2 = 4
  else:
    skip = 7  
    skip2 = 30
  barblength = 3.5
  window1 = 100
  window2 = 550

  units = 'mb'
  if dom == 'alaska' or dom == 'hawaii' or dom == 'puerto_rico':
    clevs = [976,980,984,988,992,996,1000,1004,1008,1012,1016,1020,1024,1028,1032,1036,1040]
  else:
    clevs = [976,980,984,988,992,996,1000,1004,1008,1012,1016,1020,1024,1028,1032,1036,1040,1044,1048,1052]
  cm = plt.cm.Spectral_r
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs1_a = ax1.pcolormesh(lon_shift,lat_shift,slp_1,transform=transform,cmap=cm,norm=norm)  
  cbar1 = fig.colorbar(cs1_a,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  cs1_b = ax1.contour(lon_shift,lat_shift,slp_1,np.arange(940,1060,4),colors='black',linewidths=0.1,transform=transform)
#  plt.clabel(cs1_b,np.arange(940,1060,4),inline=1,fmt='%d',fontsize=6)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_1[::skip,::skip],vwind_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)

  rrfs_plot_utils.plt_highs_and_lows(lon_shift,lat_shift,slp_1,xmin,xmax,ymin,ymax,offset,ax1,transform,mode='reflect',window=window1)

  ax1.text(.5,1.03,'Parent NAM SLP ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs2_a = ax2.pcolormesh(lon2_shift,lat2_shift,slp_2,transform=transform,cmap=cm,norm=norm)  
  cbar2 = fig.colorbar(cs2_a,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  cs2_b = ax2.contour(lon2_shift,lat2_shift,slp_2,np.arange(940,1060,4),colors='black',linewidths=0.1,transform=transform)
#  plt.clabel(cs2_b,np.arange(940,1060,4),inline=1,fmt='%d',fontsize=6)
  ax2.barbs(lon2_shift[::skip2,::skip2],lat2_shift[::skip2,::skip2],uwind_2[::skip2,::skip2],vwind_2[::skip2,::skip2],length=barblength,linewidth=0.5,color='black',transform=transform)

  rrfs_plot_utils.plt_highs_and_lows(lon2_shift,lat2_shift,slp_2,xmin,xmax,ymin,ymax,offset,ax2,transform,mode='reflect',window=window2)

  ax2.text(.5,1.03,'RRFS SLP ('+units+') and 10-m Winds (kts) \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save('compareslp_'+dom+'_f'+fhour)
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
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)

  units = '\xb0''F'
  if dom == 'alaska':
    clevs = np.linspace(-46,98,25)
    clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  elif dom == 'hawaii' or dom == 'puerto_rico':
    clevs = np.linspace(18,99,28)
    clevsdif = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
  else:
    clevs = np.linspace(-16,134,26)
    clevsdif = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12]
  cm = rrfs_plot_utils.cmap_t2m()
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,tmp2m_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'Parent NAM 2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,tmp2m_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('white')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFS 2-m Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))       
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save('compare2mt_'+dom+'_f'+fhour)
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
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,tmpsfc_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'Parent NAM Surface Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,tmpsfc_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('white')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFS Surface Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save('comparetsfc_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot tsfc for: '+dom) % t3)

# #################################
#   # Plot 2-m Dew Point
# #################################
  if dom != "hawaii" and dom != "puerto_rico":
    t1 = time.perf_counter()
    print(('Working on 2mdew for '+dom))
 
   # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)

    units = '\xb0''F'
    if dom == 'alaska':
      clevs = np.linspace(-25,65,31)
    else:
      clevs = np.linspace(-10,80,19)
    cm = rrfs_plot_utils.cmap_q2m()
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,dew2m_1,transform=transform,cmap=cm,norm=norm)
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'Parent NAM 2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,dew2m_2,transform=transform,cmap=cm,norm=norm)
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,extend='both')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFS 2-m Dew Point Temperature ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
 
    rrfs_plot_utils.convert_and_save('compare2mdew_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot 2mdew for: '+dom) % t3)
  else:
    print (f"no DPT plot is available for {dom}")
 
 #################################
   # Plot 10-m WSPD
 #################################
  t1 = time.perf_counter()
  print(('Working on 10mwspd for '+dom))
 
  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
 
  # Wind barb density settings
  if dom == 'conus':
      skip = 80
      skip2 = 80
  elif dom == 'alaska':
      skip = 20
      skip2 = 40
  elif dom == 'southeast':
      skip = 18
      skip2 = 35
  elif dom == 'colorado' or dom == 'la_vegas' or dom =='mid_atlantic' or dom == 'south_florida':
      skip = 6
      skip2 = 12
  elif dom == 'puerto_rico':
      skip = 8
      skip2 = 15
  elif dom == 'boston_nyc' or dom == 'hawaii':
      skip = 5
      skip2 = 10
  elif dom == 'seattle_portland':
      skip = 5
      skip2 = 9
  elif dom == 'sf_bay_area':
      skip = 2
      skip2 = 3
  else:
      skip = 10
      skip2 = 20
  barblength = 3.5
 
  units = 'kts'
  clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
  colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
 
  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,wspd10m_1,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=6)
  ax1.barbs(lon_shift[::skip,::skip],lat_shift[::skip,::skip],uwind_1[::skip,::skip],vwind_1[::skip,::skip],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax1.text(.5,1.03,'Parent NAM 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
     
  cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,wspd10m_2,transform=transform,cmap=cm,vmin=5,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=6)
  ax2.barbs(lon2_shift[::skip2,::skip2],lat2_shift[::skip2,::skip2],uwind_2[::skip2,::skip2],vwind_2[::skip2,::skip2],length=barblength,linewidth=0.5,color='black',transform=transform)
  ax2.text(.5,1.03,'RRFS 10-m Winds ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
 
  rrfs_plot_utils.convert_and_save('compare10mwind_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot 10mwspd for: '+dom) % t3)
 
# #################################
#   # Plot surface wind gust
# #################################
  if dom != "hawaii":
    t1 = time.perf_counter()
    print(('Working on surface wind gust for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)

    units = 'kts'
    clevs = [5,10,15,20,25,30,35,40,45,50,55,60]
    colorlist = ['turquoise','dodgerblue','blue','#FFF68F','#E3CF57','peru','brown','crimson','red','fuchsia','DarkViolet']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
 
    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,gust_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_under('white',alpha=0.)
    cs_1.cmap.set_over('black')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.05,'Parent NAM Surface Wind Gust ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,gust_2,transform=transform,cmap=cm,norm=norm)
    cs_2.cmap.set_under('white',alpha=0.)
    cs_2.cmap.set_over('black')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.05,'RRFS Surface Wind Gust ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.convert_and_save('comparegust_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot surface wind gust for: '+dom) % t3)
  else:
    print (f"no GUST data is available for {dom}")

# #################################
#   # Plot Most Unstable CAPE/CIN
# #################################
  t1 = time.perf_counter()
  print(('Working on mucapecin for '+dom))
 
  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
 
  units = 'J/kg'
  clevs = [100,250,500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
  clevs2 = [-2000,-500,-250,-100,-25]
  colorlist = ['blue','dodgerblue','cyan','mediumspringgreen','#FAFAD2','#EEEE00','#EEC900','darkorange','crimson','darkred','darkviolet']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,mucape_1,transform=transform,cmap=cm,vmin=100,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=4)
  ax1.text(.5,1.05,'Parent NAM Most Unstable CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
  cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,mucape_2,transform=transform,cmap=cm,vmin=100,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=4)
  ax2.text(.5,1.05,'RRFS Most Unstable CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save('comparemucape_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot mucapecin for: '+dom) % t3)

# #################################
#   # Plot Surface-Based CAPE/CIN
# #################################
  t1 = time.perf_counter()
  print(('Working on sfcapecin for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
 
  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,cape_1,transform=transform,cmap=cm,vmin=100,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=4)
  ax1.text(.5,1.05,'Parent NAM Surface-Based CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
  cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,cape_2,transform=transform,cmap=cm,vmin=100,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=4)
  ax2.text(.5,1.05,'RRFS Surface-Based CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
  rrfs_plot_utils.convert_and_save('comparesfcape_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot sfcapecin for: '+dom) % t3)
# 
# #################################
#   # Plot Mixed Layer CAPE/CIN
# #################################
  t1 = time.perf_counter()
  print(('Working on mlcapecin for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,mlcape_1,transform=transform,cmap=cm,vmin=100,norm=norm)
  cs_1.cmap.set_under('white',alpha=0.)
  cs_1.cmap.set_over('black')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=4)
  ax1.text(.5,1.05,'Parent NAM Mixed Layer CAPE ('+units+') \n  initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
  cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,mlcape_2,transform=transform,cmap=cm,vmin=100,norm=norm)
  cs_2.cmap.set_under('white',alpha=0.)
  cs_2.cmap.set_over('black')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=4)
#  cs_2b = ax2.contourf(lon_shift,lat_shift,mlcin_2,clevs2,colors='none',hatches=['**','++','////','..'],transform=transform)
  ax2.text(.5,1.05,'RRFS Mixed Layer CAPE ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
  rrfs_plot_utils.convert_and_save('comparemlcape_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot mlcapecin for: '+dom) % t3)
 
# #################################
#   # Plot Surface Visibility
# #################################
  t1 = time.perf_counter()
  print(('Working on Surface Visibility for '+dom))

  # Clear off old plottables but keep all the map info
  cbar1.remove()
  cbar2.remove()
  rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
  rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
 
  units = 'miles'
  clevs = [0.25,0.5,1,2,3,4,5,10]
  colorlist = ['salmon','goldenrod','#EEEE00','palegreen','darkturquoise','blue','mediumpurple']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,vis_1,transform=transform,cmap=cm,vmax=10,norm=norm)
  cs_1.cmap.set_under('firebrick')
  cs_1.cmap.set_over('white',alpha=0.)
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='min')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=6)
  ax1.text(.5,1.03,'Parent NAM Surface Visibility ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
  cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,vis_2,transform=transform,cmap=cm,vmax=10,norm=norm)
  cs_2.cmap.set_under('firebrick')
  cs_2.cmap.set_over('white',alpha=0.)
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='min')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.set_xticklabels(clevs)
  cbar2.ax.tick_params(labelsize=6)
  ax2.text(.5,1.03,'RRFS Surface Visibility ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  rrfs_plot_utils.convert_and_save('comparevis_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot Surface Visibility for: '+dom) % t3)

# ######################################################
 
def plot_set_2():
 
  t1dom = time.perf_counter()
  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))
 
# #################################
#   # Plot PW
# #################################
  t1 = time.perf_counter()
  print(('Working on PW for '+dom))

  units = 'in'
  clevs = [0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25]
  colorlist = ['lightsalmon','khaki','palegreen','cyan','turquoise','cornflowerblue','mediumslateblue','darkorchid','deeppink']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,pw_1,transform=transform,cmap=cm,norm=norm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('hotpink')
  cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'Parent NAM Precipitable Water ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,pw_2,transform=transform,cmap=cm,norm=norm)
  cs_2.cmap.set_under('white')
  cs_2.cmap.set_over('hotpink')
  cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='both')
  cbar2.set_label(units,fontsize=6)
  cbar2.ax.tick_params(labelsize=5)
  ax2.text(.5,1.03,'RRFS Precipitable Water ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
  rrfs_plot_utils.convert_and_save('comparepw_'+dom+'_f'+fhour)
  t2 = time.perf_counter()
  t3 = round(t2-t1, 3)
  print(('%.3f seconds to plot PW for: '+dom) % t3)

# #################################
#   # Plot snow depth
# #################################
  if (fhr > 0):
    t1 = time.perf_counter()
    print(('Working on snow depth for '+dom))

    # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
 
    units = 'in'
    clevs = [0.5,1,2,3,4,6,8,12,18,24,30,36]
    colorlist = ['#adc4d9','#73bdff','#0f69db','#004da8','#002673','#ffff73','#ffaa00','#e64c00','#e60000','#730000','#e8beff']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
   
    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,snow_1,transform=transform,cmap=cm,vmin=0.5,norm=norm)
    cs_1.cmap.set_under('white',alpha=0.)
    cs_1.cmap.set_over('#CA7AF5')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels(clevs)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'Parent NAM Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,snow_2,transform=transform,cmap=cm,vmin=0.5,norm=norm)
    cs_2.cmap.set_under('white',alpha=0.)
    cs_2.cmap.set_over('#CA7AF5')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.set_xticklabels(clevs)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFS Snow Depth ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.convert_and_save('comparesnow_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot snow depth for: '+dom) % t3)


# #################################
#   # Plot total precipitation
# #################################
  if fhr > 0 and fhr % 12 == 0: 
    t1 = time.perf_counter()
    print(('Working on qpf for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
 
    units = 'in'
    clevs = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,7,10,15,20]
    colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','deepskyblue','cyan','mediumpurple','mediumorchid','darkmagenta','darkred','crimson','orangered','darkorange','goldenrod','gold','yellow']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
   
    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,qpf_1,transform=transform,cmap=cm,vmin=0.01,norm=norm)
    cs_1.cmap.set_under('white',alpha=0.)
    cs_1.cmap.set_over('pink')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'Parent NAM' +fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,qpf_2,transform=transform,cmap=cm,vmin=0.01,norm=norm)
    cs_2.cmap.set_under('white',alpha=0.)
    cs_2.cmap.set_over('pink')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFS '+fhour+'-hr Accumulated Precipitation ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.convert_and_save('compareqpf_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot qpf for: '+dom) % t3)
  else:
    print (f"no qpf data is calculated for {dom} at {fhr}")


# #################################
#   # Plot Cloud Ceiling Height
# #################################
  if dom != "hawaii" and dom != "puerto_rico":
    t1 = time.perf_counter()
    print(('Working on Cloud Ceiling Height for '+dom))
    
    # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
    
    units = 'kft'
    clevs = [0,0.1,0.3,0.5,1,5,10,15,20,25,30,35,40]
    colorlist = ['firebrick','tomato','salmon','lightsalmon','goldenrod','khaki','gold','yellow','palegreen','mediumspringgreen','lime','limegreen']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,zceil_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_over('white')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.set_xticklabels(clevs)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'Parent NAM Cloud Ceiling Height ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,zceil_2,transform=transform,cmap=cm,norm=norm)
    cs_2.cmap.set_over('white')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,ticks=clevs,extend='max')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.set_xticklabels(clevs)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFS Cloud Ceiling Height ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
 
    rrfs_plot_utils.convert_and_save('comparezceil_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Cloud Ceiling Height for: '+dom) % t3)
  else:
    print (f"no HGT data at cload ceiling level is available for {dom}")

# #################################
#   # Plot PBL height
# #################################
  if dom != "hawaii" and dom != "puerto_rico" and dom != "alaska":
    t1 = time.perf_counter()
    print(('Working on PBL height for '+dom))
  
  # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
 
    units = 'm'
    clevs = [50,100,250,500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
    colorlist= ['gray','blue','dodgerblue','cyan','mediumspringgreen','#FAFAD2','#EEEE00','#EEC900','darkorange','crimson','darkred','darkviolet']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
# 
    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,hpbl_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_under('white')
    cs_1.cmap.set_over('black')
    cbar1 = fig.colorbar(cs_1,ax=ax1,ticks=clevs,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=4)
    ax1.text(.5,1.03,'Parent NAM PBL Height ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
    cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,hpbl_2,transform=transform,cmap=cm,norm=norm)
    cs_2.cmap.set_under('white')
    cs_2.cmap.set_over('black')
    cbar2 = fig.colorbar(cs_2,ax=ax2,ticks=clevs,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=4)
    ax2.text(.5,1.03,'RRFS PBL Height ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
    rrfs_plot_utils.convert_and_save('comparehpbl_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot PBL height for: '+dom) % t3)
  else:
    print (f"no PBL height data is available for {dom}")

# 
# #################################
#   # Plot 0-3 km Storm Relative Helicity
# #################################
  if dom != "hawaii" and dom != "puerto_rico" and dom != "alaska":
    t1 = time.perf_counter()
    print(('Working on 0-3 km SRH for '+dom))
 
  # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)

    units = 'm${^2}$ s$^{-2}$'
    clevs = [50,100,150,200,250,300,400,500,600,700,800]
    colorlist = ['mediumblue','dodgerblue','chartreuse','limegreen','darkgreen','#EEEE00','orange','orangered','firebrick','darkmagenta']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,hel3km_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_under('white')
    cs_1.cmap.set_over('black')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'Parent NAM 0-3 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
    cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,hel3km_2,transform=transform,cmap=cm,norm=norm)
    cs_2.cmap.set_under('white')
    cs_2.cmap.set_over('black')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFS 0-3 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
    rrfs_plot_utils.convert_and_save('comparehel3km_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot 0-3 km SRH for: '+dom) % t3)
  else:
    print (f"no HLCY data at cload ceiling level is available for {dom}")

# #################################
#   # Plot 0-1 km Storm Relative Helicity
# #################################
  if dom != "hawaii" and dom != "puerto_rico" and dom != "alaska":
    t1 = time.perf_counter()
    print(('Working on 0-1 km SRH for '+dom))

  # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,hel1km_1,transform=transform,cmap=cm,norm=norm)
    cs_1.cmap.set_under('white')
    cs_1.cmap.set_over('black')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'Parent NAM 0-1 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,hel1km_2,transform=transform,cmap=cm,norm=norm)
    cs_2.cmap.set_under('white')
    cs_2.cmap.set_over('black')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='both')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFS 0-1 km Storm Relative Helicity ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
# 
    rrfs_plot_utils.convert_and_save('comparehel1km_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot 0-1 km SRH for: '+dom) % t3)
  else:
    print (f"no Helicity data is available for {dom}")

# 

#################################
# Plot Echo Top Height
#################################
  if dom != "puerto_rico":
    t1 = time.perf_counter()
    print(('Working on Echo Top Height for '+dom))

# Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)

    units = 'kft'
    clevs = [1,5,10,15,20,25,30,35,40]
    colorlist = ['firebrick','tomato','lightsalmon','goldenrod','#EEEE00','palegreen','mediumspringgreen','limegreen']
    cm = matplotlib.colors.ListedColormap(colorlist)
    norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

    cs_1 = ax1.pcolormesh(lon_shift,lat_shift,retop_1,transform=transform,cmap=cm,vmin=1,norm=norm)
    cs_1.cmap.set_under('white',alpha=0.)
    cs_1.cmap.set_over('darkgreen')
    cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
    cbar1.set_label(units,fontsize=6)
    cbar1.ax.tick_params(labelsize=6)
    ax1.text(.5,1.03,'Parent NAM Echo Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,retop_2,transform=transform,cmap=cm,vmin=1,norm=norm)
    cs_2.cmap.set_under('white',alpha=0.)
    cs_2.cmap.set_over('darkgreen')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.8,extend='max')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFS Echo Top Height ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.convert_and_save('compareretop'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Echo Top Height for: '+dom) % t3)
  else:
    print (f"no RETOP data is available for {dom}")


# #################################
#   # Plot Precipitation Type
# #################################
  if dom != "puerto_rico":
    t1 = time.perf_counter()
    print(('Working on Precipitation Type for forecast hour '+fhour))
# 
  # Clear off old plottables but keep all the map info
    cbar1.remove()
    cbar2.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
# 
    clevs = [0,20,40,100]
    snowhex=["#64B3E8", "#3E7CC6", "#1945A4"]
    rainhex=["#4EEE94", "#43CD80", "#2E8B57"]
    sleethex=["#947EEC", "#6B47AB", "#42106A"]
    freezehex=["#E65956", "#D93B3A", "#CC1E1E"]
    mixhex=["#E75FD5", "#C33BA2", "#A01870"]

    csrain_1 = ax1.contourf(lon_shift,lat_shift,rain1,clevs,colors=rainhex,transform=transform)
    csmix_1 = ax1.contourf(lon_shift,lat_shift,mix1,clevs,colors=mixhex,transform=transform)
    cssnow_1 = ax1.contourf(lon_shift,lat_shift,sn1,clevs,colors=snowhex,transform=transform)
    cssleet_1 = ax1.contourf(lon_shift,lat_shift,pl1,clevs,colors=sleethex,transform=transform)
    csfrzra_1 = ax1.contourf(lon_shift,lat_shift,fr1,clevs,colors=freezehex,transform=transform)
    ax1.text(.5,1.03,'Parent NAM composite reflectivity by ptype \n initialized: '+itime +' valid: '+ vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
# 
    csrain_2 = ax2.contourf(lon2_shift,lat2_shift,rain2,clevs,colors=rainhex,transform=transform)
    csmix_2 = ax2.contourf(lon2_shift,lat2_shift,mix2,clevs,colors=mixhex,transform=transform)
    cssnow_2 = ax2.contourf(lon2_shift,lat2_shift,sn2,clevs,colors=snowhex,transform=transform)
    cssleet_2 = ax2.contourf(lon2_shift,lat2_shift,pl2,clevs,colors=sleethex,transform=transform)
    csfrzra_2 = ax2.contourf(lon2_shift,lat2_shift,fr2,clevs,colors=freezehex,transform=transform)
    ax2.text(.5,1.03,'RRFS composite reflectivity by ptype \n initialized: '+itime +' valid: '+ vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    caxrain=fig.add_axes([.09,.52,.1,.03])
    cbrain=fig.colorbar(csrain_1,cax=caxrain,ticks=clevs,orientation='horizontal',extend='max')
    cbrain.set_label('rain',fontsize=7)
    cbrain.ax.tick_params(labelsize=6)
    cbrain.ax.set_xticklabels(['light','','','heavy'])

    caxsnow=fig.add_axes([.27,.52,.1,.03])
    cbsnow=fig.colorbar(cssnow_1,cax=caxsnow,ticks=clevs,orientation='horizontal',extend='max')
    cbsnow.set_label('snow',fontsize=7)
    cbsnow.ax.tick_params(labelsize=6)
    cbsnow.ax.set_xticklabels(['light','','','heavy'])

    caxsleet=fig.add_axes([.45,.52,.1,.03])
    cbsleet=fig.colorbar(cssleet_1,cax=caxsleet,ticks=clevs,orientation='horizontal',extend='max')
    cbsleet.set_label('sleet',fontsize=7)
    cbsleet.ax.tick_params(labelsize=6)
    cbsleet.ax.set_xticklabels(['light','','','heavy'])

    caxfrzra=fig.add_axes([.63,.52,.1,.03])
    cbfrzra=fig.colorbar(csfrzra_1,cax=caxfrzra,ticks=clevs,orientation='horizontal',extend='max')
    cbfrzra.set_label('freezing rain',fontsize=7)
    cbfrzra.ax.tick_params(labelsize=6)
    cbfrzra.ax.set_xticklabels(['light','','','heavy'])

    caxmix=fig.add_axes([.81,.52,.1,.03])
    cbmix=fig.colorbar(csmix_1,cax=caxmix,ticks=clevs,orientation='horizontal',extend='max')
    cbmix.set_label('mix',fontsize=7)
    cbmix.ax.tick_params(labelsize=6)
    cbmix.ax.set_xticklabels(['light','','','heavy'])

    rrfs_plot_utils.convert_and_save('compareptype_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot Precipitation Type for: '+dom) % t3)
  else:
    print (f"no precipitation type data is available for {dom}")
# 
# #################################
#   # Plot 1-km reflectivity
# #################################
  if dom != "hawaii" and dom != "puerto_rico" and dom != "alaska":
    t1 = time.perf_counter()
    print(('Working on 1-km reflectivity for '+dom))

  # Clear off old plottables but keep all the map info
    cbrain.remove()
    cbsnow.remove()
    cbsleet.remove()
    cbfrzra.remove()
    cbmix.remove()
    rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
    rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)

    units = 'dBZ'
    clevs = np.linspace(5,70,14)
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
    ax1.text(.5,1.03,'Parent NAM 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
 
    cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,ref1km_2,transform=transform,cmap=cm,vmin=5,norm=norm)
    cs_2.cmap.set_under('white',alpha=0.)
    cs_2.cmap.set_over('black')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='max')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFS 1-km Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    rrfs_plot_utils.convert_and_save('compareref1km_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot 1-km reflectivity for: '+dom) % t3)
  else:
    print (f"no reflectivity data is available for {dom}")
# 
# #################################
#   # Plot composite reflectivity
# #################################
  if dom != "puerto_rico":
    t1 = time.perf_counter()
    print(('Working on composite reflectivity for '+dom))

  # Clear off old plottables but keep all the map info
    if dom == "alaska" or dom == "hawaii":
      cbrain.remove()
      cbsnow.remove()
      cbsleet.remove()
      cbfrzra.remove()
      cbmix.remove()
      rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
      rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)
                             
    else:
      cbar1.remove()
      cbar2.remove()
      rrfs_plot_utils.clear_plotables(ax1,keep_ax_lst_1,fig)
      rrfs_plot_utils.clear_plotables(ax2,keep_ax_lst_2,fig)

    units = 'dBZ'
    clevs = np.linspace(5,70,14)
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
    ax1.text(.5,1.03,'Parent NAM Composite Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

    cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,refc_2,transform=transform,cmap=cm,vmin=5,norm=norm)
    cs_2.cmap.set_under('white',alpha=0.)
    cs_2.cmap.set_over('black')
    cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=1.0,ticks=clevs,extend='max')
    cbar2.set_label(units,fontsize=6)
    cbar2.ax.tick_params(labelsize=6)
    ax2.text(.5,1.03,'RRFS Composite Reflectivity ('+units+') \n initialized: '+itime+' valid: '+vtime_end + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax2.transAxes,bbox=dict(facecolor='white',color='white',alpha=0.85,boxstyle='square,pad=0.2'))
    ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
    rrfs_plot_utils.convert_and_save('comparerefc_'+dom+'_f'+fhour)
    t2 = time.perf_counter()
    t3 = round(t2-t1, 3)
    print(('%.3f seconds to plot composite reflectivity for: '+dom) % t3)
  else:
    print (f"no RFC data is available for {dom}")
# ######################################################
# 
  t3dom = round(t2-t1dom, 3)
  print(("%.3f seconds to plot all set 2 variables for: "+dom) % t3dom)
  plt.clf()

######################################################

main()
