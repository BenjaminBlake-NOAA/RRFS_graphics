#!/bin/usr/env python

import numpy as np
import os,sys,multiprocessing
import metplus_utils
import multiprocessing.pool

#-------------------------------------------------------#

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
vtime_start = metplus_utils.ndate(ymdh_model,int(fhr-24))
vtime_start = str(vtime_start[0:8])
vtime_end = ymd

# Define the directory paths to the output files
STAGE_DIR = '/lfs/h2/emc/stmp/'+os.environ['USER']+'/rrfs_verif'
PARM_DIR = os.path.join(os.environ['HOMEDIR'],'parm')
DCOMmrms = os.path.join(os.environ['DCOMmrms'],'upperair','mrms')

MRMS_DIR = os.path.join(STAGE_DIR,'mrms.'+ymd)

# Set up working directories
if not os.path.exists(os.path.join(MRMS_DIR, 'tmp')):
    if not os.path.exists(MRMS_DIR):
        os.makedirs(MRMS_DIR)
    os.makedirs(os.path.join(MRMS_DIR, 'tmp'))
    os.makedirs(os.path.join(MRMS_DIR, 'logs'))

# Specify plotting domains
#domains = ['conus','boston_nyc','central','colorado','la_vegas','mid_atlantic','north_central','northeast','northwest','ohio_valley','south_central','southeast','south_florida','sf_bay_area','seattle_portland','southwest','upper_midwest']
domains = ['conus','alaska']

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

# We sub-class multiprocessing.pool.Pool (proper class) 
# rather than multiprocessing.Pool (only a wrapper function)
class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)

#-------------------------------------------------------#

def main():

    for domain in domains:
        metplus_process(domain)
    #pool = MyPool(len(domains))
    #pool.map(metplus_process,domains)

def metplus_process(domain):

  global dom
  dom = domain
  print(('Working on '+dom))

  # Process MRMS files
  if dom in ['alaska']:
      metplus_utils.prep_mrms_radar(
          ymd, hour, DCOMmrms, MRMS_DIR, PARM_DIR, dom, 'G091'
      )
  else:
      metplus_utils.prep_mrms_radar(
          ymd, hour, DCOMmrms, MRMS_DIR, PARM_DIR, dom, 'G227'
      )

main()
