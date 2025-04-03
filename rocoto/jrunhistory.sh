#!/bin/bash
set -x

mkdir -p /lfs/h2/emc/stmp/Benjamin.Blake/testrunhist
cd /lfs/h2/emc/stmp/Benjamin.Blake/testrunhist

export model=runhistory
###export job=jrunhistory
export rhistlist=rrfs

export RSYNC_MAXTIME=30
export LOG_DAYS_KEEP=5
export WRITE_LOG_DIR=NO
export RSYNC_LOG_DIR=NO
export READ_LOG_DIR=NO
export CHECK_HPSS_IDX=YES
export DRY_RUN_ONLY=NO

module reset
module load intel/19.1.3.304
module load envvar/1.0
module load PrgEnv-intel/8.1.0
module load craype/2.7.10
module load cray-mpich/8.1.9
module load prod_util/2.0.13
module load prod_envir/2.0.6

export HOMErunhistory=/lfs/h2/emc/lam/noscrub/Benjamin.Blake/rrfs_graphics
export USHrunhistory=${HOMErunhistory}/ush
export PARMrunhistory=${HOMErunhistory}/parm

${HOMErunhistory}/rocoto/JRHIST
