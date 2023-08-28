#!/bin/bash

set -x

module load python/3.8.6
module use /lfs/h1/mdl/nbm/save/apps/modulefiles
module load python-modules/3.8.6
module load proj/7.1.0
module load geos/3.8.1
module load libjpeg-turbo/2.1.0
export PYTHONPATH="${PYTHONPATH}:/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python:/lfs/h2/emc/lam/noscrub/Benjamin.Blake/PyGSI"

module use /apps/ops/test/nco/modulefiles
module load core/rocoto/1.3.5

rocotorun -v 10 -w /lfs/h2/emc/lam/noscrub/Benjamin.Blake/rrfs_graphics/rocoto/drive_plots.xml -d /lfs/h2/emc/lam/noscrub/Benjamin.Blake/rrfs_graphics/rocoto/drive_plots.db
