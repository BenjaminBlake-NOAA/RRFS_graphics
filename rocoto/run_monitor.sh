#!/bin/bash

set -x

/u/benjamin.blake/.bashrc

rocotorun -v 10 -w /lfs/h2/emc/lam/noscrub/Benjamin.Blake/rrfs_graphics/rocoto/drive_monitor.xml -d /lfs/h2/emc/lam/noscrub/Benjamin.Blake/rrfs_graphics/rocoto/drive_monitor.db
