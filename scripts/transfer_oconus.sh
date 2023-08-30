#!/bin/bash

set -x

date

cd /lfs/h2/emc/stmp/${USER}/3panel_nam/${PDY}/${cyc}
# Copy images from WCOSS to emcrzdm
rsync -t *alaska*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *hawaii*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *puerto_rico*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/

cd /lfs/h2/emc/stmp/${USER}/3panel_hrrr/${PDY}/${cyc}
# Copy images from WCOSS to emcrzdm
rsync -t *alaska*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/


date

exit
