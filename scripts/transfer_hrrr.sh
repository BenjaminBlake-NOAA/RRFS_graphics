#!/bin/bash

set -x

date

cd /lfs/h2/emc/stmp/${USER}/3panel_hrrr/${PDY}/${cyc}

# Retrieve main10.php to update cycle dates
scp bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/main10.php .

DATE=$(sed -n "286p" main10.php | cut -c 15-24)
DATEm1=$(sed -n "286p" main10.php | cut -c 28-37)
DATEm2=$(sed -n "286p" main10.php | cut -c 41-50)
DATEm3=$(sed -n "286p" main10.php | cut -c 54-63)
DATEm4=$(sed -n "286p" main10.php | cut -c 67-76)
DATEm5=$(sed -n "286p" main10.php | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

sed '286s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main10.php > tmpfile ; mv tmpfile main10.php

scp main10.php bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs


# Move images into correct directories on emcrzdm
# remove images from cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hrrr/images/*.png"

# move cycm4 images to cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/*f0*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/*f1*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/*f2*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/*f3*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/*f4*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hrrr/images/"

# move cycm3 images to cycm4 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/*f0*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/*f1*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/*f2*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/*f3*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/*f4*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/"

# move cycm2 images to cycm3 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/*f0*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/*f1*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/*f2*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/*f3*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/*f4*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/"

# move cycm1 images to cycm2 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/*f0*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/*f1*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/*f2*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/*f3*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/*f4*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/"

# move cyc images to cycm1 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/*f0*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/*f1*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/*f2*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/*f3*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/*f4*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/"


# Copy images from WCOSS to emcrzdm
rsync -t *conus*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *alaska*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *boston_nyc*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *central*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *colorado*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *la_vegas*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *mid_atlantic*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *north_central*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *northeast*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *northwest*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *ohio_valley*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *south_central*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *southeast*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *sf_bay_area*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *seattle_portland*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *southwest*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *upper_midwest*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/

rsync -t /lfs/h2/emc/stmp/${USER}/uhtracks_hrrr/${PDY}/${cyc}/*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/

date

exit
