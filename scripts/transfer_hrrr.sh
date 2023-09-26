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
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hrrr/images/*.gif"

# move cycm4 images to cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/*f0*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/*f1*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/*f2*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/*f3*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/*f4*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hrrr/images/"

# move cycm3 images to cycm4 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/*f0*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/*f1*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/*f2*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/*f3*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/*f4*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hrrr/images/"

# move cycm2 images to cycm3 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/*f0*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/*f1*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/*f2*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/*f3*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/*f4*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hrrr/images/"

# move cycm1 images to cycm2 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/*f0*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/*f1*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/*f2*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/*f3*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/*f4*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hrrr/images/"

# move cyc images to cycm1 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/*f0*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/*f1*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/*f2*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/*f3*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/*f4*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hrrr/images/"


# Copy images from WCOSS to emcrzdm
rsync -t *conus*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *alaska*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *boston_nyc*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *central*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *colorado*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *la_vegas*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *mid_atlantic*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *north_central*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *northeast*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *northwest*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *ohio_valley*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *south_central*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *southeast*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *south_florida*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *sf_bay_area*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *seattle_portland*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *southwest*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/
rsync -t *upper_midwest*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/

rsync -t /lfs/h2/emc/stmp/${USER}/uhtracks_hrrr/${PDY}/${cyc}/*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hrrr/images/

date

exit
