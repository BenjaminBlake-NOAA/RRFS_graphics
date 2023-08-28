#!/bin/bash

set -x

date

cd /lfs/h2/emc/stmp/${USER}/firewx/${PDY}/${cyc}

# Retrieve main11.php to update cycle dates
scp bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/main11.php .

DATE=$(sed -n "286p" main11.php | cut -c 15-24)
DATEm1=$(sed -n "286p" main11.php | cut -c 28-37)
DATEm2=$(sed -n "286p" main11.php | cut -c 41-50)
DATEm3=$(sed -n "286p" main11.php | cut -c 54-63)
DATEm4=$(sed -n "286p" main11.php | cut -c 67-76)
DATEm5=$(sed -n "286p" main11.php | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

sed '286s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main11.php > tmpfile ; mv tmpfile main11.php

scp main11.php bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs


# Move images into correct directories on emcrzdm
# remove images from cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/firewx/images/*.png"

# move cycm4 images to cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/firewx/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/firewx/images/"

# move cycm3 images to cycm4 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/firewx/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/firewx/images/"

# move cycm2 images to cycm3 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/firewx/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/firewx/images/"

# move cycm1 images to cycm2 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/firewx/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/firewx/images/"

# move cyc images to cycm1 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/firewx/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/firewx/images/"


# Copy images from WCOSS to emcrzdm
rsync -t *firewx*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/firewx/images/

date

exit
