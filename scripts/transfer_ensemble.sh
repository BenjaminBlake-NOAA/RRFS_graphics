#!/bin/bash

set -x

date

cd /lfs/h2/emc/stmp/${USER}/ensemble/${PDY}/${cyc}

# Retrieve main18.php to update cycle dates
scp bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/main18.php .

DATE=$(sed -n "286p" main18.php | cut -c 15-24)
DATEm1=$(sed -n "286p" main18.php | cut -c 28-37)
DATEm2=$(sed -n "286p" main18.php | cut -c 41-50)
DATEm3=$(sed -n "286p" main18.php | cut -c 54-63)
DATEm4=$(sed -n "286p" main18.php | cut -c 67-76)
DATEm5=$(sed -n "286p" main18.php | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

sed '286s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main18.php > tmpfile ; mv tmpfile main18.php

scp main18.php bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs


# Move images into correct directories on emcrzdm
# remove images from cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/ensemble/images/*.gif"

# move cycm4 images to cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/ensemble/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/ensemble/images/"

# move cycm3 images to cycm4 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/ensemble/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/ensemble/images/"

# move cycm2 images to cycm3 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/ensemble/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/ensemble/images/"

# move cycm1 images to cycm2 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/ensemble/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/ensemble/images/"

# move cyc images to cycm1 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/ensemble/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/ensemble/images/"


# Copy images from WCOSS to emcrzdm
rsync -t *members*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/ensemble/images/

date

exit
