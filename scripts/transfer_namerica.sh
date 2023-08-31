#!/bin/bash

set -x

date

cd /lfs/h2/emc/stmp/${USER}/namerica/${PDY}/${cyc}

# Retrieve main17.php to update cycle dates
scp bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/main17.php .

DATE=$(sed -n "286p" main17.php | cut -c 15-24)
DATEm1=$(sed -n "286p" main17.php | cut -c 28-37)
DATEm2=$(sed -n "286p" main17.php | cut -c 41-50)
DATEm3=$(sed -n "286p" main17.php | cut -c 54-63)
DATEm4=$(sed -n "286p" main17.php | cut -c 67-76)
DATEm5=$(sed -n "286p" main17.php | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

sed '286s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main17.php > tmpfile ; mv tmpfile main17.php

scp main17.php bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs


# Move images into correct directories on emcrzdm
# remove images from cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/namerica/images/*.png"

# move cycm4 images to cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/namerica/images/*f0*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/namerica/images/*f1*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/namerica/images/*f2*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/namerica/images/*f3*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/namerica/images/*f4*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/namerica/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/namerica/images/"

# move cycm3 images to cycm4 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/namerica/images/*f0*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/namerica/images/*f1*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/namerica/images/*f2*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/namerica/images/*f3*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/namerica/images/*f4*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/namerica/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/namerica/images/"

# move cycm2 images to cycm3 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/namerica/images/*f0*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/namerica/images/*f1*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/namerica/images/*f2*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/namerica/images/*f3*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/namerica/images/*f4*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/namerica/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/namerica/images/"

# move cycm1 images to cycm2 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/namerica/images/*f0*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/namerica/images/*f1*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/namerica/images/*f2*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/namerica/images/*f3*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/namerica/images/*f4*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/namerica/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/namerica/images/"

# move cyc images to cycm1 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/namerica/images/*f0*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/namerica/images/*f1*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/namerica/images/*f2*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/namerica/images/*f3*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/namerica/images/*f4*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/namerica/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/namerica/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/namerica/images/"


# Copy images from WCOSS to emcrzdm
rsync -t *namerica*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/namerica/images/
rsync -t *caribbean*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/namerica/images/

date

exit
