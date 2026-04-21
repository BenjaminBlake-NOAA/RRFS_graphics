#!/bin/bash

set -x

date

cd /lfs/h2/emc/stmp/${USER}/4panel_qpf/${CDATE}

# Retrieve main24.php to update cycle dates
scp bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/main24.php .

DATE=$(sed -n "226p" main24.php | cut -c 15-24)
DATEm1=$(sed -n "226p" main24.php | cut -c 28-37)
DATEm2=$(sed -n "226p" main24.php | cut -c 41-50)
DATEm3=$(sed -n "226p" main24.php | cut -c 54-63)
DATEm4=$(sed -n "226p" main24.php | cut -c 67-76)
DATEm5=$(sed -n "226p" main24.php | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

sed '226s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main24.php > tmpfile ; mv tmpfile main24.php

scp main24.php bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs


# Move images into correct directories on emcrzdm
# remove images from cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/verif/images/*.gif"

# move cycm4 images to cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/verif/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/verif/images/"

# move cycm3 images to cycm4 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/verif/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/verif/images/"

# move cycm2 images to cycm3 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/verif/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/verif/images/"

# move cycm1 images to cycm2 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/verif/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/verif/images/"

# move cyc images to cycm1 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/verif/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/verif/images/"


# Copy images from WCOSS to emcrzdm
rsync -t *.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/verif/images/

date

exit
