#!/bin/bash

set -x

date

cd /lfs/h2/emc/stmp/${USER}/4panel_qpf/${PDY}/${cyc}

# Retrieve main3.php to update cycle dates
scp bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/main3.php .

DATE=$(sed -n "226p" main3.php | cut -c 15-24)
DATEm1=$(sed -n "226p" main3.php | cut -c 28-37)
DATEm2=$(sed -n "226p" main3.php | cut -c 41-50)
DATEm3=$(sed -n "226p" main3.php | cut -c 54-63)
DATEm4=$(sed -n "226p" main3.php | cut -c 67-76)
DATEm5=$(sed -n "226p" main3.php | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

sed '226s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main3.php > tmpfile ; mv tmpfile main3.php

scp main3.php bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs


# Move images into correct directories on emcrzdm
# remove images from cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/4panel/images/*.gif"

# move cycm4 images to cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/4panel/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/4panel/images/"

# move cycm3 images to cycm4 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/4panel/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/4panel/images/"

# move cycm2 images to cycm3 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/4panel/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/4panel/images/"

# move cycm1 images to cycm2 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/4panel/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/4panel/images/"

# move cyc images to cycm1 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/4panel/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/4panel/images/"


# Copy images from WCOSS to emcrzdm
# QPF
rsync -t *.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/4panel/images/

# REFC
cd /lfs/h2/emc/stmp/${USER}/4panel_refc/${PDY}/${cyc}
rsync -t *.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/4panel/images/

# SLP
cd /lfs/h2/emc/stmp/${USER}/4panel_slp/${PDY}/${cyc}
rsync -t *.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/4panel/images/

# 2-m T
cd /lfs/h2/emc/stmp/${USER}/4panel_tmp2m/${PDY}/${cyc}
rsync -t *.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/4panel/images/

# 2-m Dew Point
cd /lfs/h2/emc/stmp/${USER}/4panel_dpt2m/${PDY}/${cyc}
rsync -t *.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/4panel/images/

# 10-m Wind Speed
cd /lfs/h2/emc/stmp/${USER}/4panel_wind10m/${PDY}/${cyc}
rsync -t *.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/4panel/images/


date

exit
