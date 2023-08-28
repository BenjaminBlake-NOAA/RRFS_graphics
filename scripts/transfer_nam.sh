#!/bin/bash

set -x

date

cd /lfs/h2/emc/stmp/${USER}/3panel_nam/${PDY}/${cyc}

# Retrieve main2.php to update cycle dates
scp bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/main2.php .

DATE=$(sed -n "286p" main2.php | cut -c 15-24)
DATEm1=$(sed -n "286p" main2.php | cut -c 28-37)
DATEm2=$(sed -n "286p" main2.php | cut -c 41-50)
DATEm3=$(sed -n "286p" main2.php | cut -c 54-63)
DATEm4=$(sed -n "286p" main2.php | cut -c 67-76)
DATEm5=$(sed -n "286p" main2.php | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

sed '286s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main2.php > tmpfile ; mv tmpfile main2.php

scp main2.php bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs


# Move images into correct directories on emcrzdm
# remove images from cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/nam/images/*.png"

# move cycm4 images to cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/nam/images/*f0*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/nam/images/*f1*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/nam/images/*f2*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/nam/images/*f3*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/nam/images/*f4*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/nam/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/nam/images/"

# move cycm3 images to cycm4 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/nam/images/*f0*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/nam/images/*f1*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/nam/images/*f2*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/nam/images/*f3*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/nam/images/*f4*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/nam/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/nam/images/"

# move cycm2 images to cycm3 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/nam/images/*f0*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/nam/images/*f1*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/nam/images/*f2*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/nam/images/*f3*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/nam/images/*f4*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/nam/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/nam/images/"

# move cycm1 images to cycm2 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/nam/images/*f0*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/nam/images/*f1*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/nam/images/*f2*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/nam/images/*f3*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/nam/images/*f4*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/nam/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/nam/images/"

# move cyc images to cycm1 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/*f0*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/*f1*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/*f2*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/*f3*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/*f4*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/nam/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/nam/images/"


# Copy images from WCOSS to emcrzdm
rsync -t *conus*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *alaska*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *hawaii*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *puerto_rico*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *boston_nyc*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *central*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *colorado*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *la_vegas*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *mid_atlantic*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *north_central*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *northeast*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *northwest*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *ohio_valley*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *south_central*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *southeast*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *sf_bay_area*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *seattle_portland*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *southwest*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/
rsync -t *upper_midwest*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/

rsync -t /lfs/h2/emc/stmp/${USER}/uhtracks_nam/${PDY}/${cyc}/*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/nam/images/

date

exit
