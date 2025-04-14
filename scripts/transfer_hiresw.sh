#!/bin/bash

set -x

date

cd /lfs/h2/emc/stmp/${USER}/2panel_hiresw/${PDY}/${cyc}

# Retrieve main4.php to update cycle dates
scp bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/main4.php .

DATE=$(sed -n "306p" main4.php | cut -c 15-24)
DATEm1=$(sed -n "306p" main4.php | cut -c 28-37)
DATEm2=$(sed -n "306p" main4.php | cut -c 41-50)
DATEm3=$(sed -n "306p" main4.php | cut -c 54-63)
DATEm4=$(sed -n "306p" main4.php | cut -c 67-76)
DATEm5=$(sed -n "306p" main4.php | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

sed '306s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main4.php > tmpfile ; mv tmpfile main4.php

scp main4.php bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs


# Move images into correct directories on emcrzdm
# remove images from cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hiresw/images/*f0*.gif"
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hiresw/images/*f1*.gif"
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hiresw/images/*f2*.gif"
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hiresw/images/*f3*.gif"
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hiresw/images/*f4*.gif"
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hiresw/images/*.gif"

# move cycm4 images to cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hiresw/images/*f0*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hiresw/images/*f1*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hiresw/images/*f2*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hiresw/images/*f3*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hiresw/images/*f4*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hiresw/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm5/hiresw/images/"

# move cycm3 images to cycm4 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hiresw/images/*f0*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hiresw/images/*f1*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hiresw/images/*f2*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hiresw/images/*f3*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hiresw/images/*f4*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hiresw/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm4/hiresw/images/"

# move cycm2 images to cycm3 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hiresw/images/*f0*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hiresw/images/*f1*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hiresw/images/*f2*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hiresw/images/*f3*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hiresw/images/*f4*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hiresw/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm3/hiresw/images/"

# move cycm1 images to cycm2 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hiresw/images/*f0*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hiresw/images/*f1*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hiresw/images/*f2*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hiresw/images/*f3*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hiresw/images/*f4*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hiresw/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm2/hiresw/images/"

# move cyc images to cycm1 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hiresw/images/*f0*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hiresw/images/*f1*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hiresw/images/*f2*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hiresw/images/*f3*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hiresw/images/*f4*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hiresw/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hiresw/images/*.gif /home/people/emc/www/htdocs/users/emc.campara/rrfs/cycm1/hiresw/images/"


# Copy images from WCOSS to emcrzdm
if [[ "${cyc}" == "06" ]] || [[ "${cyc}" == "18" ]]; then
    doms=("alaska" "puerto_rico")
else
    doms=("conus" "hawaii" "boston_nyc" "central" "colorado" "la_vegas" "mid_atlantic" "north_central" "northeast" "northwest" "ohio_valley" "south_central" "southeast" "south_florida" "sf_bay_area" "seattle_portland" "southwest" "upper_midwest")
fi

for dom in "${doms[@]}"; do
  rsync -t *${dom}*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/cyc/hiresw/images/
done

date

exit
