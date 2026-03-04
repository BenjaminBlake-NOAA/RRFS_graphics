#!/bin/bash

set -x

date

cd /lfs/h2/emc/stmp/${USER}/2panel_hiresw/${PDY}/${cyc}

# Use the a version for 00/12Z cycles, the b version for 06/18Z cycles
if [[ "${cyc}" == "00" ]] || [ "${cyc}" == "12" ]; then
    rzdmdir='hireswa'
    rzdmphp='main4a.php'
    doms=("conus" "hawaii" "boston_nyc" "central" "colorado" "la_vegas" "mid_atlantic" "north_central" "northeast" "northwest" "ohio_valley" "south_central" "southeast" "south_florida" "sf_bay_area" "seattle_portland" "southwest" "upper_midwest")
elif [[ "${cyc}" == "06" ]] || [[ "${cyc}" == "18" ]]; then
    rzdmdir='hireswb'
    rzdmphp='main4b.php'
    doms=("alaska" "puerto_rico")
fi

# Retrieve main4.php to update cycle dates
scp bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/regional/restricted/rrfs/${rzdmphp} .

DATE=$(sed -n "307p" ${rzdmphp} | cut -c 15-24)
DATEm1=$(sed -n "307p" ${rzdmphp} | cut -c 28-37)
DATEm2=$(sed -n "307p" ${rzdmphp} | cut -c 41-50)
DATEm3=$(sed -n "307p" ${rzdmphp} | cut -c 54-63)
DATEm4=$(sed -n "307p" ${rzdmphp} | cut -c 67-76)
DATEm5=$(sed -n "307p" ${rzdmphp} | cut -c 80-89)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5


sed '307s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' ${rzdmphp} > tmpfile ; mv tmpfile ${rzdmphp}

scp ${rzdmphp} bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/regional/restricted/rrfs


# Move images into correct directories on emcrzdm
# remove images from cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm5/${rzdmdir}/images/*f0*.gif"
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm5/${rzdmdir}/images/*f1*.gif"
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm5/${rzdmdir}/images/*f2*.gif"
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm5/${rzdmdir}/images/*f3*.gif"
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm5/${rzdmdir}/images/*f4*.gif"
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm5/${rzdmdir}/images/*.gif"

# move cycm4 images to cycm5 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm4/${rzdmdir}/images/*f0*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm5/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm4/${rzdmdir}/images/*f1*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm5/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm4/${rzdmdir}/images/*f2*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm5/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm4/${rzdmdir}/images/*f3*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm5/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm4/${rzdmdir}/images/*f4*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm5/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm4/${rzdmdir}/images/*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm5/${rzdmdir}/images/"

# move cycm3 images to cycm4 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm3/${rzdmdir}/images/*f0*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm4/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm3/${rzdmdir}/images/*f1*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm4/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm3/${rzdmdir}/images/*f2*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm4/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm3/${rzdmdir}/images/*f3*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm4/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm3/${rzdmdir}/images/*f4*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm4/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm3/${rzdmdir}/images/*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm4/${rzdmdir}/images/"

# move cycm2 images to cycm3 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm2/${rzdmdir}/images/*f0*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm3/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm2/${rzdmdir}/images/*f1*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm3/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm2/${rzdmdir}/images/*f2*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm3/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm2/${rzdmdir}/images/*f3*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm3/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm2/${rzdmdir}/images/*f4*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm3/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm2/${rzdmdir}/images/*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm3/${rzdmdir}/images/"

# move cycm1 images to cycm2 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm1/${rzdmdir}/images/*f0*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm2/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm1/${rzdmdir}/images/*f1*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm2/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm1/${rzdmdir}/images/*f2*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm2/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm1/${rzdmdir}/images/*f3*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm2/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm1/${rzdmdir}/images/*f4*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm2/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm1/${rzdmdir}/images/*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm2/${rzdmdir}/images/"

# move cyc images to cycm1 directory
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cyc/${rzdmdir}/images/*f0*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm1/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cyc/${rzdmdir}/images/*f1*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm1/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cyc/${rzdmdir}/images/*f2*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm1/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cyc/${rzdmdir}/images/*f3*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm1/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cyc/${rzdmdir}/images/*f4*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm1/${rzdmdir}/images/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/regional/restricted/rrfs/cyc/${rzdmdir}/images/*.gif /home/people/emc/www/htdocs/regional/restricted/rrfs/cycm1/${rzdmdir}/images/"


# Copy images from WCOSS to emcrzdm
for dom in "${doms[@]}"; do
  rsync -t *${dom}*.gif bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/regional/restricted/rrfs/cyc/${rzdmdir}/images/
done

date

exit
