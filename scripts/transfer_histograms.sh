#!/bin/bash

#USER=Benjamin.Blake
#CDATE=2019051700
#cyc=00

cd /lfs/h2/emc/stmp/${USER}/damonitor/${PDY}/${cyc}

# Retrieve main7.php to update cycle dates
scp bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/main7.php .

DATE=$(sed -n "190p" main7.php | cut -c 15-24)
DATEm1=$(sed -n "190p" main7.php | cut -c 28-37)
DATEm2=$(sed -n "190p" main7.php | cut -c 41-50)
DATEm3=$(sed -n "190p" main7.php | cut -c 54-63)
DATEm4=$(sed -n "190p" main7.php | cut -c 67-76)
DATEm5=$(sed -n "190p" main7.php | cut -c 80-89)
DATEm6=$(sed -n "190p" main7.php | cut -c 93-102)
DATEm7=$(sed -n "190p" main7.php | cut -c 106-115)
DATEm8=$(sed -n "190p" main7.php | cut -c 119-128)
DATEm9=$(sed -n "190p" main7.php | cut -c 132-141)
DATEm10=$(sed -n "190p" main7.php | cut -c 145-154)
DATEm11=$(sed -n "190p" main7.php | cut -c 158-167)
DATEm12=$(sed -n "190p" main7.php | cut -c 171-180)
DATEm13=$(sed -n "190p" main7.php | cut -c 184-193)
DATEm14=$(sed -n "190p" main7.php | cut -c 197-206)
DATEm15=$(sed -n "190p" main7.php | cut -c 210-219)
DATEm16=$(sed -n "190p" main7.php | cut -c 223-232)
DATEm17=$(sed -n "190p" main7.php | cut -c 236-245)
DATEm18=$(sed -n "190p" main7.php | cut -c 249-258)
DATEm19=$(sed -n "190p" main7.php | cut -c 262-271)
DATEm20=$(sed -n "190p" main7.php | cut -c 275-284)
DATEm21=$(sed -n "190p" main7.php | cut -c 288-297)
DATEm22=$(sed -n "190p" main7.php | cut -c 301-310)
DATEm23=$(sed -n "190p" main7.php | cut -c 314-323)
echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5
echo $DATEm6
echo $DATEm7
echo $DATEm8
echo $DATEm9
echo $DATEm10
echo $DATEm11
echo $DATEm12
echo $DATEm13
echo $DATEm14
echo $DATEm15
echo $DATEm16
echo $DATEm17
echo $DATEm18
echo $DATEm19
echo $DATEm20
echo $DATEm21
echo $DATEm22
echo $DATEm23

sed '190s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'","'${DATEm6}'","'${DATEm7}'","'${DATEm8}'","'${DATEm9}'","'${DATEm10}'","'${DATEm11}'","'${DATEm12}'","'${DATEm13}'","'${DATEm14}'","'${DATEm15}'","'${DATEm16}'","'${DATEm17}'","'${DATEm18}'","'${DATEm19}'","'${DATEm20}'","'${DATEm21}'","'${DATEm22}'","'${DATEm23}'"\]/var cyclist=\["'${CDATE}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'","'${DATEm6}'","'${DATEm7}'","'${DATEm8}'","'${DATEm9}'","'${DATEm10}'","'${DATEm11}'","'${DATEm12}'","'${DATEm13}'","'${DATEm14}'","'${DATEm15}'","'${DATEm16}'","'${DATEm17}'","'${DATEm18}'","'${DATEm19}'","'${DATEm20}'","'${DATEm21}'","'${DATEm22}'"\]/' main7.php > tmpfile ; mv tmpfile main7.php

scp main7.php bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs


# Move images into correct directories on emcrzdm
# remove images from cycm23 directory
ssh bblake@emcrzdm.ncep.noaa.gov "rm /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm23/*.png"

# move cycm10 images to cycm11 directory, cycm9 to cycm10, etc.
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/mnitor/cycm22/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm23/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm21/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm22/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm20/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm21/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm19/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm20/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm18/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm19/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm17/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm18/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm16/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm17/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm15/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm16/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm14/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm15/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm13/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm14/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm12/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm13/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm11/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm12/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm10/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm11/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm9/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm10/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm8/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm9/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm7/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm8/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm6/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm7/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm5/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm6/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm4/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm5/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm3/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm4/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm2/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm3/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm1/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm2/"
ssh bblake@emcrzdm.ncep.noaa.gov "mv /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cyc/*.png /home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cycm1/"


# Copy images from WCOSS to emcrzdm
rsync -t RRFS_A*.png bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/monitor/cyc/


exit
