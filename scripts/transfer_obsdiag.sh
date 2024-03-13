#!/bin/bash

set -x

date

#USER=Benjamin.Blake
#PDYm1=20230910
#yyyy=2023
#mm=09
#dd=10

cd /lfs/h2/emc/stmp/${USER}

# Retrieve main19.php to update cycle dates
scp bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs/main19.php .

# Read in PDY, yyyy, mm, dd from Rocoto xml

DATE=$(sed -n "197p" main19.php | cut -c 15-22)
YEAR=`echo $DATE | cut -c1-4`
MONTH=`echo $DATE | cut -c5-6`
DAY=`echo $DATE | cut -c7-8`

DATEm1=$(sed -n "197p" main19.php | cut -c 26-33)
YEARm1=`echo $DATEm1 | cut -c1-4`
MONTHm1=`echo $DATEm1 | cut -c5-6`
DAYm1=`echo $DATEm1 | cut -c7-8`

DATEm2=$(sed -n "197p" main19.php | cut -c 37-44)
YEARm2=`echo $DATEm2 | cut -c1-4`
MONTHm2=`echo $DATEm2 | cut -c5-6`
DAYm2=`echo $DATEm2 | cut -c7-8`

DATEm3=$(sed -n "197p" main19.php | cut -c 48-55)
YEARm3=`echo $DATEm3 | cut -c1-4`
MONTHm3=`echo $DATEm3 | cut -c5-6`
DAYm3=`echo $DATEm3 | cut -c7-8`

DATEm4=$(sed -n "197p" main19.php | cut -c 59-66)
YEARm4=`echo $DATEm4 | cut -c1-4`
MONTHm4=`echo $DATEm4 | cut -c5-6`
DAYm4=`echo $DATEm4 | cut -c7-8`

DATEm5=$(sed -n "197p" main19.php | cut -c 70-77)
YEARm5=`echo $DATEm5 | cut -c1-4`
MONTHm5=`echo $DATEm5 | cut -c5-6`
DAYm5=`echo $DATEm5 | cut -c7-8`

echo $DATE
echo $DATEm1
echo $DATEm2
echo $DATEm3
echo $DATEm4
echo $DATEm5

# Change dates in php file
sed '197s/var cyclist=\["'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'","'${DATEm5}'"\]/var cyclist=\["'${PDY}'","'${DATE}'","'${DATEm1}'","'${DATEm2}'","'${DATEm3}'","'${DATEm4}'"\]/' main19.php > tmpfile

sed '182s/var cyc="'${YEAR}'\/'${MONTH}'\/'${DAY}'"/var cyc="'${yyyy}'\/'${mm}'\/'${dd}'"/' tmpfile > tmpfile2
sed '183s/var cycm1="'${YEARm1}'\/'${MONTHm1}'\/'${DAYm1}'"/var cycm1="'${YEAR}'\/'${MONTH}'\/'${DAY}'"/' tmpfile2 > tmpfile3
sed '184s/var cycm2="'${YEARm2}'\/'${MONTHm2}'\/'${DAYm2}'"/var cycm2="'${YEARm1}'\/'${MONTHm1}'\/'${DAYm1}'"/' tmpfile3 > tmpfile4
sed '185s/var cycm3="'${YEARm3}'\/'${MONTHm3}'\/'${DAYm3}'"/var cycm3="'${YEARm2}'\/'${MONTHm2}'\/'${DAYm2}'"/' tmpfile4 > tmpfile5
sed '186s/var cycm4="'${YEARm4}'\/'${MONTHm4}'\/'${DAYm4}'"/var cycm4="'${YEARm3}'\/'${MONTHm3}'\/'${DAYm3}'"/' tmpfile5 > tmpfile6
sed '187s/var cycm5="'${YEARm5}'\/'${MONTHm5}'\/'${DAYm5}'"/var cycm5="'${YEARm4}'\/'${MONTHm4}'\/'${DAYm4}'"/' tmpfile6 > tmpfile7 ; mv tmpfile7 main19.php

scp main19.php bblake@emcrzdm.ncep.noaa.gov:/home/people/emc/www/htdocs/users/emc.campara/rrfs


date

exit
