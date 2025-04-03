#!/bin/sh
# This script echos error status for HPSS transfer jobs, calls the
# script that calculates transfer rates from standard output, and
# sends that information to ecflow.

model=$1
cyc=$2
pgm=$model$cyc

set +x
if [ "$err" -ne 0 ]; then
  echo "****************************************************************"
  echo "******  PROBLEM ARCHIVING $pgm RETURN CODE $err at `date` ******"
  echo "****************************************************************"
  msg1="PROBLEM ARCHIVING $pgm RETURN CODE $err"
#took out GM
##  sh postmsg "$jlogfile" "$msg1"
else
  echo " --------------------------------------------- "
  echo " ********** COMPLETED ARCHIVE $pgm  **********"
  echo " --------------------------------------------- "
  msg="ARCHIVE of $pgm COMPLETED NORMALLY"
#took out GM
##  sh postmsg "$jlogfile" "$msg"
  if [[ "$SENDECF" == "YES" && $model != dcom ]]; then
    ecflow_client --event $pgm
    # Calculate and log transfer rates:
    rate_label_log=$DATA/rate_label_log
    echo "node:$HOST  PDY=$PDY " > $rate_label_log

#new for W2 
    NEWPBS_JOBID=$(echo $PBS_JOBID  | awk -F. '{print $1}')
    ${USHrunhistory}/hpss_transfer_rates.pl $PBS_O_WORKDIR/$PBS_JOBNAME.o$NEWPBS_JOBID | grep "Total for all Jobs" >> $rate_label_log
    echo "outputfilepath=$PBS_O_WORKDIR/$PBS_JOBNAME.o$NEWPBS_JOBID"

#old wcoss    
#    ${USHrunhistory}/hpss_transfer_rates.pl $LSB_OUTPUTFILE | grep "Total for all Jobs" >> $rate_label_log

    ecflow_client --label info "`cat $rate_label_log | grep -v output `"
  fi
fi

exit
