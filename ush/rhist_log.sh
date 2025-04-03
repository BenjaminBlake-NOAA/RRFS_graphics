#!/bin/sh

# This script creates an empty file with name built out 
# of $1 and $2 in directory $LOGrunhistory. This file is
# used by rhist_errchk.sh to keep ecflow appraised of transfer events.

set -x

if [ $# -ne 2 ]
then
  echo "This scripts requires exactly two arguments"
  exit 1
fi 

[[ $WRITE_LOG_DIR != "YES" ]] && exit 2
[[ $DRY_RUN_ONLY == "YES" ]] && exit 3

str1=`echo $1|sed s:/:_:g`
str2=$2
file="${str1}__${str2}"
> ${LOGrunhistory:?}/$file
exit 0
