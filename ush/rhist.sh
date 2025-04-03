#!/bin/bash
################################################################################
#  This script archives data to HPSS based on a 'parmfile' defined for a given model.
#  Archiving can be with htar or with tar+rsync.
#
#  Usage:
#    $ rhist.sh modelname
#    where 'modelname' is an ASCII file located in ${PARMrunhistory};
#    don't prepend any paths to the filenames.
#
#  Variables defined upstream that we need:
#    $job (e.g., runhistory00); $HPSSOUT; $PDY's; path dir's (COM, etc.)
#
#  Example parmfile:
#    runjobs       runhistory00
#    whichpdy      PDYm2
#    storetypes    2yr
#    dir           $(compath.py lmp/prod/lmp).%thepdy%
#    files         *grib2*
#    dotransfer
#
#  Parmfile directives/parameters:
#
#  * indicates required
#  + indicates support of multiple, comma-delimited values
#  % indicates special name that can be used as variable in filenames (incl. from file lists), 'dir', 'runjobs', and 'tarmod' by using '%varname%', singular, no quotes:
#
#    runjobs*+: names of runhistory jobs so we know which parts of each parmfile to us
#    method*: htar (self-explanatory); some day this might support tarrsync (tar files locally, then rsync)
#    storetypes*+: which storage term on HPSS we want; one of '1yr', '2yr', '5yr', 'perm'
#    whichpdys*+%: PDY variables to use
#    dir*: local directory from which to get data; supports shell variables
#    tarmod: single-word argument that modifies output tar file name (e.g., {normalname}{addedbit}.tar); can include '%thepdy%' as a special variable to sub in the PDY based on whichpdys
#    tarsed: sed argument that will get applied to final tarfile name (e.g., 's/gpfs_hps_nco_ops_//', not including the single quotes)
#    files: POSIX filename patterns to include; clobbers any previously defined filename patterns; default is *
#    morefiles: POSIX filename patterns to include; gets appended to running list of filename patterns
#    filelist: name of file in $PARMrunhistory that contains a list of file patterns to be included; clobbers any previously defined included filename patterns
#    morefilelist: name of file in $PARMrunhistory that contains a list of file patterns which get appended to running list of included filename patterns
#    excludefiles: POSIX filename patterns to exclude; clobbers any previously defined filename patterns; default is none
#    moreexcludefiles: POSIX filename patterns to exclude; gets appended to list of excluded filename patterns
#    excludefilelist: name of file in $PARMrunhistory that contains a list of file patterns to be excluded; clobbers any previously defined excluded filename patterns
#    moreexcludefilelist: name of file in $PARMrunhistory that contains a list of file patterns to be excluded; patterns get appended to list of excluded filename patterns
#    modelcycs%+: HH-format cycle times to iterate over
#    dotransfer: do the transfer based on the parameters defined so far; please note parameters will not be reset. This can take an optional argument "optional", which will quietly skip a transfer if no matching files exist.
#    reset: resets associative array containing parameters; can be used for sanity checking
#    var: set a shell variable; should typically not be needed
#    shell: some Bash text to 'eval'; should typically not be needed
#    quit: exit quietly; for debugging
#    ignore72: treat exit code 72 from htar as exit code 0; used for dcom due to frequent file changes.
#    logarg1, logarg2: customize first and second arguments for logging script rhist_log.sh. Can accommodate shell variables/expansions, as well as several special rhist.sh variables (e.g., %thepdy%).
#    lognow: log based on the latest 'dotransfer' (default is to log after the parm file has been fully processed)
#    skipnodir: quietly exit if the directory does not exist
################################################################################
#   Possible future modifications:
#   -Wrap "find" commands in filelist creation with "eval" to allow variables to be used (useful for multi-cycle transfers, e.g., {00,01,02,03,04,05})
#   -filterin/filterout: allow user to provide regexes for filtering file paths
#   -Incorporate remaining models
################################################################################
#rsyncserver=my-little-server.noaa.gov
#########################################

set -x

defcyc=defcyc_$$_$(date +%s_%n)
defpdy=defpdy_$$_$(date +%s_%n)

function echoerr(){
 code=${1:-1} ; shift 1
 echo -e "ERROR in rhist.sh: $*"
 #echo (other stuff like which job/cycle/etc.)
 exit $code
}

if [ $# -ne 1 ]; then echo "Usage: rhist.sh modelname" ; exit 1 ; fi
if [ -z $job ]; then echoerr 1 "'\$job' env. variable not set; aborting!" ; exit 1 ; fi
echo "Firing up rhist.sh with model file '$1' for runhistory job $job"

function checkvars(){
 TRANSFERPARAMS=$1
 for var in runjob dir storetypes thepdy
 do
  if [ -z "${TRANSFERPARAMS[$var]}" ] ; then echoerr 1 "Parameter '$var' has not been defined for '${TRANSFERPARAMS[parmfilename]}'! Skipping this dotransfer!" ; fi
 done
}

function dotransfer(){
 TRANSFERPARAMS=$1
 if [ "${TRANSFERPARAMS[modelcyc]}" == "$defcyc" ]; then TRANSFERPARAMS[modelcyc]=""; fi
 if [ "${TRANSFERPARAMS[whichpdy]}" == "$defpdy" ]; then TRANSFERPARAMS[whichpdy]=""; fi
 if [ -z ${TRANSFERPARAMS[method]} ]; then TRANSFERPARAMS[method]=htar ; fi
 year=$(echo ${TRANSFERPARAMS[thepdy]} | cut -c 1-4)
 yearmo=$(echo ${TRANSFERPARAMS[thepdy]} | cut -c 1-6)
 yrmoday=$(echo ${TRANSFERPARAMS[thepdy]} | cut -c 1-8)
 IFS=', ' read -r -a storetypes <<< $(echo ${TRANSFERPARAMS[storetypes]})
 unset hpssdirs
 if [ "${TRANSFERPARAMS[hpssout_obs]}" == YES ]; then whichhpssout=$HPSSOUT_OBS; else whichhpssout=$HPSSOUT; fi
 for storetype in ${storetypes[@]};
 do
  case "$storetype" in
   perm) hpssdirs+=( ${whichhpssout}/rh${year}/${yearmo}/$yrmoday ) ;;
   1yr) hpssdirs+=( ${whichhpssout}/1year/rh${year}/${yearmo}/$yrmoday ) ;;
   2yr) hpssdirs+=( ${whichhpssout}/2year/rh${year}/${yearmo}/$yrmoday ) ;;
   5yr) hpssdirs+=( ${whichhpssout}/5year/rh${year}/${yearmo}/$yrmoday ) ;;
#   *) hpssdirs+=( ${whichhpssout}/$storetype ) ;; # NEEDS TESTING
  esac
 done
 tarbase=$(echo ${TRANSFERPARAMS[dir]} | cut -c 2- | tr "/" "_" | perl -pe 's|_$||')
 tarname=${tarbase}${TRANSFERPARAMS[tarmod]}.tar
# tarname=$(echo $tarname | perl -pe 's|lfs_[^_]+_nco_ops_(d?com)|\1|')

 #new?
# tarname=$(echo $tarname | perl -pe 's|lfs_[^]+_ops(d?prod)|\1|')

tarname=$(echo $tarname | perl -pe "s|(d?com)|\1|")

 #old wcoss
 # tarname=$(echo $tarname | perl -pe 's|gpfs_[^_]+_nco_ops_(d?com)|\1|')
 if [ ! -z "${TRANSFERPARAMS[tarsed]}" ]; then tarname=$(echo $tarname | sed "${TRANSFERPARAMS[tarsed]}"); fi
 if [ -z "$tarname" ]; then echoerr 1 "For some reason \$tarname didn't get properly set."; fi
 oldpwd=$PWD #
 logarg1=${TRANSFERPARAMS[dir]}
 logarg2=${TRANSFERPARAMS[thepdy]}
 if [ ! -z "${TRANSFERPARAMS[modelcyc]}" ]; then logarg2=${logarg2}${TRANSFERPARAMS[modelcyc]}; fi
 if [ ! -z "${TRANSFERPARAMS[logarg1]}" ]; then
  logarg1=$(echo ${TRANSFERPARAMS[logarg1]} | sed "s|%thepdy%|${TRANSFERPARAMS[thepdy]}|g;s|%modelcyc%|${TRANSFERPARAMS[modelcyc]}|g;s|%whichpdy%|${TRANSFERPARAMS[whichpdy]}|g;s|%dir%|${TRANSFERPARAMS[dir]}|g")
 fi
 if [ ! -z "${TRANSFERPARAMS[logarg2]}" ]; then
  logarg2=$(echo ${TRANSFERPARAMS[logarg2]} | sed "s|%thepdy%|${TRANSFERPARAMS[thepdy]}|g;s|%modelcyc%|${TRANSFERPARAMS[modelcyc]}|g;s|%whichpdy%|${TRANSFERPARAMS[whichpdy]}|g")
 fi
 cd ${TRANSFERPARAMS[dir]}
 filepatterns=$(echo ${TRANSFERPARAMS[filepatterns]} | sed "s|%modelcyc%|${TRANSFERPARAMS[modelcyc]}|g;s|%thepdy%|${TRANSFERPARAMS[thepdy]}|g;s|%thepdym1%|${TRANSFERPARAMS[thepdym1]}|g;s|%thepdyp1%|${TRANSFERPARAMS[thepdyp1]}|g;s|%dir%|${TRANSFERPARAMS[dir]}|g")
 excludepatterns=$(echo ${TRANSFERPARAMS[excludepatterns]} | sed "s|%modelcyc%|${TRANSFERPARAMS[modelcyc]}|g;s|%thepdy%|${TRANSFERPARAMS[thepdy]}|g;s|%dir%|${TRANSFERPARAMS[dir]}|g")
 if [ -z "$filepatterns" ] ; then filepatterns="./" ; fi #
 if [ -z "$excludepatterns" ] #
 then #
  FILELIST=$(eval "find $filepatterns \( -type l -o -type f \) | sort | uniq")
 else #
  FILELIST=$(eval "find $filepatterns \( -type l -o -type f \) | grep -vFxf <(find $excludepatterns \( -type l -o -type f \)) | sort | uniq") #
 fi #
 if [[ -z ${FILELIST} && ${TRANSFERPARAMS[optional]} == YES ]]; then return 0; fi
 for hpssdir in ${hpssdirs[@]}; do
  fulltarpath="${hpssdir}/$tarname"
  if [ "$READ_LOG_DIR" == YES ]; then
   if [ $(grep -cFx "$fulltarpath" $JOBTARLOGFILE) -eq 1 ]; then echo "rhist.sh: skipping file '$tarname', which is logged as having been transferred to '$hpssdir'"; continue; fi
  fi
  if [[ "$DRY_RUN_ONLY" == YES ]]; then
   echo -e "rhist.sh: Doing dry run only. Model: ${TRANSFERPARAMS[parmfilename]}.\nThe following files from directory ${TRANSFERPARAMS[dir]}\nwould be placed in tar file ${tarname}\nin HPSS directory $hpssdir:\n"$FILELIST"\n"
   dolog=YES
   return
  fi
  case "${TRANSFERPARAMS[method]}" in
#   tarrsync)
#    echoerr 1 "HPSS does not yet support rsync. It's nice to have dreams, though."
#    echo tar -cvf $tarname $FILELIST --exclude=$tarname
#    if [ $? -ne 0 ]; then echoerr 1 "'$tarname' creation failed for '${TRANSFERPARAMS[parmfilename]}' (non-zero exit for 'tar')." ; fi 
#    for try in 1 2 3; do
#     echo rsync -vtou $tarname ${rsyncserver}:${hpssdir}/$tarname ; status = $?
#     if [ $status -eq 0 ]; then break; else sleep 30; fi
#    done
#    if [ $status -eq 0 ]; then
#     rm -f $tarname
#    else
#     echo "rhist.sh: '$tarname' transfer failed for '${TRANSFERPARAMS[parmfilename]}' (non-zero exit for 'rsync')."
#     exit 0
#    fi
#    ;;
   htar)
    hsi "ls -l $fulltarpath.idx" ; status=$?
    if [[ $status -ne 0 || "$CHECK_HPSS_IDX" != YES ]]; then
     for try in 1 2 3; do
#add echo for testing
      echo  "   htar -Pcvf $fulltarpath $(echo $FILELIST)  "
      htar -Pcvf $fulltarpath $(echo $FILELIST)
      status=$?
      if [[ $status -eq 72 && ${TRANSFERPARAMS[ignore72]} == YES ]]; then status=0 ; fi
      if [[ $status -eq 0 ]]; then break; else sleep 30; fi
     done
     if [ $status -ne 0 ]; then
      echoerr $status "'$tarname' creation failed for '${TRANSFERPARAMS[parmfilename]}' (non-zero exit for 'htar')." ;
     fi
#add echo for testing
#echo   "  htar -tvf $hpssdir/$tarname "
     htar -tvf $hpssdir/$tarname
     if [ $? -ne 0 ]; then echoerr 4 "tar file '$tarname' could not be read with htar -t" ; fi
    else
     echo "File '$tarname' already saved."
    fi ;;
   *)
    echoerr 1 "Method '${TRANSFERPARAMS[method]}' not found for '${TRANSFERPARAMS[parmfilename]}'\nShould be 'htar'"  ;;
  esac
  ${USHrunhistory}/rhist_restrict.sh $hpssdir/$tarname ${TRANSFERPARAMS[method]} # this will need to be updated if rsync is added
  dolog=YES
  if [[ "$WRITE_LOG_DIR" != NO && "$DRY_RUN_ONLY" != YES ]]; then echo $fulltarpath >> $JOBTARLOGFILE; fi
  cd $oldpwd
 done
}

# Open and parse parmfile:
parmfilename=$1
if [[ ! -f ${PARMrunhistory}/models/$parmfilename || -z $parmfilename ]]; then echoerr 1 "parm file ${PARMrunhistory}/models/$parmfilename not found!"; fi
mapfile -t parmfilelines < <(sed "s/[[:space:]]*#.*//g;/^$/d" ${PARMrunhistory}/models/$parmfilename )
# TRANSFERPARAMS is an associative array that will contain various parameters from each parmfile.
# The parmfile is read sequentially, so parameters can get overwritten (see header doco for details).
unset TRANSFERPARAMS
declare -A TRANSFERPARAMS
TRANSFERPARAMS[parmfilename]=$(echo $parmfilename)
modelcycs=( $defcyc )
whichpdys=( $defpdy )
echo defcyc $defcyc defpdy $defpdy

for line in "${parmfilelines[@]}"
do
 linetype=$(echo $line | grep -Eo "^[^[:space:]]*")
 lineparams=$(echo "$line" | perl -pe 's|^[^[:space:]]+[[:space:]]+||')
 # These cases correspond with the directives/parameters allowed for our parmfiles
 case "$linetype" in
  runjobs) IFS=', ' read -r -a runjobs <<< $(echo $lineparams) ;;
  method) TRANSFERPARAMS[method]=$(echo $lineparams | grep -Eo "^[[:alpha:]]*") ;;
  modelcycs) IFS=', ' read -r -a modelcycs <<< $(echo $lineparams) ;;
  whichpdys)
   IFS=', ' read -r -a whichpdys <<< $(echo $lineparams)
   for whichpdy in ${whichpdys[@]}; do
   if [ -z ${!whichpdy} ]; then num=$(echo $whichpdy | sed 's|PDYm||g'); export $whichpdy=$(date -d "$PDY-$num days" +%Y%m%d); fi
   done ;;
  dir)
   dir=$(eval "echo $lineparams") ;; # expanding shell variables specified in parmfile
  tarmod)
   tarmod=$(echo $lineparams | grep -Eo "^[^[:space:]]+") ;;
  tarsed) TRANSFERPARAMS[tarsed]="$lineparams" ;;
  files) TRANSFERPARAMS[filepatterns]="$lineparams" ;;
  morefiles) TRANSFERPARAMS[filepatterns]+=" $lineparams" ;;
  filelist)
   filelistname=$(echo $lineparams | grep -Eo "^[^[:space:]]*")
   mapfile filelist < <(sed "s/[[:space:]]*#.*//g;/^$/d" ${PARMrunhistory}/$filelistname)
   TRANSFERPARAMS[filepatterns]=""
   for filepattern in "${filelist[@]}"; do TRANSFERPARAMS[filepatterns]+=" $filepattern"; done ;;
  morefilelist)
   filelistname=$(echo $lineparams | grep -Eo "^[^[:space:]]*")
   mapfile filelist < <(sed "s/[[:space:]]*#.*//g;/^$/d" ${PARMrunhistory}/$filelistname)
   for filepattern in "${filelist[@]}"; do TRANSFERPARAMS[filepatterns]+=" $filepattern"; done;;
  excludefiles)
   TRANSFERPARAMS[excludepatterns]="$lineparams" ;;
  moreexcludefiles)
   TRANSFERPARAMS[excludepatterns]+=" $lineparams" ;;
  excludefilelist)
   filelistname=$(echo $lineparams | grep -Eo "^[^[:space:]]*")
   mapfile filelist < <(sed "s/[[:space:]]*#.*//g;/^$/d" ${PARMrunhistory}/$filelistname)
   TRANSFERPARAMS[filepatterns]=""
   for filepattern in "${filelist[@]}"; do TRANSFERPARAMS[excludepatterns]+=" $filepattern"; done;;
  moreexcludefilelist)
   filelistname=$(echo $lineparams | grep -Eo "^[^[:space:]]*")
   mapfile filelist < <(sed "s/[[:space:]]*#.*//g;/^$/d" ${PARMrunhistory}/$filelistname)
   for filepattern in "${filelist[@]}"; do TRANSFERPARAMS[excludepatterns]+=" $filepattern"; done;;
  var)
   varname=$(echo $lineparams | grep -Eo "^\w+")
   varvalue=$(echo $lineparams | grep -Po '^\w+ \K.*')
   export $varname=$varvalue ;;
  shell) eval "$lineparams" ;;
  quit) exit 0 ;;
  lognow)
   if [ "$dolog" == YES ]; then
    if [ -z "$logarg1" ]; then echo "rhist.sh: log argument not set; are you sure you set 'dir' and ran 'dotransfer' before trying to log?" ; exit 1; fi
    if [ "$DRY_RUN_ONLY" != YES ]; then ${USHrunhistory}/rhist_log.sh $logarg1 $logarg2
    else echo "rhist.sh: For dry run, job ${TRANSFERPARAMS[parmfilename]} would have been logged now with arguments: $logarg1 $logarg2"
    fi
   fi
   dolog=NO
   ;;
  skipnodir) skipnodir=YES ;;
  reset)
   unset TRANSFERPARAMS
   declare -A TRANSFERPARAMS 
   TRANSFERPARAMS[parmfilename]=$(echo $parmfilename) ;;
  storetypes) TRANSFERPARAMS[storetypes]="$lineparams" ;;
  logarg1) TRANSFERPARAMS[logarg1]=$(eval "echo $lineparams") ;;
  logarg2) TRANSFERPARAMS[logarg2]=$(eval "echo $lineparams") ;;
  ignore72) TRANSFERPARAMS[ignore72]="YES" ;;
  hpssout_obs) TRANSFERPARAMS[hpssout_obs]="YES" ;;
  dotransfer)
   if [ "$lineparams" == "optional" ]; then TRANSFERPARAMS[optional]=YES; fi
   for modelcyc in ${modelcycs[@]}; do
    isrightjob=NO
    if [[ "${runjobs[0]}" == *"%modelcyc%"* ]]; then
     runjob=$(echo ${runjobs[$key]} | sed "s|%modelcyc%|$modelcyc|g")
#added n in front of job GM
     if [ "$runjob" == "j${job:1}" ]; then isrightjob=YES; fi
    else
#added n in front of job GM
     if [[ " ${runjobs[@]} " =~ " j${job:1} " ]]; then isrightjob=YES; fi
    fi
    if [[ "$isrightjob" == YES ]]; then
#added n in front of job GM
     TRANSFERPARAMS[runjob]=j${job:1}
     for whichpdy in ${whichpdys[@]}; do
      TRANSFERPARAMS[whichpdy]=$whichpdy
      TRANSFERPARAMS[modelcyc]=$modelcyc
      TRANSFERPARAMS[thepdy]=${!whichpdy} # converting, e.g., 'PDYm1' to 20190703
      TRANSFERPARAMS[thepdym1]=$(date -d "${!whichpdy}-1 days" +%Y%m%d)
      TRANSFERPARAMS[thepdyp1]=$(date -d "${!whichpdy}+1 days" +%Y%m%d)
      TRANSFERPARAMS[tarmod]=$(echo $tarmod | sed "s|%modelcyc%|${TRANSFERPARAMS[modelcyc]}|g;s|%whichpdy%|${TRANSFERPARAMS[whichpdy]}|g;s|%thepdy%|${TRANSFERPARAMS[thepdy]}|g")
      TRANSFERPARAMS[dir]=$(echo $dir | sed "s|%modelcyc%|${TRANSFERPARAMS[modelcyc]}|g;s|%whichpdy%|${TRANSFERPARAMS[whichpdy]}|g;s|%thepdy%|${TRANSFERPARAMS[thepdy]}|g")
      if [ ! -d ${TRANSFERPARAMS[dir]} ]; then
       if [ "$skipnodir" == YES ]; then echo "rhist.sh: Exit 0 due to 'skipnodir' directive"; exit 0; fi
       echoerr 2 "Directory ${TRANSFERPARAMS[dir]} does not exist for model $parmfilename"
      fi
      checkvars $TRANSFERPARAMS
      dotransfer $TRANSFERPARAMS
     done
    fi
   done
   TRANSFERPARAMS[optional]=""
  ;;
  *)
   echo "rhist.sh: parameter '$linetype' not recognized. Hope it's nothing important!"
  ;;
 esac
done

if [[ "$dolog" == YES && "$WRITE_LOG_DIR" != NO ]]; then
 if [ "$DRY_RUN_ONLY" != YES ]; then ${USHrunhistory}/rhist_log.sh $logarg1 $logarg2
 else echo "rhist.sh: For dry run, job ${TRANSFERPARAMS[parmfilename]} would have been logged with arguments: $logarg1 $logarg2"
 fi
fi

echo "rhist.sh is done with model file '$1' for runhistory job $job"
exit 0
