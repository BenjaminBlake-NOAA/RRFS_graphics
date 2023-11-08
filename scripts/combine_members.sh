#!/bin/bash

set -x

fhr=$1
fhrp6=$2
dom=$3

vars="slp 2mt 2mdew 10mwind mucape 850t 500 250wind refc uh25 maxuvv qpf"

for var in ${vars}; do
  convert Control_no_${var}_${dom}_f${fhr}.gif Control_yes_${var}_${dom}_f${fhrp6}.gif -background white +append mem1-2_${var}_${dom}_f${fhr}.gif
  convert 1_no_${var}_${dom}_f${fhr}.gif 1_yes_${var}_${dom}_f${fhrp6}.gif -background white +append mem3-4_${var}_${dom}_f${fhr}.gif
  convert mem1-2_${var}_${dom}_f${fhr}.gif mem3-4_${var}_${dom}_f${fhr}.gif +append mem1-4_${var}_${dom}_f${fhr}.gif

  convert 2_no_${var}_${dom}_f${fhr}.gif 2_yes_${var}_${dom}_f${fhrp6}.gif -background white +append mem5-6_${var}_${dom}_f${fhr}.gif
  convert 3_no_${var}_${dom}_f${fhr}.gif 3_yes_${var}_${dom}_f${fhrp6}.gif -background white +append mem7-8_${var}_${dom}_f${fhr}.gif
  convert mem5-6_${var}_${dom}_f${fhr}.gif mem7-8_${var}_${dom}_f${fhr}.gif +append mem5-8_${var}_${dom}_f${fhr}.gif

  convert 4_no_${var}_${dom}_f${fhr}.gif 4_yes_${var}_${dom}_f${fhrp6}.gif -background white +append mem9-10_${var}_${dom}_f${fhr}.gif
  convert 5_no_${var}_${dom}_f${fhr}.gif 5_yes_${var}_${dom}_f${fhrp6}.gif -background white +append mem11-12_${var}_${dom}_f${fhr}.gif
  convert mem9-10_${var}_${dom}_f${fhr}.gif mem11-12_${var}_${dom}_f${fhr}.gif +append mem9-12_${var}_${dom}_f${fhr}.gif

  convert mem1-4_${var}_${dom}_f${fhr}.gif mem5-8_${var}_${dom}_f${fhr}.gif -append mem1-8_${var}_${dom}_f${fhr}.gif
# Make sure row of images with colorbars is the same width as the other 2 rows
  width=$(identify -ping -format "%w" mem1-8_${var}_${dom}_f${fhr}.gif)
  convert mem9-12_${var}_${dom}_f${fhr}.gif -resize ${width} mem9-12_${var}_${dom}_f${fhr}.gif
  convert mem1-8_${var}_${dom}_f${fhr}.gif mem9-12_${var}_${dom}_f${fhr}.gif -append mem1-12_${var}_${dom}_f${fhr}.gif

  width2=$(identify -ping -format "%w" Control_no_${var}_${dom}_f${fhr}.gif)
  convert HRRR_no_${var}_${dom}_f${fhr}.gif -resize ${width2} HRRR_no_${var}_${dom}_f${fhr}.gif
  convert HRRR_yes_${var}_${dom}_f${fhr}.gif -resize ${width2} HRRR_yes_${var}_${dom}_f${fhr}.gif
  convert HRRR_no_${var}_${dom}_f${fhr}.gif HRRR_yes_${var}_${dom}_f${fhrp6}.gif -background white +append mem13-14_${var}_${dom}_f${fhr}.gif
# Place final image in parent directory
  convert mem1-12_${var}_${dom}_f${fhr}.gif mem13-14_${var}_${dom}_f${fhr}.gif -background white -append ../${var}_members_${dom}_f${fhr}.gif

done

exit
