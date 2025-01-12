#!/usr/bin/env bash
# Copy data and correct with numerically calculcated geometric factor

# Define data files
ert="SCH2010-07-21_u_i.txt"
rst="SCH2010-07-21_tt.txt"

# Copy seismic data
cp -f ../../field_data_timelapse/$rst rst.data

# Copy ERT data and calculate K / rhoa based on primary potentials in a
# temporary folder
mkdir -p bert
cd bert
cp ../../../field_data_timelapse/$ert ert.ohm
bertNew2DTopo ert.ohm > bert.cfg
bert bert.cfg meshs pot filter
mv -f ert.data ..
cd ..
rm -rf bert
