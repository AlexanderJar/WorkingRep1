#!/usr/bin/env bash
# Copy data and correct with numerically calculcated geometric factor

# Define data files
#ert="ert_t1.data"
#rst="rst_t1.data"

# Copy seismic data
#cp ../../../data_timelapse_SCH/edited/$rst rst_t1.data

# Copy ERT data and calculate K / rhoa based on primary potentials in a
# temporary folder
mkdir -p bert
cd bert
cp ../ert_t1.data ert_t1.data
bertNew2DTopo ert_t1.data > bert.cfg
bert bert.cfg meshs pot filter
mv -f ert_t1.data.data ../ert_t1.data
cd ..
rm -rf bert

mkdir -p bert
cd bert
cp ../ert_t2.data ert_t2.data
bertNew2DTopo ert_t2.data > bert.cfg
bert bert.cfg meshs pot filter
mv -f ert_t2.data.data ../ert_t2.data
cd ..
rm -rf bert

mkdir -p bert
cd bert
cp ../ert_t3.data ert_t3.data
bertNew2DTopo ert_t3.data > bert.cfg
bert bert.cfg meshs pot filter
mv -f ert_t3.data.data ../ert_t3.data
cd ..
rm -rf bert
