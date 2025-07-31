#!/bin/bash

# This script downloads the THINGS dataset from OSF and unzips into the data directory.

cd data
mkdir -p THINGS
cd THINGS
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE # i know this is super sus but the zip file is safe
osf -p jum2f fetch images_THINGS.zip
unzip images_THINGS.zip
rm images_THINGS.zip
