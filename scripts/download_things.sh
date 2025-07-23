#!/bin/bash

# This script downloads the necessary data files for the project.

cd data
mkdir -p THINGS
cd THINGS
osf -p jum2f fetch images_THINGS.zip
unzip images_THINGS.zip
rm images_THINGS.zip
