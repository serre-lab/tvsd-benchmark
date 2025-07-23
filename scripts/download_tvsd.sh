#!/bin/bash

# This script downloads the necessary data files for the project.

mkdir -p data
cd data
mkdir -p TVSD
cd TVSD
mkdir -p monkeyF monkeyN monkeyF/_logs monkeyN/_logs
wget -c https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyF/THINGS_normMUA.mat monkeyF/THINGS_normMUA.mat
wget -c https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyN/THINGS_normMUA.mat monkeyN/THINGS_normMUA.mat
wget -c https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyF/_logs/things_imgs.mat monkeyF/_logs/things_imgs.mat
wget -c https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyN/_logs/things_imgs.mat monkeyN/_logs/things_imgs.mat
echo "Download complete. Data is stored in data/TVSD."
