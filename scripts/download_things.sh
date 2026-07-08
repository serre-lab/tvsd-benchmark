#!/bin/bash
set -euo pipefail

# This script downloads the THINGS dataset from OSF and unzips into the data directory.
#
# The OSF archive is password-protected. The password is public ("things4all"),
# but can be overridden via the THINGS_ZIP_PASSWORD environment variable.
THINGS_ZIP_PASSWORD="${THINGS_ZIP_PASSWORD:-things4all}"

cd data
mkdir -p THINGS
cd THINGS
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE # i know this is super sus but the zip file is safe
osf -p jum2f fetch images_THINGS.zip
unzip -o -P "$THINGS_ZIP_PASSWORD" images_THINGS.zip
rm images_THINGS.zip
