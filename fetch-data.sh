#!/usr/bin/env bash

# Written by Julian

# Print usage
if [[ $1 == '-h' ]]; then
	echo 'Usage: bash fetch-data.sh [modified]'
	echo 'Downloads the original dataset by default'
	exit
fi

check_and_remove_dir(){
	test -e $1
	if [[ $? -eq 0 ]]; then
		rm -rf $1
	fi
}

RAW_DATA_PATH='raw_data'

# Remove existing dataset folder
check_and_remove_dir $RAW_DATA_PATH

# Download links
if [[ $1 == 'modified' ]]; then
	LINK='https://github.com/ECS171-Team-15/Preprocessed-Dataset/raw/master'
else
	LINK='https://github.com/UCSD-AI4H/COVID-CT/raw/master/Images-processed'
fi
POSITIVE_ZIP='CT_COVID.zip'
NEGATIVE_ZIP='CT_NonCOVID.zip'

mkdir $RAW_DATA_PATH
cd $RAW_DATA_PATH

# Download dataset
wget "${LINK}/${POSITIVE_ZIP}"
wget "${LINK}/${NEGATIVE_ZIP}"

# Extract
unzip $POSITIVE_ZIP
unzip $NEGATIVE_ZIP

# Only save useable folders in modified dataset
if [[ $1 == 'modified' ]]; then
	# Positive data
	mv CT_COVID/Useable .
	rm CT_COVID -rf
	mv Useable CT_COVID
	# Negative data
	mv CT_NonCOVID/Usable .
	rm CT_NonCOVID -rf
	mv Usable CT_NonCOVID
fi

# Don't need zip files anymore
rm $POSITIVE_ZIP $NEGATIVE_ZIP

# Remove useless metadata
check_and_remove_dir __MACOSX

echo "Done."

