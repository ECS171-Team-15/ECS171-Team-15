#!/usr/bin/env bash

# Written by Julian
# Retrieves both modified and original datasets

# Print usage
if [[ $1 == '-h' ]]; then
	echo 'Usage: bash fetch-data.sh'
	echo 'Downloads both the original and modified dataset'
	exit
fi

check_and_remove_dir(){
	test -e $1
	if [[ $? -eq 0 ]]; then
		rm -rf $1
	fi
}

RAW_DATA_PATH='./raw_data'
check_and_remove_dir $RAW_DATA_PATH

mkdir $RAW_DATA_PATH
cd $RAW_DATA_PATH

# Assume zip files have these names
POSITIVE_ZIP='CT_COVID.zip'
NEGATIVE_ZIP='CT_NonCOVID.zip'

retrieve_dataset(){
	# Make directory for this dataset
	mkdir $2
	cd $2

	# Download zip files from link
	wget "${LINK}/${POSITIVE_ZIP}"
	wget "${LINK}/${NEGATIVE_ZIP}"

	unzip $POSITIVE_ZIP
	unzip $NEGATIVE_ZIP

	# Only save useable folders in modified dataset
	if [[ $1 == 'modified' ]]; then
		# Make sure they are formatted properly
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

	cd ..
}

MODIFIED_LINK='https://github.com/ECS171-Team-15/Preprocessed-Dataset/raw/master'
ORIGINAL_LINK='https://github.com/UCSD-AI4H/COVID-CT/raw/master/Images-processed'

retrieve_dataset $MODIFIED_LINK modified
retrieve_dataset $ORIGINAL_LINK original

echo "Done."
