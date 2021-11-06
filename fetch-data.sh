#!/usr/bin/env bash

# Written by Julian
# Retrieves both modified and original datasets

# Print usage
# if [[ $1 == '-h' ]]; then
# 	echo 'Usage: bash fetch-data.sh'
# 	echo 'Downloads both the original and modified dataset'
# 	exit
# fi

check_and_remove_dir(){
	test -e $1
	if [[ $? -eq 0 ]]; then
		rm -rf $1
	fi
}

# number_imgs(){
# 	cd $1
# 	i=1
# 	for file in *; do
# 		test -e "$i.png"
# 		if [[ $? -eq 0 ]]; then
# 			# Indexed image already exists
# 			# Change file name to avoid deleting it
# 			echo $1
# 			echo $file
# 			mv "$i.png" "$i-1.png"
# 		fi

# 		mv "$file" "$i.png"
# 		i=$((i+1))
# 	done
# 	cd ..
# }

# Assume zip files have these names
POSITIVE_ZIP='CT_COVID.zip'
NEGATIVE_ZIP='CT_NonCOVID.zip'

download_dataset(){
	echo -n "Downloading $2 data..."

	# Make directory for this dataset
	mkdir $2
	cd $2

	# Download zip files from link
	wget "$1/${POSITIVE_ZIP}" &> /dev/null
	wget "$1/${NEGATIVE_ZIP}" &> /dev/null

	echo "done."

	echo -n "Extracting $2 data..."
	unzip $POSITIVE_ZIP &> /dev/null
	unzip $NEGATIVE_ZIP &> /dev/null
	echo "done."

	cd ..
}

split_data(){
	cd $1

	echo -n "Splitting $1 data..."

	# Get number of images
	num_imgs=$(ls -l | wc -l)
	num_imgs=$((num_imgs - 1))

	files=(*)

	# Training data
	mkdir training
	end_idx=$((num_imgs / 2))
	for ((i = 0 ; i < end_idx ; i++)); do
		mv "${files[$i]}" "training"
	done

	# Testing data
	mkdir testing
	start_idx=$((num_imgs / 2 + 1))
	end_idx=$((num_imgs * 3 / 4))
	for ((i = start_idx ; i < end_idx ; i++)); do
		mv "${files[$i]}" "testing"
	done

	# Validation data
	# Redirect errors in case files don't exist
	mkdir validation
	mv *.png validation
	mv *.jpg validation

	echo "done."

	cd ..
}

clean_dataset(){
	cd $1

	# Only use useable images in modified dataset
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

	# Number all the images
	# number_imgs 'CT_COVID'
	# number_imgs 'CT_NonCOVID'

	# Split into training, testing, and validation data
	split_data 'CT_COVID'
	split_data 'CT_NonCOVID'

	# Remove useless metadata
	check_and_remove_dir __MACOSX

	# Don't need zip files anymore
	rm $POSITIVE_ZIP $NEGATIVE_ZIP

	cd ..
}

# Remove data folder
DATA_PATH='./data'
check_and_remove_dir $DATA_PATH
mkdir $DATA_PATH
cd $DATA_PATH

MODIFIED_LINK='https://github.com/ECS171-Team-15/Preprocessed-Dataset/raw/master'
ORIGINAL_LINK='https://github.com/UCSD-AI4H/COVID-CT/raw/master/Images-processed'

download_dataset $MODIFIED_LINK modified
download_dataset $ORIGINAL_LINK original

clean_dataset modified
clean_dataset original

echo "Script finished."
