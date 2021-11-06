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

	# Remove useless metadata
	check_and_remove_dir __MACOSX

	# Don't need zip files anymore
	rm $POSITIVE_ZIP $NEGATIVE_ZIP

	cd ..
}

create_dataset_dir(){
	mkdir $1
	cd $1

	mkdir CT_COVID
	mkdir CT_NonCOVID

	cd ..
}

sort_class_imgs(){
	cd $1

	files=(*)
	num_imgs=${#files[@]}

	# Move training data to the right folder
	end_idx=$((num_imgs / 2))
	for ((i = 0 ; i < end_idx ; i++)); do
		mv "${files[$i]}" "../training/$1"
	done

	# Testing data
	start_idx=$((num_imgs / 2 + 1))
	end_idx=$((num_imgs * 3 / 4))
	for ((i = start_idx ; i < end_idx ; i++)); do
		mv "${files[$i]}" "../testing/$1"
	done

	# Validation data
	mv *.{png,jpg} "../validation/$1"

	cd ..
}

# data directory should look like this:
# data
# - training
# -- CT_COVID
# -- CT_NonCOVID
# - testing
# -- CT_COVID
# -- CT_NonCOVID
# - validation
# -- CT_COVID
# -- CT_NonCOVID
# Each set has the class so image generators can distinguish them
split_data(){
	cd $1

	echo -n "Splitting $1 data..."

	create_dataset_dir training
	create_dataset_dir testing
	create_dataset_dir validation

	sort_class_imgs CT_COVID
	sort_class_imgs CT_NonCOVID

	# No longer using image directories
	rmdir CT_COVID
	rmdir CT_NonCOVID

	echo "done."

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
clean_dataset modified
split_data modified

download_dataset $ORIGINAL_LINK original
clean_dataset original
split_data original

echo "Script finished."
