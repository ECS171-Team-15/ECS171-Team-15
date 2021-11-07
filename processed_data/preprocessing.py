import os
import numpy as np
import csv
from PIL import Image
import sys

# Output the average width and height of all images in specified folders
def get_mean_dimensions(dirs: list) -> list:
    i=0
    height = []
    width = []
    for dir in dirs:
	    # Open directory and get all image names
	    image_names = os.listdir(dir)
	    for name in image_names:
	    	# Store image dimensions
		    im = Image.open(f"{dir}/{name}")
		    w, h = im.size
		    width.append(w)
		    height.append(h)
		    i = i + 1
    # Return averages
    print("Averaged images:", i)
    return sum(width)//len(width), sum(height)//len(height)

# Resize image and convert to grayscale
def preprocess_image(image_path: str, width: int, height: int) -> Image:
	image = Image.open(image_path)
	image = image.resize((width, height))
	image = image.convert("L")
	return image

# Preprocess images and generate dataset as CSV file
if __name__ == '__main__':
	csv_rows = []

	# Input validation
	# Must take proper keyword as argument
	if len(sys.argv) < 2 and (sys.argv[1] != "original" or sys.argv[1] != "modified"):
		print("Usage: python3 preprocessing.py (original|modified)")

	# Generate either modified or original dataset
	COVID_PATH = f'../raw_data/{sys.argv[1]}/CT_COVID'
	NONCOVID_PATH = f'../raw_data/{sys.argv[1]}/CT_NonCOVID'
	image_dirs = [COVID_PATH, NONCOVID_PATH]
	positive_dirs = [COVID_PATH]

	print("Generating mean dimensions...")
	mean_width, mean_height = get_mean_dimensions(image_dirs)
	print("Width:", mean_width, "height:", mean_height)
	print("Creating CSV dataset file...")
	# Create CSV header
	num_pixels = mean_width * mean_height
	header = [f"pixel{i+1}" for i in range(num_pixels)]
	header.append("class")
	csv_rows.append(header)

	for dir_name in image_dirs:
		image_names = os.listdir(dir_name)
	    # Preprocess each image and add as row to CSV
		for image_name in image_names:
			processed_image_obj = preprocess_image(f"{dir_name}/{image_name}", mean_width, mean_height)

			# Convert image object to a row of pixel values
			pixels = list(processed_image_obj.getdata())

			# Include image class value in each row
			if positive_dirs.count(dir_name) > 0:
				# This image is COVID-positive
				class_value = 1
			else:
				class_value = 0
			pixels.append(class_value)

    		# Add example to dataset
			csv_rows.append(pixels)

	# Parse list of rows into csv file
	with open(f"{sys.argv[1]}.csv", "w+") as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=',')
		csv_writer.writerows(csv_rows)

	print("Done.")
