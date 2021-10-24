import os
import numpy as np
import csv
from PIL import Image

COVID_PATH = "../Images-processed/COVID/Useable"
NONCOVID_PATH = "../Images-processed/CT_NonCOVID/Usable"

# Preprocess all our images in specified folders
# Return a dictionary mapping directory path to preprocessed image objects
def preprocess_images(dirs: list) -> dict:
	mean_width, mean_height = get_mean_dimensions(dirs)
    
    # Preprocess each image
	preprocessed_images = {}
	for dir in dirs:
		image_names = os.listdir(dir)
		preprocessed_images[dir] = []
		for name in image_names:
			processed_image = preprocess_image(f"{dir}/{name}", mean_width, mean_height)
			preprocessed_images[dir].append(processed_image)
	
	return preprocessed_images

# Output the average width and height of all images in specified folders
def get_mean_dimensions(dirs: list) -> list:
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
		    
	# Return averages
    return sum(width)//len(width), sum(height)//len(height)

# Resize image and convert to grayscale
def preprocess_image(image_path: str, width: int, height: int) -> Image:
	image = Image.open(image_path)
	image = image.resize((width, height))
	image = image.convert("L")
	return image

# Generate CSV file of preprocessed images (pixels + labels)
# CSV file is our dataset to be used for training
# positive_dirs: Images in these directories will be labelled as 1 in dataset
def create_dataset(preprocessed_images: dict, positive_dirs: list):
    images_pixel_format = []
    
    header = [f"pixel{i+1}" for i in range(width*height)] + ["class"]
    csv_file.append(header)

	# Create list of image pixels    
    for dirname, images in preprocessed_images.items():
    	for image in images:
    		# Convert images to list of pixel values
    		pixels = list(image.getdata())
    		if positive_dirs.count(dirname) > 0:
    			# These images are positive, so mark them
    			class_value = 1
    		else:
    			class_value = 0
    		pixels = pixels.append(class_value)
    		
    		# Add example to dataset
    		images_pixel_format.append(pixels)

	# Parse list of image pixels into csv file
    with open("full_data.csv", "w+") as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',')
        csvWriter.writerows(images_pixel_format)

if __name__ == '__main__':
	preprocessed_images = preprocess_images([COVID_PATH, NONCOVID_PATH])
	create_dataset(preprocessed_images, positive_dirs=[COVID_PATH])

