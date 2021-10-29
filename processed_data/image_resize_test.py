import preprocessing
import os

UPSCALED_NAME = 'upscaled.png'
DOWNSCALED_NAME = 'downscaled.png'

# Manually check if resized images appear correctly
if __name__ == '__main__':
	test_image_path = '../raw_data/COVID/Useable_with_marks/2020.03.04.20030395-p27-108_8.png'
	# Original dimensions (172, 124)
	upscaled_image = preprocessing.preprocess_image(test_image_path, 300, 300)
	upscaled_image.save(UPSCALED_NAME)

	downscaled_image = preprocessing.preprocess_image(test_image_path, 100, 100)
	downscaled_image.save(DOWNSCALED_NAME)

	# Open both original and resized versions
	os.system(f'xdg-open {test_image_path}')
	os.system(f'xdg-open {UPSCALED_NAME}')
	os.system(f'xdg-open {DOWNSCALED_NAME}')

