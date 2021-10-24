import unittest

import preprocessing

class MeanDimensions(unittest.TestCase):
	def test_one_folder(self):
		test_dir = '../raw_data/COVID/Useable_with_marks'
		mean_width, mean_height = preprocessing.get_mean_dimensions([test_dir])

		# (172, 124), (172, 124), (172, 124), (172, 124)
		# (172, 124), (172, 124), (512, 512)
		self.assertEqual(mean_width, 220)
		self.assertEqual(mean_height, 179)

if __name__ == '__main__':
	unittest.main()
	
