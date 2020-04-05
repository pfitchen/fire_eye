import unittest
import os
import cv2
from fire_eye.preprocess import preprocess


class TestPreprocess(unittest.TestCase):


	# don't change these names! these are unittest specific!
	# runs before every single test.
	def setUp(self):
		# assume removing works, empty dataset
		preprocess.remove_all_from_dataset()


	# don't change these names! these are unittest specific!
	# runs after every single test
	def tearDown(self):
		# assume removing works, empty dataset
		preprocess.remove_all_from_dataset()


	def test_add_to_dataset(self):
		# add new files
		preprocess.add_to_dataset('test_dataset', 'test_pos', 'test_neg')

		# get lists of files in data
		pos_fnames = list(os.listdir(preprocess.POS_DIR))
		neg_fnames = list(os.listdir(preprocess.NEG_DIR))

		# make sure new fnames lists are correct
		# there are 10 valid positive images in this test dataset,
		# and 10 valid negative images
		correct_pos_fnames = [f'pos{idx}.png' for idx in range(10)]
		correct_neg_fnames = [f'neg{idx}.png' for idx in range(10)]
		self.assertEqual(sorted(pos_fnames), sorted(correct_pos_fnames))
		self.assertEqual(sorted(neg_fnames), sorted(correct_neg_fnames))
		# although using sets instead of lists would be faster,
		# this could hide potential duplication errors...

		# now add files to dataset without emptying first
		preprocess.add_to_dataset('test_dataset', 'test_pos', 'test_neg')

		# update list of fnames
		pos_fnames = list(os.listdir(preprocess.POS_DIR))
		neg_fnames = list(os.listdir(preprocess.NEG_DIR))

		# make sure new fnames lists are correct
		# there should now be 20 valid positive images in the test dataset,
		# and 20 valid negative images
		correct_pos_fnames = [f'pos{idx}.png' for idx in range(20)]
		correct_neg_fnames = [f'neg{idx}.png' for idx in range(20)]
		self.assertEqual(sorted(pos_fnames), sorted(correct_pos_fnames))
		self.assertEqual(sorted(neg_fnames), sorted(correct_neg_fnames))


	def test_preprocess_img(self):
		# get a path to an image in the test dataset
		fname = 'test_pos0.png'
		fpath = os.path.join(preprocess.RAW_DATA_DIR, 'test_dataset', 'test_pos', fname)

		# call preprocess_img() on the fpath
		img = preprocess.preprocess_img(fpath)

		# correct dimensions are (w, h, 3) (3 color channels are used still)
		(w, h) = preprocess.IMG_SIZE
		d = 3

		# check that the image has been properly resized
		self.assertEqual(img.shape, (w, h, d))


	def test_get_highest_file_idx(self):
		# first test with empty dataset (no files, so index is -1)
		self.assertEqual(preprocess.get_highest_file_idx(), -1)

		# now add some data
		preprocess.add_to_dataset('test_dataset', 'test_pos', 'test_neg')

		# check that highest file index is now 9 (after adding 10 files)
		self.assertEqual(preprocess.get_highest_file_idx(), 9)


	def test_remove_from_dataset(self):
		# so far it has been assumed that this works (it's used by remove_all)
		# but test anyway
		# start by adding some data
		preprocess.add_to_dataset('test_dataset', 'test_pos', 'test_neg')

		# remove a few indices from dataset
		pos_idx_to_remove = [0, 4]
		neg_idx_to_remove = [1, 7]
		preprocess.remove_from_dataset(pos_indices=pos_idx_to_remove,
									   neg_indices=neg_idx_to_remove)

		# grab a list of fnames
		pos_fnames = list(os.listdir(preprocess.POS_DIR))
		neg_fnames = list(os.listdir(preprocess.NEG_DIR))

		# correct fnames are:
		correct_pos_fnames = [f'pos{idx}.png' for idx in range(10) if not idx in pos_idx_to_remove]
		correct_neg_fnames = [f'neg{idx}.png' for idx in range(10) if not idx in neg_idx_to_remove]

		# check that they are equal
		self.assertEqual(sorted(pos_fnames), sorted(correct_pos_fnames))
		self.assertEqual(sorted(neg_fnames), sorted(correct_neg_fnames))


	def test_remove_all_from_dataset(self):
		# this has been assumed to work for other tests...
		# this is probably bad practice I know!
		# start by adding some data
		preprocess.add_to_dataset('test_dataset', 'test_pos', 'test_neg')

		# assuming that worked, remove all data
		preprocess.remove_all_from_dataset()

		# check that the lists of fnames are empty
		self.assertEqual(len(os.listdir(preprocess.POS_DIR)), 0)
		self.assertEqual(len(os.listdir(preprocess.NEG_DIR)), 0)


if __name__ == '__main__':
	unittest.main()