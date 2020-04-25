import unittest
import os
import shutil
import cv2
from fire_eye.preprocessor.preprocessor import Preprocessor


class TestPreprocessor(unittest.TestCase):


	def test_preprocess(self):
		test_data_path = os.path.join(os.getcwd(), 'data',
									  'test_preprocessor_data')
		preprocessed_path = os.path.join(test_data_path, 'test_preprocessed')
		# ensure preprocessed_path is empty, but exists
		if os.path.isdir(preprocessed_path):
			shutil.rmtree(preprocessed_path)
		os.mkdir(preprocessed_path)
		# treat an exception as a failure rather than crashing
		try:
			prep = Preprocessor(preprocessed_path=preprocessed_path,
								img_size=(32, 32))
		except:
			self.fail('Issue in Preprocessor() constructor')
		self.assertTrue(os.path.isdir(preprocessed_path))

		# IDEA: replace with a separate test dir outside of data/ ?
		# specify raw test dataset paths
		pos_paths = [os.path.join(test_data_path, 'test_pos')]
		neg_paths = [os.path.join(test_data_path, 'test_neg')]
		# treat an exception as a failure rather than crashing
		try:
			num_imgs = prep.preprocess(pos_paths=pos_paths, neg_paths=neg_paths)
		except:
			self.fail('Issue with Preprocessor.preprocess()')
		
		# check that ./data/preprocessed/ pos and neg dirs exist
		pos_path = os.path.join(preprocessed_path, 'pos')
		neg_path = os.path.join(preprocessed_path, 'neg')
		self.assertTrue(os.path.isdir(pos_path))
		self.assertTrue(os.path.isdir(neg_path))

		# check that num_imgs were actually copied over from test dataset
		img_paths = [os.path.join(pos_path, fname)
					 for fname in os.listdir(pos_path)]
		img_paths += [os.path.join(neg_path, fname)
					  for fname in os.listdir(neg_path)]
		self.assertEqual(len(img_paths), num_imgs)

		# check that all images were properly resized
		for img_path in img_paths:
			img = cv2.imread(img_path)
			self.assertIsNotNone(img)
			self.assertEqual(img.shape[0:2], prep.img_size)

		# again, treat an exception as a failure rather than crashing
		try:
			prep = Preprocessor(preprocessed_path=preprocessed_path,
								img_size=None) # No resizing this time
		except:
			self.fail('Issue in Preprocessor() constructor')
		try:
			prep.preprocess(pos_paths=pos_paths, neg_paths=neg_paths)
		except:
			self.fail('Issue with Preprocessor.preprocess()')
		# compare list of image sizes in raw and preprocessed datasets. 
		# check that they are the same (after sorting).
		# while technically not a guaranteee, this means no resizing was done
		orig_img_paths = [os.path.join(pos_paths[0], fname) 
						  for fname in os.listdir(pos_paths[0])]
		orig_img_paths += [os.path.join(neg_paths[0], fname) 
						   for fname in os.listdir(neg_paths[0])]
		orig_img_sizes = set([cv2.imread(img_path).shape
						  for img_path in orig_img_paths
						  if cv2.imread(img_path) is not None])
		prep_img_sizes = set([cv2.imread(img_path).shape
						  for img_path in img_paths])
		# check that set of new sizes is a subset of original sizes
		# (possibly not all images were successfully copied)
		self.assertTrue(prep_img_sizes.issubset(orig_img_sizes))


if __name__ == '__main__':
	unittest.main()