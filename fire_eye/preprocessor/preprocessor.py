import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm


class Preprocessor():


	# set up directory structure for preprocessed data.
	# preprocessed_path is where preprocessed images will show up
	def __init__(self, preprocessed_path:str, img_size:tuple=None):
		# set the img_size resize target. if its none, no resizing happens
		self.img_size = img_size

		# check that preprocessed target path exists
		if not os.path.isdir(preprocessed_path):
			raise ValueError(f'{preprocessed_path} is not a valid path.')
		self.preprocessed_path = preprocessed_path

		# set up the pos and neg dirs in ./data/preprocessed/
		self.pos_path = os.path.join(self.preprocessed_path, 'pos')
		self.neg_path = os.path.join(self.preprocessed_path, 'neg')
		if not os.path.isdir(self.pos_path):
			os.mkdir(self.pos_path)
		if not os.path.isdir(self.neg_path):
			os.mkdir(self.neg_path)

		# this will get updated in preprocess() step
		self.num_imgs = None


	# Preprocess all images in the lists of pos and neg paths.
	# Previous preprocessed_path images are removed!
	# Invalid paths are skipped, rather than throwing an error.
	def preprocess(self, pos_paths:list=[], neg_paths:list=[]):
		# helper function:
		# this does all of the resizing and copying. it takes a list of paths
		# with images to resize and copy to a target dir.
		# by default, these are the pos_paths and preprocessed pos dir
		def resize_and_copy(paths:str=pos_paths, dest_path:str=self.pos_path):
			img_paths = list()
			for path in paths:
				if not os.path.isdir(path):
					continue # skip if not valid dir
				for fname in os.listdir(path):
					if fname.endswith('png') or fname.endswith('jpg'):
						img_paths.append(os.path.join(path, fname))

			# preprocessed images are named <dest_dir>x.png 
			# where x is image count up to that point (index)
			img_count = 0
			fname_prefix = os.path.basename(os.path.normpath(dest_path))
			for img_path in tqdm(img_paths):
				try:
					img = cv2.imread(img_path)
					# don't resize if img_size is specified as None
					if self.img_size is not None:
						img = cv2.resize(img, self.img_size)
					# save image to appropriate preprocessed dir
					img_fname = f'{fname_prefix}{img_count}.png'
					cv2.imwrite(os.path.join(dest_path, img_fname), img)
					img_count += 1
				except:
					# if an issue occurs, just move on to the next image
					continue
			return img_count

		# first remove preprocessed data if it exists
		if os.path.isdir(self.pos_path):
			shutil.rmtree(self.pos_path)
		if os.path.isdir(self.neg_path):
			shutil.rmtree(self.neg_path)
		os.mkdir(self.pos_path)
		os.mkdir(self.neg_path)

		# could use image counts in a print statement if so desired...
		num_pos_imgs = resize_and_copy(paths=pos_paths, dest_path=self.pos_path)
		num_neg_imgs = resize_and_copy(paths=neg_paths, dest_path=self.neg_path)

		# calculate total number of images
		self.num_imgs = num_pos_imgs + num_neg_imgs

		return self.num_imgs


if __name__ == '__main__':
	prep = Preprocessor()
	pos_paths = [os.path.join(os.getcwd(), 'data', 'test_preprocess',
							  'test_pos')]
	neg_paths = [os.path.join(os.getcwd(), 'data', 'test_preprocess',
							  'test_neg')]
	prep.preprocess(pos_paths=pos_paths, neg_paths=neg_paths)
