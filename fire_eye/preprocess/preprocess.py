import os
import re
import numpy
import cv2
from tqdm import tqdm


# Global Variables/Parameters for Preprocessing
DATA_DIR = os.path.join(os.getcwd(), 'data')
POS_DIR = os.path.join(DATA_DIR, 'pos')
NEG_DIR = os.path.join(DATA_DIR, 'neg')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_datasets')
IMG_SIZE = (32, 32)


'''
All data file names are of the form: 'posXXX.png' or 'negXXX.png'
where XXX is the file index (integer greater than or equal to 0).
This helper function is used when appending new preprocessed data
or when removing data from the dataset.
'''
def get_highest_file_idx(pos_bool=True):
	global POS_DIR, NEG_DIR
	target_dir = POS_DIR if pos_bool else NEG_DIR

	if not os.path.isdir(target_dir):
		os.mkdir(target_dir)

	idx_re = re.compile(r'\d+')
	highest_idx = -1
	for fname in os.listdir(target_dir):
		idx = int(re.findall(idx_re, fname)[0])
		if idx > highest_idx:
			highest_idx = idx
	return highest_idx


'''
This function handles all of the preprocessing for a single image.
It takes a full path to a raw image and resizes it to fit the rest
of the preprocessed dataset according to the IMG_SIZE global variable.
Other preprocessing tasks such as colormap changes should be done here too.
'''
def preprocess_img(fpath):
	if not isinstance(fpath, str):
		raise TypeError(f'fpath must be a string')
	if not (fpath.endswith('.png') or fpath.endswith('.jpg')):
		raise ValueError(f'fpath must end in .jpg or .png (be an image file)')
	if not os.path.isfile(fpath):
		raise OSError(f'fpath must point to an existing file')

	global IMG_SIZE

	raw_img = cv2.imread(fpath)
	try:
		img = cv2.resize(raw_img, IMG_SIZE)
	except:
		return None
	return img


def add_to_dataset(set_dir, pos_dirname, neg_dirname):
	if not isinstance(set_dir, str):
		raise TypeError(f'set_dir must be a string')
	if not isinstance(pos_dirname, str):
		raise TypeError(f'pos_dirname must be a string')
	if not isinstance(neg_dirname, str):
		raise TypeError(f'neg_dirname must be a string')

	global RAW_DATA_DIR

	if not os.path.isdir(set_dir):
		set_dir = os.path.join(RAW_DATA_DIR, set_dir)
	if not os.path.isdir(set_dir):
		raise ValueError(f'{os.path.abspath(set_dir)} does not exist!')

	raw_pos_dir = os.path.join(set_dir, pos_dirname)
	if not os.path.isdir(raw_pos_dir):
		raise ValueError(f'{os.path.abspath(raw_pos_dir)} does not exist!')

	raw_neg_dir = os.path.join(set_dir, neg_dirname)
	if not os.path.isdir(raw_neg_dir):
		raise ValueError(f'{os.path.abspath(raw_neg_dir)} does not exist!')

	print(f'Preprocessing positive images from {set_dir}')
	pos_file_idx = get_highest_file_idx(True) + 1
	idx = 0
	for fname in tqdm(os.listdir(raw_pos_dir)):
		if fname.endswith('.png') or fname.endswith('.jpg'):
			new_img_fname = f'pos{pos_file_idx + idx}.png'
			new_img = preprocess_img(os.path.join(raw_pos_dir, fname))
			if new_img is not None:
				cv2.imwrite(os.path.join(POS_DIR, new_img_fname), new_img)
				idx += 1
	print(f'Successfully preprocessed {idx}/{len(os.listdir(raw_pos_dir))} positive images')

	print(f'Preprocessing negative images from {set_dir}')
	neg_file_idx = get_highest_file_idx(False) + 1
	idx = 0
	for fname in tqdm(os.listdir(raw_neg_dir)):
		if fname.endswith('.png') or fname.endswith('.jpg'):
			new_img_fname = f'neg{neg_file_idx + idx}.png'
			new_img = preprocess_img(os.path.join(raw_neg_dir, fname))
			if new_img is not None:
				cv2.imwrite(os.path.join(NEG_DIR, new_img_fname), new_img)
				idx += 1
	print(f'Successfully preprocessed {idx}/{len(os.listdir(raw_neg_dir))} negative images')


def remove_from_dataset(pos_indices=[], neg_indices=[]):
	if not isinstance(pos_indices, list):
		raise TypeError(f'pos_indices should be a list of positive filename indices to remove.')
	if not isinstance(neg_indices, list):
		raise TypeError(f'neg_indices should be a list of negative filename indices to remove.')

	global POS_DIR, NEG_DIR

	if len(pos_indices) > 0:
		print(f'Removing specified positive files from dataset...')
		pos_remove_count = 0
		for idx in tqdm(pos_indices):
			pos_fname = os.path.join(POS_DIR, f'pos{idx}.png')
			if os.path.isfile(pos_fname):
				os.remove(pos_fname)
				pos_remove_count += 1
		print(f'Successfully removed {pos_remove_count}/{len(pos_indices)} target positive files')

	if len(neg_indices) > 0:
		print(f'Removing specified negative files from dataset...')
		neg_remove_count = 0
		for idx in tqdm(neg_indices):
			neg_fname = os.path.join(NEG_DIR, f'neg{idx}.png')
			if os.path.isfile(neg_fname):
				os.remove(neg_fname)
				neg_remove_count += 1
		print(f'Successfully removed {neg_remove_count}/{len(neg_indices)} target negative files')


def remove_all_from_dataset(pos=True, neg=True):
	if not isinstance(pos, bool):
		raise TypeError('pos should be a boolean to indicate whether to clear positive dataset')
	if not isinstance(neg, bool):
		raise TypeError('neg should be a boolean to indicate whether to clear negative dataset')

	if pos:
		pos_file_max_idx = get_highest_file_idx(True)
		pos_indices = list(range(pos_file_max_idx + 1))
		remove_from_dataset(pos_indices=pos_indices)
	
	if neg:
		neg_file_max_idx = get_highest_file_idx(False)
		neg_indices = list(range(neg_file_max_idx + 1))
		remove_from_dataset(neg_indices=neg_indices)

