import pandas as pd
import os
import cv2
import numpy as np
from fire_eye.preprocess.preprocess import DATA_DIR, POS_DIR, NEG_DIR


TALLY_DIR = os.path.join(DATA_DIR, 'tallies')
POS_COL_NAMES = ['Image Index', 'Setting', 'Smoke? (Y/N)',
				 'Time (D/N)', 'Observations (if any)']
NEG_COL_NAMES = ['Image Index', 'Setting', 'Time (D/N)',
				 'Observations (if any)']


def import_tallies(fname):
	if not isinstance(fname, str):
		raise TypeError(f'fname must be a string')
	if not fname.endswith('.csv'):
		raise ValueError(f'{fname} must be a csv file (include .csv in fname arg)')

	fpath = os.path.join(TALLY_DIR, fname)
	if not os.path.isfile(fpath):
		raise ValueError(f'{os.path.abspath(fpath)} does not exist!')

	df = pd.read_csv(fpath, skiprows=4)
	return df


def import_all_tallies(fnames):
	if not isinstance(fnames, list):
		raise TypeError('fnames must be a list of filenames')

	pos_df = pd.DataFrame(columns=POS_COL_NAMES)
	neg_df = pd.DataFrame(columns=NEG_COL_NAMES)

	for fname in fnames:
		try:
			df = import_tallies(fname)
		except:
			raise IOError(f'Couldn\'t import tallies from {fname}.')
		if 'Smoke? (Y/N)' in df.columns:
			pos_df = pd.concat([pos_df, df])
		else:
			neg_df = pd.concat([neg_df, df])

	return pos_df, neg_df


def calc_statistics(df, colname):
	if not isinstance(df, pd.DataFrame):
		raise TypeError(f'df should be a Pandas DataFrame')
	if not colname in df.columns:
		raise ValueError(f'colname must be exactly a column name in the DataFrame')

	counts = df[colname].value_counts().to_dict()
	total = sum(counts.values(), 0)
	statistics = {k: v / total for k, v in counts.items()}
	
	return statistics


def calc_pixel_value_statistics():
	img_means = []
	img_stds = []
	fpaths = [os.path.join(POS_DIR, fname) for fname in os.listdir(POS_DIR)] +\
			 [os.path.join(NEG_DIR, fname) for fname in os.listdir(NEG_DIR)]
	for fpath in fpaths:
		img = cv2.imread(fpath)
		img_mean = img.mean(axis=(0,1)).reshape(-1, 1)
		img_means.append(img_mean)
		img_std = img.std(axis=(0,1)).reshape(-1, 1)
		img_stds.append(img_std)
	img_means= np.array(img_means).mean(axis=0)
	img_stds= np.array(img_stds).mean(axis=0)
	return img_means, img_stds



def calc_mean_raw_img_dimensions():
	pass