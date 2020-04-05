import os
from fire_eye.preprocess import preprocess
# from fire_eye.test import test_preprocess


if __name__ == '__main__':
	print(os.getcwd())
	preprocess.remove_all_from_dataset()
	preprocess.add_to_dataset('fire_dataset', 'fire_images', 'non_fire_images')
	preprocess.add_to_dataset('Fire-Detection', '1', '0')
	preprocess.remove_from_dataset([-1], [1])