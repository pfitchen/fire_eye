from fire_eye.fire_eye import FireEye
import os


if __name__ == '__main__':
	eye = FireEye()
	eye.scrape_more_data(num_images=1, pos_queries=['Forest Fire'],
						 neg_queries=['Forest'])
	# download_path = os.path.join(os.getcwd(), 'data', 'scraped')
	# preprocessed_path = os.path.join(os.getcwd(), 'data', 'preprocessed')
	# pos_data_path = os.path.join(preprocessed_path, 'pos')
	# neg_data_path = os.path.join(preprocessed_path, 'neg')

