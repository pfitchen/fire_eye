from fire_eye.fire_eye import FireEye
import os
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
	eye = FireEye()
	eye.scrape_more_data(num_images=100,
						 pos_queries=['Forest Fire', 'wildfire', 'grass fire'],
						 neg_queries=['Forest', 'landscape', 'sunset', 
						 			  'grass land'])
	eye.load_kaggle_data()
	print(eye.get_dataset_sizes())
	# eye.evaluate_model()
	fig0 = eye.train_model(num_epochs=10)
	fig1, fig2 = eye.evaluate_model()
	eye.save_model('first_model')
	plt.show()
	# eye.load_model('test_model')
	# download_path = os.path.join(os.getcwd(), 'data', 'scraped')
	# preprocessed_path = os.path.join(os.getcwd(), 'data', 'preprocessed')
	# pos_data_path = os.path.join(preprocessed_path, 'pos')
	# neg_data_path = os.path.join(preprocessed_path, 'neg')

