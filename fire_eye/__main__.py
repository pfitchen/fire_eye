from fire_eye.fire_eye import FireEye
import os
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
	# create an instance of a FireEye object
	eye = FireEye()

	# scrape some new images
	eye.scrape_more_data(num_images=2,
						 pos_queries=['Forest Fire', 'wildfire'],
						 neg_queries=['Forest', 'autumn forest'])
	
	# load any previously scraped images into the preprocessed dataset
	eye.load_scraped_data(pos_queries=['Forest Fire', 'wildfire'],
						  neg_queries=['Forest', 'autumn forest'])
	# load kaggle dataset images into the preprocessed dataset
	eye.load_kaggle_data()
	# print the dataset breakdown of train/validation/test
	print(eye.get_dataset_sizes())

	# train the model for a specified number of epochs
	fig0 = eye.train_model(num_epochs=2)
	
	# evaluate the model and produce relevant figures
	fig1, fig2 = eye.evaluate_model()

	# optionally save the model
	eye.save_model('demo_model')
	
	# display all figures
	plt.show()


	'''
	The following is an example script that might be run after training
	to further evaluate the trained model. In general, scripts should first
	load data, then either load or train a model, and finally evaluate the
	performance of the model.

	the preprocessed directory should be cleaned before starting a new dataset.
	Refer to the Makefile and README.md

	# after training (and cleaning preprocessed data), consider loading new data
	eye.load_scraped_data(pos_queries=['Forest Fire'],
 						  neg_queries=['autumn forest'])
	# load the trained model as well
	eye.load_model('full_dataset_model')
	# evaluate the model on the new dataset
	fig1, fig2 = eye.evaluate_model(title_prefix='Kaggle Datasets ')
	plt.show()
	'''

