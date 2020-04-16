import os
from fire_eye.preprocess import preprocess
from fire_eye.statistics import statistics
from fire_eye.model import loader, model
from fire_eye.model import model2
import torch


if __name__ == '__main__':
	# print(f'Preprocessing Data...')
	# preprocess.remove_all_from_dataset()
	# # Consider changing add_to_dataset to check for existence so remove_all doesn't need to be called so often...
	# preprocess.add_to_dataset('fire_dataset', 'fire_images', 'non_fire_images')
	# preprocess.add_to_dataset('Fire-Detection', '1', '0')
	# print(f'Both fire_dataset and Fire-Detection datasets' +\
	# 	  f'are combined and properly formatted for training.')

	# print(f'Importing tallies to pandas...')
	# pos_df, neg_df = statistics.import_all_tallies(['fire_dataset_pos_tallies.csv',
	# 												'fire_dataset_neg_tallies.csv',
	# 												'Fire-Detection_pos_tallies.csv',
	# 												'Fire-Detection_neg_tallies.csv'])
	# print(f'Calculating tally statistics...')
	# pos_setting_statistics = statistics.calc_statistics(pos_df, colname='Setting')
	# neg_setting_statistics = statistics.calc_statistics(neg_df, colname='Setting')
	# pos_time_statistics = statistics.calc_statistics(pos_df, colname='Time (D/N)')
	# neg_time_statistics = statistics.calc_statistics(neg_df, colname='Time (D/N)')
	# pos_smoke_statistics = statistics.calc_statistics(pos_df, colname='Smoke? (Y/N)')
	# print(f'\nPositive setting stats: \n{pos_setting_statistics}')
	# print(f'\nNegative setting stats: \n{neg_setting_statistics}')
	# print(f'\nPositive time-of-day stats: \n{pos_time_statistics}')
	# print(f'\nNegative time-of-day stats: \n{neg_time_statistics}')
	# print(f'\nPositive smoke stats: \n{pos_smoke_statistics}')

	# print(statistics.calc_pixel_value_statistics())


	# print('loading data to model...')

	# model.model_run()

	datasets, dataset_sizes, dataloaders = model2.load_data()
	model = model2.init_model()
	model, train_df, best_epoch = model2.train_model(model, dataset_sizes, dataloaders, num_epochs=2)
	model2.evaluate_model(model, dataset_sizes, dataloaders)
	model2.plot_results(train_df, best_epoch)
	model2.visualize_model(model, dataloaders)

