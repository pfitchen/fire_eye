import os
from fire_eye.preprocess import preprocess
from fire_eye.statistics import statistics


if __name__ == '__main__':
	print(f'Preprocessing Data...')
	# preprocess.remove_all_from_dataset()
	# preprocess.add_to_dataset('fire_dataset', 'fire_images', 'non_fire_images')
	# preprocess.add_to_dataset('Fire-Detection', '1', '0')
	# print(f'Both fire_dataset and Fire-Detection datasets' +\
	# 	  f'are combined and properly formatted for training.')
	print(f'Importing tallies to pandas...')
	pos_df, neg_df = statistics.import_all_tallies(['fire_dataset_pos_tallies.csv',
													'fire_dataset_neg_tallies.csv',
													'Fire-Detection_pos_tallies.csv',
													'Fire-Detection_neg_tallies.csv'])
	print(f'Calculating tally statistics...')
	pos_setting_statistics = statistics.calc_statistics(pos_df, colname='Setting')
	neg_setting_statistics = statistics.calc_statistics(neg_df, colname='Setting')
	pos_time_statistics = statistics.calc_statistics(pos_df, colname='Time (D/N)')
	neg_time_statistics = statistics.calc_statistics(neg_df, colname='Time (D/N)')
	pos_smoke_statistics = statistics.calc_statistics(pos_df, colname='Smoke? (Y/N)')
	print(f'\nPositive setting stats: \n{pos_setting_statistics}')
	print(f'\nNegative setting stats: \n{neg_setting_statistics}')
	print(f'\nPositive time-of-day stats: \n{pos_time_statistics}')
	print(f'\nNegative time-of-day stats: \n{neg_time_statistics}')
	print(f'\nPositive smoke stats: \n{pos_smoke_statistics}')