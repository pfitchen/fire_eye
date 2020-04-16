import unittest
import torch
from fire_eye.model import loader
from fire_eye.model.loader import DEVICE
from fire_eye.preprocess import preprocess


class TestModel(unittest.TestCase):


	# don't change these names! these are unittest specific!
	# runs before every single test.
	def setUp(self):
		# start by preprocessing/ensuring there is data
		preprocess.add_to_dataset('test_dataset', 'test_pos', 'test_neg')


	# don't change these names! these are unittest specific!
	# runs after every single test
	def tearDown(self):
		# clean out dataset at the end
		preprocess.remove_all_from_dataset()


	def test_load_data(self):
		# call load dataset
		DATASETS, DATALOADERS = loader.load_data()
		train_loader = DATALOADERS['train']

		# if this runs properly, loading data probably worked properly...
		try:
			torch.manual_seed(0)
			num_epochs = 2
			for epoch in range(num_epochs):
				for batch_idx, (x, y) in enumerate(train_loader):
					print('Epoch:', epoch+1, end='')
					print(' | Batch index:', batch_idx, end='')
					print(' | Batch size:', y.size()[0])

					x = x.to(DEVICE)
					y = y.to(DEVICE)

					print('break minibatch for-loop')
					break
		except:
			self.fail('DataLoaders not properly set up')