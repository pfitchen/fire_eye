import unittest
from fire_eye.classifier.classifier import Classifier
import os
from torchvision import datasets, models, transforms


class TestClassifier(unittest.TestCase):
	def test_classifier(self):
		test_data_path = os.path.join(os.getcwd(), 'data',
									  'test_classifier_data')
		pos_data_path = os.path.join(test_data_path, 'test_pos')
		neg_data_path = os.path.join(test_data_path, 'test_neg')
		transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
										transforms.RandomRotation(5),
										transforms.ColorJitter(brightness=0.05, 
															   contrast=0.05, 
															   saturation=0.05, 
															   hue=0.05),
										transforms.ToTensor()])
		# just test is all code runs on a test dataset
		try:
			clssfr = Classifier(pos_data_path=pos_data_path, 
								neg_data_path=neg_data_path, 
								transform=transform)
			clssfr.train(num_epochs=2)
			clssfr.evaluate()
		except Exception as e:
			self.fail(f'Error occurred in evaluation of classifier - {e}')


if __name__ == '__main__':
	unittest.main()