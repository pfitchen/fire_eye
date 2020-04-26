import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import os
import time
import copy
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm


# Manages all pytorch steps, including dataloaders, training, and evaluating
class Classifier():
	# positive and negative data paths must be specified to set up the
	# dataloaders. The combined dataset from these two paths must not be empty.
	# Optionally, a previously trained model can be loaded by specifying
	# a path to the saved model/parameters. Also a transform for data
	# data augmentation can be passed in as well.
	def __init__(self, pos_data_path:str, neg_data_path:str, 
				 model_path:str=None, transform=None):
		# create a dataset object of inner class FireDataset
		dataset = self.FireDataset(pos_data_path=pos_data_path,
								   neg_data_path=neg_data_path,
								   transform=transform)
		# raise a ValueError if dataset is empty.
		if len(dataset) == 0:
			raise ValueError(f'Empty dataset!')

		# split the dataset into 60% train, 20% validation, and 20% test
		total_dataset_size = len(dataset)
		train_dataset_size = int(0.6*total_dataset_size)
		val_dataset_size = int(0.2*total_dataset_size)
		test_dataset_size = total_dataset_size - train_dataset_size \
							- val_dataset_size
		dataset_split = [train_dataset_size,
						 val_dataset_size,
						 test_dataset_size]
		train_dataset, test_dataset, valid_dataset = \
						torch.utils.data.random_split(dataset, dataset_split)

		# create the dataloaders.
		train_loader = DataLoader(dataset=train_dataset,
								  batch_size=16,
								  shuffle=True,
								  num_workers=0)
		valid_loader = DataLoader(dataset=valid_dataset,
								  batch_size=128,
								  shuffle=False,
								  num_workers=0)
		test_loader = DataLoader(dataset=test_dataset,
								 batch_size=128,
								 shuffle=False,
								 num_workers=0)

		# store datasets and dataloaders in dictionaries for convenience
		self.datasets = {'train': train_dataset, 'test': test_dataset, 
						 'val': valid_dataset}
		self.dataloaders = {'train': train_loader, 'test': test_loader, 
							'val': valid_loader}

		# set up a ResNet18 model for binary classification
		model = models.resnet18(pretrained=True)
		num_in_fc = model.fc.in_features
		model.fc = nn.Linear(num_in_fc, 2) # two classes: Fire and NoFire

		# if a previous model is specified by model_path, load it
		if model_path is not None:
			if not os.path.isfile(model_path):
				raise ValueError(f'{model_path} is not a valid model path')
			model.load_state_dict(torch.load(model_path))

		self.model = model


	# train for a specified number of epochs. For simplicity, 
	# cross entropy loss, SGD, and a learning rate scheduler are always used.
	# This will run on a GPU if possible, but it shouldn't be necessary.
	# Returns a figure of training/validation accuracy and loss versus epochs.
	# Internally updates model parameters to those with highest validation
	# accuracy.
	def train(self, num_epochs=25):
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.model = self.model.to(device)
		criterion = nn.CrossEntropyLoss()
		# Observe that all parameters are being optimized
		optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
		# Decay LR by a factor of 0.1 every 7 epochs
		scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

		start_time = time.time()
		best_model = copy.deepcopy(self.model.state_dict())
		best_acc = 0.0
		best_loss = 0.0
		best_epoch = 0

		train_stats = {'train_acc': [], 'val_acc': [], 
					   'train_loss': [], 'val_loss': [],
					   'epoch': []}

		for epoch in range(num_epochs):
			print(f'Epoch {epoch}/{num_epochs-1}')
			print('-' * 80)
			
			# one training and one validation phase per epoch
			for phase in ['train', 'val']:
				if phase == 'train':
					self.model.train()  # Set model to training mode
				else:
					self.model.eval()   # Set model to evaluate mode

				running_loss = 0.0
				running_corrects = 0

				for inputs, labels in tqdm(self.dataloaders[phase]):
					inputs = inputs.to(device)
					labels = labels.to(device)

					optimizer.zero_grad()

					# Forward
					with torch.set_grad_enabled(phase == 'train'):
						outputs = self.model(inputs)
						_, predictions = torch.max(outputs, 1)
						loss = criterion(outputs, labels)

						# Backward
						if phase == 'train':
							loss.backward()
							optimizer.step()

					# Statistics
					running_loss += loss.item()*inputs.size(0)
					running_corrects += torch.sum(predictions == \
												  labels.data).item()

				if phase == 'train':
					scheduler.step()

				epoch_loss = running_loss/len(self.datasets[phase])
				epoch_acc = running_corrects/len(self.datasets[phase])

				print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

				# deepcopy/save the model if it is current best
				if phase == 'val' and epoch_acc > best_acc:
					best_acc = epoch_acc
					best_loss = epoch_loss
					best_model = copy.deepcopy(self.model.state_dict())
					best_epoch = epoch

				# store epoch statistics for plotting later
				train_stats[phase + '_acc'].append(epoch_acc)
				train_stats[phase + '_loss'].append(epoch_loss)
			
			train_stats['epoch'].append(epoch)
			print()

		time_elapsed = time.time() - start_time
		print()
		print(f'Training complete in {(time_elapsed//60):.0f}m ' + \
			  f'{(time_elapsed%60):.0f}s')
		print(f'Best val Acc: {best_acc:4f}')

		# load best model
		self.model.load_state_dict(best_model)
		
		# put training stats into a dataframe
		train_df = pd.DataFrame(train_stats)
		with plt.style.context('seaborn'):
			fig = plt.figure()
			plt.subplot(121)
			plt.plot(train_df['epoch'],
					 train_df[['train_acc', 'val_acc']])
			plt.plot([best_epoch], 
					 [best_acc], 'ro')
			plt.title('Accuracy')
			plt.legend(['Train', 'Validation', 'Best Epoch'])
			plt.xlabel('Epoch')
			
			plt.subplot(122)
			plt.plot(train_df['epoch'],
					 train_df[['train_loss', 'val_loss']])
			plt.plot([best_epoch],
					 [best_loss], 'ro')
			plt.title('Cross Entropy Loss')
			plt.legend(['Train', 'Validation', 'Best Epoch'])
			plt.xlabel('Epoch')
			
			plt.tight_layout()

		return fig


	# Calculates accuracy and cross entropy loss for all three datasets.
	# Calculates rate of false positives and negatives, i.e. confusion.
	# Will run on a GPU if possible.
	# Returns two figures: confusion bar plot and array of classified images.
	def evaluate(self, num_images:int=16, title_prefix:str=''):
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.model = self.model.to(device)
		criterion = nn.CrossEntropyLoss()
		self.model.eval()

		performance_stats = {'train_acc': [], 'val_acc': [], 'test_acc': [],
					   		 'train_loss': [], 'val_loss': [], 'test_loss': []}

		num_per_label = np.zeros(2)
		num_correct_per_label = np.zeros(2)

		for phase in ['train', 'val', 'test']:
			running_loss = 0.0
			running_corrects = 0

			for inputs, labels in tqdm(self.dataloaders[phase]):
				inputs = inputs.to(device)
				labels = labels.to(device)

				outputs = self.model(inputs)
				_, predictions = torch.max(outputs, 1)
				loss = criterion(outputs, labels)

				running_loss += loss.item()*inputs.size(0)
				running_corrects += torch.sum(predictions == labels.data)

				if phase == 'test':
					for j in range(inputs.size()[0]):
						num_per_label[labels[j]] += 1
						if predictions[j] == labels[j]:
							num_correct_per_label[labels[j]] += 1

			phase_loss = running_loss/len(self.datasets[phase])
			phase_acc = running_corrects.double()/len(self.datasets[phase])

			performance_stats[phase + '_acc'].append(phase_acc)
			performance_stats[phase + '_loss'].append(phase_loss)

			print(f'{phase} Loss: {phase_loss:.4f} Acc: {phase_acc:.4f}')

		percent_correct_per_label = 100*num_correct_per_label/num_per_label
		percent_wrong_per_label = 100*np.ones(2) - percent_correct_per_label
		with plt.style.context('seaborn'):
			fig1= plt.figure()
			plt.bar([0.25, 0.5], percent_correct_per_label, width=0.2)
			plt.bar([0.25, 0.5], percent_wrong_per_label, 
					bottom=percent_correct_per_label, width=0.2, color='r')
			plt.legend(['Correct', 'Wrong'], loc='upper right')
			plt.xticks([0.25, 0.5], ('No Fire', 'Fire'))
			plt.xlim((0, 0.75))
			plt.title(title_prefix + 'Model Confusion')
			plt.ylabel('%')

		print(f'False Positive Rate: {percent_wrong_per_label[0]}%')
		print(f'False Negative Rate: {percent_wrong_per_label[1]}%')


		# num_images = 16
		num_rows = int(np.sqrt(num_images))
		images_so_far = 0
		class_names = {0: 'No Fire', 1: 'Fire'}
		with torch.no_grad():
			with plt.style.context('seaborn'):
				fig2 = plt.figure()
				plt.title(title_prefix + 'Example Classifications')
				for i, (inputs, labels) in enumerate(self.dataloaders['test']):
					inputs = inputs.to(device)
					labels = labels.to(device)

					outputs = self.model(inputs)
					_, predictions = torch.max(outputs, 1)
					for j in range(inputs.size()[0]):
						images_so_far += 1
						ax = plt.subplot(num_rows, num_images//num_rows,
										 images_so_far)
						ax.axis('off')
						color = 'k' if predictions[j] == labels[j] else 'r'
						ax.set_title(f'{class_names[int(predictions[j])]}',
									 color=color)
						img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
						plt.imshow(img)

						if images_so_far == num_images:
							break
					if images_so_far == num_images:
							break
				plt.tight_layout()

		return fig1, fig2


	def save(self, model_path:str):
		# save the model
		if not os.path.isdir(os.path.dirname(model_path)):
			raise ValueError(f'{model_path} is not a valid path.')
		torch.save(self.model.state_dict(), model_path)


	def get_dataset_sizes(self):
		return {'train':len(self.datasets['train']),
				'val':len(self.datasets['val']),
				'test':len(self.datasets['test'])}


	class FireDataset(Dataset):
		def __init__(self, pos_data_path:str, neg_data_path:str,
					 transform=None):
			if not os.path.isdir(pos_data_path):
				raise ValueError(f'{pos_data_path} is not a valid path.')
			if not os.path.isdir(neg_data_path):
				raise ValueError(f'{neg_data_path} is not a valid path.')
			if transform is None:
				transform = transforms.Compose([transforms.ToTensor()])
			pos_fnames = [os.path.join(pos_data_path, fname)
						  for fname in os.listdir(pos_data_path)]
			neg_fnames = [os.path.join(neg_data_path, fname)
						  for fname in os.listdir(neg_data_path)]
			# self.img_dir = data_path
			self.img_names = pos_fnames + neg_fnames
			self.y = [1]*len(pos_fnames) + [0]*len(neg_fnames)
			self.transform = transform


		def __getitem__(self, index:int):
			img = Image.open(os.path.join(#self.img_dir,
										  self.img_names[index]))
			if self.transform is not None:
				img = self.transform(img)
			label = self.y[index]
			return img, label


		def __len__(self):
			return len(self.y)


		def __str__(self):
			return f'img_names: {self.img_names}\n' +\
				   f'y: {self.y}\n' +\
				   f'transform: {self.transform}\n' #+\
				   # f'img_dir: {self.img_dir}\n'


if __name__ == '__main__':
	transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
									transforms.RandomRotation(5),
									transforms.ColorJitter(brightness=0.05, 
														   contrast=0.05, 
														   saturation=0.05, 
														   hue=0.05),
									transforms.ToTensor()])
	pos_data_path = os.path.join(os.getcwd(), 'data', 'preprocessed', 'pos')
	neg_data_path = os.path.join(os.getcwd(), 'data', 'preprocessed', 'neg')
	clssfr = Classifier(pos_data_path=pos_data_path, 
						neg_data_path=neg_data_path, 
						transform = transform)
	# ret_status = clssfr.load_data()
	# if ret_status:
	# clssfr.init_model()
	fig0 = clssfr.train(num_epochs=5)
	fig1, fig2 = clssfr.evaluate()
	plt.show()