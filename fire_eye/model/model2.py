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
from tqdm import tqdm


class FireDataset(Dataset):


	def __init__(self, data_path, pos_dir='pos', neg_dir='neg', transform=None):
		pos_fnames = [os.path.join(data_path, pos_dir, fname)
					  for fname in os.listdir(os.path.join(data_path, pos_dir))]
		neg_fnames = [os.path.join(data_path, neg_dir, fname)
					  for fname in os.listdir(os.path.join(data_path, neg_dir))]
		self.img_dir = data_path  # fire_eye/data
		self.img_names = pos_fnames + neg_fnames
		self.y = [1]*len(pos_fnames) + [0]*len(neg_fnames)
		self.transform = transform


	def __getitem__(self, index):
		img = cv2.imread(os.path.join(self.img_dir,
									  self.img_names[index]))
		if self.transform is not None:
		    img = self.transform(img)
		label = self.y[index]
		return img, label


	def __len__(self):
		return len(self.img_names)


	def __str__(self):
		return f'img_dir: {self.img_dir}\n' +\
			   f'img_names: {self.img_names}\n' +\
			   f'y: {self.y}\n' +\
			   f'transform: {self.transform}\n'


def load_data():
	# already divides pixels by 255 internally
	# any data augmentation should occur here with transforms
	custom_transform = transforms.Compose([transforms.ToTensor()])

	dataset = FireDataset(data_path=os.path.join(os.getcwd(), 'data'), transform=custom_transform)
	dataset_split = [int(percent*len(dataset)) for percent in [0.6, 0.2, 0.2]]
	train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(dataset, dataset_split)

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

	datasets = {'train': train_dataset, 'test': test_dataset, 'val': valid_dataset}
	dataset_sizes = {'train': dataset_split[0], 'test': dataset_split[1], 'val': dataset_split[2]}
	dataloaders = {'train': train_loader, 'test': test_loader, 'val': valid_loader}

	return datasets, dataset_sizes, dataloaders


def init_model(num_classes=2):
	model = models.resnet18(pretrained=True)
	num_in_fc = model.fc.in_features
	model.fc = nn.Linear(num_in_fc, num_classes)
	return model


def train_model(model, dataset_sizes, dataloaders, num_epochs=25):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	# Observe that all parameters are being optimized
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	# Decay LR by a factor of 0.1 every 7 epochs
	scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	start_time = time.time()
	best_model = copy.deepcopy(model.state_dict())
	best_acc = 0.0
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
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			for inputs, labels in tqdm(dataloaders[phase]):
				inputs = inputs.to(device)
				labels = labels.to(device)

				optimizer.zero_grad()

				# Forward
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, predictions = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# Backward
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# Statistics
				running_loss += loss.item()*inputs.size(0)
				running_corrects += torch.sum(predictions == labels.data).item()

			if phase == 'train':
				scheduler.step()

			epoch_loss = running_loss/dataset_sizes[phase]
			epoch_acc = running_corrects/dataset_sizes[phase]

			print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

			# deepcopy/save the model if it is current best
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model = copy.deepcopy(model.state_dict())
				best_epoch = epoch

			# store epoch statistics for plotting later
			train_stats[phase + '_acc'].append(epoch_acc)
			train_stats[phase + '_loss'].append(epoch_loss)
		
		train_stats['epoch'].append(epoch)
		print()

	time_elapsed = time.time() - start_time
	print()
	print(f'Training complete in {(time_elapsed//60):.0f}m {(time_elapsed%60):.0f}s')
	print(f'Best val Acc: {best_acc:4f}')

	# load best model
	model.load_state_dict(best_model)
	# put training stats into a dataframe
	train_df = pd.DataFrame(train_stats)
	return model, train_df, best_epoch


def evaluate_model(model, dataset_sizes, dataloaders):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	model.eval()

	num_per_label = np.zeros(2)
	num_correct_per_label = np.zeros(2)

	for phase in ['train', 'val', 'test']:
		running_loss = 0.0
		running_corrects = 0

		for inputs, labels in tqdm(dataloaders[phase]):
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs)
			_, predictions = torch.max(outputs, 1)
			loss = criterion(outputs, labels)

			running_loss += loss.item()*inputs.size(0)
			running_corrects += torch.sum(predictions == labels.data)

			if phase == 'test':
				for j in range(inputs.size()[0]):
					num_per_label[labels[j]] += 1
					if predictions[j] == labels[j]:
						num_correct_per_label[labels[j]] += 1

		phase_loss = running_loss/dataset_sizes[phase]
		phase_acc = running_corrects.double()/dataset_sizes[phase]

		print(f'{phase} Loss: {phase_loss:.4f} Acc: {phase_acc:.4f}')

	percent_correct_per_label = 100*num_correct_per_label/num_per_label
	percent_wrong_per_label = 100*np.ones(2) - percent_correct_per_label
	with plt.style.context('seaborn'):
		fig= plt.figure()
		plt.bar([0.25, 0.5], percent_correct_per_label, width=0.2)
		plt.bar([0.25, 0.5], percent_wrong_per_label, bottom=percent_correct_per_label,
				width=0.2, color='r')
		plt.legend(['Correct', 'Wrong'], loc='upper right')
		plt.xticks([0.25, 0.5], ('No Fire', 'Fire'))
		plt.xlim((0, 0.75))
		plt.title('Model Confusion')
		plt.ylabel('%')
		plt.show()

	return fig


def plot_results(train_df, best_epoch):
	fig = plt.figure()
	with plt.style.context('seaborn'):
		plt.subplot(121)
		plt.plot(train_df['epoch'], train_df[['train_acc', 'val_acc']])
		plt.plot([best_epoch], [train_df['val_acc'].loc[best_epoch]], 'ro')
		plt.title('Accuracy')
		plt.legend(['Train', 'Validation', 'Best Epoch'])
		plt.xlabel('Epoch')
		
		plt.subplot(122)
		plt.plot(train_df['epoch'], train_df[['train_loss', 'val_loss']])
		plt.plot([best_epoch], [train_df['val_loss'].loc[best_epoch]], 'ro')
		plt.title('Cross Entropy Loss')
		plt.legend(['Train', 'Validation', 'Best Epoch'])
		plt.xlabel('Epoch')
		
		plt.tight_layout()
		plt.show()
	return fig


def visualize_model(model, dataloaders):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	model.eval()

	num_images = 16
	num_rows = 4
	images_so_far = 0
	class_names = {0: 'No Fire', 1: 'Fire'}
	with torch.no_grad():
		with plt.style.context('seaborn'):
			fig = plt.figure()
			for i, (inputs, labels) in enumerate(dataloaders['test']):
				inputs = inputs.to(device)
				labels = labels.to(device)

				outputs = model(inputs)
				_, predictions = torch.max(outputs, 1)
				for j in range(inputs.size()[0]):
					images_so_far += 1
					ax = plt.subplot(num_rows, num_images//num_rows, images_so_far)
					ax.axis('off')
					color = 'k' if predictions[j] == labels[j] else 'r'
					ax.set_title(f'{class_names[int(predictions[j])]}', color=color)
					plt.imshow(inputs.cpu().data[j].numpy().transpose((1, 2, 0)))

					if images_so_far == num_images:
						break
				if images_so_far == num_images:
						break
			plt.tight_layout()
			plt.show()
	return fig



