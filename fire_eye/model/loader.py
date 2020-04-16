import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


def load_data(dataset_percents=[0.6, 0.2, 0.2]):
	# already divides pixels by 255 internally
	custom_transform = transforms.Compose([transforms.ToTensor()])

	dataset = FireDataset(data_path=os.path.join(os.getcwd(), 'data'), transform=custom_transform)
	dataset_split = [int(percent*len(dataset)) for percent in dataset_percents]
	train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(dataset, dataset_split)

	train_loader = DataLoader(dataset=train_dataset,
							  batch_size=32,
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
	dataloaders = {'train': train_loader, 'test': test_loader, 'val': valid_loader}

	return datasets, dataloaders


