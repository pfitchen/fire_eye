import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from fire_eye.model import loader
from fire_eye.model.loader import DEVICE
from fire_eye.statistics import statistics


DATASETS = None
DATALOADERS = None
DATASET_SIZES = None
CLASS_NAMES = None

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			for inputs, labels in DATALOADERS[phase]:
				inputs = inputs.to(DEVICE)
				labels = labels.to(DEVICE)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
			if phase == 'train':
				scheduler.step()

			epoch_loss = running_loss / DATASET_SIZES[phase]
			epoch_acc = running_corrects.double() / DATASET_SIZES[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))

			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean, std = statistics.calc_pixel_value_statistics()
    mean = mean.flatten()/255
    std = std.flatten()/255
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    fig = plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated
    return fig


def visualize_model(model, num_images=6):
	was_training = model.training
	model.eval()
	images_so_far = 0
	fig = plt.figure()

	with torch.no_grad():
		for i, (inputs, labels) in enumerate(DATALOADERS['val']):
			inputs = inputs.to(DEVICE)
			labels = labels.to(DEVICE)

			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			print(type(preds[0]))
			for j in range(inputs.size()[0]):
				images_so_far += 1
				ax = plt.subplot(num_images//2, 2, images_so_far)
				ax.axis('off')
				ax.set_title('predicted: {}'.format(CLASS_NAMES[int(preds[j])]))
				imshow(inputs.cpu().data[j])

				if images_so_far == num_images:
					model.train(mode=was_training)
					return
		model.train(mode=was_training)


def model_run():
	global DATASETS, DATALOADERS, DATASET_SIZES, CLASS_NAMES
	DATASETS, DATALOADERS = loader.load_data()
	DATASET_SIZES = {x: len(DATASETS[x]) for x in ['train', 'test', 'val']}
	CLASS_NAMES = {0: 'NoFire', 1: 'Fire'}

	model_ft = models.resnet18(pretrained=True)
	num_ftrs = model_ft.fc.in_features
	# Here the size of each output sample is set to 2.
	# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(CLASS_NAMES)).
	model_ft.fc = nn.Linear(num_ftrs, 2)

	model_ft = model_ft.to(DEVICE)

	criterion = nn.CrossEntropyLoss()

	# Observe that all parameters are being optimized
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

	# Decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

	model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
	                       num_epochs=2)

	visualize_model(model_ft)

	# Get a batch of training data
	inputs, classes = next(iter(DATALOADERS['train']))
	# Make a grid from batch
	out = torchvision.utils.make_grid(inputs)
	labels = [CLASS_NAMES[int(x)] for x in classes]
	plt.figure()
	fig = imshow(out, title=labels)
	print(type(fig))
