import config

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class network_CNN(nn.Module):

	def __init__(self, input_dim, output_dim):
		super(network_CNN, self).__init__()

		torch.manual_seed(51)
		torch.cuda.manual_seed_all(51)

		self.layer_convFeatExtractor = nn.Sequential(
			nn.Conv2d(input_dim, 32, kernel_size = 3, stride = 1).float(),
			nn.BatchNorm2d(32).float(),
			nn.ReLU(),
			nn.MaxPool2d(2,2).float(),

			nn.Conv2d(32, 64, kernel_size = 3, stride = 1).float(),
			nn.BatchNorm2d(64).float(),
			nn.ReLU(),
			nn.MaxPool2d(2,2).float(),

			nn.Conv2d(64, 128, kernel_size = 3, stride = 1).float(),
			nn.BatchNorm2d(128).float(),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d((2,2)).float(),
		)

		self.layer_fullyConnected = nn.Sequential(
			nn.Dropout(p=0.3),
			nn.Linear(512, 32, bias=True).float(),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(32, output_dim, bias=True).float(),
		)

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.name = 'networkCNN'
		self.input_dim = input_dim
		self.output_dim = output_dim

	def forward(self, X):

		samplesInCurrentBatch = X.shape[0]

		feat = self.layer_convFeatExtractor(X)
		y_hat = self.layer_fullyConnected(feat.reshape(samplesInCurrentBatch, -1))

		return y_hat

