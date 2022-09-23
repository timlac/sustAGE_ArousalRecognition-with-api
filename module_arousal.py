import os 
import sys
import pdb
sys.path.append('src')

import json
import pandas as pd

from numpy import argmax, mean

import config

from module_AudioTools import generate_spectrograms

from PIL import Image
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import network_arousal as netArch


def infer_arousal(X, modelPath):

	model = netArch.network_CNN(3,3)

	model.load_state_dict(torch.load(modelPath))

	model.eval()

	with torch.no_grad():
		y_hat = model(X)

	probs = F.softmax(y_hat, dim=1)
	confidence = torch.max(probs, dim=1)[0]
	pred = torch.argmax(y_hat, dim=1)

	if y_hat.shape[0] == 1:

		confidence = confidence.item()

		if pred.item() == 0: label = 'low'	
		elif pred.item() == 1: label = 'mid'	
		elif pred.item() == 2: label = 'high'	

	else:

		res_summary_df = pd.DataFrame({'Predictions':pred.numpy(), 'Confidence':confidence.numpy()})

		pred_0_df = res_summary_df[res_summary_df['Predictions'] == 0]
		pred_1_df = res_summary_df[res_summary_df['Predictions'] == 1]
		pred_2_df = res_summary_df[res_summary_df['Predictions'] == 2]

		mostVotedClass = argmax([pred_0_df.shape[0], pred_1_df.shape[0], pred_2_df.shape[0]])

		if mostVotedClass == 0: 	confidence = mean(pred_0_df['Confidence'])
		elif mostVotedClass == 1:	confidence = mean(pred_1_df['Confidence'])
		elif mostVotedClass == 2:	confidence = mean(pred_2_df['Confidence'])

		if mostVotedClass == 0: label = 'low'	
		elif mostVotedClass == 1: label = 'mid'	
		elif mostVotedClass == 2: label = 'high'	

	confidence = '{:.3f}'.format(confidence)

	return label, confidence


def load_sample(dtype, info):

	if dtype == 'openSMILEfeatures':

		X = torch.from_numpy(info.values).float()

	elif dtype == 'melSpectrograms':

		numOfMelSpectrograms = len(info)
		X = torch.zeros(numOfMelSpectrograms,3,config._spectrogramSize,config._spectrogramSize)

		for ID, sample in enumerate(info):
			img = Image.open(sample)
			X[ID,:,:,:] = TF.to_tensor(img)[:3,:,:].float()

	return X


def arousalInference(audioPath, featuresPath, storagePath):

	spectrogramFiles = generate_spectrograms(audioPath, featuresPath)
	X = load_sample('melSpectrograms', spectrogramFiles)

	modelPath = os.path.join(storagePath, 'EnglishSER_arousal.pth')

	AS_label, AS_confidence = infer_arousal(X, modelPath)

	for file in os.listdir(featuresPath): os.remove(os.path.join(featuresPath, file))

	return AS_label, AS_confidence


def API(audioPath):

	storagePath = 'src/Models_Arousal'

	featuresPath = 'FeaturesFolder'
	if not os.path.exists(featuresPath): os.mkdir(featuresPath)

	arousal_label, arousal_confidence = arousalInference(audioPath, featuresPath, storagePath)

	return arousal_label, arousal_confidence
