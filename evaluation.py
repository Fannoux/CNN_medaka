# import the necessary packages
from pyimagesearch.classifier import Larval_MLClassifier
from pyimagesearch.ziramutils import get_dataloader, ZiramDataset
from pyimagesearch.metrics import MetricRecorder
from torchvision import transforms
from torch.nn import Softmax
from torch import nn
import matplotlib.pyplot as plt
import argparse
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os
import torchmetrics
import yaml
from pyimagesearch.config import params_fromYAML
import mlflow
import datetime


def main(yaml_file, model_path=None):

####################### PARAMS settings #####################################

	# Load parameters from YAML (same as train_metrics.py)
	train_kwargs, model_kwargs, hyperparams, params, metrics_kwargs = params_fromYAML(yaml_file)
	numClasses = train_kwargs.pop('num_class')

	# Get configuration parameters
	DEVICE = params.get('DEVICE', 'cuda')
	OUTPUT_PATH = params['OUTPUT_PATH']
	run_name = params['NAME'] + '_evaluation'
	exp_name = params.get('EXP_NAME', 'evaluation')
	MLFLOW = metrics_kwargs.get('MLFLOW', False)

	# Default to the model saved during training if not specified
	if model_path is None:
		model_path = os.path.join(OUTPUT_PATH, 'model.pth')
		print(f'[INFO] Using default model from training: {model_path}')
	else:
		print(f'[INFO] Using specified model: {model_path}')

	print(f'[INFO] Evaluation run: {run_name}')
	print(f'[INFO] Output path: {OUTPUT_PATH}\n')

	# Verify model exists
	assert os.path.exists(model_path), f'Model file not found: {model_path}'

	# Initialize MLflow if enabled
	ABS_PATH = os.path.dirname(os.path.abspath(__file__))
	if MLFLOW:
		mlflow.set_tracking_uri(metrics_kwargs.get('MLFLOW_OUT', './mlruns'))
		mlflow.start_run(run_name=run_name)
		mlflow.log_artifact(yaml_file)
		mlflow.log_param('model_path', model_path)
		for key, item in {**train_kwargs, **hyperparams, **params}.items():
			mlflow.log_param(key, item)

#################### Initialisation of evaluation ################################

	# create the test dataset
	testDataset = ZiramDataset(
		dataset_path=train_kwargs['dataset_path'],
		mode='test',
		im_size=hyperparams['IMAGE_SIZE'],
		mean=model_kwargs['mean'],
		std=model_kwargs['std'],
		filename=train_kwargs['filename'],
		label=train_kwargs['label'],
		augment=False
	)

	# initialize the test data loader
	testLoader = get_dataloader(testDataset, batch_size=hyperparams.get('PRED_BATCH_SIZE', 32))

	# build the custom model
	model = Larval_MLClassifier(numClasses=numClasses, **model_kwargs).to(DEVICE)
	model.load_state_dict(torch.load(model_path))  # load the model state

	# initialize loss function (criterion)
	criterion_classif = torch.nn.CrossEntropyLoss()

	# initialize test data loss
	eval_metrics = metrics_kwargs.get('EVALUATION', [])
	test_metrics = MetricRecorder(num_class=numClasses, prefix='Eval', metric_ls=eval_metrics)
	# testCorrect, totalTestLoss  = 0, 0
	# testAuroc = torchmetrics.AUROC(task="multiclass", num_classes=run_params['NUM_CLASS'])
	# soft = Softmax()

	# switch off autograd
	res_df = pd.DataFrame()
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()
		tqdm_object = tqdm(testLoader, total=len(testLoader), desc='Evaluation', colour='magenta')

		# loop over the testing set
		# tot_encoding, tot_logits, tot_coef = np.empty([run_params['PRED_BATCH_SIZE'], 1000]), np.empty(shape=[run_params['PRED_BATCH_SIZE'], run_params['NUM_CLASS']]), np.empty([run_params['PRED_BATCH_SIZE'], 1])
		tot_encoding = np.empty([0, 1000])
		tot_pth = []
		for batch in tqdm_object:

			# send the input to the device
			image_batch = batch['image'].to(DEVICE)
			target_class = batch['label_categorical'].to(DEVICE)
         	
			# Compute loss
			out_dict = model(image_batch)
			# make the predictions and calculate the evaluation loss
			logits_class = out_dict['logits']

			#### TEST encodings
			results_batch = pd.DataFrame({key:item for key, item in batch.items() if key in ['img_path', 'label_categorical', 'label_regression']})
			results_batch[['logits_' + str(n) for n in range(numClasses)]] = logits_class.cpu().numpy()
			res_df = res_df.append(results_batch)
			# print(results_batch.head().to_markdown())

			batch_encoding = out_dict['encoding']
			# print(len(batch_encoding))
			tot_encoding = np.append(tot_encoding, batch_encoding, axis=0)
			tot_pth += list(batch['img_path'])
			# tot_logits = np.append(tot_logits, logits_class, axis=0)
			# tot_coef = np.append(tot_coef, coefficient_reg, axis=0)

			classif_loss = criterion_classif(logits_class, target_class)
			test_metrics.metric_update(logits=logits_class, targets=target_class, loss=classif_loss)
			# totalTestLoss += loss.item()
			# output logits through the softmax layer to get output

			mean_test_metrics = test_metrics.compute_array()
			# print(mean_test_metrics)

			# predictions, and calculate the number of correct predictions
			# testCorrect += (logit.argmax(dim=-1) == target_batch).sum().item()
			# testAuroc.update(torch.nn.functional.softmax(logit, dim=-1), target_batch)
			# testAuroc_value = testAuroc.compute().cpu().numpy()		
			# tqdm_object.set_postfix(TestAuroc=testAuroc_value.mean())
			# if config.MLFLOW:
			# 	mlflow.log_metric(key='totalTestLoss', value=totalTestLoss, step=batch)
			# 	mlflow.log_metric(key='testCorrect', value=testCorrect, step=batch)
			# 	for n in range(run_params.NUM_CLASS): 
			# 		mlflow.log_metric(key=f'testAuroc_{n}', value=float(testAuroc_value[n]), step=batch)
	
	print('\n\n Results Summary')
	print(res_df.shape)
	res_df.to_csv(os.path.join(OUTPUT_PATH, 'results_evaluation.csv'), index=False)

	print('\n\n Encodings Summary')
	print(tot_encoding.shape)
	print(len(tot_pth))
	encoding_df = pd.DataFrame(tot_encoding, index=tot_pth)
	encoding_df.to_csv(os.path.join(OUTPUT_PATH, 'encodings_evaluation.csv'))
	sys.exit()

	# TODO: ADD extraction of logits / coeff (in a table?) and encoddings (how ?) + saving		
	
	# print test data accuracy
	# print('\n[INFO] Final Accuracy', testCorrect/len(testDataset), 'Auroc', testAuroc_value.mean())
	
	# initialize iterable variable
	sweeper = iter(testLoader)
	
	# grab a batch of test data
	batch = next(sweeper)
	(images, labels) = (batch['image'], batch['label_categorical'])
	
	# initialize a figure
	from math import ceil
	fig, axs = plt.subplots(ceil(len(images)/6), 6, figsize=(50,50))

	# calculate the inverse mean and standard deviation to define our denormalization transform
	invMean = [-m/s for (m, s) in zip(model_kwargs['mean'], model_kwargs['std'])]
	invStd = [1/s for s in model_kwargs['std']]
	deNormalize = transforms.Normalize(mean=invMean, std=invStd)

	# switch off autograd
	with torch.no_grad():
		# send the images to the device
		images = images.to(DEVICE)
		# make the predictions
		preds = model(images)
		# loop over all the batch
		for i, (image, ax) in enumerate(zip(images, axs.flatten())):
			# grab the image, de-normalize it, scale the raw pixel
			# intensities to the range [0, 255], and change the channel
			# ordering from channels first tp channels last
			# image = images[i]
			image = deNormalize(image).cpu().numpy()
	

			# grab the ground truth label
			idx = labels[i].cpu().numpy()

			# grab the predicted label
			pred = preds[i].argmax().cpu().numpy()
			result = idx == pred
			# add the results and image to the plot
			info = "Result: {} - (Ground Truth: {}, Predicted: {})".format(result, idx, pred)
			# print(info)
			image_perm = (image.permute((1, 2, 0)) / torch.max(torch.abs(image))+1)/2
			ax.imshow(image_perm, cmap='Greys')
			ax.title.set_text(info)
			# ax.axis("off")
		[axi.axis('off') for axi in axs.ravel()]
		
		# # show the plot
	plt.tight_layout()
	fig.suptitle(f'Evaluation Results - {run_name}')
	fig.savefig(os.path.join(OUTPUT_PATH, 'evaluation_predictions.png'))

	print('[INFO] Results saved!')
	if MLFLOW:
		mlflow.log_artifact(os.path.join(OUTPUT_PATH, 'results_evaluation.csv'))
		mlflow.log_artifact(os.path.join(OUTPUT_PATH, 'encodings_evaluation.csv'))
		mlflow.log_artifact(os.path.join(OUTPUT_PATH, 'evaluation_predictions.png'))
		mlflow.end_run()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Evaluation of CNN on test dataset using YAML config")
	parser.add_argument("-Y", "--yaml_file", required=True, help="Path to YAML configuration file (same as used for training)")
	parser.add_argument("-M", "--model_path", default=None, help="Path to trained model state dict (.pth file). Defaults to OUTPUT_PATH/model.pth from YAML config.")
	args = parser.parse_args()
	main(**vars(args))