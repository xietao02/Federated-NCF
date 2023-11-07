# ------------------------------------------------- #
# Author: xietao                					#
# Repo: https://github.com/xietao02/Federated-NCF/  #
# ------------------------------------------------- #


import os
import time
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import logging
import flwr as fl
from collections import OrderedDict

import model
import config 
import util
import data_utils
import evaluate

fl.common.logger.FLOWER_LOGGER.setLevel(logging.CRITICAL)

parser = argparse.ArgumentParser(description="Client")
parser.add_argument("--cid", 
	type=int,
	help="Client ID. Distribute unique dataset by cid")
parser.add_argument("--seed", 
	type=int, 
	default=516, 
	help="Seed")
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.2,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=128,
	help="batch size for training")
parser.add_argument("--num_workers", 
	type=int,
	default=0,  
	help="num_workers")
parser.add_argument("--epochs", 
	type=int,
	default=1,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--layers",
    nargs='+', 
    default=[64,32,16,8],
    help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="Number of negative samples for training set")
parser.add_argument("--num_ng_test", 
	type=int,
	default=100, 
	help="Number of negative samples for test set")
parser.add_argument("--out", 
	default=True,
	help="save model or not")

# set device and parameters
args = parser.parse_args()
if __name__ == "__main__":
    cid = args.cid
    if cid is None:
        print("Client ID not provided. Use --cid to specify a client.")
        exit(1)
    elif not (0 < cid <= config.NUM_CLIENTS):
        raise ValueError("Invalid client ID")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# seed for Reproducibility
util.seed_everything(args.seed)

# load data
ml_1m = pd.read_csv(
	"../data/ml-1m/ratings.dat", 
	sep="::", 
	names = ['user_id', 'item_id', 'rating', 'timestamp'], 
	engine='python')

# set the num_users, items
num_users = ml_1m['user_id'].nunique()+1
num_items = ml_1m['item_id'].nunique()+1

# construct the train and test datasets
data = data_utils.NCF_Data(args, ml_1m, config.NUM_CLIENTS)

train_loader =data.get_train_instance()
train_dataset_length = len(train_loader.dataset)
print(f"[Client {args.cid}] Length of training dataset: {train_dataset_length}")

test_loader =data.get_test_instance()
test_dataset_length = len(test_loader.dataset)
print(f"[Client {args.cid}] Length of testing dataset: {test_dataset_length}")

# set model and loss, optimizer
model = model.NeuMF(args, num_users, num_items)
model = model.to(device)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
total_epoch = 0

def train(model, train_loader, epochs):
	global total_epoch
	for epoch in range(1, epochs + 1):
		total_epoch = total_epoch + 1
		model.train() # Enable dropout (if have).
		start_time = time.time()

		for user, item, label in train_loader:
			user = user.to(device)
			item = item.to(device)
			label = label.to(device)

			optimizer.zero_grad()
			prediction = model(user, item)
			if prediction.size() != label.size():
				continue

			loss = loss_function(prediction, label)
			loss.backward()
			optimizer.step()

		elapsed_time = time.time() - start_time
		print(f"[Client {args.cid}] Time elapse of epoch {epoch:03d}/{total_epoch} is: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")


def test(model, testloader):
	model.eval()
	HR, NDCG = evaluate.metrics(model, test_loader, args.top_k, device)
	# Calculate the average loss for the test set
	test_loss = 0.0
	num_batches = 0
	for user, item, label in test_loader: 
		user = user.to(device)
		item = item.to(device)
		label = label.to(device)

		with torch.no_grad():
			prediction = model(user, item)
			test_loss += loss_function(prediction, label).item()
			num_batches += 1
	test_loss /= num_batches  # Calculate the average test loss
	torch.cuda.empty_cache()
	print(f"[Client {args.cid}] ", "HR: {:.3f}\tNDCG: {:.3f}\tTest Loss: {:.6f}".format(np.mean(HR), np.mean(NDCG), test_loss))
	return test_loss, len(test_loader.dataset), {"HR": float(np.mean(HR)), "NDCG": float(np.mean(NDCG))}

# Create a client
class FlowerClient(fl.client.NumPyClient):
	def __init__(self, cid, model, trainloader, valloader):
		self.cid = cid
		self.model = model
		self.trainloader = trainloader
		self.valloader = valloader
		# print(f"[Client {self.cid}] initiates successfully")
    
	def get_parameters(self, config):
		# print(f"[Client {self.cid}] receives get instructions")
		return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

	def set_parameters(self, parameters):
		# print(f"[Client {self.cid}] receives set instructions")
		params_dict = zip(self.model.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.model.load_state_dict(state_dict, strict=True)

	def fit(self, parameters, config):
		# print(f"[Client {self.cid}] receives fit instruction")
		self.set_parameters(parameters)
		train(self.model, self.trainloader, args.epochs)
		return self.get_parameters(config={}), len(self.trainloader.dataset), {}

	def evaluate(self, parameters, config):
		# print(f"[Client {self.cid}] receives evaluate instruction")
		self.set_parameters(parameters)
		loss, length, metrics = test(self.model, self.valloader)
		return loss, length, metrics

# Start client
fl.client.start_numpy_client(
	server_address="127.0.0.1:8080",
	client=FlowerClient(args.cid, model, train_loader, test_loader),
)