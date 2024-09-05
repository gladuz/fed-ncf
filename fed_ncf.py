import os, sys
# from google.colab import drive
# drive.mount('/content/drive')
# nb_path = '/content/notebooks'
# os.symlink('/content/drive/My Drive/Colab Notebooks', nb_path)
# sys.path.insert(0,nb_path)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import syft as sy
import time
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from syft.frameworks.torch.fl import utils
import torch
import torch.nn as nn
import torch.nn.functional as f
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

from tensorboardX import SummaryWriter
writer = SummaryWriter()

# Initialize the PySyft hook. This is a crucial step in setting up the federated learning environment.
# PySyft extends PyTorch to enable privacy-preserving machine learning and federated learning.
hook = sy.TorchHook(torch)

# Define the hyperparameters and configuration for our federated learning setup.
# These parameters control various aspects of the training process and model architecture.
class Arguments():
    def __init__(self):
        self.seed = 50  # Set a random seed for reproducibility
        self.device = "cpu"  # Use CPU for computations (can be changed to "cuda" for GPU)
        self.batch_size = 512  # Number of samples per batch for training
        self.test_batch_size = 512  # Number of samples per batch for testing
        self.learning_rate = 0.01  # Learning rate for the optimizer
        self.epochs = 15  # Number of times to iterate over the entire dataset
        self.embed_size = 30  # Size of the embedding vectors for users and items
        self.no_workers = 100  # Number of federated workers (simulated clients)
        self.test_size = 0.2  # Proportion of data to use for testing
        self.federate_after_n_batches = 50  # Number of batches to process before aggregating models
        self.weight_decay = 1e-5  # L2 regularization term
        self.top_k = 10  # Top K items to consider for recommendation metrics

args = Arguments()

# Set the random seed for reproducibility across runs
random.seed(args.seed)

# Define a custom Dataset class for handling user-item interactions
# This class will be used to create both the federated and centralized datasets
class UserItemRatingDataset(Dataset):
    """
    This custom Dataset class is designed to handle user-item-rating triples.
    It's a crucial component in our federated learning setup, as it allows us to
    easily distribute the data across multiple workers.
    """
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        Initialize the dataset with user, item, and rating tensors.
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
 
    def __getitem__(self, index):
        """
        Retrieve a single (user, item, rating) triple from the dataset.
        This method is called by DataLoader to construct batches.
        """
        data_tensor = torch.stack([self.user_tensor[index], self.item_tensor[index]], dim=0)
        return data_tensor, self.target_tensor[index]
 
    def __len__(self):
        """
        Return the total number of user-item-rating triples in the dataset.
        """
        return len(self.user_tensor)

# Load and preprocess the MovieLens dataset
# We're using the MovieLens 100K dataset, which contains 100,000 ratings from 943 users on 1682 movies
rs_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
train_data = pd.read_csv('ua.base', sep='\t', names=rs_cols, engine='python')

# Create mappings between original user/movie IDs and encoded versions
# This step is necessary to convert the original IDs into contiguous integer indices,
# which is required for the embedding layers in our neural network
user_ids = train_data["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = train_data["movie_id"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

# Apply the mappings to our dataset
train_data["user"] = train_data["user_id"].map(user2user_encoded)
train_data["movie"] = train_data["movie_id"].map(movie2movie_encoded)

# Get the number of unique users and movies
# This information will be used to define the size of our embedding layers
num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)

# Convert ratings to float and find the min and max ratings
# This will be used later for scaling our model's output
train_data["rating"] = train_data["rating"].values.astype(np.float32)
max_rating = train_data["rating"].max()
min_rating = train_data["rating"].min()

# Split the data into training and testing sets
# In a real-world federated learning scenario, this split would typically happen on each client's local data
# Here, we're simulating that process with a central dataset
train_single_epoch, test = train_test_split(train_data, test_size=args.test_size, random_state=42, shuffle=True)

# Define the Neural Collaborative Filtering (NCF) model
class NeuralCF(nn.Module):
    """
    This class defines the Neural Collaborative Filtering model.
    NCF combines matrix factorization and multi-layer perceptron to model user-item interactions.
    In our federated learning setup, this model will be trained locally on each worker and then aggregated.
    """
    def __init__(self, num_users=num_users, num_movies=num_movies, embedding_size=args.embed_size):
        super(NeuralCF, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        
        # Define embedding layers for users and movies
        # We have separate embeddings for the matrix factorization (MF) and multi-layer perceptron (MLP) parts
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_size)
        self.movie_embedding_mlp = nn.Embedding(num_movies, embedding_size)
        self.user_embedding_mf = nn.Embedding(num_users, embedding_size)
        self.movie_embedding_mf = nn.Embedding(num_movies, embedding_size)

        # Initialize the embedding weights
        # Proper initialization is crucial for model convergence
        self.user_embedding_mlp.weight.data.uniform_(-0.1, 0.1)
        self.movie_embedding_mlp.weight.data.uniform_(-0.1, 0.1)
        self.user_embedding_mf.weight.data.uniform_(-0.1, 0.1)
        self.movie_embedding_mf.weight.data.uniform_(-0.1, 0.1)

        # Define the architecture of the multi-layer perceptron
        config = {}
        config['layers'] = [60, 32, 16, 8]  # Sizes of hidden layers

        # Create the MLP layers
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # Final output layer
        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + self.embedding_size, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, train_data):
        """
        Define the forward pass of the NCF model.
        This method will be called during both training and inference.
        """
        user_indices = train_data[:, 0]
        movie_indices = train_data[:, 1]
        
        # Get embeddings for users and movies
        user_embedding_mlp = self.user_embedding_mlp(user_indices)
        movie_embedding_mlp = self.movie_embedding_mlp(movie_indices)
        user_embedding_mf = self.user_embedding_mf(user_indices)
        movie_embedding_mf = self.movie_embedding_mf(movie_indices)

        # Concatenate user and movie embeddings for MLP
        mlp_vector = torch.cat([user_embedding_mlp, movie_embedding_mlp], dim=-1)
        
        # Element-wise product for matrix factorization part
        mf_vector = torch.mul(user_embedding_mf, movie_embedding_mf)

        # Pass through MLP layers with ReLU activation and dropout
        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = F.dropout(mlp_vector, p=0.2)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        # Concatenate MLP and MF vectors
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        
        # Final prediction
        logits = self.affine_output(vector)
        
        # Scale the output to match the original rating scale
        logits = logits * (max_rating - min_rating + 1) + min_rating - 1
        return logits.view(-1)

# Define a custom RMSE loss function
# RMSE (Root Mean Square Error) is a common metric for recommender systems
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self, yhat, y, reduction='mean'):
        loss = torch.sqrt(f.mse_loss(yhat, y, reduction=reduction) + self.eps)
        return loss

RMSE_Loss = RMSELoss()

# Function to create and initialize virtual workers for federated learning
def startWorkers(no_workers):
    """
    This function creates virtual workers to simulate a federated learning environment.
    In a real-world scenario, these would represent actual client devices or institutions.
    """
    for i in range(no_workers):
        exec(f"w{i}=sy.VirtualWorker(hook, id='w{i}')")
    workers = []
    for i in range(no_workers):
        workers.append(eval(f"w{i}"))
    return workers

# Create the specified number of virtual workers
workers = startWorkers(args.no_workers)

# Create a federated dataset and dataloader
# This is a crucial step in setting up the federated learning environment
# The data is distributed across the virtual workers, simulating a real-world scenario
# where each client has its own local dataset
federated_train_loader = sy.FederatedDataLoader(
    UserItemRatingDataset(
        torch.LongTensor(train_single_epoch["user"].values),
        torch.LongTensor(train_single_epoch["movie"].values),
        torch.FloatTensor(train_single_epoch["rating"].values)
    ).federate(tuple(workers)),
    batch_size=args.batch_size,
    shuffle=True,
    iter_per_worker=True
)

# Create a regular (non-federated) dataloader for the test set
# This will be used to evaluate the model's performance after each round of federated learning
test_loader = torch.utils.data.DataLoader(
    UserItemRatingDataset(
        torch.LongTensor(test["user"].values),
        torch.LongTensor(test["movie"].values),
        torch.FloatTensor(test["rating"].values)
    ),
    batch_size=args.test_batch_size,
    shuffle=True
)

def get_next_batches(fdataloader: sy.FederatedDataLoader, nr_batches: int):
    # This function is a crucial part of our federated learning setup. In a real-world scenario,
    # data would be naturally distributed across different clients (e.g., user devices or institutions).
    # Here, we're simulating that distribution by grouping batches of data for each of our virtual workers.
    
    # Initialize a dictionary to hold batches for each worker
    batches = {}
    for worker_id in fdataloader.workers:
        worker = fdataloader.federated_dataset.datasets[worker_id].location
        batches[worker] = []
    
    # Attempt to retrieve the specified number of batches for each worker
    try:
        for i in range(nr_batches):
            next_batches = next(fdataloader)
            for worker in next_batches:
                batches[worker].append(next_batches[worker])
    except StopIteration:
        # If we've run out of data, we simply stop adding batches
        # This could happen if some workers have less data than others
        pass
    
    return batches


def train_worker(worker, batches, model_in, device, lr):
    # This function represents the local training process that would occur on each client
    # in a federated learning system. In our simulation, each "worker" is a virtual entity,
    # but in a real-world scenario, this could be running on a user's device or in a separate institution.
    
    # We create a copy of the input model for this worker. This is important in federated learning
    # as each worker needs to train on its local data without directly affecting the global model.
    model = model_in.copy()
    
    # Set up the optimizer. We're using Adam with weight decay for regularization.
    # The choice of optimizer can significantly impact the model's convergence and performance.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    
    # Prepare the model for training and send it to the worker
    model.train()
    model.send(worker)
    
    loss_local = False
    LOG_INTERVAL = 25

    # Iterate through the batches assigned to this worker
    for batch_idx, (data, target) in enumerate(batches):
        # Move data to the appropriate device (CPU/GPU)
        data, target = data.to(device), target.to(device)
        
        # Standard PyTorch training loop:
        # 1. Zero the parameter gradients
        # 2. Forward pass
        # 3. Compute loss
        # 4. Backward pass
        # 5. Optimize
        optimizer.zero_grad()
        output = model(data)
        loss = RMSE_Loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Periodically print the training progress
        if batch_idx % LOG_INTERVAL == 0:
            # Note that we need to use .get() to retrieve the loss value
            # This is because the loss is computed on the worker, not locally
            loss = loss.get()
            loss_local = True
            print(
                "Train Worker {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    worker.id,
                    batch_idx,
                    len(batches),
                    100.0 * batch_idx / len(batches),
                    loss.item(),
                )
            )

    # Ensure we get the final loss if it wasn't retrieved in the loop
    if not loss_local:
        loss = loss.get()
    
    # Retrieve the trained model from the worker
    model.get()
    
    return model, loss


def train_single_epoch(model, device, federated_train_loader, lr, federate_after_n_batches, abort_after_one=False):
    # This function orchestrates the federated learning process for a single epoch.
    # It coordinates the distribution of data to workers, local training on each worker,
    # and the aggregation of worker models to update the global model.
    
    model.train()

    nr_batches = federate_after_n_batches

    # These dictionaries will store the locally trained models and their corresponding losses
    models = {}
    loss_values = {}

    # Initialize the federated data loader
    iter(federated_train_loader)
    
    # Get the first set of batches for each worker
    batches = get_next_batches(federated_train_loader, nr_batches)
    counter = 0

    while True:
        print(f"Starting training round, batches [{counter}, {counter + nr_batches}]")
        data_for_all_workers = True
        
        # Train the model locally on each worker
        for worker in batches:
            curr_batches = batches[worker]
            if curr_batches:
                models[worker], loss_values[worker] = train_worker(
                    worker, curr_batches, model, device, lr
                )
            else:
                # If a worker has no more data, we'll stop after this round
                data_for_all_workers = False
        
        counter += nr_batches
        
        if not data_for_all_workers:
            print("At least one worker ran out of data, stopping.")
            break
        
        # This is where the "federated" part of federated learning happens.
        # We aggregate the locally trained models to update our global model.
        # The federated_avg function computes a weighted average of the model parameters.
        model = utils.federated_avg(models)
        
        # Prepare the next set of batches
        batches = get_next_batches(federated_train_loader, nr_batches)
        
        if abort_after_one:
            # This is useful for debugging - we can check what happens in a single round
            break
    
    return model

def metrics(model, test_loader, top_k):
    # This function evaluates the model's performance on the test set.
    # In a recommendation system, we're often interested in how well our model
    # can predict user ratings. Here, we use RMSE (Root Mean Square Error) as our metric.
    
    model.eval()
    losses = []
    
    # Disable gradient computation for efficiency during evaluation
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            user = data[:, 0]
            item = data[:, 1]

            # Generate predictions and compute the loss
            predictions = model(data)
            loss = RMSE_Loss(predictions, target)
            losses.append(loss.item())

    # Return the mean RMSE across all test samples
    return np.mean(losses)

def test(model, device, test_loader):
    # This function provides a more detailed evaluation of the model.
    # While similar to the metrics function, it computes the total loss
    # across the entire test set, which can be useful for tracking overall performance.
    
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Accumulate the total loss
            test_loss += RMSE_Loss(output, target, reduction="sum").item()

    # Compute the average loss per sample
    test_loss /= len(test_loader.dataset)
    print(f"Test set RMSE: {test_loss}")

# Set up the device for training
# In this case, we're using CPU, but this could be changed to GPU if available
device = torch.device(args.device)

# Initialize our Neural Collaborative Filtering model
# This creates the global model that will be used throughout the federated learning process
model = NeuralCF(num_users, num_movies, args.embed_size).to(device)

# Main training loop
# This is where we bring everything together to perform federated learning
import time
for epoch in range(1, args.epochs):
    t0 = time.time()
    print(f"Starting epoch {epoch}/{args.epochs}")
    
    # Perform one epoch of federated training
    # This involves distributing data to workers, training locally, and aggregating results
    model = train_single_epoch(model, device, federated_train_loader, args.learning_rate, args.federate_after_n_batches)
    
    # Evaluate the updated model on the test set
    loss = metrics(model, test_loader, args.top_k)
    
    print(f'Epoch completed in {time.time() - t0} seconds')
    
    # Log the loss for this epoch
    # This allows us to track the model's performance over time
    writer.add_scalar("Loss", loss, epoch)

# After training is complete, save the results
# This JSON file will contain the loss values for each epoch, which can be used for analysis or visualization
writer.export_scalars_to_json(f"results/fed_ncf_{args.no_workers}_{args.epochs}_{args.batch_size}_{args.learning_rate}_{args.federate_after_n_batches}_{args.weight_decay}_{args.top_k}.json")
writer.close()