import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import warnings
# Suppress the deprecation warning from the cryptography module.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import cryptography

from tensorboardX import SummaryWriter
writer = SummaryWriter()

# Define the hyperparameters and configuration for our centralized learning setup.
# We've removed parameters specific to federated learning and adjusted others as needed.
class Arguments():
    def __init__(self):
        self.seed = 50
        self.device = "cpu"
        self.batch_size = 512
        self.test_batch_size = 512
        self.learning_rate = 0.01
        self.epochs = 15
        self.embed_size = 30
        self.test_size = 0.2
        self.weight_decay = 1e-5
        self.top_k = 10

args = Arguments()

# Set the random seed for reproducibility
random.seed(args.seed)
torch.manual_seed(args.seed)

# Define the custom Dataset class for handling user-item interactions
# This remains largely the same as in the federated version
class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
 
    def __getitem__(self, index):
        return (
            torch.tensor([self.user_tensor[index], self.item_tensor[index]]),
            self.target_tensor[index]
        )
 
    def __len__(self):
        return len(self.user_tensor)

# Load and preprocess the MovieLens dataset
# This part remains largely unchanged
rs_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
train_data = pd.read_csv('ua.base', sep='\t', names=rs_cols, engine='python')

user_ids = train_data["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = train_data["movie_id"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

train_data["user"] = train_data["user_id"].map(user2user_encoded)
train_data["movie"] = train_data["movie_id"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)
train_data["rating"] = train_data["rating"].values.astype(np.float32)
max_rating = train_data["rating"].max()
min_rating = train_data["rating"].min()

# Split the data into training and testing sets
train_data, test_data = train_test_split(train_data, test_size=args.test_size, random_state=42)

# The Neural Collaborative Filtering (NCF) model remains the same
class NeuralCF(nn.Module):
    def __init__(self, num_users=num_users, num_movies=num_movies, embedding_size=args.embed_size):
        super(NeuralCF, self).__init__()
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_size)
        self.movie_embedding_mlp = nn.Embedding(num_movies, embedding_size)
        self.user_embedding_mf = nn.Embedding(num_users, embedding_size)
        self.movie_embedding_mf = nn.Embedding(num_movies, embedding_size)
        
        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip([60, 32, 16], [32, 16, 8])):
            self.fc_layers.append(nn.Linear(in_size, out_size))
        
        self.affine_output = nn.Linear(in_features=8 + embedding_size, out_features=1)
        self.logistic = nn.Sigmoid()
        
    def forward(self, user_indices, movie_indices):
        user_embedding_mlp = self.user_embedding_mlp(user_indices)
        movie_embedding_mlp = self.movie_embedding_mlp(movie_indices)
        user_embedding_mf = self.user_embedding_mf(user_indices)
        movie_embedding_mf = self.movie_embedding_mf(movie_indices)
        
        mlp_vector = torch.cat([user_embedding_mlp, movie_embedding_mlp], dim=-1)
        mf_vector = torch.mul(user_embedding_mf, movie_embedding_mf)
        
        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = F.relu(mlp_vector)
        
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        logits=logits*(max_rating-min_rating+1)+min_rating-1
        return logits.view(-1)

# Define the RMSE loss function
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

# Create DataLoaders for training and testing
train_dataset = UserItemRatingDataset(
    torch.LongTensor(train_data["user"].values),
    torch.LongTensor(train_data["movie"].values),
    torch.FloatTensor(train_data["rating"].values)
)
test_dataset = UserItemRatingDataset(
    torch.LongTensor(test_data["user"].values),
    torch.LongTensor(test_data["movie"].values),
    torch.FloatTensor(test_data["rating"].values)
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

criterion = RMSELoss()

# Training and evaluation functions
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data[:, 0], data[:, 1])
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader)
    print(f'Train Epoch: {epoch}\tAverage Loss: {avg_loss:.6f}')
    return model, avg_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data[:, 0], data[:, 1])
            test_loss += criterion(output, target).item()
    
    test_loss /= len(test_loader)
    print(f'Test set: Average loss: {test_loss:.4f}')
    return test_loss

device = torch.device(args.device)
model = NeuralCF(num_users, num_movies, args.embed_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# Main training loop
for epoch in range(1, args.epochs + 1):
    model, train_loss = train(model, device, train_loader, optimizer, epoch)
    test_loss = test(model, device, test_loader)
    
    # Log the losses
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)

# Save the results
writer.export_scalars_to_json(f"results/centralized_ncf_{args.epochs}_{args.embed_size}_{args.batch_size}_{args.learning_rate}_{args.weight_decay}.json")
writer.close()

print("Training complete. Model and results have been saved.")