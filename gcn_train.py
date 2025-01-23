# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch_geometric.data import Data, DataLoader
# from torch_geometric.nn import GCNConv
# import os
# import numpy as np

# # Define a GNN Model
# class LandmarkGNN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(LandmarkGNN, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv2(x, edge_index)
#         return x

# # Load Data Function
# def load_landmark_data(data_dir="landmark_data"):
#     data_list = []
#     for file in os.listdir(data_dir):
#         landmarks = np.load(os.path.join(data_dir, file))
#         x = torch.tensor(np.nan_to_num(landmarks), dtype=torch.float)
#         edge_index = torch.tensor(
#             [[i, i + 1] for i in range(len(x) - 1)], dtype=torch.long
#         ).t().contiguous()
#         data = Data(x=x, edge_index=edge_index)
#         data_list.append(data)
#     return data_list

# # Training Function
# def train(model, loader, optimizer, criterion, epochs=20):
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in loader:
#             optimizer.zero_grad()
#             output = model(batch.x, batch.edge_index)
#             loss = criterion(output, batch.x)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader)}")

# # Initialize Model, Optimizer, and Loss Function
# in_channels = 3
# hidden_channels = 64
# out_channels = 3
# model = LandmarkGNN(in_channels, hidden_channels, out_channels)
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# criterion = nn.MSELoss()

# # Load dataset and create DataLoader
# data_list = load_landmark_data()
# loader = DataLoader(data_list, batch_size=16, shuffle=True)

# # Train the model
# train(model, loader, optimizer, criterion)

# # Save the trained model
# torch.save(model.state_dict(), "landmark_gnn_model.pth")
# print("Model saved to landmark_gnn_model.pth")






#version 2 ---------------------------------------------------->

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch_geometric.data import Data, DataLoader
# from torch_geometric.nn import GCNConv
# import os
# import numpy as np

# # Define a GNN Model
# class LandmarkGNN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(LandmarkGNN, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)  # Additional layer
#         self.conv3 = GCNConv(hidden_channels, hidden_channels)  # Additional layer
#         self.conv4 = GCNConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv2(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv3(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv4(x, edge_index)
#         return x

# # Load Data Function
# def load_landmark_data(data_dir="landmark_data"):
#     data_list = []
#     for file in os.listdir(data_dir):
#         landmarks = np.load(os.path.join(data_dir, file))
#         x = torch.tensor(np.nan_to_num(landmarks), dtype=torch.float)
#         edge_index = torch.tensor(
#             [[i, i + 1] for i in range(len(x) - 1)], dtype=torch.long
#         ).t().contiguous()
#         data = Data(x=x, edge_index=edge_index)
#         data_list.append(data)
#     return data_list

# # Training Function
# def train(model, loader, optimizer, criterion, epochs=100):
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         total_mse = 0
#         for batch in loader:
#             optimizer.zero_grad()
#             output = model(batch.x, batch.edge_index)
#             loss = criterion(output, batch.x)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#             mse = ((output - batch.x) ** 2).mean().item()
#             total_mse += mse
#         average_loss = total_loss / len(loader)
#         average_mse = total_mse / len(loader)
#         print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}, MSE: {average_mse:.4f}")

# # Initialize Model, Optimizer, and Loss Function
# in_channels = 3
# hidden_channels = 64
# out_channels = 3
# model = LandmarkGNN(in_channels, hidden_channels, out_channels)
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# criterion = nn.MSELoss()

# # Load dataset and create DataLoader
# data_list = load_landmark_data()
# loader = DataLoader(data_list, batch_size=16, shuffle=True)

# # Train the model
# train(model, loader, optimizer, criterion, epochs=100)

# # Save the trained model
# torch.save(model.state_dict(), "landmark_gnn_model.pth")
# print("Model saved to landmark_gnn_model.pth")



#########version 3------------------------------------------->
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.data import Data, DataLoader
import os
import numpy as np

# Define a GNN Model
class LandmarkGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LandmarkGNN, self).__init__()
        # Start with an input convolution
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.norm1 = BatchNorm(hidden_channels)

        # Add more layers
        self.middle_convs = nn.ModuleList()
        self.middle_norms = nn.ModuleList()
        for _ in range(10):  # Adding ten more layers
            self.middle_convs.append(GCNConv(hidden_channels, hidden_channels))
            self.middle_norms.append(BatchNorm(hidden_channels))
        
        # End with an output convolution
        self.conv_last = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = torch.relu(x)

        for conv, norm in zip(self.middle_convs, self.middle_norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)

        x = self.conv_last(x, edge_index)
        return x

# Load Data Function
def load_landmark_data(data_dir="/Users/akhilesh/Desktop/superjoin/gcn/data/youtube_landmarks"):
    data_list = []
    for person_folder in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_folder)
        if os.path.isdir(person_path):  # Check if it is a directory
            for file in os.listdir(person_path):
                file_path = os.path.join(person_path, file)
                landmarks = np.load(file_path)
                x = torch.tensor(np.nan_to_num(landmarks), dtype=torch.float)
                edge_index = torch.tensor(
                    [[i, i + 1] for i in range(len(x) - 1)], dtype=torch.long
                ).t().contiguous()
                data = Data(x=x, edge_index=edge_index)
                data_list.append(data)
    return data_list

# Training Function
def train(model, loader, optimizer, criterion, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_mse = 0
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index)
            loss = criterion(output, batch.x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            mse = ((output - batch.x) ** 2).mean().item()
            total_mse += mse
        average_loss = total_loss / len(loader)
        average_mse = total_mse / len(loader)
        print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}, MSE: {average_mse:.4f}")

# Initialize Model, Optimizer, and Loss Function
model = LandmarkGNN(in_channels=3, hidden_channels=64, out_channels=3)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Load dataset and create DataLoader
data_list = load_landmark_data()
loader = DataLoader(data_list, batch_size=16, shuffle=True)

# Train the model
train(model, loader, optimizer, criterion, epochs=100)

# Save the trained model
torch.save(model.state_dict(), "landmark_gnn_model.pth")
print("Model saved to landmark_gnn_model.pth")
