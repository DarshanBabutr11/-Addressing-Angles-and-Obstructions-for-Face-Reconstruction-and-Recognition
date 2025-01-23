# import cv2
# import torch
# import mediapipe as mp
# import numpy as np
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv

# # Load the trained GCN model
# class LandmarkGNN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(LandmarkGNN, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv2(x, edge_index)
#         return x

# model = LandmarkGNN(in_channels=3, hidden_channels=64, out_channels=3)
# model.load_state_dict(torch.load("landmark_gnn_model.pth"))
# model.eval()

# # Load saved landmarks
# saved_landmarks = np.load("landmark_data/landmarks.npy")

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# # Function to preprocess landmarks for the model
# def preprocess_landmarks(landmarks):
#     x = torch.tensor(landmarks, dtype=torch.float)
#     edge_index = torch.tensor(
#         [[i, i + 1] for i in range(len(x) - 1)], dtype=torch.long
#     ).t().contiguous()
#     return Data(x=x, edge_index=edge_index)

# # Open webcam
# cap = cv2.VideoCapture(0)
# print("Press 'q' to quit")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to RGB for MediaPipe
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Extract and preprocess landmarks
#             landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
#             data = preprocess_landmarks(landmarks)

#             # Use the model for inference
#             with torch.no_grad():
#                 output = model(data.x, data.edge_index)
#                 similarity = torch.cosine_similarity(
#                     torch.tensor(saved_landmarks, dtype=torch.float).flatten(),
#                     output.flatten(),
#                     dim=0
#                 )
#                 print(f"Similarity Score: {similarity.item()}")

#             # Draw landmarks on the frame
#             for lm in face_landmarks.landmark:
#                 x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
#                 cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

#     # Display the frame
#     cv2.imshow("Reidentification", frame)

#     # Quit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()








# import cv2
# import torch
# import mediapipe as mp
# import numpy as np
# import os
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv

# # Load the trained GCN model
# class LandmarkGNN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(LandmarkGNN, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)  # Corrected
#         self.conv3 = GCNConv(hidden_channels, hidden_channels)  # Added
#         self.conv4 = GCNConv(hidden_channels, out_channels)      # Added

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv2(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv3(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv4(x, edge_index)
#         return x

# model = LandmarkGNN(in_channels=3, hidden_channels=64, out_channels=3)
# model.load_state_dict(torch.load("landmark_gnn_model.pth"))
# model.eval()

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# # Function to preprocess landmarks for the model
# def preprocess_landmarks(landmarks):
#     x = torch.tensor(landmarks, dtype=torch.float)
#     edge_index = torch.tensor(
#         [[i, i + 1] for i in range(len(x) - 1)], dtype=torch.long
#     ).t().contiguous()
#     return Data(x=x, edge_index=edge_index)

# # Open webcam
# cap = cv2.VideoCapture(0)
# print("Press 'q' to quit")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to RGB for MediaPipe
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Extract and preprocess landmarks
#             landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
#             data = preprocess_landmarks(landmarks)

#             # Use the model for inference
#             with torch.no_grad():
#                 output = model(data.x, data.edge_index).flatten()

#             # Compare with saved landmarks
#             for filename in os.listdir("landmark_data"):
#                 saved_landmarks = np.load(os.path.join("landmark_data", filename))
#                 saved_data = torch.tensor(saved_landmarks, dtype=torch.float).flatten()
#                 similarity = torch.cosine_similarity(output, saved_data, dim=0) * 100  # similarity as a percentage
#                 if similarity.item() >= 95:
#                     print(f"High similarity ({similarity.item():.2f}%) with file: {filename}")

#             # Draw landmarks on the frame
#             for lm in face_landmarks.landmark:
#                 x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
#                 cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

#     # Display the frame
#     cv2.imshow("Reidentification", frame)

#     # Quit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()











import cv2
import torch
import mediapipe as mp
import numpy as np
import os
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm
import torch.nn as nn

# Define the GNN Model with added layers and normalization
class LandmarkGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LandmarkGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.norm1 = BatchNorm(hidden_channels)
        self.middle_convs = nn.ModuleList()
        self.middle_norms = nn.ModuleList()
        for _ in range(10):
            self.middle_convs.append(GCNConv(hidden_channels, hidden_channels))
            self.middle_norms.append(BatchNorm(hidden_channels))
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

model_path = "/Users/akhilesh/Desktop/superjoin/gcn/landmark_gnn_model.pth"
model = LandmarkGNN(in_channels=3, hidden_channels=64, out_channels=3)
model.load_state_dict(torch.load(model_path))
model.eval()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def preprocess_landmarks(landmarks):
    x = torch.tensor(landmarks, dtype=torch.float)
    edge_index = torch.tensor([[i, i + 1] for i in range(len(x) - 1)], dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

# Path to the directory containing npy files
data_path = "/Users/akhilesh/Desktop/superjoin/gcn/landmark_test/faces/"

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
            data = preprocess_landmarks(landmarks)

            with torch.no_grad():
                output = model(data.x, data.edge_index).flatten()

            for filename in os.listdir(data_path):
                if filename.endswith(".npy"):
                    file_path = os.path.join(data_path, filename)
                    saved_landmarks = np.load(file_path)
                    saved_data = torch.tensor(saved_landmarks, dtype=torch.float).flatten()
                    similarity = torch.cosine_similarity(output, saved_data, dim=0)
                    if similarity.item() >= 0.965:
                        print(f"High similarity ({similarity.item():.2f}%) with file: {filename}")

            for lm in face_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Reidentification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
