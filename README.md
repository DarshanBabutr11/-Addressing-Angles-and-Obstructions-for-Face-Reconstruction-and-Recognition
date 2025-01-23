Facial Landmark Reidentification using Graph Neural Networks (GNN)
Overview
This project focuses on real-time facial landmark reidentification using Graph Neural Networks (GNN) and MediaPipe Face Mesh. It extracts 3D facial landmarks, processes them as graph data, and uses a trained GNN model to reidentify and match facial features against saved data with high accuracy.

Features
Real-time facial landmark detection using MediaPipe Face Mesh.
Graph-based processing of landmarks using a Graph Convolutional Network (GCN).
High-accuracy reidentification using cosine similarity.
Save and load landmarks for future reidentification.
Dynamic user folder creation for organizing saved data.
Visualization of landmarks on live webcam feed.
Technologies Used
Python: Core programming language.
OpenCV: For webcam integration and frame visualization.
MediaPipe Face Mesh: For extracting 3D facial landmarks.
PyTorch & PyTorch Geometric:
Building and evaluating the GNN model.
Graph-based operations for facial landmark processing.
NumPy: For handling landmark data.
Cosine Similarity: For measuring similarity between real-time and saved landmarks.
File I/O and os Module: For managing saved data files.
datetime Module: For creating dynamic folders.

Future Enhancements
Extend the GNN model for multi-face reidentification.
Add a web-based user interface for easier interaction.
Incorporate more advanced similarity metrics for robust comparisons.
