# import cv2
# import mediapipe as mp
# import numpy as np
# import os

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# # Create directory to save landmarks if it doesn't exist
# os.makedirs("landmark_data", exist_ok=True)

# # Open webcam
# cap = cv2.VideoCapture(0)
# print("Press 's' to save landmarks, 'q' to quit")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to RGB for MediaPipe
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Extract landmarks
#             landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
            
#             # Draw landmarks on the frame
#             for lm in face_landmarks.landmark:
#                 x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
#                 cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

#             # Save landmarks when 's' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('s'):
#                 file_path = os.path.join("landmark_data", "landmarks.npy")
#                 np.save(file_path, landmarks)
#                 print(f"Landmarks saved to {file_path}")

#     # Display the frame
#     cv2.imshow("Capture Landmarks", frame)

#     # Quit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import mediapipe as mp
# import numpy as np
# import os

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# # Create directory to save landmarks if it doesn't exist
# os.makedirs("landmark_data", exist_ok=True)

# # Open webcam
# cap = cv2.VideoCapture(0)
# print("Press 's' to save landmarks, 'q' to quit")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to RGB for MediaPipe
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Extract landmarks
#             landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
            
#             # Draw landmarks on the frame
#             for lm in face_landmarks.landmark:
#                 x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
#                 cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

#     # Display the frame
#     cv2.imshow("Capture Landmarks", frame)

#     # Check if 's' was pressed to save landmarks
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#         # Ask user for filename input
#         filename = input("Enter filename to save landmarks: ")
#         file_path = os.path.join("landmark_data", f"{filename}.npy")
#         np.save(file_path, landmarks)
#         print(f"Landmarks saved to {file_path}")

#     # Quit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()









import cv2
import mediapipe as mp
import numpy as np
import os
import datetime  # To generate unique folder names based on date and time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Base directory for saving data
base_dir = "landmark_test/faces"

# Create base directory to save landmarks if it doesn't exist
os.makedirs(base_dir, exist_ok=True)

# Function to create a new folder for storing current session's landmarks
def create_new_person_folder(base_dir):
    # Use the current date and time as a unique folder name
    folder_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    person_dir = os.path.join(base_dir, folder_name)
    os.makedirs(person_dir, exist_ok=True)
    return person_dir

# Create a new folder for the current session
current_session_folder = create_new_person_folder(base_dir)

# Open webcam
cap = cv2.VideoCapture(0)
print("Press 's' to save landmarks, 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmarks
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
            
            # Draw landmarks on the frame
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Capture Landmarks", frame)

    # Check if 's' was pressed to save landmarks
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Ask user for filename input
        filename = input("Enter filename to save landmarks: ")
        file_path = os.path.join(current_session_folder, f"{filename}.npy")
        np.save(file_path, landmarks)
        print(f"Landmarks saved to {file_path}")

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
