import face_recognition
import os
import numpy as np
import pickle
from pathlib import Path

def train_model(input_folder, database_file):
    faces = {
        'encs': [],
        'names': []
    }

    # Iterate through the input folder and process each image
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(input_folder, filename)
            name = Path(filename).stem

            # Load the image and get the face encodings
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                faces['encs'].append(encodings[0])
                faces['names'].append(name)
            else:
                print(f"No faces found in {filename}")

    # Convert lists to numpy arrays for easier handling
    faces['encs'] = np.array(faces['encs'])
    faces['names'] = np.array(faces['names'])

    # Save the encodings to the database file
    with open(database_file, 'wb') as f:
        pickle.dump(faces, f)

    print("Training complete. Encodings saved to", database_file)

# Define the input folder and database file
input_folder = 'faces'
database_file = 'face_encodings.pkl'

train_model(input_folder, database_file)
