import face_recognition
import os
import numpy as np
import pickle
from pathlib import Path

def load_database(database_file):
    with open(database_file, 'rb') as f:
        return pickle.load(f)

def recognize_faces(input_folder, database_file, tolerance=0.48):
    faces = load_database(database_file)
    recognition_dict = {}

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(input_folder, filename)
            image = face_recognition.load_image_file(filepath)
            locs = face_recognition.face_locations(image)
            encs = face_recognition.face_encodings(image, known_face_locations=locs)

            if not encs:
                print(f"No faces found in {filename}")
                continue

            for enc in encs:
                matches = face_recognition.compare_faces(faces['encs'], enc, tolerance=tolerance)
                if True in matches:
                    first_match_index = matches.index(True)
                    name = faces['names'][first_match_index]
                else:
                    name = "unknown"

                if name in recognition_dict:
                    recognition_dict[name].append(filename)
                else:
                    recognition_dict[name] = [filename]

                print(f"Recognized {name} in {filename}")

    return recognition_dict

# Define the input folder and database file
input_folder = 'input'
database_file = 'face_encodings.pkl'

# Get the recognition dictionary
recognition_dict = recognize_faces(input_folder, database_file)

# Print the recognition dictionary
for person, images in recognition_dict.items():
    print(f"{person}: {images}")
