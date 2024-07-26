# face-tag

then run following commands - 

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip3 install -q face_recognition

pip3 install -q fer

pip3 install facenet-pytorch

pip3 install numpy dlib piexif face_recognition

brew install jhead

to do without training -

create input folder and add .jpg images

python3 facetag.py

to do with training -

create input folder and add .jpg images

create faces images, where give file name as person name

python3 training.py

python3 testing.py
