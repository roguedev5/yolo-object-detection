import numpy as np
import argparse
import time
import cv2
import os
import dlib
import face_recognition
from flask import Flask, request, Response, jsonify, flash, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import jsonpickle
import io as StringIO
import base64
from io import BytesIO
import io
import json
import pickle
from PIL import Image,ImageDraw, ImageFont
import pytesseract
import hashlib

os.environ['LC_ALL'] = 'C.UTF-8'

confthres = 0.3
nmsthres = 0.1
yolo_path = './'

def get_labels(labels_path):
    lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='PNG')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr


def get_predection(image,net,LABELS,COLORS):
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    print(layerOutputs)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confthres:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)
                            
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            print(boxes)
            print(classIDs)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    return image


labelsPath="coco.names"
cfgpath="yolov3.cfg"
wpath="yolov3.weights"
Lables=get_labels(labelsPath)
CFG=get_config(cfgpath)
Weights=get_weights(wpath)
nets=load_model(CFG,Weights)
Colors=get_colors(Lables)


# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/api/test', methods=['POST'])
def main():
    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))
    npimg=np.array(img)
    image=npimg.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    res=get_predection(image,nets,Lables,Colors)
    image=cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
    np_img=Image.fromarray(image)
    img_encoded=image_to_byte_array(np_img)
    return Response(response=img_encoded, status=200,mimetype="image/jpeg")

@app.route('/detecttext', methods=['POST'])
def detect_text():
    try:
        image_data = request.files['image'].read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        text = pytesseract.image_to_string(image)
        response_data = {'detected_text': text}
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# Directory to store the known faces
known_faces_dir = 'known_faces'

# Load the known faces and their names from a pickle file or create an empty one
known_faces_path = "known_faces.pkl"
if os.path.exists(known_faces_path):
    with open(known_faces_path, "rb") as file:
        known_faces_data = pickle.load(file)
        known_face_encodings = known_faces_data["encodings"]
        known_face_names = known_faces_data["names"]
else:
    known_face_encodings, known_face_names = [], []

known_face_image_hashes = []
@app.route('/api/train', methods=['POST'])
def train():
    if 'name' not in request.form:
        return jsonify({"error": "Name is required"}), 400

    if 'image' not in request.files:
        return jsonify({"error": "Image is required"}), 400

    name = request.form['name']
    image_file = request.files['image']

    if name == '':
        return jsonify({"error": "Name cannot be empty"}), 400

    if image_file.filename == '':
        return jsonify({"error": "No selected image file"}), 400

    if image_file:
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(known_faces_dir, filename)
        image_file.save(file_path)

        # Calculate the hash of the image content
        image_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

        # Check if this image hash exists in the list
        if image_hash in known_face_image_hashes:
            # Find the index of the existing image hash
            index = known_face_image_hashes.index(image_hash)
            # Update the associated name with the latest name
            known_face_names[index] = name
        else:
            # Load and encode the newly added known face
            img = face_recognition.load_image_file(file_path)
            face_encodings = face_recognition.face_encodings(img)

            if len(face_encodings) == 0:
                flash('No faces found in the training image')
                os.remove(file_path)
                return jsonify({"error": "No faces found in the training image"}), 400

            known_face_encodings.extend(face_encodings)
            known_face_names.extend([name] * len(face_encodings))
            known_face_image_hashes.extend([image_hash] * len(face_encodings))

        # Save the updated known faces, names, and image hashes
        known_faces_data = {
            "encodings": known_face_encodings,
            "names": known_face_names,
            "image_hashes": known_face_image_hashes
        }
        with open(known_faces_path, "wb") as file:
            pickle.dump(known_faces_data, file)

        flash('Training successful!')
        os.remove(file_path)

        return jsonify({"message": "Training successful"}), 200


@app.route('/api/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        img = face_recognition.load_image_file(file)
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)

        # Create a copy of the image to draw labels on
        img_with_labels = np.copy(img)

        recognized_results = []

        # Define a color map for labels
        label_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # You can add more colors as needed

        for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
            # Compare the face encoding with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            color = label_colors[i % len(label_colors)]  # Cycle through available colors

            if any(matches):
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            recognized_results.append({"name": name})

            # Draw a rectangle and label on the image
            top, right, bottom, left = face_location
            cv2.rectangle(img_with_labels, (left, top), (right, bottom), color, 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            label_size, _ = cv2.getTextSize(name, font, 0.5, 1)
            cv2.rectangle(img_with_labels, (left, top - label_size[1]), (right, top), color, cv2.FILLED)
            cv2.putText(img_with_labels, name, (left + 6, top - 6), font, 0.5, (255, 255, 255), 1)

        # Convert the image with labels to bytes
        _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(img_with_labels, cv2.COLOR_RGB2BGR))
        img_bytes = img_encoded.tobytes()

        return Response(response=img_bytes, status=200, mimetype="image/jpeg")


    # start flask app
if __name__ == '__main__':
    app.secret_key = 'your_secret_key_here'
    app.run(debug=True, host='0.0.0.0')

