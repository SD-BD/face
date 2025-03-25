from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition
import os
import mediapipe as mp

app = Flask(__name__)

# 📌 Initialize Face Detection Model
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# 📌 Load Known Faces
KNOWN_FACES_DIR = "known_faces"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{filename}")
    encoding = face_recognition.face_encodings(image)
    if encoding:
        known_face_encodings.append(encoding[0])
        known_face_names.append(filename.split(".")[0])

# 📌 Real-time Face Match Function
def face_match(image):
    unknown_encodings = face_recognition.face_encodings(image)
    
    if not unknown_encodings:
        return {"match": "No Face Detected"}
    
    results = face_recognition.compare_faces(known_face_encodings, unknown_encodings[0])
    match_name = "Unknown"

    if True in results:
        match_name = known_face_names[results.index(True)]

    return {"match": match_name}

# 📌 Fake Face Detection Function
def detect_fake_face(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)

    if not results.detections:
        return {"fake": True, "message": "No face detected, possibly fake"}

    for detection in results.detections:
        if detection.score[0] < 0.6:
            return {"fake": True, "message": "Low face detection confidence, possibly fake"}

    return {"fake": False, "message": "Real face detected"}

@app.route("/live_face_match", methods=["POST"])
def live_face_match():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    image_np = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    face_result = face_match(frame)
    fake_result = detect_fake_face(frame)

    return jsonify({"face_match": face_result, "fake_detection": fake_result})

if __name__ == "__main__":
    app.run(debug=True)
