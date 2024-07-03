from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import tempfile
import mediapipe as mp
import tensorflow as tk
import keras
import sys

app = Flask(__name__)

# Create a directory to save uploaded videos
os.makedirs('uploads', exist_ok=True)

parent_folder = 'videos'
actions = np.array([name for name in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, name))])

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

model = keras.models.load_model('models/slr_model.keras')
vid_features = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    file_bytes = np.frombuffer(file.read(), np.uint8)

    # Write the bytes to a temporary file
    temp_video_path = os.path.join(tempfile.gettempdir(), 'temp_video.webm')
    with open(temp_video_path, 'wb') as temp_video_file:
        temp_video_file.write(file_bytes)

    # Open the temporary video file with cv2.VideoCapture
    video = cv2.VideoCapture(temp_video_path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        out.write(frame)

    video.release()
    out.release()

    os.remove(temp_video_path)
    
    vid_features.clear()

    try:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            cap = cv2.VideoCapture('output.mp4')
            
            frame_num = 0
            while frame_num < 50:
                ret, new_frame = cap.read()
                if not ret:
                    # print(f"Failed to read frame {frame_num}")
                    break
                # print(f"Processing frame {frame_num}")
                image, results = mediapipe_detection(new_frame, holistic)
                keypoints = extract_keypoints(results)
                vid_features.append(keypoints)
                frame_num += 1

            cap.release()
    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({'error': 'Processing error'}), 500
    
    try:
        res = model.predict(np.expand_dims(vid_features, axis=0))[0]
        print(res)
        print(actions)
        print(np.argmax(res))

        if res[np.argmax(res)] < 0.5:
            prediction = "Low accuracy"
        else:
            prediction = actions[np.argmax(res)]
        return jsonify({'sign': prediction})
    except Exception as e:
        print(e)
        return jsonify({'error': 'Error Occured'})

if __name__ == '__main__':
    app.run(debug=True)
