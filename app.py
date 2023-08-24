from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import json
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# Load the pre-trained emotion detection model from JSON and H5 files
with open('fer.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights('fer (1).h5')

# Initialize the face Haar Cascade
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

## Emotion-to-playlist mapping
emotion_to_playlist = {
    "anger": "2Cwf7HSoZ4pd7GgqrMaPak",
    "disgust": "2cJJ0lRhuwABm8LLgVI9xh",
    "sadness": "0ViQOZ7tk78qoPb4fkDbCO",
    "happiness": "0kD8G6OB9jHEvmVqRlIlAh",
    "fear": "0SDf28nEtw4R2xTGGBiPGz",
    "surprise": "6NVKSIpaRyOfgd55b00aWH",
    "neutral": "7p9jOTYVsEFa6If9ik78Jv"
}

detected_emotion = None
playlist_id = None

@app.route('/')
def index():
    return render_template('index.html')

def generate():
    global detected_emotion, playlist_id
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))

        for (x, y, w, h) in faces_detected:
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0

            predictions = model.predict(img_pixels)
            max_index = int(np.argmax(predictions))

            emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
            detected_emotion = emotions[max_index]
            playlist_id = emotion_to_playlist.get(detected_emotion)

            cv2.putText(img, detected_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

            # Emit the detected emotion and playlist ID via WebSocket
            socketio.emit('emotion_detected', {'detected_emotion': detected_emotion, 'playlist_id': playlist_id})

            # Break out of the loop after emotion detection
            break

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def connect():
    print('Client connected')

@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True)
