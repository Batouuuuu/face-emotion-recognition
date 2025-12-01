"""REST API"""

from flask import Flask
from flask import Response
import matplotlib.pyplot as plt 
import cv2
import io

app = Flask(__name__)
webcam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def pie_graph():

    labels = ["Angry", "Happy", "Sad", "Neutral", "Fear", "Surprise", "Disgust"]
    sizes = [10, 10, 40, 10, 10, 10, 10]
    plt.pie(sizes, labels=labels, autopct="%1.1f%%")
    plt.title("Emotion")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    graph_byte = buf.getvalue()
    plt.close()
    return graph_byte


def video_capture() -> bytes:
        try:
            ret, frame = webcam.read()
            ret, jpeg = cv2.imencode('.jpg', frame)
            jpeg_bytes = jpeg.tobytes()

        except Exception as e:
            print(e)

        return jpeg_bytes
        

def generator():
     while True:
        bytes = video_capture()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytes + b"\r\n"


@app.route("/video_feed")
def create_mjpeg():
    return Response(generator(), mimetype="multipart/x-mixed-replace; boundary=frame")
          


@app.route("/pie_graph")
def create_graphe():
    graph_bytes = pie_graph()
    return Response(graph_bytes, mimetype="image/png")

@app.route("/")  
def home_page():
    return 'Go to /video_feed'

if __name__ == '__main__':
    app.run()