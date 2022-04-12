import multiprocessing
from flask import Flask, Response, render_template, request
from flask_sockets import Sockets
import cv2
import numpy as np
import threading
import queue
imq = queue.LifoQueue()
app = Flask(__name__)
sockets = Sockets(app)

@app.route('/')
def hello():
    return render_template('index.html')

def gen_vision(imq):
    while True:
        image = imq.get()
        frame = cv2.imencode('.jpg', image)[1]
        yield ( b'--frame\r\n'
                b'Content-Type:image/jpeg\r\n'
                b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n'
                b'\r\n' + frame.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_vision(imq), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()

def push(image):
    imq.put(image) 