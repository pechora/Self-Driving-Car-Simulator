import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import model as br
import cv2
import cv2 as cv
from keras.models import load_model
from sklearn.preprocessing import scale

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#init our model and image array as empty
model = None
prev_image_array = None

#set min/max speed for our autonomous car
MAX_SPEED = 25
MIN_SPEED = 10
avgspeed = 0.0
steering_rot = 0.0

#and a speed limit
speed_limit = MAX_SPEED

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)      # from PIL image to numpy array
            image = np.array((image[20:140:2, ::2]), dtype = 'float32')
            #image = utils.preprocess(image) # apply the preprocessing
            image = cv.cvtColor(image, cv2.COLOR_RGB2YUV)
            cv.imshow("Dusk", image/256)
            cv.waitKey(1)
            image = np.array([image])
            image -= 128
            image /= 128

            # predict the steering angle for the image
            nxx = model.predict(image, batch_size=1)
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            '''global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2'''
            global steering_rot
            steering_angle = nxx[0][0]
            steering_angle = (steering_angle/abs(steering_angle)) * (abs(steering_angle) ** (1/1))
            steering_rot = (0.0 * steering_rot) + (1 * steering_angle)
            global avgspeed
            avgspeed = (avgspeed * 0.8) + (speed * 0.2)
            throttle = (1 - abs(steering_angle**2)) * ((30 - avgspeed)/90)

            if speed < 2:
                throttle = 1
            if throttle < 0:
                throttle = 0

            print('{} {} {}'.format(steering_rot, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:

        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    #load model
    br.model.load_weights('model')
    model = br.model

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
