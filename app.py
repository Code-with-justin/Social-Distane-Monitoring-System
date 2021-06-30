from social import social_distancing_config as config
from social.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
from playsound import playsound
import os
from flask import Flask, render_template, Response
from tensorflow import keras
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def gen_frames():
    camera = cv2.VideoCapture(0)

    print('[INFO] Loading model')
    print('[INFO] Accessing webcam')
    labelspath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
    Labels = open(labelspath).read().strip().split("\n")
    weightspath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
    configpath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # loop over the frames from the video stream
    while True:
        (grabbed, frame) = camera.read()

        if not grabbed:
            break

        # resize the frame and detect people
        frame = imutils.resize(frame, width=800)
        results = detect_people(frame, net, ln, personIdx=Labels.index("person"))

        violate = set()

        # ensure there are at least two people detections
        if len(results) >= 2:
            # calculate euclidean distance
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check distance < min distance
                    if D[i, j] < config.MIN_DISTANCE:
                        violate.add(i)
                        violate.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):
            # initialize the color for bounding box
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

            text = "Social Distancing Violations: {}".format(len(violate))

            cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            # if violations occurs
            if i in violate:
                color = (0, 0, 255)
                # Play the mp3 file
                playsound('alarm.wav', False)

                # draw a bounding box and centroid for each people
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)

                # show total no. of violations
                text = "Social Distancing Violations: {}".format(len(violate))

                cv2.putText(frame, text,  (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                cv2.imwrite(os.path.join('C:\\Users\\ASUS\\Desktop\\social distance\\captured', 'image.jpg'), frame)
                """
                email = 'justin.lmca1921@saintgits.org'
                password = 'Just#1997'
                subject = 'social distance violation'
                message = 'one guest violates the rule of social distance'

                msg = MIMEMultipart()
                msg['From'] = email
                msg['To'] = email
                msg['Subject'] = subject

                msg.attach(MIMEText(message, 'plain'))

                # Setup the attachment
                filename = 'C:\\Users\\ASUS\\Desktop\\social distance\\captured\\image.jpg'
                attachment = open(filename, "rb").read()
                image = MIMEImage(attachment, name=filename)
                msg.attach(image)

                # Attach the attachment to the MIMEMultipart object
                # msg.attach(part)

                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.ehlo()
                server.starttls()
                server.login(email, password)
                text = msg.as_string()
                server.sendmail(email, email, text)
                server.quit()

        """
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def upload():

    # initialize the video stream and pointer to output video file
    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture('pedestrians.mp4')

    print('[INFO] Loading model')

    labelspath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
    Labels = open(labelspath).read().strip().split("\n")
    weightspath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
    configpath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # loop over the frames from the video stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # resize the frame and then detect people (and only people) in it
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln, personIdx=Labels.index("person"))

        # initialize the set of indexes that violate the minimum social
        # distance
        violate = set()

        # ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps)
        if len(results) >= 2:
            # extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of pixels
                    if D[i, j] < config.MIN_DISTANCE:
                        # update our violation set with the indexes of
                        # the centroid pairs
                        violate.add(i)
                        violate.add(j)

        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation set, then
            # update the color
            if i in violate:
                color = (0, 0, 255)
                # Play the mp3 file
                playsound('alarm.wav', False)

            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        # draw the total number of social distancing violations on the
        # output frame
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def mask_det():
    model = keras.models.load_model('model-036.model')
    face_clsfr = cv2.CascadeClassifier(r"C:\Users\ASUS\Desktop\social distance\haarcascade_frontalface_default.xml")

    labels_dict = {0: 'MASK', 1: 'NO MASK'}
    color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
    source = cv2.VideoCapture(0)

    while True:

        grabbed, frame = source.read()


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:

            face_img = gray[y:y + w, x:x + w]
            resized = cv2.resize(face_img, (100, 100))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 100, 100, 1))
            result = model.predict(reshaped)

            label = np.argmax(result, axis=1)[0]


            cv2.rectangle(frame, (x, y), (x + w, y + h), color_dict[label], 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), color_dict[label], -1)
            cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            acc = round(np.max(result, axis=1)[0] * 100, 2)
            cv2.putText(frame, str(acc), (x + 150, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            SUBJECT = "Subject"
            TEXT = "One Visitor violated Face Mask Policy.A person has been detected without mask"
            """
            if label == 1:
                # messagebox.showwarning("Warning", "Access Denied!!")

                message = 'Subject:{}\n\n{}'.format(SUBJECT, TEXT)
                mail = smtplib.SMTP('smtp.gmail.com', 587)
                mail.ehlo()
                mail.starttls()
                mail.login('justin.lmca1921@saintgits.org', 'Just#1997')
                mail.sendmail('justin.lmca1921@saintgits.org', 'justin.lmca1921@saintgits.org', message)
                mail.close()
            else:
                pass
                """

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary = frame')


@app.route('/upload_file')
def upload_file():
    return Response(upload(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/mask')
def mask():
    return Response(mask_det(), mimetype='multipart/x-mixed-replace; boundary = frame')


if __name__ == '__main__':
    app.run(debug=True)


