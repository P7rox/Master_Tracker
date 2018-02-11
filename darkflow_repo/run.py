import face_recognition
import cv2
import numpy as np
import time
import sys
from darkflow.net.build import TFNet
from math import sqrt

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

skip_tracker_init = 0
failure = 1
yolo_init = True

    # Set up tracker.
    # Instead of MIL, you can also use
 
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[1]



# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("./faces/obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("./faces/biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Load a custom sample picture and learn how to recognize it.
p7rox_image = face_recognition.load_image_file("./faces/p7rox.jpg")
p7rox_face_encoding = face_recognition.face_encodings(p7rox_image)[0]

# Load a custom sample picture and learn how to recognize it.
aman_image = face_recognition.load_image_file("./faces/aman.jpg")
aman_face_encoding = face_recognition.face_encodings(aman_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    p7rox_face_encoding,
    aman_face_encoding,
    jay_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Prajwal Goswami",
    "Aman Tanwar",
    "Jay"
]

trackObj = [
   "person" 
]
options = {
    'model': 'darkflow_repo/cfg/tiny-yolo-voc.cfg',
    'load': 'darkflow_repo/bin/tiny-yolo-voc.weights',
    'threshold': 0.2,
    'gpu': 0.7
}

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]


def tracker_init(frame, bbox) :

    # Initialize tracker with first frame and bounding box
    global failure
    global tracker
    global tracker_type
    failure = 0

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

    ok = tracker.init(frame, bbox)
    return ok
 
def tracker_final(frame, identity) :
    global failure
    global tracker


    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        cv2.putText(frame, identity, p1, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        failure = 0
        print("tracker success")
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        failure = 1
        print("tracker failed")

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
 
    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    return frame

def facRec(roi, frame, tl, br) :

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    major_face_length = 0


    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(roi, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]


    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    boxIdentity = "Unknown Person"
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        diag_length = sqrt( (left - right)**2 + (bottom - top)**2 )
        top *= 4
        top += tl[1]
        right *= 4
        right += tl[0]
        bottom *= 4
        bottom += tl[1]
        left *= 4
        left += tl[0]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        if diag_length > major_face_length :
            boxIdentity = name
            major_face_length = diag_length

    return frame, boxIdentity

def yolo(frame, unt_frame) :
    global failure
    # global frame
    # global unt_frame
    global text
    global options
    global yolo_init
    global tfnet
    if (failure == 1) :
        print("failure dectected")
        if yolo_init :
            print(yolo_init)
            tfnet = TFNet(options)
            yolo_init = False
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            if label not in trackObj :
                continue
            if confidence > 0.2:
                roi = frame[tl[1]:br[1], tl[0]:br[0]]
            else :
                continue
            frame, text = facRec(roi, frame, tl, br)
            if text not in known_face_names :
            #if (text=="Prajwal Goswami") :
                pass
            else :
                continue
            #text = '{}: {:.0f}%'.format(label, confidence * 100)
            #frame = cv2.rectangle(frame, tl, br, color, 5)
            #frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            bbox = (tl[0], tl[1], br[0]-tl[0], br[1]-tl[1])
            print (bbox)
            print (str(br[1]) + "  sdd  " + str((br[1]-tl[1])))
            print (str(br[0]) + "  sdd  " + str((br[0]-tl[0])))
            print(failure)
            ok = tracker_init(unt_frame, bbox)
            if (ok) :
               break
            print(failure)

    if (failure == 0) :
        trackFrame = tracker_final(unt_frame, text)
    else :
        trackFrame = unt_frame

    return trackFrame
s