#!/usr/bin/python3

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import threading
import pickle
import time
import cv2

# Try importing the GPIO library.
try:
    import RPi.GPIO as GPIO
except ModuleNotFoundError:
    print("RPi module not found. Ignoring blow jobs.\n")


# Blow.
def blow():
    try:
        # Set up GPIO
        servo_pin = 21  # GPIO Pin where servo is connected
        GPIO.setmode(GPIO.BCM)

        # Define the Pin numbering type and define Servo Pin as output pin
        GPIO.setup(servo_pin, GPIO.OUT)
        p = GPIO.PWM(servo_pin, 50)  # PWM channel at 50 Hz frequency
        p.start(9)  # Zero duty cycle initially

        # Blow.
        time.sleep(0.1)
        p.ChangeDutyCycle(12.5)
        time.sleep(3)
        p.ChangeDutyCycle(9)
        time.sleep(0.4)

    except NameError:
        print("GPIO not defined. Ignoring blow job.\n")

    finally:
        GPIO.cleanup()


# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
# Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"

# Load the known faces and embeddings along with OpenCV's Haar cascade for
# face detection.
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# Initialize the video stream and allow the camera sensor to warm up.
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# Start the FPS counter.
fps = FPS().start()

# Start the video stream.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video.avi', fourcc, 2.0, (320, 240))

# Loop over frames from the video file stream.
while True:
    # Grab the frame from the threaded video stream.
    frame = vs.read()
    # Detect the face boxes.
    boxes = face_recognition.face_locations(frame)
    # Compute the facial embeddings for each face bounding box.
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    # Loop over facial embeddings.
    for encoding in encodings:
        # Attempt to match each face in the input image to known encodings.
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"  # if face is not recognized, print "Unknown"

        # Check to see if we found a match.
        if True in matches:
            # Find the indexes of all matched faces, then initialize a
            # dictionary to count the total number of times each face matched.
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # Loop over the matched indexes and maintain a count for
            # each recognized face.
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary).
            name = max(counts, key=counts.get)

            # Print name on screen if someone is identified.
            if currentname != name:
                currentname = name
                print(currentname)

                # Trigger the blower in a background thread.
                # NOTE: Not safe if you identify someone multiple times.
                if (currentname == "Jeff"):
                    blow_thread = threading.Thread(target=blow, name="blower")
                    blow_thread.start()

        # Update the list of names.
        names.append(name)

    # Loop over the recognized faces.
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image - color is in BGR
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (0, 255, 255), 2)

    # Write frame to stream.
    out.write(frame)

    # Display the image to our screen.
    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit when 'q' key is pressed.
    if key == ord("q"):
        break

    # Update FPS counter.
    fps.update()

# Stop the timer and display FPS information.
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Do a bit of cleanup.
out.release()
cv2.destroyAllWindows()
vs.stop()
