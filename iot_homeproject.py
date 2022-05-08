'''
IoT Smart Home Security Project
By Brandon Dusek, Robert Hayek, & Jerry Urizar

Within the program, majority of the code involving
the use of opencv and other facial recognition
libraries were directly referenced from the following
github repositiory. From this same repository the creation
of our systems training set and the code which trained our
system are also found within this repository:
https://github.com/carolinedunn/facial_recognition

This program emulates a safe with IoT and facial
recognition capabilities in order to provide
two factor authentication.
'''

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from RPLCD import CharLCD
from time import sleep
from signal import pause
import RPi.GPIO as GPIO
import face_recognition
import smtplib
import _thread
import imutils
import pickle
import passcred
import emailcred
import time
import cv2

#Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
namecount = 0
#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"

#LCD Display Initialization
lcd = CharLCD(cols=16, rows=2, pin_rs=37, pin_e=35, pins_data=[33, 31, 29, 23], numbering_mode=GPIO.BOARD)

#GPIO Initialization
GPIO.setwarnings(False)

#SMS Module
#The SMS module contains the send_msg function which contains
#all of the necessary commands and credentials for the function
#to send the user a message that the garage is opened. This
#module also includes the function to send a 5-digit random
#code to the user for the first part of the authentication
#process.

def send_msg(name):
    messName = name
    # we will use Gmial accounts and SMTP protocol
    server = smtplib.SMTP_SSL( 'smtp.gmail.com', 465)

    # get login credentials from the file "emailcred.py"
    server.login( emailcred.FROM, emailcred.PASS )
    #Compile message string to print and send.
    #Ex: 'Button was pressed at 5:50:20 PM'
    actionMessage = ''.join([ '\n' + messName + ' opened the safe door. Time: ',
                        time.strftime('%I:%M:%S %p')])
    print(actionMessage)
    server.sendmail(emailcred.FROM, emailcred.TO, actionMessage)
    server.quit()


# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# initialize the video stream and allow the camera sensor to warm up
# Set the ser to the followng
# src = 0 : for the build in single web cam, could be your laptop webcam
# src = 2 : I had to set it to 2 inorder to use the USB webcam attached to my laptop
vs = VideoStream(src=0,framerate=10).start()
#vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	
	#Initialization of User ID which is passed in through the passcred.py file
	userID = passcred.NUMOID
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	# Detect the fce boxes
	boxes = face_recognition.face_locations(frame)
	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(frame, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown" #if face is not recognized, then print Unknown

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)

			#If someone in your dataset is identified, print their name on the screen
			
			if currentname != name:
				currentname = name
				print(currentname)
		# update the list of names
		names.append(name)
		
	if currentname == 'unknown':    
		#Do nothing except for clearing the lcd
		#Fail-Safe Default, if it does not recognize you
		#base level access and prompt will not be granted.
		lcd.clear()
		
	elif currentname != 'unknown':
		
		if namecount == 1:        
			currentname = 'unknown'
			namecount = 0
		else:
			lcd.write_string('Hi ' + currentname + '      Please Scan ID:')
			tempID = input("Scan ID Now: ")
			tmpID = int(tempID)
			namecount += 1
			lcd.clear()
			if userID == tmpID:
				lcd.write_string('Safe Unlocked')
				send_msg(currentname)
				sleep(5)
				lcd.clear()
				lcd.write_string('Safe locking    Keep Away')
				sleep(2)
				lcd.clear()
				continue
			else:
				lcd.write_string('ID not valid    Restart')
				sleep(2)
				lcd.clear()
				continue
	
	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image - color is in BGR
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)

	# display the image to our screen
	cv2.imshow("Facial Recognition is Running", frame)
	key = cv2.waitKey(1) & 0xFF

	# quit when 'q' key is pressed
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
GPIO.cleanup()
cv2.destroyAllWindows()
vs.stop()
