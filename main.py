import cv2
from simple_facerec import SimpleFacerec
import pandas as pd
from datetime import datetime

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(0)

# Coloumn headers in the csv file
df = pd.DataFrame(columns=['Name', 'Time'])

last_detected_name = None  # Variable to store the last detected name

while True:
    ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        if name == 'Unknown':   
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        else:
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 4)
            # Check if the detected name is different from the last detected name
            if name != last_detected_name:
                now = datetime.now()
                data = [name, now.strftime("%d/%m/%Y %H:%M:%S")]
                df.loc[len(df)] = data
                df.to_csv('data.csv', index=False)
                last_detected_name = name  # Update the last detected name

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
