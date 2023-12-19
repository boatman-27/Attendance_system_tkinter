import tkinter as tk
from tkinter import ttk, font
import cv2
from simple_facerec import SimpleFacerec
import pandas as pd
from datetime import datetime

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Column headers in the csv file
df = pd.DataFrame(columns=['Name', 'Time'])

def startCamera():
    cap = cv2.VideoCapture(0)
    last_detected_name = None
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

def show_attendance():
    data = pd.read_csv('data.csv')
    show_table(data)

def show_table(data):
    # Destroy existing Treeview widget
    for widget in window.winfo_children():
        if isinstance(widget, ttk.Treeview):
            widget.destroy()

    # Create a new Treeview widget
    tree = ttk.Treeview(window, columns=list(data.columns), show="headings")
    for col in data.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    # Insert data into the Treeview
    for index, row in data.iterrows():
        tree.insert("", "end", values=list(row))

    tree.pack(expand=True, fill="both")

# Tkinter window setup
window = tk.Tk(className='Face Recognition')
window.geometry('600x600')
label_font = font.Font(weight="bold", size=36)
label = tk.Label(text="Face Recognition", font=label_font, fg="#a89b22", bg="#030945")
label.pack()
window.configure(bg='#030945')
convert_button = tk.Button(window, text="Take Attendance", command=startCamera, font=("Helvetica", 14), bg="#a89b22", fg="white")
convert_button.pack(pady=20)
convert_button = tk.Button(window, text="Show Attendance", command=show_attendance, font=("Helvetica", 14), bg="#a89b22", fg="white")
convert_button.pack(pady=20)
window.mainloop()
