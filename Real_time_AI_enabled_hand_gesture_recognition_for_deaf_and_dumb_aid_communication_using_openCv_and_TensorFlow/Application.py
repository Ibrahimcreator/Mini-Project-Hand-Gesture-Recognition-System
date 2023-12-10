import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from sklearn.ensemble import RandomForestClassifier

class HandGestureRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time AI-Enabled Hand Gesture Recognition")
        self.root.attributes("-fullscreen", True)

        self.project_info_label = tk.Label(root, text="Real-time AI-Enabled Hand Gesture Recognition\nDeveloped by A Syed Ibrahim, NT Suryaa, Jefry, Gowtham", font=("Arial", 20))
        self.project_info_label.pack(pady=20)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.panel = tk.Label(root)
        self.panel.pack(padx=10, pady=10)

        self.exit_button = tk.Button(root, text="Exit", command=self.exit_app, font=("Arial", 16))
        self.exit_button.pack(side="bottom")

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, min_tracking_confidence=0.3, max_num_hands=10)

        self.labels_dict = {
            0: "Hello: Greeting, friendly welcome.",
            1: "OK: Approval, agreement signal.",
            2: "Yes: Affirmative response.",
            3: "No: Negative response, disagreement.",
            4: "Need Help: Seeking assistance or support.",
            5: "Peace: Peaceful, tranquility symbol.",
            6: "Like: Positive approval, enjoyment.",
            7: "Dislike: Negative reaction, aversion.",
            8: "Call Me: Request for contact.",
            9: "You: Pointing to someone.",
            10: "Good Luck: Wishing success, fortune.",
            11: "Where: Inquiring about location.",
            12: "I Hate: Expressing strong dislike."
        }
        
        # Define 'W' and 'H' attributes
        self.W = None
        self.H = None

        # Load the model
        model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = model_dict['model']

    def draw_landmark_connections(self, frame, hand_landmarks):
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20)  # Little finger
        ]

        for connection in connections:
            x1, y1 = hand_landmarks.landmark[connection[0]].x, hand_landmarks.landmark[connection[0]].y
            x2, y2 = hand_landmarks.landmark[connection[1]].x, hand_landmarks.landmark[connection[1]].y

            cx1, cy1 = int(x1 * self.W), int(y1 * self.H)
            cx2, cy2 = int(x2 * self.W), int(y2 * self.H)

            cv2.line(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), 2, cv2.LINE_AA)  # White lines with anti-aliasing

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.H, self.W, _ = frame.shape  # Update 'H' and 'W'
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x, y, z = landmark.x, landmark.y, landmark.z
                        cx, cy = int(x * self.W), int(y * self.H)
                        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)  # Larger green landmarks

                    self.draw_landmark_connections(frame, hand_landmarks)

                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    x1 = int(min(x_) * self.W) - 10
                    y1 = int(min(y_) * self.H) - 10

                    x2 = int(max(x_) * self.W) - 10
                    y2 = int(max(y_) * self.H) - 10

                    prediction = self.model.predict([np.asarray(data_aux)])

                    predicted_character = self.labels_dict[int(prediction[0])]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.panel.img = img
            self.panel.config(image=img)
            self.panel.after(10, self.update_frame)

    def exit_app(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureRecognitionApp(root)
    app.update_frame()
    root.mainloop()
