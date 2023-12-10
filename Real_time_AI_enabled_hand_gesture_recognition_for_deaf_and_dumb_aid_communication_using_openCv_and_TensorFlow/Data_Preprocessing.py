import os
import pickle
import mediapipe as mp
import cv2

class HandLandmarkExtractor:
    def __init__(self, data_dir, output_file):
        self.data_dir = data_dir
        self.output_file = output_file
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    def extract_hand_landmarks(self):
        data = []
        labels = []

        for label in os.listdir(self.data_dir):
            for img_filename in os.listdir(os.path.join(self.data_dir, label)):
                image_data = self.process_image(label, img_filename)
                if image_data is not None:
                    data.append(image_data)
                    labels.append(label)

        self.save_data(data, labels)

    def process_image(self, label, img_filename):
        img_path = os.path.join(self.data_dir, label, img_filename)
        data_aux = []

        x_coordinates = []
        y_coordinates = []

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    x_coordinates.append(x)
                    y_coordinates.append(y)
                    data_aux.extend([x - min(x_coordinates), y - min(y_coordinates)])

        return data_aux if data_aux else None

    def save_data(self, data, labels):
        with open(self.output_file, 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)

if __name__ == "__main__":
    DATA_DIR = './data'
    OUTPUT_FILE = 'data.pickle'

    extractor = HandLandmarkExtractor(data_dir=DATA_DIR, output_file=OUTPUT_FILE)
    extractor.extract_hand_landmarks()
