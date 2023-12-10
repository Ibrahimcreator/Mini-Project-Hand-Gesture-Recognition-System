import os
import cv2

class DataCollection:
    def __init__(self, data_dir='./data', number_of_classes=13, dataset_size=100):
        self.data_dir = data_dir
        self.number_of_classes = number_of_classes
        self.dataset_size = dataset_size
        self.cap = cv2.VideoCapture(0)

    def create_data_directories(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        for class_id in range(self.number_of_classes):
            class_dir = os.path.join(self.data_dir, str(class_id))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

    def capture_data(self):
        self.create_data_directories()

        for class_id in range(self.number_of_classes):
            print(f'Collecting data for class {class_id}')
            self.wait_for_key_press()
            self.capture_images_for_class(class_id)

        self.cap.release()
        cv2.destroyAllWindows()

    def wait_for_key_press(self):
        done = False
        while not done:
            ret, frame = self.cap.read()
            cv2.putText(frame, 'Press S To Start', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('s'):
                done = True

    def capture_images_for_class(self, class_id):
        counter = 0
        while counter < self.dataset_size:
            ret, frame = self.cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            img_path = os.path.join(self.data_dir, str(class_id), f'{counter}.jpg')
            cv2.imwrite(img_path, frame)
            counter += 1

if __name__ == "__main__":
    data_collector = DataCollection(data_dir='./data', number_of_classes=13, dataset_size=100)
    data_collector.capture_data()
