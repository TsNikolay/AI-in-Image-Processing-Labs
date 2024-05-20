import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

class ImageClassifier:
    def __init__(self, dataset_name='imagenette/160px-v2', data_dir='D:/tensorflow_datasets'):
        try:
            self.data, self.info = tfds.load(dataset_name, with_info=True, as_supervised=True, data_dir=data_dir)
            self.trainSet, self.validationSet = self.data['train'], self.data['validation']
            self.classNames = ['tench', 'English springer', 'cassette player', 'chainsaw',
                                'church', 'horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
        except Exception as e:
            print(f"Помилка завантаження даних: {e}")
            raise SystemExit

        self.model = None

    def processImage(self, image):
        image = tf.image.per_image_standardization(image)
        image = tf.image.resize(image, (64, 64))
        return image

    def preprocessDataset(self):
        trainDataset = self.trainSet.map(lambda image, label: (self.processImage(image), label)).batch(32)
        validationDataset = self.validationSet.map(lambda image, label: (self.processImage(image), label)).batch(32)
        return trainDataset, validationDataset

    def createAlexNet(self):
        self.model = models.Sequential([
            layers.Conv2D(filters=128, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(64, 64, 3)),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3)),
            layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary() 

    def trainModel(self, epochs):
        trainDataset, validationDataset = self.preprocessDataset()
        history = self.model.fit(trainDataset, epochs=epochs, validation_data=validationDataset, validation_freq=1)
        self.model.save('class_recognizer_imagenet1.h5')
        return history

    def loadModel(self, path):
        self.model = models.load_model(path)

    def displayImage(self, image, prediction):
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.xlabel(f'{prediction}') 
        plt.imshow(image)
        plt.show()

    def showAccuracy(self, history):
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()

    def makePredictions(self, numPredictions=10):
        self.validationSet = self.validationSet.shuffle(buffer_size=len(self.validationSet))
        for _ in range(numPredictions):
            image, label = next(iter(self.validationSet))
            processedImage = self.processImage(image)
            processedImage = tf.expand_dims(processedImage, axis=0)
            prediction = np.argmax(self.model.predict(processedImage)[0])
            self.displayImage(image.numpy(), self.classNames[prediction])


def main():
    classifier = ImageClassifier()
    #classifier.createAlexNet()
    #history = classifier.trainModel(20)
    #classifier.showAccuracy(history)
    classifier.loadModel('class_recognizer_imagenet1.h5')
    classifier.makePredictions(10)

main()
