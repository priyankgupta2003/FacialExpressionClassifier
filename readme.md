**Facial Expression Recognition with Deep Learning**

This code uses deep learning techniques to perform facial expression recognition on the FER2013 dataset, which contains images of faces labeled with one of seven possible emotions (angry, disgust, fear, happy, sad, surprise, neutral). The code uses two different models for classification: a custom convolutional neural network (CNN) and a pre-trained MobileNetV2 model.

**Installation**

Clone this repository: git clone this repository
Install the required packages: pip install -r requirements.txt
Usage
Download the dataset from Kaggle <https://www.kaggle.com/datasets/samaneheslamifar/facial-emotion-expressions> and extract the files into a directory called archive.
Run the following command to train and test the custom CNN model: python train_custom_model.py
Run the following command to train and test the pre-trained MobileNetV2 model: python train_pretrained_model.py
Run the following command to train and test the custom CNN model with oversampling: python train_custom_model_oversampling.py
Note: Training the models can take a significant amount of time, depending on your hardware.

**Results**

The custom CNN model achieved a test accuracy of approximately 62% after 50 epochs, while the pre-trained MobileNetV2 model achieved a test accuracy of approximately 67% after 50 epochs. After oversampling the data, the custom CNN model achieved a test accuracy of approximately 65% after 50 epochs.

**Dataset used**

The FER2013 dataset is used in this code, which contains 35,887 grayscale images of faces labeled with one of seven emotions. The dataset is split into three sets: training, validation, and test.

**Approach**

Two different models are used for classification: a custom convolutional neural network (CNN) and a pre-trained MobileNetV2 model.

For the custom CNN, we first load and preprocess the dataset, splitting it into training and test sets. We then define the architecture of the CNN, compile it, and train it on the training set with data augmentation. Finally, we evaluate the CNN model on the test set.

For the pre-trained MobileNetV2 model, we define the architecture of the model and load the pre-trained weights. We then compile the model and train it on the training set. Finally, we evaluate the pre-trained model on the test set.

To handle imbalanced data, we apply oversampling using the RandomOverSampler method from the imblearn library.

**Technical Challenges**

One of the main technical challenges was handling the imbalanced data in the FER2013 dataset. We used the RandomOverSampler method from the imblearn library to oversample the minority classes and balance the dataset.

Another technical challenge was converting the grayscale images to RGB for the pre-trained MobileNetV2 model, which expects RGB input images. We concatenated the same grayscale image three times along the channel axis to convert it to RGB.


**Conclusion**

Facial expression recognition is an important task in computer vision and has many potential applications, such as in human-computer interaction, psychology, and marketing. This code provides an example of how to perform facial expression recognition using deep learning techniques and demonstrates the use of oversampling to handle imbalanced data.
