# Using-CNN-in-Keras-for-object-Recognition
Background:
Computer vision applications heavily rely on object classification, which involves categorizing objects from different object categories. In Duckietown, object classification can be useful for tasks such as object detection and tracking, especially for informative objects like traffic signs, Duckiebot, duckies, house models, etc. One of the most popular techniques for object classification is Convolutional Neural Networks (CNN), a variant of neural networks that uses convolutional layers, pooling layers, fully connected layers, and normalization layers.

Problem Statement:
The objective of this project is to build a CNN-based classification model for furniture objects. The model will classify furniture into 5 classes using a dataset of furniture images.

Approach:
To accomplish the objective, the project will use the Keras library in Python to build the CNN model. The dataset will be preprocessed by cleaning and standardizing the images, then it will be split into training and validation sets. The CNN model will be trained on the training set, and its performance will be evaluated on the validation set. The hyperparameters of the CNN model will be tuned to achieve the best possible accuracy.


Methodology:
To implement the model, the first step involves downloading the dataset from the drive, followed by importing relevant libraries, such as TensorFlow Keras, NumPy, matplotlib.pyplot, and others.

Building the Model:
The input images for the convnet are resized to 150x150 color images during the data preprocessing step. To stack the convolutional layers, we will use the {convolution + relu + maxpooling} modules. Each convolution will operate on 3x3 windows, and each maxpooling layer will operate on 2x2 windows. The first convolution will extract 16 filters, followed by the second one which extracts 32 filters, and the last one which extracts 64 filters. Two fully connected layers are added before the output layer. The output layer will use softmax activation to classify images.

Data Preprocessing:
To ensure that all the images have the same dimensions, we will resize them to 150x150 pixels before feeding them into the neural network. This step ensures consistency in the dataset and avoids any size-related issues that might affect the performance of the model.

Expected Outcome:
The expected outcome of the model is accurate image classification for the given dataset of furniture images. The trained model will be able to classify images into their respective classes using softmax activation. The model's performance will be evaluated based on its accuracy, precision, and recall. The results will be visualized using matplotlib.pyplot library, enabling easy interpretation of the performance metrics.

Standard HyperParameters
HyperParameters	-
Conv Layers	3 Layers
Filter Size	3
Epochs	5
L2 Regularisation	0 (REMOVE)
Dropout	0 (REMOVE)
Batch	size 16
Optimizer	adam
Loss	Categorical crossentropy

Outcomes:
The standard results represent the model's performance without any additional hyperparameters. These results are obtained by using standard hyperparameters that are commonly used in the model.

Graphical Representation:
The accuracy of the model across epochs is represented in a graph, displaying the changes in accuracy over time.

Analysis:
By altering the hyperparameters of the model, we can observe changes in the results and analyze for the best possible model. However, the impact of these changes may vary with different datasets, so it is essential to choose the hyperparameters carefully.

Number of Convolutional Layers:
We compared the model's performance with 5 and 2 convolutional layers, and the accuracy tweaks with epochs are shown below. The other hyperparameters in each comparison remain standard.

L2 Regularization:
We added L2 regularization of 0.01 in the first 2 hidden layers of the model and compared the results with the model without L2 regularization.

Dropout:
We added a dropout of 0.2 between the input layer and the first hidden layer and compared the results with the model without dropout.

Filter/Kernel Size:
We compared the model's performance with filter size 6 and filter size 3.

Analysis of Results:
From the above results, we can infer that adding or removing L2 regularization, changing the layer count, or altering the filter size did not significantly impact the model's performance in the last epoch. The accuracies ranged between 0.83 to 0.89. However, adding a dropout of 0.2 increased the train accuracy of the model up to 0.945 and validation accuracy up to 0.885 in the last epoch. Therefore, the addition of dropout regularization can be considered for improving the model's performance.







