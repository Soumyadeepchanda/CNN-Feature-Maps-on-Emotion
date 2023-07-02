# CNN Feature Maps Visualization for Emotions using FER 2013 Dataset
This project aims to visualize the feature maps of a Convolutional Neural Network (CNN) trained on the FER 2013 dataset to predict emotions. The FER 2013 dataset contains facial images labeled with various emotions, including angry, happy, sad, and more.

## Dataset
The FER 2013 dataset used in this project can be obtained from the following link: <br>[FER 2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)<br>

The dataset consists of grayscale images of size 48x48 pixels, and each image is labeled with one of the following emotions:<br>

Angry<br>
Happy<br>
Sad<br>


For this project, we will focus on the emotions of angry, happy, and sad.

## Data Preprocessing
Before visualizing the feature maps, it's necessary to preprocess the dataset. This step involves loading and resizing the images, converting them to grayscale, normalizing the pixel values, and splitting the dataset into training and testing sets.

## Model Architecture
The CNN model used for this project will have multiple convolutional and pooling layers followed by fully connected layers. The specific architecture details and hyperparameters can be found in the accompanying notebook or script.

## Model Training
Train the CNN model using the preprocessed dataset. Use appropriate training techniques such as batch normalization, dropout, and optimization algorithms like Adam to improve the model's performance.

## Feature Maps
To visualize the feature maps of the trained CNN model, we can follow the following steps:<br><br>
Load the pre-trained CNN model: Load the weights of the pre-trained CNN model that was trained on the FER 2013 dataset.<br><br>
Select sample images: Choose a few sample images from the dataset that represent the emotions of angry, happy, and sad.<br><br>
Extract feature maps: Pass the selected sample images through the pre-trained CNN model and extract the feature maps from intermediate layers. These feature maps capture the learned representations of different features in the images.<br><br>
Visualize the feature maps: Plot the feature maps for each emotion category (angry, happy, and sad). Each feature map represents a specific learned feature or pattern that the model uses to classify the emotions.<br><br>
Interpretation: Analyze the patterns and activations in the feature maps to gain insights into what the model focuses on when predicting different emotions. This can help in understanding which facial features or regions contribute to the model's decision-making process.
