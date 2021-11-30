# Materialist_2021
The goal of this project is to segment images into different pieces of clothing. I use the [iMaterialist-2021](https://www.kaggle.com/c/imaterialist-fashion-2021-fgvc8/data) dataset, which contains over 45K images with each image segmented into 47 classes. 
I use a subset of these classes limiting my scope to classifying 6 classes. Many similar classes were clubbed into one single class.
UNET architecture was used, where the pretrained RESNET-34 was used for downsampling and CNN's were used for upsampling.
In order to reduce the noise in output (the noise being the ghost classes labeled for few pixels) a multi-output random forest was trained on the output of the images in the development set. The percent area of each class in the image as predicted by UNET was used as the input to the model, while the ground truth classes were considered to be the output for training.

