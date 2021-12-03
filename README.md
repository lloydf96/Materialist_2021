# Materialist_2021
The goal of this project is to segment images into different pieces of clothing. I use the [iMaterialist-2021](https://www.kaggle.com/c/imaterialist-fashion-2021-fgvc8/data) dataset, which contains over 45K images with each image segmented into 47 classes.
The Solution employed involves:
1. Use a subset of these classes limiting my scope to classifying 6 classes. Many similar classes were clubbed into one single class.
2. UNET architecture was used, where the pretrained RESNET-34 was used for downsampling and CNN's were used for upsampling.
3. In order to reduce the noise in output (the noise being the ghost classes labeled for few pixels) a multi-output random forest was trained on the output of the images in the development set. The percent area of each class in the image as predicted by UNET was used as the input to the model, while the ground truth classes were considered to be the output for training.

Results:
1. Achieved an IOU of 0.703 on the test set using just the UNET and an IOU score of 0.801 by using both UNET and post processing using multi-output random forest.
2. On an average the model accurately identifies 5  out of 6 classes.
3. Refer to rf_on_unet.ipynb for detailed analysis of model performance.





