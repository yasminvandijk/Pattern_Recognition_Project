# Classifying GTSRB images by using data augmentation, DCGAN and a CNN.
### Project for the pattern recognition course.

This project contains several python files. Most of them are to get the restuls found in the paper, except for the Pix2Pix and the Sqeezenet files. We tried to implement those frameworks, but were not able to make them work for us. Still, we left them in the repository for potential later purposes. 

**DataLoad**: allows to import the dataset files. 

**CNN**: contain the CNN model used in the paper.

**DataAugmentation**: contains the training of the CNN, using traditional data augmentation methods.

**DCGAN**: Notebook that we used for defining and training the DCGAN.

**CNN + DCGAN**: Notebook that we used for training the CNN with generated images from the DCGAN as input data.

**trained_generator_model_200.h5**: trained generator model after 200 epochs that we used for generating new images for the CNN input.


