# SpiderTree

Project Overview: 
This project is an attempt to create a data classification pipeline for identifying spiders, specifically in the Infraorder Entelegynae. 

The project will use Python 3 to grab data from the iNaturalist api in the form of image + metadata, to construct a statistical model using a CNN, such as VGG16 or VGG19, that can be used to categorize further images. 

The model will categorize images in such a way that a distance table can be created from their categorizations, which can then be used to create a phyologenetic tree. When a new image is analyzed by the model, it is given such a distance score. 

The final output of the pipeline will be a phylogenetic tree with the analyzed image placed on the tree where the model thinks it fits in. 

Data
The project will begin with 10 categories of spider from the Infraorder Entelegynae set, each with 100 images. 
I will be choosing these images by selecting a random batch of 120 for each category and then manually trimming some from the sets.
120 was chosen as an arbitrary but educated-guess threshold; I manually looked through a number of images and noted one in five were unsuitable. I am simply assuming I will have to do the same with the random data sets I collect. 
Regardless, I will then manually trim or add images until I meet 100 per category.

I will be using the iNaturalist API found here: https://api.inaturalist.org/v1/docs/
When I figure out how it works I will update this section of the README to reflect my process and results.

The data will be stored in CSV format, with references to the folder holding the images, and to the image names themselves.
I will be heavily referencing this project: https://github.com/valanm/off-the-shelf-insect-identification/
In the data sets directory of the above project, there is a dated but workable python script for assembling the appropriate data to pass to VGG16. 

Note that I am not being as precise as the above project - I don't care about the specific view or the clarity of image. I may alter the training dataset later but hastiness for now is the key.
