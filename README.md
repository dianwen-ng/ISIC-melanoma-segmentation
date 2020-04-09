# ISIC melanoma segmentation
**An example of building a semantic skin lesion segmentation model with CNN**

This is a quick hands on example of using deep learning modelling techniques to train an AI machine to perform detection and semantic segmentation on dermoscopy skin images.
The training dataset used in this hands on guide is provided by the `ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection`[1][2] and you can find your dataset [here](https://challenge2018.isic-archive.com/task1/).

In the dataset, there is about 2594 training RGB images with labelled mask and each images come in different resolution.
Here, we simply resize all images to 224x224x3. This may not be the best way to handle your image data as it introduces lots of distortion for elongated images.
However, we are only interested to give a quick and good way to produce a model that can learn to detect our skin lesion.
Thus, you may want to try out other processing techniques to further improve this guide.

A sample of the images
* Top row: Input images of dermoscopy skin images
* Bottom row: Labelled mask of the corresponding images (Target truth)
<img src="https://github.com/DW-Hwang/ISIC-melanoma-segmentation/blob/master/screenshots/image1.png" width= "640" height="350"/> 

For the deep learning architecture, we implemented a U-Net model to perform our detection and semantic segmentation and the details of the computation and all the other information about the convolutional network can be found in `Medical Image Recognition: An Explanation and Hands-on Example on Convolutional Networks`. A view of the U-Net architecture is shown below
<img src="https://github.com/DW-Hwang/ISIC-melanoma-segmentation/blob/master/screenshots/unet.png" width= "600" height="450"/>
During the construction of our model, we change the input shape to 224x224x3 as it is sufficient to preserve the information from the original resolution. 

The guide to the training can be seen in this [repo](https://github.com/DW-Hwang/ISIC-melanoma-segmentation/blob/master/U-net/segmentation.ipynb) 

## Results
A sample result of our model is shown 
<img src="https://github.com/DW-Hwang/ISIC-melanoma-segmentation/blob/master/screenshots/image2.png" width= "640" height="350"/> 

Have a good day ahead!

References

[1] Noel C. F. Codella, David Gutman, M. Emre Celebi, Brian Helba, Michael A. Marchetti, Stephen W. Dusza, Aadi Kalloo, Konstantinos Liopyris, Nabin Mishra, Harald Kittler, Allan Halpern: “Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC)”, 2017; arXiv:1710.05006.

[2] Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).

