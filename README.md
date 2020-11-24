## Image Classification:
Image classification is a widely used and very meaningful task to train our deep learning models on. In this problem we have an input image with a label, from a fixed set of categories. This is one of the core problems in Computer Vision, and has very wide practical applications. 
- ``Classification`` : A classifier is a system that inputs a vector of discrete and/or continuous feature values and outputs a discrete value, which is the class. Example: Classify an email as Spam or No Spam

### Image Classification Pipeline:
In this task, we take an input as an array of pixels that represent a single image and add a label to it.
1. **Input**: Our input consists of a set of images, each one is labeled as one of the classes. It is in the form of rows, columns and channels 
1. **Learning**: We want to train the model to learn what every one of the classes looks like.
1. **Evaluation**: We evaluate the quality of the classifier by asking it to predict labels for a new set of images. We will compare the true labels with the predicted ones.   

## DataSet
The dataset used in the project are as follows:
- **ImageNet** is a dataset of over 15 million labeled high-resolution images belonging to roughly 22,000 categories. 
- [Kaggle dataset](https://www.kaggle.com/c/siim-isic-melanoma-classification/data)

## Important topics:
1. **Filter(kernel)**:
A filter is multiplied with the input image and it results in a single output. The input has similar number of channels and fewer number of rows than input image. This filter is repreatedly applied to the input image resulting in 2-D output map of activations called **feature maps**. The weights in this filter represents the structure or the feature that the filter will detect. In the layers we specify the number of filters and the shape of the filter. The filter isinitialized with randomw weights.
- **Multiple Filters**: CNN's can learn multiple filters in parallel for a given input. This gives out model the ability to learn multiple features simultaneously. 
- **Multiple channels**: In colored images the, input comes in with multiple channels, ex. 3 in a colored image - RGB. In this case the filter also need to be in similar shape, with same number of channels [3*3*3]. If the input had 3 channels, then each filter will have 3 channels, one for each layer. 
- **Multiple Layers**: Convolution layer can be applied in hierarchial form, stacked upon each other. Filters applied directly on the input image, will extract low level features such as lines, filters on top of these features which can detect more detailed features like faces, hourses etc

1. **Strides**: the amount by which the filter shifts is the stride. Stride is normally set in a way so that the output volume is an integer and not a fraction.

1. **Pooling**: It is the laye where we do the resampling. It is a basically a filter of size (2*2)  and strides of same length.  One one of the most common sampling methods is Maxpooling. Maxpooling selects the maximun value in the block and outputs that. Some other examples of pooling are ``Average pooling`` & ``L-2 Norm pooling``.
- The reason for adding this layer is that the exact location of the feature should not be important, its relative position to other features is more important to be detected.
- This layer reduces the amount of parameters, hence reducing the computational cost and reduces ``over-fitting``. 

1. **Transfer Learning** :
It is a very important concept, which leverages the pre-trained models . Here the tasks could be different, but it should be related. 
- There are essentially 2 ways:
    - Use the pre trained model completely
    - Fine-tune the pre-trained model: uses atleast 50% or more if the tasks are quite different. To implement this we remove the last predicting layer of the pre-trained model and replace them with our own predicting layers.
- Example: If the pre-trained model is trained for cars, and you want to use it for trucks. The tasks are similar , so we can probably fine tune, 30% of the original model. The fine tuning takes time and effort and most importantly depends on nature of the task. If we do not want to load the last fully connected layers which act as the classifier. 
    - We accomplish that by using **include_top=False**. We can add our own fully connected layers on the top of ResNet50 model for our specific classification.
    - We freeze the weights of the model by setting trainable as “False”. This stops any updates to the pre-trained weights during training. This is because we want to leverage the knowledge learnt by the network from previous data.
[Hands on guide to Transfer Learning](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a)
[link in python](https://github.com/dipanjanS/hands-on-transfer-learning-with-python/blob/master/notebooks/Ch05%20-%20Unleash%20the%20Power%20of%20Transfer%20Learning/CNN%20with%20Transfer%20Learning.ipynb)

1. **Vanishing Gradient or Degradation Problem**:
This is one of the problems in training neural networks. When we add more layers using activation functions to the Neural network, the gradient of the loss function approaches to 0., making it hard to train. Specifically this problem makes it really hard to learn and tune the parameters of the earlier layers in the network.
- Gradient based methods learn a parameter's value by understanding how a small change in the parameter's value will affect the network's output. If a change in the parameter's value causes very small change in the network's output - the network just can't learn the parameter effectively, which is a problem.
- The gradients of the network's output with respect to the parameters in the early layers become extremely small. That's a fancy way of saying that even a large change in the value of parameters for the early layers doesn't have a big effect on the output.

[Gradient problem](https://www.quora.com/What-is-the-vanishing-gradient-problem)
[How to Fix the Vanishing Gradients Problem Using the ReLU](https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/)

1. **Translation Invariance** : 
It is the ability to ignore the positional shifts or transitions in the image. Example: A car is a car, regarless of whether its stright or upside down.
- Transitional invariance is acheived by a combination of convolutional layers and max pooling layers. 
    - Firstly the ConvD reduces the image to a set of features 
    - MaxPooling takes this output from CovD and reduces its resolution and the complexity
- If we train a convolutional Neural Network on images of a target, it will work on the shifted images as well
- CovD layers work better for transitional invariance
[Translation Invariance in Convolutional Neural Networks](https://medium.com/@divsoni2012/translation-invariance-in-convolutional-neural-networks-61d9b6fa03df)

1. **Data Augmentation** :
It is an similar to regularization steps, which are used to make a model more robust. It is done by add some kind of distortion in the training images, and train the model with this augmented dataset. When we have our test images with high distortion, this step becomes essential to make a robust model.
- **Keras** has an amazing feature known as *"ImageDataGenerator"* which generates batches of tensor image data with real world data augmentation
    - We can zoom the image randomly by a factor of 0.3 using the **zoom_range** parameter. 
    - We rotate the image randomly by 50 degrees using the **rotation_range** parameter. 
    - Translating the image randomly horizontally or vertically by a 0.2 factor of the image’s width or height using the **width_shift_range** and the **height_shift_range** parameters. 
    - Applying shear-based transformations randomly using the **shear_range** parameter. 
    - Randomly flipping half of the images horizontally using the **horizontal_flip parameter**. 
    - Leveraging the **fill_mode** parameter to fill in new pixels for images after we apply any of the preceding operations (especially rotation or translation). 
- Example: [image augmentation with Keras](https://keras.io/api/preprocessing/image/)
""train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                    horizontal_flip=True, fill_mode=’nearest’)""

1. **Callback**:
- History Callback: It remembers and records training metrics for each epoch. It includes the loss and the accuracy (for classification problems) as well as the loss and accuracy for the validation dataset.

1. **Parameters in a CNN model**:
In all the different CNN's we have differemt number of parameters. In CNN's we are just trying to learn the values of the filters using bakcpropagation or we can say that we are trying to learn these weight matrices. ``The number of parameters in each layer is equal to the count of these learnable elements in a filter ``. These parameters are generally the weights learnt during the training process. 
- Lets go step by step to ``count the number of parameters``:
    - Input layer :  does not have any learnable parameters so its 0
    - ConvD layer : This leyer will learn , so we have parameters int his layer. ( ( (shape of width of the filter * shape of height of the filter * number of filters in the previous layer ) + 1)*number of filters)
    - POOL Layer: This layer doesnt have any learning parameters. It is used for downsampling and just reduces the size of the feature map by 2.
    - Fully connected layer: It has learnable parameters, as every neuron in this layer is connected with each other.  ((current layer neurons c * previous layer neurons p)+1*c).
[Count the number of parameters](https://stackoverflow.com/questions/42786717/how-to-calculate-the-number-of-parameters-for-convolutional-neural-network)

## Models
<img_src="Images/imagenet_winner_models.png">

``Therte are multiple Concolution Network Architectures such as :``

1. VGG 16:
**VGG**, which won the 2014 Imagenet competition, and is a very simple model to create and understand.
<img src="Images/vgg16.png">

- ``Parameters :`` 138 MILLION

1. VGG 19:
It is a model built by stacking convolutionals together, but the depth of the model is limited because of issue called "Diminishing Gradient". This issue makes the CNN training hard. 
1. Resnet50:
Resnet50 is used to solve the issue of diminishing gradient.  The idea is to skip the connection and pass the residual to the next layer so that the model can continue to train. With Resnet models, CNN models can go deeper and deeper.
- ``Parameters:`` 25 million 
1. Inception V3:
It is another type of CNN. It consits of many Convolution Layers, and max pooling layers. Finally it includes fully connected neural network. 
1. Xception:
Xception stands for the "extreme inception"

## Convolution Neural Network Tutorial with Tensorflow:
https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/

## Library used:
- **Keras** is a high-level neural networks API, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation.
- **Theano** is a joint work done by some high profile researchers such as Yoshua Bengio and others Montereal Institute for Learning Algorithms (MILA). [Tutorial on Theano](https://archive.org/details/Scipy2010-JamesBergstra-TransparentGpuComputingWithTheano)
- **TensorFlow** was developed by researchers and engineers working on the Google Brain Team within Google’s Machine Intelligence research organization. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them


## Resources:
1. [Keras guide to building powerful image classification model using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

topics:
loss function
optimizers

