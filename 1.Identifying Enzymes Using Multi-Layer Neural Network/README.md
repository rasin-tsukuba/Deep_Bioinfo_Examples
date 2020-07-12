# Fully Connected Psepssm Predict Enzyme

The paper claimed that:

> In our implementation, ... We use ReLU as the activation function, cross-entropy loss as the loss function, and Adam as the optimizer. We utilize dropout, batch normalization and weight decay to prevent overfitting. With the help of Keras, we build and train the model in 10 lines. Training the model on a Titan X for 2 minutes, we can reach around 94.5% accuracy, which is very close to the state-of-the-art performance. Since bimolecular function prediction and annotation is one of the main research directions of bioinformatics, researchers can easily adopt this example and develop the applications for their own problems.

The network architecture of their implemetation is like:

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200712100558.png)

In fact, we didn't see the implementation of **batch normalization**, **weight decay** in their code. The accuracy is soon reach to *94.5%* and can hardly get higher. Using *1024* neuron in this trivial task is unnecessary and it converges very fast.

In our PyTorch implementation, we using the same network structure as shown below:

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200712100927.png)

The training process is here:

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200712101110.png)

