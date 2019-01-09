# TPU-VAE
A MNIST example of variational auto encoder in TensorFlow/Keras by TPUs.

# Overview
Random numbers for sampling and reparameterization trick, and KL divergence of the middle layer are connected by skip-connection, respectively.

![](https://github.com/koshian2/TPU-VAE/blob/master/images/model.png)

This makes it possible to define VAEs in one model, making it easier to train with TPU.
