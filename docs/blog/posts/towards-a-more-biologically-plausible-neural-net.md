---
title: Towards a (More) Biologically Plausible Neural Net
date: 2013-10-04
description: Proposing an entropy-based cost function for neural networks that is more biologically plausible than back-propagation
categories:
  - ml
tags:
  - ml
  - neuro
---

Of the many machine learning models, the [artificial neural network](http://ufldl.stanford.edu/wiki/index.php/Neural_Networks) (ANN) is of particular interest because of the obvious analogy to the function of the brain. However, the standard [supervised cost function and error back-propagation algorithm](http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm) are entirely implausible from a biological perspective, and in practice the performance of back-prop decreases sharply with the number of hidden-layers, requiring more and more labeled training examples which are often in short supply.

<!-- more -->

So we have two sorts of problems with error back-prop, some biological and some computational. In Machine Learning we can try to reduce the need for labeled training data by constructing an autoencoder network and training it on unlabeled data. The [standard approach to training autoencoders](http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity) is to use an output layer with the same number of nodes as the input layer and then apply the input data as the target output values in a standard error back-propagation technique. It is well-known that this trick works best if we constrain the hidden-layer nodes to fire only sparsely, using an information theoretic measure ([KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)).

[Applied iteratively to multilayer networks](https://www.youtube.com/watch?v=ZmNOAtZIgIk), sparse autoencoders can learn to extract structure from unlabeled data and in so doing learn very good features from which traditional supervised learning may then be applied. This final supervised learning step may only require a handful of labeled training examples per class to attain high accuracy. This is the so-called [semi-supervised](https://en.wikipedia.org/wiki/Semi-supervised_learning) approach and has enjoyed some recent success in computer vision.

These backprop-based sparse autoencoders suffer from two problems however: they are still computationally inefficient to train, and moreover they are not really any more biologically plausible than the standard ANN. While the sparsity requirement is on the right track, being analogous to biological energy minimization, the cost-function and back-propagation are still implausible.

So, I propose using an [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory))-based cost function directly on the hidden layer, plus some terms to reward conservation of energy (sparsity of outputs) and materials (sparsity of input weights). I believe that just this principle of parsimony should be sufficient to learn a compressed representation at each layer of the network.

Unfortunately the full entropy cost function computation is exponential in the number of nodes. Nonetheless I would like to assess this model's performance in comparison with the Stanford sparse autoencoders. The partial derivatives of this cost function can be worked out in closed-form, and should be implementable for some low dimensionality. If this model proves interesting, it would be worth finding a more computationally efficient form of the cost function. Crucially, if it is found that the full-blown entropy cost function tends to train hidden units with "[localized](https://en.wikipedia.org/wiki/Receptive_field)" input responses (due to, for example, correlation between nearby input pixels of natural images), then we may keep the cost function but instead constrain the geometry of the network along the lines of [Convolutional NNs](http://deeplearning.net/tutorial/lenet.html).

**References:**

- [General Formalism for Neural Networks](http://ufldl.stanford.edu/wiki/index.php/Neural_Networks) (Stanford UFLDL)
- [Supervised Cost Function and Back-Propagation](http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm) (Stanford UFLDL)
- [Convolutional Neural Networks](http://deeplearning.net/tutorial/lenet.html) (deeplearning.net)

---

*Originally published on [Quasiphysics](https://quasiphysics.wordpress.com/2013/10/04/towards-a-more-biologically-plausible-neural-net/).*
