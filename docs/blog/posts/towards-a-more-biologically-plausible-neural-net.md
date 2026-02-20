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

Of the many machine learning models, the [artificial neural network](https://en.wikipedia.org/wiki/Artificial_neural_network) (ANN) is of particular interest because of the obvious analogy to the function of the brain. However, the standard supervised cost function and error [back-propagation](https://en.wikipedia.org/wiki/Backpropagation) algorithm are entirely implausible from a biological perspective, and in practice the performance of back-prop decreases sharply with the number of hidden layers, requiring more and more labeled training examples which are often in short supply.

<!-- more -->

So there are two problems: biological implausibility and computational inefficiency. [Semi-supervised learning](https://en.wikipedia.org/wiki/Semi-supervised_learning) approaches such as [sparse autoencoders](http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity) can reduce reliance on labeled data by leveraging information-theoretic constraints such as [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). Applied iteratively to multilayer networks, sparse autoencoders can learn to extract structure from unlabeled data and in so doing learn very good features from which traditional supervised learning may then be applied.

However, these approaches remain biologically implausible. My proposal is an entropy-based cost function directly on the hidden layer, plus some terms to reward conservation of energy (sparsity of outputs) and materials (sparsity of input weights). The idea is that hidden units should discover [localized](https://en.wikipedia.org/wiki/Receptive_field) input responses, potentially enabling [convolutional network](https://en.wikipedia.org/wiki/Convolutional_neural_network) constraints to emerge naturally.

The full [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) cost function computation is exponential in the number of nodes. However, partial derivatives are implementable for some low dimensionality, and the approach may still prove useful for exploring whether biologically motivated cost functions can yield practical learning algorithms.

**References:**

- [General Formalism for Neural Networks](http://ufldl.stanford.edu/wiki/index.php/Neural_Networks) (Stanford UFLDL)
- [Supervised Cost Function and Back-Propagation](http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm) (Stanford UFLDL)
- [Convolutional Neural Networks](http://deeplearning.net/tutorial/lenet.html) (deeplearning.net)

---

*Originally published on [Quasiphysics](https://quasiphysics.wordpress.com/2013/10/04/towards-a-more-biologically-plausible-neural-net/).*
