---
title: Clustering Jane Austen
date: 2011-08-17
description: Applying hierarchical clustering algorithms to the 100 most common words in Jane Austen's works
categories:
  - nlp
tags:
  - clustering
  - machine-learning
  - jane-austen
---

## Motivation

I'm curious about unsupervised [word sense disambiguation](https://en.wikipedia.org/wiki/Word-sense_disambiguation), and unsupervised machine learning in general. For that, Manning and Schuetze tell us we need [clustering](https://en.wikipedia.org/wiki/Cluster_analysis). I jumped ahead to Chapter 14 to experiment with clustering algorithms.

<!-- more -->

## A Survey of Clustering Algorithms

I generated some random 2D points and applied various [hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) algorithms, using inverse Euclidean distance as the similarity metric. I visualized the results using pygame/python, with each algorithm configured to produce $k = 10$ clusters.

**Key to the figures:** White dots are elements to be clustered. Red circles are cluster centers, where the area of a circle is proportional to the number of elements in the cluster. Green lines show cluster membership and hierarchy.

![Single-link binary hierarchical clustering of random points in 2-space. Note that most points have merged into a single cluster.](../../assets/clustering-single-link.png)

*Single-link binary hierarchical clustering of random points in 2-space. Note that most points have merged into a single cluster.*

![Complete-link binary hierarchical clustering of random points in 2-space](../../assets/clustering-complete-link.png)

*Complete-link binary hierarchical clustering of random points in 2-space.*

![Group-average binary hierarchical clustering of random points in 2-space](../../assets/clustering-group-avg.png)

*Group-average binary hierarchical clustering of random points in 2-space.*

![Single-link clustering, but with k=35](../../assets/clustering-single-link-35.png)

*Single-link clustering, but with $k = 35$. This higher $k$ brings the results more into agreement with the other algorithms for comparison purposes.*

## Dendrogram of Common English Words

I then applied [complete-link clustering](https://en.wikipedia.org/wiki/Complete-linkage_clustering) to the 100 most common words from [Jane Austen](https://en.wikipedia.org/wiki/Jane_Austen)'s complete works, attempting to replicate an analysis from Manning and Schuetze's textbook. The similarity function measures the overlap between [bigram](https://en.wikipedia.org/wiki/Bigram) distributions for words immediately preceding and following the words in question:

$$\text{sim}(a, b) = \frac{\sqrt{\sum_k P(k,a) P(k,b) \cdot \sum_k P(a,k) P(b,k)}}{P(a) P(b)}$$

I precomputed a similarity matrix covering the 300 most common words for efficiency. The resulting [dendrogram](https://en.wikipedia.org/wiki/Dendrogram) was too wide to display linearly, so I reformatted it as a circular "polar dendrogram":

![Polar Dendrogram of Complete-Link Clustering of the 100 Most Common Words in Jane Austen's Works](../../assets/austen-polar-dendrogram.png)

*Polar Dendrogram of Complete-Link Clustering of the 100 Most Common Words in Jane Austen's Works. Similarity is intuitively defined as the overlap between the bigram distributions for words immediately preceding and following the words in question. Note in particular the accurate separation of prepositions from conjunctions.*

## Embedding in 2-Space

I also investigated 2D embedding techniques for visualization. Treating embedding quality as potential energy:

$$Q = -V$$

we can derive forces from the potential:

$$\frac{\partial V}{\partial x} = F$$

This gives a second-order dampened differential system that treats word pairs as particles with attractive and repulsive forces, optimizing initially random embeddings for inverse-similarity distances.

![Embedding of common words using force-directed placement](../../assets/austen-word-cloud.png)

*Embedding of common words obtained using a soft attractive force plus a force which strongly repels words which approach closer than $1/\text{sim}$ pixels from each other.*

The major problem with this approach is that our data is not easily squashed down into two dimensions. While visualizations must ultimately be two-dimensional, alternative space-filling techniques for displaying hierarchical clusters could prove superior to traditional dendrograms.

Source code available upon request.

---

*Originally published on [Quasiphysics](https://quasiphysics.wordpress.com/2011/08/17/clustering-jane-austen/).*
