---
title: Playing Genetic Doublets in 3D
date: 2010-12-05
description: Using a genetic algorithm in Scheme to embed Lewis Carroll's word-ladder game into 3D space
categories:
  - programming
tags:
  - bio
  - comp
  - math
---

Since reading about [Lewis Carroll](https://en.wikipedia.org/wiki/Lewis_Carroll)'s game of [Doublets](https://en.wikipedia.org/wiki/Word_ladder) in Pappas' Joy of Mathematics, I have loved the game and often thought about graphs (networks) in terms of Doublets. Here I present the basics of embeddings, and demonstrate using a [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) to optimize an embedding of a list of three-letter words from a few games of Doublets.

<!-- more -->

## Theory of Embedding

Suppose we have some set of $N$ points in an abstract space. Instead of taking as given coordinates for each point as we would in a concrete space, we take as given a distance function among the points. So, for any $i$ and $j$ in $\mathbb{Z}^N$ we have a distance $\Delta_{ij}$. Of this distance function we demand three properties: non-negativity, symmetry, and the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality).

The problem of embedding is to take the $N$ points and find for them some correspondents in a space some small number of dimensions that in some way resembles the relationships (distances) between the points in their native, abstract space. For example, an embedding of the points in $\mathbb{R}^3$ is the association of a 3D vector $x_i$ with each point $i$.

The quality $Q$ of the embedding is a real number which indicates how closely the distances within the embedding $|x_i - x_j|$ are in agreement with the abstract distances $\Delta_{ij}$. We measure using a weighted mean-square error function that emphasizes closely-spaced points more heavily than distant ones:

$$Q = -\sum_{ij} \frac{1}{\Delta_{ij}^2}\left(\Delta_{ij} - |x_i - x_j|\right)^2$$

## What does this have to do with Doublets?

First take a set of words $W$ = {pig, pin, pen, fen, fey, fly, dig, dog, dot, cot, cat, rat, mat, mit, pit, fry, cry, coy, peg, leg, let, lot}. Now define an undirected graph where the vertices are the words and an edge exists between two words iff the words differ by exactly one letter. And so this graph maps out all possible games of Doublets using only words from $W$.

Now define the distance $\Delta_{ij}$ between two words to be the length of the shortest path from one to the other within the graph. That is, the distance is the length of the shortest game of Doublets that can be played with those words as end-points.

## Applying a Genetic Algorithm

I don't want to go into the theory of GAs right now, but the basic idea is to establish a biological analogy in which potential solutions are individuals of a population. The quality of a solution is the genetic fitness of the individual. Relatively unfit individuals die without reproducing, and relatively fit individuals will reproduce. The offspring are mutated versions of their parents. Over many generations, the population converges toward good solutions.

## Implementation in Scheme and Results

[Scheme](https://en.wikipedia.org/wiki/Scheme_(programming_language)) is particularly well-suited to problems of this kind. Due to the [functional](https://en.wikipedia.org/wiki/Functional_programming) nature of Scheme, you can write a function which takes other functions as input. The code to execute the genetic algorithm might look like:

```scheme
(darwin init-pop fitness mutator)
```

Here `darwin` is a higher-order function implementing a genetic algorithm that takes an initial population, a fitness function, and a mutation operator as arguments --- all the problem-specific details are abstracted away, making the GA code completely reusable.

Here is a video of the results: *(Editor's note: the original video embed is no longer available.)*

![Genetic Doublets 3D embedding](../../assets/genetic-doublets-3d.jpg)

---

*Originally published on [Quasiphysics](https://quasiphysics.wordpress.com/2010/12/05/playing-genetic-doublets/).*
