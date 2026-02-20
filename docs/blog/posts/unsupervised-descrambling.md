---
title: Unsupervised Image Descrambling and the Retina
date: 2013-10-09
description: Framing retinal design as an unsupervised descrambling problem and proposing a genetic algorithm approach
categories:
  - ml
tags:
  - neuro
  - ml
  - math
---

One of the primary goals of contemporary neuroscience is the reverse-engineering of the brain's functional architecture. Our understanding has evolved from descriptive to functional, particularly through borrowing ideas from computer science and [information theory](https://en.wikipedia.org/wiki/Information_theory). V. Balasubramanian and P. Sterling's [paper](http://jp.physoc.org/content/587/12/2753.short) explains several aspects of retinal design using information- and selection-theoretic arguments in conjunction with computer simulation.

<!-- more -->

I advocate for reducing a complicated biological problem to a [toy model](https://en.wikipedia.org/wiki/Toy_model) and then asking basic mathematical questions about that model, provided the model captures essential features while remaining simple enough to be analytically or computationally tractable.

## Statement of Problem

Let $A$ denote a set of $N$ digital images of naturalistic scenes, each with $D \times D$ pixel geometry. A [permutation](https://en.wikipedia.org/wiki/Permutation) $P$ scrambles pixel locations, producing a set $B$ of scrambled images from $A$.

**Goal:** Recover permutation $P$ from set $B$ alone.

Solutions are judged by:

1. Accuracy in recovering permutation $P$
2. Computational complexity relative to $D$
3. Convergence rate as a function of $N$

## Discussion

For $100 \times 100$ pixel images, pixel brightness correlation decreases with distance, suggesting one could learn a descrambling by maximizing the correlation between neighboring pixel brightnesses.

Key computational challenges:

- Candidate permutations are lists of $10^4$ integers
- Evaluating candidates requires approximately $4 \times 10^4$ pixel comparisons
- Solution space size: approximately $10^{35659}$ possible permutations

**Proposed approach:** A [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) with multiple crossover operations:

1. Initialize with random permutation populations
2. Compute neighboring pixel correlations for each individual
3. Isolate best-performing sub-regions
4. Stochastically recombine best sub-regions into new generation
5. Introduce "point mutations" as needed

A properly designed sub-region recombination procedure might nearly recover the original images with surprisingly few training samples. This is a completely [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning) problem --- no labeled training data is required. Achieving high accuracy in strict permutation recovery may prove extremely difficult, though candidate reconstructions may prove interesting to observe.

The connection to retinal design rests on the premise that understanding brain organization requires computational theory that can handle similar --- if drastically simplified --- problems.

---

*Originally published on [Quasiphysics](https://quasiphysics.wordpress.com/2013/10/09/unsupervised-descrambling/).*
