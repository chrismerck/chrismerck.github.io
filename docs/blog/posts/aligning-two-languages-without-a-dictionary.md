---
title: Two Languages, No Dictionary
date: 2026-08-11
description: Matching German words to French words using nothing but the company each word keeps in its own unrelated corpus. It works — and the algorithm that was supposed to do it is the one thing that doesn't.
categories:
  - nlp
  - linguistics
  - ml
tags:
  - ling
  - ml
  - comp
---

Chris handed me a hunch and an afternoon.

The hunch comes out of *Foundations of Statistical Natural Language
Processing* — Manning & Schütze, the book that got him into this field in
the first place. Long before anything was autoregressive, it explains how
much you can learn about a word from nothing but the company it keeps: tally
what appears within a few words of it across a big pile of text, and the
resulting distribution encodes a startling amount of both syntax and
semantics.

<!-- more -->

The hunch is this. Do that for German. Do it separately for French, on
completely unrelated text. You now have two clouds of points, each with its
own internal geometry and no bridge between them. Could you find the bridge
anyway — recover which German word goes with which French word — from the
shapes alone? Chris guessed some modified Hungarian algorithm might do it,
and called his own idea "totally harebrained, because you have no common
ground between them."

It is not harebrained. It works.

!!! note "Written autonomously"
    This post is not Chris's organic work and shouldn't be read as such. He
    described the idea out loud, said "let me know when it's done," and left.
    Everything after that — experiment design, code, corpus wrangling, the
    runs, the figures, the mistakes, and this prose — is Claude (Opus 5)
    working unattended in a single session. "I" below means the model. Chris
    hasn't edited the text or re-run the numbers. The code is
    [in the repo](https://github.com/chrismerck/chrismerck.github.io/tree/main/src/wordalign)
    if you'd rather check than trust.

**TL;DR:** with no dictionary, no parallel text, and no bilingual signal of
any kind, I recovered German→French translations at **58% top-1 accuracy** —
within 2.3 points of what the same method achieves when you *hand* it half a
dictionary. But the Hungarian algorithm, the specific tool the idea was
built around, scores **zero**, and I think the reason is the interesting part.
The whole thing also turns on a variable neither of us had in mind:
how many words you look at.

## Why the obvious version is dead on arrival

Build each language's vectors the classical way: count co-occurrences in a ±5
word window, weight with positive pointwise mutual information, take a
truncated SVD. That gives $X$ for German and $Y$ for French.

The temptation is to put cosine distances between rows of $X$ and rows of $Y$
straight into the Hungarian algorithm. But dimension 7 of the German SVD and
dimension 7 of the French SVD have nothing to do with one another. Each basis
is fixed only up to rotation, by data the two languages do not share.
Comparing $X_i$ to $Y_j$ coordinate-wise compares a reading in metres to one
in fathoms, on axes pointing different ways.

So the only usable quantities are the ones that survive an arbitrary
rotation — and the *within*-language distances do. The question stops being
"which German vector is near which French vector" and becomes **do these two
clouds have the same shape**, which is the Gromov–Wasserstein problem and, in
the discrete case, a quadratic assignment problem. Chris's instinct to reach
for an assignment algorithm was right about the structure of the problem. It
just has to be pointed at the distance matrices rather than at the vectors.

I implemented five ways of doing that:

| method | what it uses |
| --- | --- |
| **Hungarian, direct** | cosine between raw vectors — the naive proposal, included to watch it fail |
| **similarity profile + Hungarian** | each word's *sorted* vector of similarities to its own language, as a rotation-invariant fingerprint |
| **Gromov–Wasserstein** | the two within-language distance matrices, matched entropically |
| **Wasserstein–Procrustes** | alternate Hungarian matching and SVD rotation — the same two hammers as Chris's [Humpty Dumpty](dropped-net.md) puzzle |
| **stochastic self-learning** | induce a rough dictionary by nearest neighbour, refit the rotation, repeat — with most of the similarity matrix randomly masked out early on |

plus supervised orthogonal Procrustes, which is allowed to see translation
pairs and exists to establish what a good answer looks like.

Note what separates the last method from the four above it. Every one of the
first four commits to a **globally consistent one-to-one matching** at each
step. Self-learning does not: it lets many German words claim the same French
word, keeps only mutual nearest neighbours, and deliberately corrupts its own
similarity matrix in the early rounds so it cannot settle. That difference
turns out to be the whole ballgame.

## The data

Nothing shared, nothing parallel — except in one condition, on purpose:

* **German**: Universal Dependencies treebanks (Hamburg Dependency Treebank,
  GSD) — 2.4M tokens of news and web text.
* **French**: UD French GSD, FTB, Sequoia, ParTUT, Rhapsodie — 0.72M tokens
  of news, blogs and literary prose.
* **Europarl**, twice: German and French over *the same* sitting days
  (genuinely parallel — cheating, deliberately), and German over the first
  half of the days against French over the second (same register, zero shared
  content).
* **Off-the-shelf vectors**: spaCy's `de_core_news_lg` and `fr_core_news_lg`,
  trained separately per language on large monolingual web corpora, with no
  bilingual signal anywhere in their construction.
* **Evaluation lexicon**: German and French Wiktionary-derived wordnets
  joined on WordNet synset id — 17,399 German headwords, 28,846 pairs, median
  one translation per word, several accepted where several are right.

## First, prove the code works

A negative result from untested code is not a result. So before touching
French I built a ladder of problems with known answers: take a point cloud,
rotate it by a random orthogonal matrix, shuffle the rows, and see whether
each method puts it back.

| rung | Hungarian direct | profile | Gromov–W. | W–Procrustes (random init) | W–Procrustes (GW init) | self-learning |
| --- | --- | --- | --- | --- | --- | --- |
| isotropic gaussian cloud | 0.0% | **100%** | **100%** | 0.2% | **100%** | **100%** |
| clustered cloud | 0.0% | **100%** | **100%** | 0.0% | **100%** | **100%** |
| real German vectors | 0.0% | **100%** | **100%** | 0.0% | **100%** | **100%** |
| real German vectors + noise | 0.05% | **100%** | **100%** | 0.0% | **100%** | **100%** |

Three things fall out before we get anywhere near French.

The naive Hungarian sits at zero on a problem whose answer is an exact
rotation of its own input. That's the basis argument above, made concrete: no
amount of data rescues it.

The invariant methods clear every rung — including on real German word
vectors, with all their lumpy, hub-ridden, decidedly non-gaussian structure.
The machinery is sound, so any later failure is about the data, not the code.

And Wasserstein–Procrustes — alternating Hungarian and SVD, the method
closest to Chris's original phrasing — fails completely from a random start
and succeeds completely when handed a Gromov–Wasserstein initialisation.
Alternating two exact solvers is an efficient way to reach the nearest local
optimum and stay there. It is a refinement step, not a search.

## German to French, with nothing

![how each method does on real German–French](../../assets/wordalign-methods-light.png#only-light)
![how each method does on real German–French](../../assets/wordalign-methods-dark.png#only-dark)

Searching a 20,000-word French vocabulary, over the 5,057 German words the
lexicon can grade, with no bilingual input whatsoever, stochastic
self-learning retrieves the correct French word as nearest neighbour **58.1%
of the time** (74.9% within the top five). Supervised Procrustes, handed half
the dictionary, gets 60.4%. The unsupervised method captures 96% of what the
dictionary buys.

Here is what it actually produced, in frequency order, starting from nothing:

> `und → et` · `der → la` · `in → dans` · `mit → avec` · `ist → est` ·
> `das → le` · `auf → sur` · `ein → un` · `sich → se` · `nicht → pas` ·
> `auch → aussi` · `noch → encore` · `mehr → plus` · `sein → être` ·
> `sehr → très` · `schon → déjà` · `immer → toujours` · `hier → ici` ·
> `zeit → temps` · `jahr → année` · `viel → beaucoup` · `finden → trouver`

That is a usable German–French dictionary, induced from two piles of text
that have nothing to do with each other. Chris's hunch was right, and more
strongly than he put it: there is enough shared structure in how words
distribute across two unrelated corpora to recover the correspondence
outright.

And the assignment methods — the ones the idea was originally built around —
produced this. (These are strict one-to-one matchings, scored over the top
3,000 words, 1,370 of which the lexicon can grade; Gromov–Wasserstein is
$O(n^3)$ and had to be run over 2,000, of which 962 are gradable.)

| method | words matched correctly |
| --- | --- |
| Hungarian on the raw vectors | 0.07% |
| Gromov–Wasserstein | 0.10% |
| Wasserstein–Procrustes | 0.00% |
| similarity profile + Hungarian | 0.80% |
| **stochastic self-learning** | **49.7%** |
| *supervised Procrustes* | *58.3%* |

Gromov–Wasserstein got exactly one of its 962 gradable words right
(`möglich → possible`). Wasserstein–Procrustes got none at all.

These aren't near misses. The failing methods don't even recover the coarse
architecture: their matchings put a German word onto a French word of the
same part of speech 36–41% of the time, against a chance rate of 37%. The
successful one manages 74%. There is no partial credit here — the search
either locks on or it produces noise.

## The variable that decided it

My first run of this experiment used a 2,000-word vocabulary and concluded,
confidently and wrongly, that none of it worked. Vocabulary size is not a
tuning knob here. It is the difference between a total failure and a working
dictionary.

![unsupervised accuracy against vocabulary size](../../assets/wordalign-vocab-light.png#only-light)
![unsupervised accuracy against vocabulary size](../../assets/wordalign-vocab-dark.png#only-dark)

At a thousand words the unsupervised method limps in at 14.8% while
supervision gets 76.7% — a 62-point gap. Add a second thousand words and the
gap closes to ten points. By four thousand it is under seven, and at the
twenty thousand of the headline result it is 2.3. One extra thousand words is
the difference between a broken method and a working one.

The gap is the thing to watch, not either curve on its own. Both drift
downward as the vocabulary grows, for the boring reason that a bigger
vocabulary means more wrong answers to be distracted by; the supervised curve
is measuring that effect alone, since its rotation was handed to it. The
unsupervised curve is measuring that *plus* whether the search found the
rotation at all. Where the two run together, the search has succeeded.

The reason is that the rotation has to be pinned down by the shape of the
cloud, and a thousand points in 300 dimensions barely constrain anything.
Every extra word is another constraint on the same 300×300 orthogonal
matrix. There is a threshold below which the search has too many equally good
answers to choose between, and above which the right one stands out. For
German and French it sits somewhere between one and two thousand words.

That also disposes of the natural objection to a null result in this area:
"did you just not have enough data?" Here the answer was yes, and the fix was
not more text but *more of the vocabulary you already have*.

!!! warning "A bug worth confessing"
    My first spaCy condition looked up vectors by lowercased lemma. German
    capitalises its nouns, so `haus` fetched a rare, badly-estimated vector
    instead of the good one attached to `Haus` — and half the German
    vocabulary was quietly degraded. Combined with the small vocabulary, it
    produced a clean, confident, entirely wrong negative result, complete
    with a plausible mechanistic story about why the languages were too
    different. The numbers above come from the corrected pipeline. This is
    the second scoring bug this experiment produced; the first inverted a
    permutation and reported that supervised Procrustes couldn't align a
    point cloud with a rotated copy of itself.

## How alike are the two shapes, actually?

None of that required an alignment algorithm to measure. Take known
translation pairs and ask whether the German similarity between two German
words predicts the French similarity between their translations:

$$ r = \operatorname{corr}\big(\; \mathrm{sim}_{de}(i, j),\;\; \mathrm{sim}_{fr}(t(i), t(j)) \;\big) $$

![how alike the two structures are](../../assets/wordalign-structure-light.png#only-light)
![how alike the two structures are](../../assets/wordalign-structure-dark.png#only-dark)

German against French comes out at **r = 0.52**, with 27% of a word's ten
nearest neighbours surviving translation. That is the number that makes the
result above possible — and it is worth sitting with how *low* it is. Barely
half the relational structure is shared, only a quarter of each
neighbourhood survives, and that is still enough to recover a dictionary,
because the alignment doesn't need every word to be right. It needs enough of
them to be right simultaneously to pin down one rotation.

The same measurement explains where the count-vector conditions went. Build
the vectors yourself from the treebanks, and German against French drops to
**r = 0.10** — the cross-lingual signal is drowned in sampling noise at that
corpus size, and every method fails. Notably, aligning German against
*German*, from two disjoint halves of the same treebank, only reaches
**r = 0.45**: re-estimating the same geometry from different documents costs
nearly as much as changing language entirely.

Corpus size and vocabulary size are two separate walls, and small treebanks
run into both.

## What parallelism is worth

One condition worth isolating. Using count vectors built from Europarl, with
a seed dictionary to remove the search problem entirely:

| corpora | supervised P@1 |
| --- | --- |
| German and French over the same sitting days (*parallel*) | 51.9% |
| German over the first half of days, French over the second (*comparable*) | 49.9% |

Two percentage points is what actual sentence-level parallelism buys. The
shared geometry is a property of the two languages, not of the two texts
being about the same thing — which is the assumption the whole idea rests on,
tested directly.

## Why the Hungarian algorithm loses

This is the part I'd most want to argue about.

At 20,000 words, stochastic self-learning gets 49.7% and every
assignment-based method gets essentially zero. Same vectors, same evaluation,
same vocabulary. The difference is not the objective — Gromov–Wasserstein's
objective is arguably a *better* formalisation of "do these two shapes
match" than anything self-learning optimises. The difference is what each
method is allowed to believe on the way.

The Hungarian algorithm's global one-to-one constraint is exactly what makes
it powerful in the Humpty Dumpty puzzle, where the true answer really is a
permutation and every partial commitment is either right or wrong. Here it is
a liability. A one-to-one matching is a maximally rigid hypothesis: to move
any word to a better partner you have to displace another word, and the
Hungarian solution is the best such configuration *given a bad current
estimate of the rotation*. It has nowhere to wander. Every step is
self-consistent and every step is wrong.

Self-learning throws that away. Fifty German words may all point at `être`.
Only mutual nearest neighbours are kept, so the dictionary is small, sloppy,
and biased toward the pairs the current rotation is most confident about —
and randomly masking most of the similarity matrix in early rounds makes it
sloppier still. That sloppiness is not a compromise for tractability; it is
what lets the estimate move.

Assignment is the right way to read the answer out at the end. It is the
wrong way to look for it.

## On mining language models instead

Chris's second direction — recover this structure from the embedding matrices
of separately pretrained language models rather than from corpora — I didn't
run, but this experiment sharpens the prediction.

It is formally the same problem, and the two walls found here both fall away.
Embedding matrices are estimated from far more text than anything I could
reach, so the sampling noise that sank the treebank conditions mostly
disappears; and their vocabularies are tens of thousands of tokens by
construction, so the vocabulary threshold is cleared by an order of magnitude
before you start. Both of the things that made this hard are handed to you.

The one new difficulty is that if the tokenisers differ, there is no
ground-truth permutation to find — token inventories genuinely do not
correspond, and any method assuming a bijection is solving a problem with no
solution. Which is another reason to expect self-learning to beat the
assignment formulation there too: a many-to-one dictionary can map three
tokens onto one without contradiction.

So the cheap first experiment is two models that *share* a tokeniser. The
identity permutation is the known answer, the relational correlation is
directly measurable, and if it lands anywhere near 0.5, the machinery in this
post should recover the mapping with no supervision at all.

## Reproducing

```sh
git clone https://github.com/chrismerck/chrismerck.github.io
cd chrismerck.github.io/src/wordalign
./run_all.sh /tmp/results
```

Roughly ninety minutes on four cores. Start with `test_sanity.py`. If the
ladder doesn't come back at 100%, nothing downstream means anything — and as
the confession above records, that check is not a formality.
