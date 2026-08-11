# Cross-language word alignment from distributional structure

Can you match German words to French words using nothing but the company
each word keeps in its own, unrelated corpus?

This is the code for the experiment written up in
[the blog post](../../docs/blog/posts/aligning-two-languages-without-a-dictionary.md).

## The question

Build a Manning & Schütze style distributional model for each language
independently -- count what appears within ±5 words, weight by PPMI, reduce
with an SVD. You get two clouds of points with no shared coordinate system.
Then try to find the correspondence between them with an assignment
algorithm, without ever showing the method a single translation pair.

## Layout

| file | what it does |
| --- | --- |
| `corpora.py` | load CoNLL-U treebanks and raw Europarl into `(lemma, POS)` sentences |
| `distrib.py` | co-occurrence counting, PPMI, truncated SVD |
| `gold.py` | build a German–French gold lexicon by joining Wiktionary-derived wordnets on synset id |
| `align.py` | the matching methods (Hungarian, Gromov–Wasserstein, Wasserstein–Procrustes, self-learning, supervised Procrustes) |
| `isomorphism.py` | alignment-free measures of how alike two spaces are |
| `evaluate.py` | accuracy, POS agreement, frequency-rank correlation, retrieval |
| `test_sanity.py` | difficulty ladder on problems with known answers |
| `distortion.py` | calibrate the real cases against controlled synthetic distortion |
| `run_experiment.py` | count-vector conditions |
| `run_embeddings.py` | large-vocabulary conditions on off-the-shelf vectors |
| `sweep.py` | how the ceiling moves with corpus size |

## Data

Everything is fetched from public mirrors into `$WORDALIGN_DATA`
(default `/tmp/corp`):

* Universal Dependencies German (HDT, GSD) and French (GSD, FTB, Sequoia,
  ParTUT, Rhapsodie) treebanks
* the Europarl sample shipped in NLTK's `europarl_raw`
* NLTK's `extended_omw`, for the Wiktionary-derived wordnets used to build
  the evaluation lexicon
* spaCy `de_core_news_lg` and `fr_core_news_lg` vector tables, trained
  separately per language with no bilingual signal

## Running it

```sh
./run_all.sh /tmp/results
```

Start with `test_sanity.py`. It checks that the methods can recover a
matching when one provably exists; without that, a negative result on real
languages means nothing.
