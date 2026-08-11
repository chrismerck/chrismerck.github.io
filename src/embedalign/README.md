# Aligning the embedding matrices of independently pretrained transformers

Part 2 of the experiment written up in
[the Part 1 post](../../docs/blog/posts/aligning-two-languages-without-a-dictionary.md).
Part 1 asked whether two languages' distributional word spaces can be aligned
with no bilingual supervision (they can, at 58% top-1). This asks the same
question of two *pretrained transformers*: take the input embedding matrices
of two models trained separately by different groups, and ask whether their
geometries share enough structure to be matched with no supervision at all.

Because the two vocabularies contain thousands of identical token strings,
the ground truth here is a **known permutation**, not a fuzzy translation
lexicon. A method either recovers it or it does not.

## Where the weights come from

`huggingface.co` is unreachable from the machine this ran on. spaCy's
transformer pipelines are distributed as GitHub release assets and each
bundles one complete pretrained transformer, so that is the route used. Ten
models, all 768-dimensional:

| name | backbone | tokenizer | vocab |
| --- | --- | --- | --- |
| `en-roberta` | `roberta-base` | byte-BPE | 50,265 |
| `de-bert` | `bert-base-german-cased` | WordPiece | 30,000 |
| `es-bert` | `dccuchile/bert-base-spanish-wwm-cased` | WordPiece | 31,002 |
| `fr-camembert` | `camembert-base` | SentencePiece | 32,005 |
| `zh-bert` | `bert-base-chinese` | WordPiece | 21,128 |
| `ja-bert` | `cl-tohoku/bert-base-japanese-char-v2` | WordPiece | 6,144 |
| `da-botxo` | `Maltehb/danish-bert-botxo` | WordPiece | 31,748 |
| `da-danskbert` | `vesteinn/DanskBERT` | SentencePiece | 50,005 |
| `ca-plantl` | `PlanTL-GOB-ES/roberta-base-ca` | byte-BPE | 52,000 |
| `ca-aina` | `projecte-aina/roberta-base-ca-v2` | byte-BPE | 50,262 |

The last four are the interesting ones: two independent Danish pretraining
runs and two independent Catalan ones. Same language, different groups,
different corpora, different tokenizers, no shared initialisation.

Two caveats stated up front. These are masked-language-model encoders, not
decoder LLMs. And spaCy fine-tuned each one on a tagging/parsing treebank
before shipping it, which perturbs the embedding table slightly; it does not
retrain it, but it is not the pristine pretrained checkpoint either.

## Layout

| file | what it does |
| --- | --- |
| `weights.py` | download the wheels; read a `torch.save` archive without torch; decode WordPiece / byte-BPE / SentencePiece vocabularies |
| `models.py` | the model registry, tokenizer normalisation, and the shared-token ground truth |
| `experiment.py` | assemble a pair, run the Part 1 methods against it, score |
| `sanity.py` | the difficulty ladder — run this first |
| `run_all.py` | survey, full method comparison, sweeps, positional embeddings, open setting |
| `scale.py` | the same pairs over the whole shared vocabulary, up to 44k tokens |
| `extras.py` | script-confound check and postprocessing ablation |
| `figures.py` | the figures in the post |

Alignment methods themselves are imported unchanged from
[`../wordalign/align.py`](../wordalign/align.py) — nothing in them knows the
input used to be a PPMI count matrix.

## Running it

```sh
./run_all.sh /tmp/embedalign-results
```

About two hours on four cores, plus ~4 GB of model downloads into
`$EMBEDALIGN_DATA` (default `/tmp/embedalign`). Start with `sanity.py`: if
the ladder does not come back at ~100%, nothing downstream means anything.

## The trap

Fairseq-trained models keep their SentencePiece piece ids and their embedding
rows in different coordinate systems — CamemBERT shifts every real piece up
by 4, XLM-R-style models by 1. Get that wrong and every matrix still has the
right shape, every method still runs, and the relational correlation between
`roberta-base` and `camembert-base` reads 0.017 instead of 0.626. The first
version of these numbers had exactly that bug. `run_all.py`'s `offset_scan`
stage exists to show how sharp the peak is.
