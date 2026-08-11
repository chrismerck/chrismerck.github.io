"""Corpus loading.

Everything downstream wants the same thing: a list of sentences, where a
sentence is a list of `(lemma, upos)` pairs.  Two sources are supported.

* CoNLL-U treebanks (Universal Dependencies).  These arrive already
  tokenised, lemmatised and POS-tagged by human annotators, which removes a
  whole class of "is the result just a tokeniser artefact?" worries.
* Raw Europarl text, lemmatised with spaCy.

The German and French halves never come from the same documents.  That is
the entire point of the exercise, so the loaders make it hard to cheat by
accident.
"""

import glob
import os
import re

# Categories that carry no useful distributional signal for this task, or
# that differ so much between corpora that they are pure noise.
SKIP_POS = {"PUNCT", "SYM", "X", "NUM", "PROPN"}

_NUMERIC = re.compile(r"\d")


def _clean(lemma, upos):
    if upos in SKIP_POS:
        return None
    lemma = lemma.strip().lower()
    if not lemma or _NUMERIC.search(lemma):
        return None
    if len(lemma) > 30:
        return None
    return lemma


def load_conllu(paths, max_tokens=None):
    """Read CoNLL-U files into [[(lemma, upos), ...], ...]."""
    sentences = []
    current = []
    n = 0
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    if current:
                        sentences.append(current)
                        current = []
                    continue
                if line.startswith("#"):
                    continue
                cols = line.split("\t")
                if len(cols) < 4:
                    continue
                # skip multiword-token ranges (e.g. "1-2 du") and empty nodes
                if "-" in cols[0] or "." in cols[0]:
                    continue
                lemma, upos = cols[2], cols[3]
                w = _clean(lemma, upos)
                if w is None:
                    continue
                current.append((w, upos))
                n += 1
            if max_tokens and n >= max_tokens:
                break
        if current:
            sentences.append(current)
            current = []
        if max_tokens and n >= max_tokens:
            break
    return sentences


def load_europarl(files, spacy_model, max_tokens=None):
    """Read raw Europarl text files and lemmatise them with spaCy."""
    import spacy

    nlp = spacy.load(spacy_model, exclude=["parser", "ner"])

    raw = []
    for path in sorted(files):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    raw.append(line)

    sentences = []
    n = 0
    for doc in nlp.pipe(raw, batch_size=200):
        sent = []
        for tok in doc:
            w = _clean(tok.lemma_, tok.pos_)
            if w is not None:
                sent.append((w, tok.pos_))
        if sent:
            sentences.append(sent)
            n += len(sent)
        if max_tokens and n >= max_tokens:
            break
    return sentences


def europarl_files(directory, ext, half=None):
    """List Europarl day-files, optionally taking only one disjoint half.

    `half="a"` and `half="b"` return non-overlapping sets of *days*, so a
    German "a" corpus and a French "b" corpus share no content at all --
    they are comparable (same register) but not parallel.
    """
    files = sorted(glob.glob(os.path.join(directory, f"*.{ext}")))
    if half == "a":
        return files[: len(files) // 2]
    if half == "b":
        return files[len(files) // 2 :]
    return files


def token_count(sentences):
    return sum(len(s) for s in sentences)


def pos_map(sentences):
    """Majority POS tag per lemma, for diagnostics."""
    import collections

    counts = collections.defaultdict(collections.Counter)
    for sent in sentences:
        for w, p in sent:
            counts[w][p] += 1
    return {w: c.most_common(1)[0][0] for w, c in counts.items()}
