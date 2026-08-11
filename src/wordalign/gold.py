"""Build a German-French gold translation lexicon.

Source: the Wiktionary-derived wordnets shipped in NLTK's `extended_omw`
package (CC BY-SA).  Each file maps a Princeton WordNet synset id to lemmas
in one language, so joining the German and French files on synset id yields
a many-to-many translation lexicon.

We keep it many-to-many on purpose: `Frau` is legitimately `femme` and
`dame`, and any evaluation that insists on a single right answer is lying
about the problem.
"""

import collections
import os
import re

WORD_RE = re.compile(r"^[a-zà-öø-ÿœæß]+$")


def _read_tab(path):
    """Return {synset_id: set(lemmas)} from an OMW .tab file."""
    out = collections.defaultdict(set)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            synset, _kind, lemma = parts[0], parts[1], parts[2]
            lemma = lemma.strip().lower()
            # single tokens only -- multiword expressions can't be matched by
            # a word-level assignment anyway
            if not WORD_RE.match(lemma):
                continue
            out[synset].add(lemma)
    return out


def build_gold(omw_dir):
    """Return {german_word: set(french_words)}."""
    deu = _read_tab(os.path.join(omw_dir, "wn-wikt-deu.tab"))
    fra = _read_tab(os.path.join(omw_dir, "wn-wikt-fra.tab"))
    gold = collections.defaultdict(set)
    for synset, de_lemmas in deu.items():
        fr_lemmas = fra.get(synset)
        if not fr_lemmas:
            continue
        for d in de_lemmas:
            gold[d] |= fr_lemmas
    return dict(gold)


def build_gold_reverse(gold):
    rev = collections.defaultdict(set)
    for d, fs in gold.items():
        for f in fs:
            rev[f].add(d)
    return dict(rev)


if __name__ == "__main__":
    import sys

    gold = build_gold(sys.argv[1])
    print(f"{len(gold)} German headwords")
    print(f"{sum(len(v) for v in gold.values())} (de, fr) pairs")
    for w in ["frau", "wasser", "haus", "gehen", "gut", "und", "aber", "zeit"]:
        print(f"  {w:10s} -> {sorted(gold.get(w, []))[:8]}")
