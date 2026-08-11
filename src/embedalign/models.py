"""The set of pretrained transformers we can actually get hold of, and the
normalisation that lets two different tokenizers be compared.

huggingface.co is unreachable from this machine, so every model here arrives
as a spaCy pipeline wheel from a GitHub release.  Each wheel embeds one
complete pretrained transformer; spaCy fine-tuned it on a tagging/parsing
treebank, which nudges the embedding table slightly but does not retrain it.
That caveat is real and is stated in the write-up.

The interesting pairs are the two where the *same language* was pretrained
twice by different groups -- Danish and Catalan.  There the ground truth is
not a translation dictionary but literal token identity: if both vocabularies
contain the word-initial piece `bare`, that is the same token, full stop.
"""

import os
import unicodedata

import numpy as np

import weights

CACHE = weights.CACHE

# name -> (spaCy package, version, backbone, language, note)
MODELS = {
    "en-roberta": ("en_core_web_trf", "3.8.0", "roberta-base", "en", ""),
    "de-bert": ("de_dep_news_trf", "3.8.0", "bert-base-german-cased", "de", ""),
    "es-bert": ("es_dep_news_trf", "3.8.0",
                "dccuchile/bert-base-spanish-wwm-cased", "es", ""),
    "fr-camembert": ("fr_dep_news_trf", "3.8.0", "camembert-base", "fr", ""),
    "zh-bert": ("zh_core_web_trf", "3.8.0", "bert-base-chinese", "zh", ""),
    "ja-bert": ("ja_core_news_trf", "3.8.0",
                "cl-tohoku/bert-base-japanese-char-v2", "ja", ""),
    # the two same-language pairs
    "da-botxo": ("da_core_news_trf", "3.3.0", "Maltehb/danish-bert-botxo",
                 "da", "Danish, run 1"),
    "da-danskbert": ("da_core_news_trf", "3.8.0", "vesteinn/DanskBERT",
                     "da", "Danish, run 2"),
    "ca-plantl": ("ca_core_news_trf", "3.3.0", "PlanTL-GOB-ES/roberta-base-ca",
                  "ca", "Catalan, run 1"),
    "ca-aina": ("ca_core_news_trf", "3.8.0", "projecte-aina/roberta-base-ca-v2",
                "ca", "Catalan, run 2"),
}


# ---------------------------------------------------------------- token forms

_SPECIAL = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<unk>", "<s>",
            "</s>", "<pad>", "<mask>", "<0x00>"}


def normalise_piece(piece, kind):
    """Map a subword token to (surface string, is word-initial).

    The three tokenizer families mark the same distinction three ways:

      wordpiece       `Haus`   word-initial     `##es`   continuation
      byte-BPE        `Ġhaus`  word-initial     `haus`   continuation
      sentencepiece   `▁haus`  word-initial     `haus`   continuation

    Reducing all three to the same pair makes `Ġbare` in a RoBERTa vocabulary
    and `bare` in a BERT vocabulary recognisably the same token.  Returns
    None for specials, unused slots and byte-fallback pieces, which carry no
    comparable surface form.
    """
    if not piece or piece in _SPECIAL:
        return None
    if piece.startswith("[unused") or piece.startswith("<unused"):
        return None
    if piece.startswith("<0x") and piece.endswith(">"):
        return None
    if kind == "wordpiece":
        if piece.startswith("##"):
            return (piece[2:], False)
        return (piece, True)
    if kind == "byte-bpe":
        # RoBERTa's byte-level alphabet maps the space to U+0120 and other
        # control bytes into the U+0100 block; decode back to real text.
        text = _byte_decode(piece)
        if text is None:
            return None
        if text.startswith(" "):
            return (text[1:], True)
        return (text, False)
    if kind == "sentencepiece":
        if piece.startswith("▁"):
            return (piece[1:], True)
        return (piece, False)
    raise ValueError(kind)


def _byte_decode_table():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_TABLE = _byte_decode_table()


def _byte_decode(piece):
    try:
        raw = bytes(_BYTE_TABLE[ch] for ch in piece)
    except KeyError:
        return None
    try:
        return raw.decode("utf8")
    except UnicodeDecodeError:
        return None


def token_forms(vocab, kind, casefold=False):
    """Vocabulary index -> (surface, initial) key, with None for unusable ids."""
    out = []
    for piece in vocab:
        form = normalise_piece(piece, kind)
        if form is not None and casefold:
            form = (form[0].casefold(), form[1])
        if form is not None:
            s = unicodedata.normalize("NFC", form[0])
            if not s:
                form = None
            else:
                form = (s, form[1])
        out.append(form)
    return out


# ---------------------------------------------------------------- loading


class Model:
    def __init__(self, name, vocab, kind, mats, meta):
        self.name = name
        self.vocab = vocab
        self.kind = kind
        self.mats = mats
        self.meta = meta

    @property
    def backbone(self):
        return self.meta["backbone"]

    @property
    def lang(self):
        return self.meta["lang"]

    def matrix(self, which="input"):
        return self.mats[which]

    def __repr__(self):
        return (f"<Model {self.name} {self.backbone} {self.kind} "
                f"V={len(self.vocab)} d={self.mats['input'].shape[1]}>")


def cache_path(name):
    return os.path.join(CACHE, f"{name}.npz")


def prepare(name, verbose=True):
    """Download, unpack and cache one model's embedding matrices."""
    path = cache_path(name)
    if os.path.exists(path):
        return path
    package, version, backbone, lang, note = MODELS[name]
    if verbose:
        print(f"[{name}] {backbone}", flush=True)
    whl = weights.fetch_wheel(package, version, quiet=not verbose)
    vocab, kind, mats = weights.load_pipeline_embeddings(whl, verbose=verbose)
    os.makedirs(CACHE, exist_ok=True)
    payload = {f"mat_{k}": v for k, v in mats.items()}
    payload["vocab"] = np.array(vocab, dtype=object)
    payload["meta"] = np.array(
        [name, package, version, backbone, lang, kind, note], dtype=object)
    np.savez(path + ".tmp.npz", **payload, allow_pickle=True)
    os.replace(path + ".tmp.npz", path)
    if verbose:
        print(f"  vocab {len(vocab):,} ({kind}), matrices "
              f"{sorted(mats)}", flush=True)
    return path


def load(name, verbose=False):
    prepare(name, verbose=verbose)
    z = np.load(cache_path(name), allow_pickle=True)
    meta_arr = list(z["meta"])
    meta = dict(zip(["name", "package", "version", "backbone", "lang",
                     "kind", "note"], meta_arr))
    mats = {k[4:]: z[k] for k in z.files if k.startswith("mat_")}
    return Model(name, list(z["vocab"]), meta["kind"], mats, meta)


# ---------------------------------------------------------------- pairing


def shared_tokens(a, b, which="input", limit=None, casefold=False,
                  initial_only=False):
    """Ground-truth index pairs: tokens with the same surface form in both.

    Note what this is *not*: it is not used by any alignment method, only to
    score one.  A method sees two anonymous matrices of floats.

    Vocabularies are frequency-ordered by construction (BPE and WordPiece
    merge the commonest things first), so `limit` truncating to the first N
    ids is a frequency cut, and the pairs come out ordered by roughly how
    common the token is.
    """
    fa = token_forms(a.vocab, a.kind, casefold=casefold)
    fb = token_forms(b.vocab, b.kind, casefold=casefold)
    va, vb = a.matrix(which).shape[0], b.matrix(which).shape[0]
    if limit:
        va, vb = min(va, limit), min(vb, limit)

    index_b = {}
    for j in range(min(vb, len(fb))):
        f = fb[j]
        if f is None or (initial_only and not f[1]):
            continue
        index_b.setdefault(f, j)  # first (= most frequent) id wins

    seen = set()
    pairs = []
    for i in range(min(va, len(fa))):
        f = fa[i]
        if f is None or (initial_only and not f[1]):
            continue
        j = index_b.get(f)
        if j is None or j in seen:
            continue
        seen.add(j)
        pairs.append((i, j))
    return pairs


def vocab_overlap(a, b, casefold=False):
    fa = {f for f in token_forms(a.vocab, a.kind, casefold) if f}
    fb = {f for f in token_forms(b.vocab, b.kind, casefold) if f}
    inter = fa & fb
    return {"a": len(fa), "b": len(fb), "shared": len(inter),
            "jaccard": len(inter) / max(len(fa | fb), 1)}
