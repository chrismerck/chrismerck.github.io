"""Getting embedding matrices out of pretrained transformers, without torch.

The models come from spaCy's transformer pipelines, which are distributed as
GitHub release assets and each bundle one complete pretrained transformer --
`roberta-base`, `bert-base-german-cased`, `camembert-base`, and so on.  That
matters here only because it is a route to real pretrained weights that does
not go through huggingface.co.

Two serialisation layouts show up:

  spacy-curated-transformers (spaCy 3.5+)   thinc msgpack, weights in a
                                            `shims[i][0]` torch archive
  spacy-transformers (spaCy 3.4 and older)  thinc msgpack, weights in a
                                            `{config, state, tokenizer}` dict

Both end in the same place: a PyTorch zip archive.  torch is not installed and
is a ~900 MB dependency for what is, in the end, a pickle and some raw float
buffers, so this module reads the archive directly.  A `torch.save` file is a
zip containing `data.pkl` (an ordinary pickle whose tensors are rebuilt by
`torch._utils._rebuild_tensor_v2`) and one flat little-endian buffer per
storage under `data/`.  Stub out the rebuild function, intercept the persistent
ids, and numpy can do the rest.
"""

import io
import json
import os
import pickle
import struct
import subprocess
import zipfile

import numpy as np

CACHE = os.environ.get("EMBEDALIGN_DATA", "/tmp/embedalign")
WHEELS = os.path.join(CACHE, "wheels")
RELEASE = "https://github.com/explosion/spacy-models/releases/download"


# ---------------------------------------------------------------- download


def wheel_path(package, version):
    fn = f"{package}-{version}-py3-none-any.whl"
    return os.path.join(WHEELS, fn)


def fetch_wheel(package, version, quiet=False):
    """Download a spaCy model wheel from its GitHub release asset."""
    path = wheel_path(package, version)
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(WHEELS, exist_ok=True)
    fn = os.path.basename(path)
    url = f"{RELEASE}/{package}-{version}/{fn}"
    if not quiet:
        print(f"  fetching {fn} ...", flush=True)
    tmp = path + ".part"
    subprocess.run(["curl", "-sS", "-L", "--max-time", "1200", "-o", tmp, url],
                   check=True)
    if os.path.getsize(tmp) < 1_000_000:
        raise RuntimeError(f"{url} did not return a wheel")
    os.replace(tmp, path)
    return path


# ------------------------------------------------------ torch archive reader


class _Stub:
    def __init__(self, *a, **k):
        pass


def _rebuild_tensor(storage, offset, size, stride, *a):
    return ("tensor", storage, offset, tuple(size), tuple(stride))


_DTYPES = {
    "FloatStorage": np.dtype("<f4"),
    "HalfStorage": np.dtype("<f2"),
    "DoubleStorage": np.dtype("<f8"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "BoolStorage": np.dtype("?"),
    "BFloat16Storage": np.dtype("<u2"),  # handled specially
}


class _TorchUnpickler(pickle.Unpickler):
    """Reads a torch.save pickle without importing torch."""

    def find_class(self, module, name):
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return ("storage_type", name)
        if module == "collections" and name == "OrderedDict":
            import collections

            return collections.OrderedDict
        return _Stub

    def persistent_load(self, pid):
        return pid


def read_torch_state_dict(blob, want=(), verbose=False):
    """Pull named float tensors out of a `torch.save` archive held in memory.

    `want` is a collection of predicates on the parameter name; only matching
    tensors are materialised, which keeps a 500 MB checkpoint from becoming
    500 MB of numpy.
    """
    z = zipfile.ZipFile(io.BytesIO(blob) if isinstance(blob, bytes) else blob)
    names = z.namelist()
    pkl_name = [n for n in names if n.endswith("data.pkl")][0]
    prefix = pkl_name[: -len("data.pkl")]
    state = _TorchUnpickler(io.BytesIO(z.read(pkl_name))).load()

    out = {}
    for key, val in state.items():
        if want and not any(w(key) for w in want):
            continue
        if not (isinstance(val, tuple) and val and val[0] == "tensor"):
            continue
        _, storage, offset, size, _stride = val
        # persistent id: ('storage', ('storage_type', name), key, location, n)
        stype = storage[1][1] if isinstance(storage[1], tuple) else "FloatStorage"
        skey = storage[2]
        dtype = _DTYPES.get(stype, np.dtype("<f4"))
        raw = z.read(f"{prefix}data/{skey}")
        arr = np.frombuffer(raw, dtype=dtype)
        n = int(np.prod(size)) if size else 1
        arr = arr[offset : offset + n].reshape(size)
        if stype == "BFloat16Storage":
            wide = np.zeros(arr.shape + (2,), dtype=np.uint16)
            wide[..., 1] = arr
            arr = wide.view(np.float32).reshape(arr.shape)
        out[key] = np.ascontiguousarray(arr, dtype=np.float32)
        if verbose:
            print(f"    {key} {arr.shape}")
    return out


# ---------------------------------------------------------------- vocabularies


def _parse_sentencepiece(blob):
    """Minimal reader for a SentencePiece ModelProto.

    Only field 1 (`repeated SentencePiece pieces`) is needed, and inside it
    only field 1 (`string piece`).  Writing 30 lines of varint parsing beats
    adding a dependency for two fields.
    """

    def varint(buf, i):
        shift = 0
        val = 0
        while True:
            b = buf[i]
            i += 1
            val |= (b & 0x7F) << shift
            if not b & 0x80:
                return val, i
            shift += 7

    pieces = []
    i, n = 0, len(blob)
    while i < n:
        key, i = varint(blob, i)
        field, wire = key >> 3, key & 7
        if wire == 2:
            ln, i = varint(blob, i)
            sub = blob[i : i + ln]
            i += ln
            if field == 1:
                j, m = 0, len(sub)
                while j < m:
                    k2, j = varint(sub, j)
                    f2, w2 = k2 >> 3, k2 & 7
                    if w2 == 2:
                        l2, j = varint(sub, j)
                        if f2 == 1:
                            pieces.append(sub[j : j + l2].decode("utf8", "replace"))
                        j += l2
                    elif w2 == 0:
                        _, j = varint(sub, j)
                    elif w2 == 5:
                        j += 4
                    elif w2 == 1:
                        j += 8
                    else:
                        break
        elif wire == 0:
            _, i = varint(blob, i)
        elif wire == 5:
            i += 4
        elif wire == 1:
            i += 8
        else:
            break
    return pieces


def _from_id_map(mapping):
    """{piece: id} -> [piece] indexed by id."""
    size = max(mapping.values()) + 1
    vocab = [""] * size
    for piece, idx in mapping.items():
        vocab[idx] = piece
    return vocab


def extract_vocab(attrs, tokenizer_blob=None):
    """Find whichever tokenizer the pipeline happens to carry."""
    import srsly

    for a in attrs:
        if not a:
            continue
        if "wordpiece_processor" in a:
            text = a["wordpiece_processor"].decode("utf8")
            return text.split("\n"), "wordpiece"
        if "byte_bpe_processor" in a:
            d = srsly.msgpack_loads(a["byte_bpe_processor"])
            return _from_id_map(d["vocab"]), "byte-bpe"
        if "sentencepiece_processor" in a:
            return _parse_sentencepiece(a["sentencepiece_processor"]), "sentencepiece"
        if "vocab" in a and isinstance(a["vocab"], (bytes, bytearray)):
            d = srsly.msgpack_loads(a["vocab"])
            return _from_id_map(d), "wordpiece"
    if tokenizer_blob is not None:
        return _vocab_from_hf_tokenizer(tokenizer_blob)
    raise RuntimeError("no tokenizer found in pipeline attrs")


_HF_KIND = {"bpe": "byte-bpe", "wordpiece": "wordpiece",
            "unigram": "sentencepiece"}


def _vocab_from_hf_tokenizer(tok):
    """spacy-transformers stores the HF tokenizer as a dict of file contents.

    Keys are the original filenames (`tokenizer.json`, `vocab.txt`, ...), so
    the tokenizer family has to be read off whichever file is present.
    """
    for key in ("tokenizer.json", "tokenizer_file"):
        if key in tok:
            d = json.loads(tok[key])
            model = d["model"]
            kind = _HF_KIND.get(model.get("type", "").lower(), "wordpiece")
            vocab = model["vocab"]
            if isinstance(vocab, list):  # unigram: [[piece, score], ...]
                return [p[0] for p in vocab], kind
            return _from_id_map(vocab), kind
    if "vocab.json" in tok:
        return _from_id_map(json.loads(tok["vocab.json"])), "byte-bpe"
    if "vocab.txt" in tok:
        raw = tok["vocab.txt"]
        if isinstance(raw, str):
            raw = raw.encode("utf8")
        return raw.decode("utf8").rstrip("\n").split("\n"), "wordpiece"
    for key in ("sentencepiece.bpe.model", "spiece.model",
                "sentencepiece_model_file"):
        if key in tok:
            return _parse_sentencepiece(tok[key]), "sentencepiece"
    raise RuntimeError(f"unrecognised HF tokenizer payload: {sorted(tok)}")


# ---------------------------------------------------------------- top level

_EMB_KEYS = (
    "embeddings.word_embeddings.weight",
    "word_embeddings.weight",
    "embeddings.word_embeddings.embeddings.weight",
)
_POS_KEYS = ("position_embeddings.weight",)
# an untied output projection, if the checkpoint kept its LM head
_OUT_KEYS = (
    "lm_head.decoder.weight",
    "cls.predictions.decoder.weight",
    "predictions.decoder.weight",
)


def _pick(state, suffixes):
    for key in state:
        for s in suffixes:
            if key.endswith(s):
                return key
    return None


# Models trained with fairseq keep their SentencePiece ids and their embedding
# rows in different coordinate systems: fairseq reserves a few leading rows for
# its own special tokens and shifts every real piece up by a constant.  The
# constants are not guessable, they are declared in the tokenizer
# implementation (curated-transformers `_CAMEMBERT_FAIRSEQ_OFFSET` = 4,
# `_XLMR_FAIRSEQ_OFFSET` = 1), and getting one wrong silently scrambles the
# entire vocabulary while leaving every matrix the right shape.
FAIRSEQ_OFFSET = {"camembert": 4, "xlmr": 1}


def _row_offset(zipf):
    """Read the pipeline's declared transformer architecture, return its offset."""
    names = [n for n in zipf.namelist() if n.endswith("config.cfg")]
    if not names:
        return 0
    text = zipf.read(names[0]).decode("utf8", "replace").lower()
    for key, off in FAIRSEQ_OFFSET.items():
        if f"{key}transformer" in text:
            return off
    return 0


def load_pipeline_embeddings(whl, verbose=True):
    """Return (vocab, kind, {matrix name: array}) for one spaCy model wheel."""
    import srsly

    z = zipfile.ZipFile(whl)
    model_file = [n for n in z.namelist() if n.endswith("/transformer/model")]
    if not model_file:
        raise RuntimeError(f"{whl} has no transformer component")
    obj = srsly.msgpack_loads(z.read(model_file[0]))

    shim = None
    for group in obj["shims"]:
        if group:
            shim = group[0]
            break
    if shim is None:
        raise RuntimeError("no torch shim in pipeline")

    tokenizer_blob = None
    payload = shim
    if shim[:1] == b"\x85" or b"tokenizer" in shim[:200]:
        # spacy-transformers layout: {config, state, tokenizer, ...}
        try:
            wrapper = srsly.msgpack_loads(shim)
            if isinstance(wrapper, dict) and "state" in wrapper:
                payload = wrapper["state"]
                tokenizer_blob = wrapper.get("tokenizer")
        except Exception:
            pass
    if payload is shim and shim[:1] not in (b"P",):
        # curated layout: {config, state}
        try:
            wrapper = srsly.msgpack_loads(shim)
            if isinstance(wrapper, dict) and "state" in wrapper:
                payload = wrapper["state"]
        except Exception:
            pass

    want = [lambda k: any(k.endswith(s) for s in _EMB_KEYS + _POS_KEYS + _OUT_KEYS)]
    state = read_torch_state_dict(payload, want=want)

    mats = {}
    for label, keys in (("input", _EMB_KEYS), ("position", _POS_KEYS),
                        ("output", _OUT_KEYS)):
        k = _pick(state, keys)
        if k is not None:
            mats[label] = state[k]
            if verbose:
                print(f"    {label:9s} {k}  {state[k].shape}")

    vocab, kind = extract_vocab(obj["attrs"], tokenizer_blob)
    # The HF-serialised pipelines carry a real id->piece table, so their ids
    # are already row indices; only the curated SentencePiece ones need the
    # fairseq shift.
    offset = 0 if tokenizer_blob is not None else _row_offset(z)
    if offset:
        n_rows = mats["input"].shape[0]
        shifted = [""] * n_rows
        for i, piece in enumerate(vocab):
            if 0 <= i + offset < n_rows:
                shifted[i + offset] = piece
        vocab = shifted
        if verbose:
            print(f"    applied fairseq row offset {offset}")
    return vocab, kind, mats
