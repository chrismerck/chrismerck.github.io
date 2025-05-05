---
title: Charformers
---

# Character-based Transformers

Way back when I had fun with little n-gram models,
that I fondly recall could write pseudo-Tolkien
when training it on a text copy of _The Hobbit_.
It was fun to watch how as n increased from 3 to 5
the words became more and more coherently spelled.
You now understand my sadness at the tokenization
step used in modern LLMs: it strips them of the 
very first learning step of discovering the morphemes,
smallest units of meaning.

So I'd like to experiment with some character-based
transformers. I have no expectation that I'd be able
to get coherent English text out of them without massive
GPU resources, but perhaps we can intepret some patterns
of morphological learning?

## Sanitizing the Data

We are concerned with English text, and we want to keep 
the language as small as possible, so let's just work
with lowercase letters plus space and a few punctuation characters:

`abcdefghijklmnopqrstuvwxyz .,?'\n`

Note that we've merged the apostrophe, single, and double quotes,
we keep distinct period (but convert exclaimation to period),
a distinct question mark,
and all other punctuation will be replaced with spaces.
Newline has its own symbol, which represents a paragraph break
or could be used as a stop token.

For my purposes I'll feed in a text copy of _The Lord of the Rings_,
because I know this book so well that it should help me interpret
the small model gibberish I'm sure to get when sampling generatively.

## N-Grams Revisited

I like to start with the simplest possible model,
which within the neural network paradigm would be
a single-layer network. (1)
{.annotate}

1. If we want to be pedantic, it would be a zero-layer
network with just a bias term predicting just based on character
frequency.




