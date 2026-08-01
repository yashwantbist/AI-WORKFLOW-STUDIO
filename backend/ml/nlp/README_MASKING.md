# Day 11: Causal Self-Attention and Transformer Variants

## Measurable objective

By the end of this lab, you can:

1. distinguish encoder-only, decoder-only, and encoder-decoder Transformers;
2. generate a causal attention permission matrix;
3. run a decoder block that cannot read future positions;
4. prove with tests that blocked attention probabilities are zero;
5. visualize both the rule and the resulting attention weights.

This code builds directly on Day 10's hand-written
`MultiHeadAttention`. Day 10 already accepts a boolean tensor shaped
`[batch, query, key]`, where `True` means attention is allowed. Therefore,
`attention.py` does not need to be changed.

## Three Transformer families

| Family | Token visibility | Typical purpose | Familiar example |
|---|---|---|---|
| Encoder-only | Every real token can see every other real token | Classification, embeddings, retrieval | BERT |
| Decoder-only | Each token sees itself and earlier tokens | Autoregressive text generation | GPT |
| Encoder-decoder | Encoder sees all source tokens; decoder sees only its generated history and reads encoder output | Translation, summarization, source-to-target generation | T5 |

### Encoder-only

```text
Input text
   ↓
Embedding + position
   ↓
Bidirectional self-attention
   ↓
Contextual representation
   ↓
Classification / embedding task
```

The encoder receives the complete input. For sentiment classification, the
word at position 1 is allowed to use information from position 5 because the
entire sentence already exists.

### Decoder-only

```text
Known tokens
   ↓
Embedding + position
   ↓
Causal multi-head self-attention
   ↓
Feed-forward network
   ↓
Vocabulary scores for the next token
```

A decoder generates from left to right. During training, the correct sentence
is available, but the causal rule prevents the model from cheating by reading
the answer on its right.

### Encoder-decoder

```text
Source text → Encoder → Source representations
                              ↓
Generated history → Decoder self-attention → Cross-attention → Next token
```

The decoder has two information sources: its earlier generated tokens and the
encoder's full representation of the source. This is useful when one complete
input must be converted into a different output sequence.

## The five-token causal attention matrix

For:

```text
The cat sat on mat
```

the permission matrix is:

```text
      The cat sat on  mat   ← key position
The    1   0   0   0   0
cat    1   1   0   0   0
sat    1   1   1   0   0
on     1   1   1   1   0
mat    1   1   1   1   1
↑
query position
```

`1` means the query may read that key. `0` means it may not. The zeroes above
the diagonal are the future positions.

## Beginner-friendly code walkthrough

### 1. `masks.py`

The central line is:

```python
torch.ones(sequence_length, sequence_length, dtype=torch.bool).tril()
```

Step by step:

1. `torch.ones(...)` creates a square matrix filled with `True`.
2. `.tril()` keeps the lower triangle, including the diagonal.
3. Everything above the diagonal becomes `False`.

For five tokens, its shape is `[5, 5]`.

A batch may also contain padding:

```text
[The, cat, sat, <pad>, <pad>]
```

`combine_causal_and_padding_masks()` requires all three conditions:

```python
causal_mask & real_queries & real_keys
```

A connection is allowed only when the key is not in the future, the query is
real, and the key is real. The returned shape is `[batch, query, key]`.

### 2. `transformer_decoder.py`

`DecoderBlock` reuses your Day 10 components:

```python
self.self_attention = MultiHeadAttention(...)
self.feed_forward = FeedForwardNetwork(...)
```

The important difference is the permission matrix passed to attention:

```python
decoder_attention_mask = combine_causal_and_padding_masks(padding_mask)
attention_output, attention_weights = self.self_attention(
    token_embeddings,
    decoder_attention_mask,
)
```

Day 10's attention function places a very negative value at every blocked
score before softmax. Softmax then gives those positions probability `0`.

The decoder keeps the two familiar residual paths:

```python
x = LayerNorm(x + CausalSelfAttention(x))
x = LayerNorm(x + FeedForward(x))
```

`TransformerDecoder` stacks several blocks. `CausalLanguageModel` adds:

1. a token embedding;
2. sinusoidal positional encoding;
3. the decoder stack;
4. a linear output projection.

If the input shape is `[batch, sequence]`, the final logits have shape:

```text
[batch, sequence, vocabulary_size]
```

Every position receives one score for every possible next token. These raw
scores are logits; do not manually apply softmax before cross-entropy loss.

This lab implements a **decoder-only block**. A T5-style decoder would also
contain cross-attention that reads the encoder output.

### 3. `transformer_utils.py`

Language-model training shifts one sequence into an input and an answer:

```text
Original: [The, cat, sat, on, mat]
Input:    [The, cat, sat, on]
Target:   [cat, sat, on, mat]
```

At input position `The`, the expected next token is `cat`. At input position
`cat`, the expected next token is `sat`.

`prepare_next_token_batch()` performs this shift. The loss utility ignores
padded targets. The verification utilities inspect the upper triangle of an
attention matrix and require every value there to be zero.

### 4. `visualize_masks.py`

This script creates the five-token example, runs a small one-layer decoder,
and saves two heatmaps:

1. the causal permission rule;
2. actual probabilities from layer 1, head 1.

The attention values are random because this visualization does not train a
language model. That is acceptable here: the purpose is to demonstrate that
the blocked cells remain exactly zero before and after learning.

### 5. `test_masking_pipeline.py`

The tests verify more than shapes:

- the exact five-token lower triangle;
- future and padding positions are blocked together;
- every future probability is zero;
- model logits have the correct shape;
- changing future tokens cannot change earlier logits;
- shifted inputs and targets are correct;
- next-token loss backpropagates gradients;
- the verifier detects intentionally invalid attention;
- the decoder block receives gradients.

The future-change test is especially important. These two inputs share their
first three tokens:

```text
[2, 3, 4, 5, 6]
[2, 3, 4, 9, 10]
```

Their logits at positions 0 through 2 must be identical. If those logits
change, information leaked backward from the future.

## Run Day 11

From the repository root with `.venv` active:

```powershell
python -m pytest backend/tests/test_masking_pipeline.py -q
```

Run every NLP test:

```powershell
python -m pytest backend/tests/test_nlp_pipeline.py backend/tests/test_transformer_pipeline.py backend/tests/test_attention_pipeline.py backend/tests/test_masking_pipeline.py -q
```

Create the visualization:

```powershell
python -m backend.ml.nlp.visualize_masks
```

If needed:

```powershell
python -m pip install matplotlib
```

Expected terminal matrix:

```text
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1
```

Expected verification:

```text
Verification passed: all future-token probabilities are zero.
```

Expected image:

```text
images/causal_attention_mask.png
```

## Lab and quiz answers

### 1. Draw the attention matrix for five tokens

```text
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1
```

### 2. Why can GPT not attend to future tokens?

GPT predicts text from left to right. If a position could read the correct
tokens to its right during training, it could copy information that will not
exist during real generation. Blocking future positions keeps training
consistent with inference and forces the model to learn next-token prediction
from the available history.

### 3. Compare BERT and GPT in one paragraph

BERT is encoder-only and uses bidirectional attention, so every input token can
use both left and right context; this makes it well suited to understanding
tasks such as classification, embeddings, semantic search, and retrieval. GPT
is decoder-only and uses causal attention, so each position can use only itself
and earlier positions; this makes it suitable for generating text one token at
a time. BERT primarily builds representations of complete inputs, while GPT
primarily predicts what should come next.

### 4. When should you use an encoder-decoder architecture?

Use an encoder-decoder architecture when the model must first understand a
complete source sequence and then generate a separate target sequence. Common
examples include translation, summarization, and converting a document into a
structured output. The encoder reads the source bidirectionally, while the
decoder generates the target from left to right and uses cross-attention to
read the source representation.

### 5. Explain causal attention without using the word “mask”

Causal attention applies a one-way visibility rule: each position may read
itself and everything to its left, but nothing to its right. This prevents a
next-token predictor from seeing answers that have not been generated yet.

## Common mistakes

1. **Reversing boolean meaning.** This project uses `True = allowed`. Some
   PyTorch APIs use `True = blocked`, so always verify the local convention.
2. **Blocking only padding.** Decoder attention needs both padding and causal
   rules.
3. **Using a diagonal of zeroes.** A token must be able to attend to itself, so
   the diagonal is `True`.
4. **Applying the rule after softmax.** Block scores before softmax so the
   resulting probabilities are normalized correctly.
5. **Calling this a full T5 decoder.** Decoder-only self-attention does not
   include encoder-decoder cross-attention.

## Definition of done

- [x] Causal permission generation implemented.
- [x] Causal and padding rules combined.
- [x] Decoder block and stack implemented.
- [x] Next-token language-model head implemented.
- [x] Visualization code implemented.
- [x] Behavioral unit tests written.
- [ ] Run the tests successfully in your `.venv`.
- [ ] Generate `images/causal_attention_mask.png`.
- [ ] Record the actual test result in this README.
