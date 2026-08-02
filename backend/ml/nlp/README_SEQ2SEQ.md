# Encoder-Decoder Transformer and Cross-Attention Lab

## Measurable objective

By the end of this lab, you can:

1. explain encoder self-attention, decoder self-attention, and cross-attention;
2. pass encoder representations into a decoder;
3. generate target tokens one position at a time;
4. inspect three different attention maps;
5. verify the attention flow with behavioral tests.

This lesson reuses:

- Day 10 `TransformerEncoder`, `MultiHeadAttention`, positional encoding, and
  feed-forward network;
- Day 11 causal and padding rules plus next-token loss.

The Day 11 decoder-only model remains unchanged. The new
`Seq2SeqDecoderBlock` adds the cross-attention sublayer required by an
encoder-decoder Transformer.

## Complete architecture

```text
Source token IDs
      ↓
Source embedding + position
      ↓
Encoder bidirectional self-attention
      ↓
EncoderOutput.hidden_states ─────────────────────┐
                                                │ keys + values
Target token IDs                                │
      ↓                                         │
Target embedding + position                    │
      ↓                                         │
Decoder causal self-attention                   │
      ↓                                         │
Cross-attention ←───────────────────────────────┘
      ↓
Feed-forward network
      ↓
Target-vocabulary logits
      ↓
Next generated token
```

## Three attention paths

| Path | Query comes from | Key/value come from | Weight shape | Purpose |
|---|---|---|---|---|
| Encoder self-attention | Source | Source | `[batch, heads, source, source]` | Understand the complete input |
| Decoder self-attention | Target history | Target history | `[batch, heads, target, target]` | Understand generated tokens without reading the future |
| Cross-attention | Decoder | Encoder | `[batch, heads, target, source]` | Retrieve relevant source information while generating |

### Self-attention

Self-attention uses one sequence for all three projections:

```text
Q = sequence × Wq
K = sequence × Wk
V = sequence × Wv
```

### Cross-attention

Cross-attention uses two sequences:

```text
Q = decoder states × Wq
K = encoder states × Wk
V = encoder states × Wv
```

The decoder asks a question with `Q`. The encoder positions advertise what
information they contain through `K`. The selected encoder information is
read from `V`.

## Beginner-friendly code walkthrough

### 1. `cross_attention.py`

`CrossAttention` looks similar to Day 10's multi-head self-attention, but its
inputs come from different places:

```python
query = self.query_projection(decoder_states)
key = self.key_projection(encoder_states)
value = self.value_projection(encoder_states)
```

For the example:

```text
Source: bonjour tout le monde       → source length 4
Target input: <bos> hello everyone  → target length 3
```

one cross-attention head has shape:

```text
[target queries, source keys] = [3, 4]
```

Each target row contains probabilities over the four source tokens. Source
padding positions receive probability zero.

### 2. `EncoderOutput`

`seq2seq_transformer.py` defines:

```python
@dataclass(frozen=True)
class EncoderOutput:
    hidden_states: Tensor
    attention_mask: Tensor
    self_attention: tuple[Tensor, ...]
```

Think of this as the encoder's completed notebook:

- `hidden_states` contains the contextual source representations;
- `attention_mask` identifies real source positions;
- `self_attention` contains maps for learning and visualization.

The encoder creates this object once. Every decoder step receives the same
object, so the original source remains available throughout generation.

### 3. `Seq2SeqDecoderBlock`

A decoder-only block from Day 11 had two sublayers:

```text
Causal self-attention → FFN
```

The encoder-decoder block has three:

```text
Causal self-attention → Cross-attention → FFN
```

The first sublayer asks, “What have I already generated?”

```python
self_attention_output, self_attention_weights = self.self_attention(
    decoder_states,
    causal_mask,
)
```

The second asks, “Which source information is useful now?”

```python
cross_attention_output, cross_attention_weights = self.cross_attention(
    decoder_states=decoder_states,
    encoder_states=encoder_output.hidden_states,
    encoder_attention_mask=encoder_output.attention_mask,
)
```

Each sublayer has a residual connection and LayerNorm:

```text
LayerNorm(original + sublayer_output)
```

### 4. `Seq2SeqTransformer.encode()`

`encode()` reads the source with bidirectional self-attention:

```python
hidden_states, self_attention = self.encoder(
    positioned_source,
    source_attention_mask,
)
```

Unlike a decoder, the encoder does not use a causal rule. Every real source
token may read every other real source token.

### 5. `Seq2SeqTransformer.decode()`

`decode()` receives target tokens and the saved `EncoderOutput`:

```python
decoder_states, decoder_self_attention, cross_attention = self.decoder(
    positioned_target,
    encoder_output,
    target_attention_mask,
)
```

The final linear layer converts each decoder representation into target
vocabulary scores:

```python
logits = self.output_projection(decoder_states)
```

The shape is:

```text
[batch, target_sequence, target_vocabulary_size]
```

### 6. `greedy_decode()`

Inference starts with `<bos>`, meaning beginning of sequence:

```text
Step 1: <bos>                 → predict hello
Step 2: <bos> hello           → predict everyone
Step 3: <bos> hello everyone  → predict <eos>
```

The source is encoded before the loop:

```python
encoder_output = model.encode(source_token_ids)
```

Only the growing target is decoded again. This is the clearest code-level
answer to why encoder outputs remain available throughout decoding.

### 7. `visualize_cross_attention.py`

The toy dataset contains six small French-to-English mappings, including:

```text
bonjour                  → hello
bonjour tout le monde    → hello everyone
merci                    → thank you
merci ami                → thank you friend
au revoir                → goodbye
```

Training uses teacher forcing:

```text
Full target:    <bos> hello everyone <eos>
Decoder input: <bos> hello everyone
Expected next: hello everyone <eos>
```

This is a learning demonstration, not a production translation dataset. The
model memorizes a few examples so you can observe the complete training,
inference, and attention flow on CPU.

The visualization contains three panels:

1. encoder self-attention: source rows by source columns;
2. decoder causal self-attention: target rows by target columns;
3. cross-attention: target rows by source columns.

### 8. `test_seq2seq_pipeline.py`

The nine tests verify:

- cross-attention output and weight shapes;
- attention probabilities sum to one;
- padded source tokens receive zero probability;
- mismatched batches are rejected;
- the encoder returns a reusable `EncoderOutput`;
- encoder, decoder, and cross-attention maps are all exposed;
- changing the source changes decoder output;
- changing future target tokens does not change earlier logits;
- gradients reach cross-attention parameters;
- greedy generation starts with `<bos>` and respects its limit.

## Run the tests

From the repository root with `.venv` active:

```powershell
python -m pytest backend/tests/test_seq2seq_pipeline.py -q
```

Run all NLP tests:

```powershell
python -m pytest backend/tests/test_nlp_pipeline.py backend/tests/test_transformer_pipeline.py backend/tests/test_attention_pipeline.py backend/tests/test_masking_pipeline.py backend/tests/test_seq2seq_pipeline.py -q
```

## Run toy training, inference, and visualization

```powershell
python -m backend.ml.nlp.visualize_cross_attention
```

For a shorter debugging run:

```powershell
python -m backend.ml.nlp.visualize_cross_attention --training-steps 50
```

The default run trains for at most 300 steps but stops early after the toy
dataset reaches 100% token accuracy with sufficiently low loss.

Expected output file:

```text
images/cross_attention.png
```

Do not copy example metrics into this README. Record only the loss, accuracy,
translation, and test result printed by your actual local run.

## Lab and quiz answers

### 1. Draw an encoder-decoder Transformer

```text
Source IDs → Embedding + Position → Encoder self-attention → Encoder output
                                                               ↓
Target IDs → Embedding + Position → Causal self-attention → Cross-attention
                                                               ↓
                                                        FFN → logits
```

### 2. Self-attention versus cross-attention

Self-attention creates queries, keys, and values from the same sequence. It
helps tokens understand other tokens within that sequence. Cross-attention
creates queries from the decoder and keys/values from the encoder. It lets the
target generation retrieve information from the source input.

### 3. How machine translation works

The encoder reads the complete source sentence and creates one contextual
representation per source token. The decoder begins with a start token and
generates the translation from left to right. At each position, causal
self-attention reads the translation generated so far, while cross-attention
selects useful source representations. A vocabulary projection produces the
scores used to select the next target token. Generation ends at `<eos>`.

### 4. Compare BERT, GPT, and T5

| Model family | Architecture | Attention visibility | Strong use cases |
|---|---|---|---|
| BERT | Encoder-only | Bidirectional over the input | Classification, embeddings, retrieval |
| GPT | Decoder-only | Causal over generated history | Chat, completion, code/text generation |
| T5 | Encoder-decoder | Bidirectional source, causal target, plus cross-attention | Translation, summarization, source-to-target tasks |

### 5. Why do encoder outputs remain available?

The source meaning does not need to be recalculated after every generated
token. The encoder runs once and its output is stored. Every decoder layer and
every generation step can use cross-attention to query those same stored
representations. This is both computationally efficient and necessary for the
decoder to continuously reference the original input.


- [ ] Generate `images/cross_attention.png`.
- [ ] Review `.gitignore` and confirm no secrets are tracked.
