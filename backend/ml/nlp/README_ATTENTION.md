# Transformer Encoder and Multi-Head Attention Lab

This lesson rebuilds the main parts hidden inside
`nn.TransformerEncoderLayer`. The implementation keeps the same batch-first
shape and padding convention as the Day 9 sentiment classifier.

## Draw the encoder from memory

```text
Token IDs
    ↓
Token Embeddings × √d_model
    +
Positional Encoding
    ↓
Multi-Head Self-Attention
    ↓
Residual Add + LayerNorm
    ↓
Feed-Forward Network
    ↓
Residual Add + LayerNorm
    ↓
Contextual Output Embeddings
```

The encoder block can be stacked. Every block receives and returns a tensor
with shape:

```text
[batch_size, sequence_length, model_dimension]
```

## The attention calculation

For one head:

```text
Q = XWq
K = XWk
V = XWv

Attention(Q, K, V) = softmax(QKᵀ / √d_head)V
```

Dividing by `√d_head` prevents large dot products from pushing softmax into
extremely confident values too early. The padding mask is applied before
softmax. In this project, `True` means a real token and `False` means padding.

With four heads and `model_dimension=32`, each head works with eight features.
The four outputs are concatenated back into 32 features and passed through one
output projection.

## Quiz explanations

### Why are residual connections needed?

A residual connection adds a sublayer's input to its output:

```text
y = LayerNorm(x + Sublayer(x))
```

The original representation therefore has a direct path through the block.
That improves gradient flow, preserves useful information, and makes deeper
networks easier to optimize.

### What does LayerNorm do?

LayerNorm normalizes the features inside each token representation, then learns
a scale and shift. It keeps activation scales controlled and makes training
more stable. Unlike BatchNorm, its result does not depend on other examples in
the batch, which works well for variable-length text.

### Why is an FFN needed after attention?

Attention mixes information **between tokens**, but its weighted combination
is mostly linear. The feed-forward network applies a nonlinear transformation
to each token independently:

```text
Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)
```

The wider hidden layer lets the model create and combine richer features after
attention has supplied context.

### One head versus multiple heads

| Property | One head | Multiple heads |
|---|---|---|
| Attention maps | One | One per head |
| Representation subspaces | One large view | Several smaller parallel views |
| Relationships | May combine patterns in one map | Can specialize in different patterns |
| Cost | Simpler to inspect | More projections and attention maps |
| Requirement | Any positive model dimension | Model dimension must divide evenly by head count |

Heads are not guaranteed to learn labels such as "grammar" or "pronouns."
Those are useful intuitions, but specialization emerges from training and must
be checked by inspecting the learned weights.

## Files

- `attention.py`: scaled dot-product attention and `MultiHeadAttention`.
- `transformer_encoder.py`: FFN, residual connections, LayerNorm, encoder
  block/stack, and `AttentionTextClassifier`.
- `visualize_attention.py`: trains on the shared sentiment data, predicts one
  sentence, measures inference latency, and saves per-head heatmaps.
- `test_attention_pipeline.py`: attention math, masks, shapes, residual paths,
  gradients, classifier outputs, and a real optimization test.

## Run the unit tests

From the repository root:

```powershell
python -m pytest backend/tests/test_attention_pipeline.py -q
```

Run all NLP tests:

```powershell
python -m pytest backend/tests/test_nlp_pipeline.py backend/tests/test_transformer_pipeline.py backend/tests/test_attention_pipeline.py -q
```

## Train and visualize attention

Install the visualization dependency if it is not already available:

```powershell
python -m pip install matplotlib
```

Then run:

```powershell
python -m backend.ml.nlp.visualize_attention
```

The default sentence is:

```text
this course is helpful and i recommend it
```

Use a different sentence or shorter training run:

```powershell
python -m backend.ml.nlp.visualize_attention --text "what a horrible movie i hate it" --epochs 5
```

The script records:

- final training accuracy;
- final validation accuracy;
- average inference latency in milliseconds per sentence;
- predicted class and confidence.

It saves:

```text
images/attention_weights.png
```

Rows are query tokens. Columns are key tokens. A darker cell means the row
token assigned more attention probability to the column token.

## Controlled baseline comparison

Keep the stretch-goal comparison fair by using the existing Day 9 command:

```powershell
python -m backend.ml.nlp.compare_models
```

The confirmed Day 9 run used the same 192/48 split, representation size 32,
seed 42, batch size 16, Adam at `0.01`, `CrossEntropyLoss`, and 10 epochs:

| Model | Final loss | Training accuracy | Validation accuracy | Training time |
|---|---:|---:|---:|---:|
| Embedding + mean pooling | 0.0096 | 100.00% | 100.00% | 0.2466 s |
| PyTorch Transformer encoder | 0.0007 | 100.00% | 100.00% | 1.9015 s |

Do not claim the Transformer improved final accuracy in this run: both models
classified all 48 validation examples correctly. The Transformer was more
confident but more expensive. Its advantage should be tested on harder examples
where meaning depends on order, negation, or long-distance relationships.

Record the new hand-built model's printed metrics after running
`visualize_attention.py`. Results depend on the computer and must not be marked
as passed until the commands complete successfully.

## Definition of done

- [x] Scaled dot-product attention implemented.
- [x] Multi-head self-attention implemented.
- [x] Two residual connections implemented.
- [x] Two LayerNorm operations implemented.
- [x] Position-wise feed-forward network implemented.
- [x] Stackable encoder block implemented.
- [x] Attention weights exposed for visualization.
- [x] Unit tests written.
- [ ] Run the tests successfully in your `.venv`.
- [ ] Train the model and record the local metrics.
- [ ] Generate `images/attention_weights.png`.
