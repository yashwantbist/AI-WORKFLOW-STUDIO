Day 9: Transformer Text Classification

This lesson replaces Day 8's learned embedding plus masked mean-pooling modelwith a Transformer encoder. Both models use the same generated sentimentdataset, vocabulary, train/validation split, random seed, batch size, Adamoptimizer, learning rate, and number of epochs.

Architecture

flowchart TD
    A["Token IDs + attention mask"] --> B["Token embeddings"]
    B --> C["Sinusoidal positional encoding"]
    C --> D["Transformer encoder × 2"]
    D --> E["Masked mean pooling"]
    E --> F["LayerNorm + dropout + linear head"]
    F --> G["Negative or positive logits"]

The Transformer encoder layer contains multi-head self-attention, afeed-forward neural network, residual connections, and layer normalization.

From text to prediction

For the sentence the course is excellent:

Tokenization produces ["the", "course", "is", "excellent"].

Vocabulary lookup changes words into integer token IDs.

Embedding changes each ID into a learned vector.

Positional encoding adds a different position signal to each vector.

Self-attention lets every token directly examine all other tokens.

Masked mean pooling combines the real token outputs into one sentencevector while ignoring <pad>.

Classification head produces two logits: negative and positive.

CrossEntropyLoss compares those logits with the correct label duringtraining.

Query, Key, and Value

Inside self-attention, each token embedding is projected into three vectors:

Vector

Beginner meaning

Job

Query (Q)

"What am I looking for?"

Searches for relevant tokens

Key (K)

"What information do I represent?"

Is compared with every query

Value (V)

"What information should I contribute?"

Is mixed into the result

The attention calculation is:

\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

QKᵀ produces relevance scores. Dividing by √dₖ keeps large vectors fromcreating extreme scores. softmax turns the scores into weights, and theweights determine how much of each value vector is used.

PyTorch creates Q, K, and V internally insidenn.TransformerEncoderLayer, so the training code does not manually buildthose matrices.

Multi-head attention

One attention head may learn sentiment words such as excellent or awful.Another may learn which subject the sentiment describes. Another can learnword-order relationships.

This project uses four heads. The 32-dimensional representation is split intofour 8-dimensional heads, processed in parallel, and combined.

Why positional encoding is required

Self-attention alone has no built-in left-to-right order. Without positioninformation, the same words in dog bites man and man bites dog would behard to distinguish.

SinusoidalPositionalEncoding creates fixed sine and cosine patterns:

token_embeddings = self.embedding(token_ids)
positioned_embeddings = self.positional_encoding(token_embeddings)

Adding these patterns tells the encoder where each token occurs while keepingall tokens parallel.

RNN, LSTM, and Transformer comparison

Model

How it reads text

Long-range memory

Parallel training

Main limitation

RNN

One token at a time

Weak

No

Vanishing gradients and forgetting

LSTM

One token at a time with gates

Better than RNN

No

Still sequential and slow to scale

Transformer

All tokens together using attention

Direct token-to-token links

Yes

Attention cost grows quickly with sequence length

LSTMs improved RNNs by adding input, forget, and output gates. These gates helpinformation survive longer, but the model must still finish token 1 beforetoken 2, then token 3, and so on.

Transformers removed that sequential bottleneck. Every token can directlyattend to every other token in the same layer. This enables GPU parallelismand short paths between distant words.

Why GPT uses Transformers

GPT means Generative Pre-trained Transformer. It uses the Transformerdecoder architecture with causal attention:

Generative: predicts the next token.

Pre-trained: learns language patterns from a large amount of text.

Transformer: uses self-attention instead of recurrent memory.

Causal mask: prevents a token from looking at future tokens duringnext-token training.

The classifier in this lesson uses an encoder because it can look at theentire input sentence. GPT generates left to right, so it uses causalTransformer blocks.

Important code

1. Embedding and position

token_embeddings = self.embedding(token_ids) * self.embedding_scale
positioned_embeddings = self.positional_encoding(token_embeddings)

The scale keeps embedding magnitudes useful relative to positional values.

2. Ignore padding during attention

encoded_tokens = self.encoder(
    positioned_embeddings,
    src_key_padding_mask=~attention_mask,
)

Our dataset mask uses True for real tokens. PyTorch's padding mask usesTrue for ignored tokens, so ~ reverses it.

3. Classification

sentence_representation = summed_tokens / real_token_counts
logits = self.classification_head(sentence_representation)

Logits are raw scores. Do not manually apply softmax beforeCrossEntropyLoss; that loss function performs the required calculationsafely.

Files

transformer_model.py: positional encoding and Transformer encoder model.

train_transformer_classifier.py: Transformer training and validation.

compare_models.py: controlled comparison against Day 8.

test_transformer_pipeline.py: shape, masking, gradient, and training tests.

Run the Transformer

From the repository root:

python backend/ml/nlp/train_transformer_classifier.py

Generated checkpoints are written under backend/ml/nlp/artifacts/ and areexcluded by the repository's *.pt rule.

Run the fair comparison

python backend/ml/nlp/compare_models.py

The controlled comparison uses:

192 training and 48 validation examples;

representation size 32;

random seed 42;

batch size 16;

Adam with learning rate 0.01;

CrossEntropyLoss;

10 epochs.

The Day 8 code called the held-out split "test." In this lesson it is called"validation" because it is checked after every epoch.

Measured results

The values below must come from an executed run of compare_models.py.

Model

Trainable parameters

Final training loss

Training accuracy

Validation accuracy

Training time

Embedding + mean pooling

1,474

0.0096

100.00%

100.00%

0.0665 seconds

Transformer encoder

18,626

0.0006

100.00%

100.00%

0.5460 seconds





What the results mean

Both models reached 100% final training and validation accuracy. Therefore,this experiment does not show an accuracy improvement from the Transformer.

The Transformer:

reached 100% validation accuracy in epoch 1; the baseline reached it inepoch 2;

finished with 93.75% lower training loss;

used about 12.64 times as many trainable parameters;

took about 8.21 times as long to train;

briefly dropped to 95.83% validation accuracy in epoch 4 before returningto 100%.

The lower loss means the Transformer produced more confident correctpredictions on this run. It does not mean its final validation accuracy wasbetter, because both models classified all 48 validation examples correctly.

This generated dataset is small, its sentences are short, and individualsentiment words reveal the label. Mean pooling can solve that kind ofbag-of-words problem very well.

The Transformer advantage becomes clearer when meaning depends on order,negation, relationships between distant words, and large-scale pre-training.That is why this lab measures both accuracy and cost instead of claiming thata larger model is automatically better.

Tests

python -m pytest backend/tests/test_nlp_pipeline.py \
  backend/tests/test_transformer_pipeline.py -q

Executed result: 14 tests passed in 1.88 seconds. The environment alsoprinted one warning because NumPy was not installed; the warning did not failor skip any test.