Day 8: NLP Tokenization and Embeddings

This module implements a beginner-friendly sentiment-classification pipelineusing PyTorch. It demonstrates how raw text becomes token IDs, embeddings, apooled sentence representation, and finally a positive or negative prediction.

Pipeline

Raw text
    ↓
Tokenization
    ↓
Vocabulary lookup
    ↓
Token IDs
    ↓
nn.Embedding
    ↓
Mean pooling
    ↓
Linear classifier
    ↓
Positive or negative prediction

Core concepts

Tokenization

Tokenization divides text into smaller pieces:

"I love this course!"
        ↓
["i", "love", "this", "course"]

The tokenizer lowercases text and keeps words, numbers, and simplecontractions.

Vocabulary and token IDs

The vocabulary is built only from training examples. Each known token receives a stable integer ID. Two special tokens are included:

<pad> has ID 0 and fills unused positions in a batch.

<unk> has ID 1 and represents words not seen during training.

Embeddings

nn.Embedding converts every token ID into a learned dense vector. During training, words connected with positive sentiment can learn different vectorpatterns from words connected with negative sentiment.

Mean pooling

Sentences have different lengths. The model averages the embeddings belongingto real tokens and ignores <pad> positions. The resulting fixed-size vectoris passed to one linear classification layer.

Dataset

The project generates a small, balanced sentiment dataset in memory fromsentence templates. It creates 120 positive and 120 negative examples, thenuses a deterministic class-balanced 80/20 training/test split.

No dataset archive or generated dataset file is stored in the repository. Thesame random seed is used for both embedding experiments, so they receive thesame examples in the same order.

Files

dataset.py: tokenization, vocabulary, generated examples, padding, masks,and data loaders.

model.py: nn.Embedding, mask-aware mean pooling, and the linearclassifier.

train_text_classifier.py: training, evaluation, timing, experimentcomparison, and checkpoint creation.

inference.py: loads a trained checkpoint and predicts new text.

Install

From the project root with your virtual environment activated:

python -m pip install -r backend/ml/requirements-ml.txt

Train one experiment

From the project root:

python backend/ml/nlp/train_text_classifier.py --embedding-size 32

Run the complete lab

Run both required embedding experiments:

python backend/ml/nlp/train_text_classifier.py --compare-embeddings

Both experiments use:

CrossEntropyLoss;

the Adam optimizer;

batch size 16;

learning rate 0.01;

10 epochs;

random seed 42.

The trainer prints the epoch, average training loss, training accuracy, testaccuracy, and measured training time. It saves the model with the highestmeasured test accuracy to:

backend/ml/nlp/artifacts/text_classifier.pt

The checkpoint is ignored by Git.

Run inference

After training:

python backend/ml/nlp/inference.py "I love this excellent course"

The script prints the tokens, token IDs, prediction, confidence, andprobability for each class.

Because this is a deliberately small template-generated learning dataset, themodel is only a demonstration. It should not be treated as a productionsentiment-analysis system.

Run tests

python -m pytest backend/tests/test_nlp_pipeline.py -q

Measured results

Record only the values printed by your completed run:

Embedding size

Final loss

Training accuracy

Test accuracy

Training time


| Embedding size | Final loss | Training accuracy | Test accuracy | Training time |
|---:|---:|---:|---:|---:|
| 32 | 0.0096 | 100.00% | 100.00% | 0.1966 seconds |
| 128 | 0.0026 | 100.00% | 100.00% | 0.2411 seconds |

### Observations

- Both models reached 100% final training and test accuracy.
- Embedding size 128 reached 100% test accuracy one epoch earlier.
- Size 128 had the lower final loss: 0.0026 versus 0.0096.
- Size 32 trained 0.0445 seconds faster.
- Size 128 took approximately 22.6% longer.
- Seven tests passed in 2.65 seconds.

Git safety

Before committing, confirm that generated checkpoints, datasets, virtualenvironments, and secrets are not staged:

git status --short
git diff --cached --name-only

Suggested branch:

feature/nlp-foundations

Suggested commit:

feat(ml): add NLP text classification pipeline

Suggested pull request:

Day 8: Add NLP tokenization and embedding pipeline