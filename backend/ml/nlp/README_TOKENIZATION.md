# Day 12: Tokenization Explorer

## Learning objective

By the end of this lab, you should be able to:

- explain what a token is;
- convert text into tokens and local token IDs;
- compare several tokenization strategies;
- estimate whether text fits inside a context window;
- generate a terminal and HTML token visualization;
- explain the central idea behind Byte Pair Encoding (BPE).

## Important accuracy note

The IDs in this project are **local demonstration IDs**. They are assigned by
the `Vocabulary` class from the tokens produced in the current example.

They are not official GPT, BERT, or SentencePiece token IDs. Production model
IDs depend on the exact tokenizer vocabulary and merge rules shipped with that
model.

## Project files

```text
backend/ml/nlp/
├── tokenizer_demo.py
├── token_visualizer.py
├── vocabulary.py
├── compare_tokenizers.py
└── README_TOKENIZATION.md

backend/tests/
└── test_tokenization_explorer.py
```

## How text becomes model input

```text
Raw text
   ↓
Tokenizer splits text
   ↓
["Artificial", "Intelligence", "!"]
   ↓
Vocabulary lookup
   ↓
[2, 3, 4]
   ↓
Embedding layer converts IDs to vectors
```

A neural network cannot directly perform matrix multiplication on a Python
string. Token IDs are integer lookup keys. An embedding layer uses each ID to
select a learned vector.

## Included strategies

### 1. Whitespace tokenization

```text
"Hello, world!" → ["Hello,", "world!"]
```

It is easy to understand, but punctuation stays attached to words. This causes
`world`, `world!`, and `world?` to look like different tokens.

### 2. Word and punctuation tokenization

```text
"Hello, world!" → ["Hello", ",", "world", "!"]
```

This is closer to a traditional NLP tokenizer. It separates words, numbers,
and punctuation.

### 3. Character tokenization

```text
"AI!" → ["A", "I", "!"]
```

It never encounters an unknown word, but it creates long token sequences.

### 4. Toy Byte Pair Encoding

The BPE tokenizer begins with characters. During training it repeatedly finds
the most frequent neighboring symbol pair and merges it.

Simplified example:

```text
l o w
l o w e r
```

The pair `l + o` may occur often, so it becomes `lo`.

```text
lo w
lo w e r
```

Later, `lo + w` may become `low`.

This project learns 60 pair merges from a small built-in corpus. It is a real
implementation of the basic BPE merge loop, but it is intentionally tiny.
Production tokenizers train on very large corpora and add byte-level,
normalization, special-token, and compatibility rules.

## BPE, WordPiece, and SentencePiece

| Method | Main idea | Common strength |
|---|---|---|
| BPE | Repeatedly merge frequent adjacent symbols | Simple, reusable subword pieces |
| WordPiece | Select pieces using a language-model-oriented vocabulary objective and use greedy longest matching | Strong handling of prefixes and continuations in BERT-style vocabularies |
| SentencePiece | Train directly from raw text and represent whitespace explicitly | Language-independent and useful for multilingual text |

SentencePiece is a tokenizer toolkit/model format rather than only one merge
algorithm. It can train unigram or BPE models. That is why it should not be
treated as exactly the same category as BPE and WordPiece.

## Run the explorer

From the repository root:

```powershell
python -m backend.ml.nlp.tokenizer_demo `
  "Artificial Intelligence is amazing!" `
  --strategy all `
  --context-window 512
```

Run only one strategy:

```powershell
python -m backend.ml.nlp.tokenizer_demo `
  "unbelievable tokenization" `
  --strategy bpe
```

Generate an HTML visualization:

```powershell
python -m backend.ml.nlp.tokenizer_demo `
  "Hello world!" `
  --strategy word `
  --html backend/ml/nlp/artifacts/tokens.html
```

Open the generated HTML file in a browser:

```powershell
start backend/ml/nlp/artifacts/tokens.html
```

Compare tokenizers without printing every token:

```powershell
python -m backend.ml.nlp.compare_tokenizers `
  "Artificial Intelligence is amazing!"
```

## Run tests

The existing ML requirements already include pytest.

```powershell
python -m pytest backend/tests/test_tokenization_explorer.py -q
```

## Read the code in this order

1. `WhitespaceTokenizer.tokenize()` — the smallest possible tokenizer.
2. `WordPunctuationTokenizer.tokenize()` — regex-based token splitting.
3. `CharacterTokenizer.tokenize()` — one character per token.
4. `Vocabulary.build()` — deterministic integer ID assignment.
5. `analyze_text()` — connects tokenization, IDs, and context estimation.
6. `BPETokenizer._learn_merges()` — the repeated-pair learning loop.
7. `token_visualizer.py` — formats terminal and HTML output.
8. `main()` — parses command-line arguments and runs the program.

## Key code explanations

### Why use a protocol?

`Tokenizer` is a `Protocol`. Any class with a `name` and `tokenize()` method can
be used by `analyze_text()`. This is polymorphism without forcing every class
to inherit from a base class.

### Why use a dataclass?

`TokenizationResult` groups related output:

- strategy name;
- original text;
- tokens;
- token IDs;
- vocabulary size;
- context window.

`frozen=True` prevents accidental changes after the result is created.

### Why sort vocabulary tokens?

A Python dictionary should not determine IDs accidentally. The vocabulary
sorts by:

1. highest frequency;
2. alphabetical token order for ties.

The same input therefore receives the same local IDs on repeated runs.

### Why preserve character whitespace?

The character tokenizer uses `list(text)`. A space is information, so it
becomes a token. The visualizer displays a space as `␠` to make it visible.

### Why is context length important?

Self-attention compares token positions. As token count grows, standard
attention requires much more computation and memory. A larger context window
can hold more instructions, examples, and documents, but it is not free.

The explorer calculates:

```text
percentage used = token count / context window × 100
remaining tokens = context window - token count
```

A real request must also reserve tokens for system instructions, chat history,
tool messages, and the model's output.

## Lab answers

### 1. What is a token?

A token is a unit of text selected by a tokenizer. It may be a whole word, part
of a word, punctuation, a number, a character, or a representation of
whitespace.

### 2. Why do LLMs not operate directly on words?

A word-only vocabulary would become very large and still fail on spelling
variants, new words, names, code, and multilingual text. Subword pieces let the
model reuse a manageable vocabulary while representing unfamiliar words.

### 3. Compare BPE, WordPiece, and SentencePiece

BPE starts with small symbols and repeatedly merges frequent neighboring
pairs. WordPiece builds a subword vocabulary using a model-oriented objective
and commonly tokenizes with greedy longest matching. SentencePiece learns from
raw text, represents spaces explicitly, and supports algorithms such as BPE
and unigram language modeling, which makes it convenient for multilingual
data.

### 4. Why does context length matter?

The context window limits how many input and output tokens the model can handle
in one request. Text beyond the limit must be shortened, chunked, summarized,
or processed in separate requests. Longer sequences also require more memory
and computation.

### 5. Paragraph token estimates

A rough English estimate is approximately one token for every three to four
characters, but language, punctuation, numbers, code, and tokenizer choice can
change the result.

Example estimates:

| Paragraph | Characters | Approximate tokens |
|---|---:|---:|
| 80-character short paragraph | 80 | 20–27 |
| 300-character medium paragraph | 300 | 75–100 |
| 800-character long paragraph | 800 | 200–267 |

Use the explorer instead of trusting the estimate. Then compare the whitespace,
word-punctuation, toy-BPE, and character counts.

## Practice task

Run this text through all strategies:

```text
Unbelievable! LLM tokenization handles words, numbers like 128K, and symbols.
```

Answer:

1. Which strategy creates the fewest tokens?
2. Which creates the most?
3. How does BPE split `unbelievable`?
4. Why are the token IDs local rather than official GPT IDs?
5. Does the text fit inside a 32-token context window?
