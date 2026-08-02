# Day 13: LLM Text Generation and Decoding Strategies

This lab adds five ways to choose the next token from a language model:

1. greedy decoding
2. beam search
3. temperature sampling
4. top-k sampling
5. top-p (nucleus) sampling

The model produces scores. The decoding strategy turns those scores into a
choice. Changing the strategy does **not** retrain or change the model.

## Mental model: autoregressive generation

An autoregressive model builds text one token at a time:

```text
Prompt token IDs
      ↓
Language model
      ↓
Scores for every possible next token
      ↓
Decoding strategy chooses one token
      ↓
Append that token to the prompt and repeat
```

For example, the model may assign:

| Token | Probability |
|---|---:|
| cat | 0.52 |
| dog | 0.28 |
| bird | 0.12 |
| fish | 0.08 |

The probabilities come from the model. The five strategies differ only in how
they use those probabilities.

## The five strategies

### 1. Greedy decoding

Greedy decoding uses `argmax`, meaning it always takes the token with the
largest score. The example above always produces `cat`.

- Fast and repeatable.
- Useful when predictability matters.
- A locally best token is not always part of the best complete sentence.
- It may become repetitive.

Main code:

```python
next_token_id = log_probabilities.argmax(dim=-1, keepdim=True)
```

### 2. Beam search

Beam search keeps several candidate sequences instead of only one. With beam
width 3, each step retains the three best paths. A path that is second best now
may become the best complete sentence later.

- Common for translation and summarization.
- More global search than greedy decoding.
- Slower and more memory-intensive because several paths run through the model.
- Still deterministic in this implementation.

The lab adds log probabilities rather than multiplying probabilities. Addition
is numerically safer because a long product of small numbers can underflow.

```text
sequence log probability
= log P(new token | tokens already generated)
```

`length_penalty` optionally prevents shorter sequences from receiving an unfair
advantage. `0.0` means no normalization.

### 3. Temperature sampling

Temperature divides the logits before softmax:

```text
scaled logits = original logits / temperature
```

- `temperature < 1`: sharper distribution, safer and more predictable.
- `temperature = 1`: original distribution.
- `temperature > 1`: flatter distribution, more variety and more risk.

Temperature does not itself remove tokens. This implementation randomly samples
from the complete vocabulary after scaling it.

### 4. Top-k sampling

Top-k keeps a fixed number of the highest-scoring tokens. `k=3` keeps `cat`,
`dog`, and `bird` in the example, blocks `fish`, then samples from the remaining
three.

- Easy to understand and control.
- The candidate set always has the same size.
- A fixed `k` may be too large when the model is confident and too small when
  several choices are reasonable.

In code, blocked tokens receive negative infinity. Softmax converts their
probability to zero:

```python
filtered_logits = torch.full_like(logits, -torch.inf)
filtered_logits = filtered_logits.scatter(-1, top_indices, top_values)
```

### 5. Top-p (nucleus) sampling

Top-p sorts tokens from most to least probable and keeps the smallest group
whose cumulative probability reaches `p`.

For `p=0.90`:

```text
cat   0.52  cumulative 0.52
dog   0.28  cumulative 0.80
bird  0.12  cumulative 0.92  ← stop here
```

The size of the candidate set adapts to the model's confidence. That makes
top-p a common choice for conversational and creative generation.

## What each file does

### `decoding.py`

This is the shared foundation.

- `DecodingConfig` holds and validates all settings.
- `GenerationResult` gives every strategy the same return type.
- `extract_next_token_logits()` accepts `[batch, sequence, vocabulary]` model
  output and selects only the final position.
- `greedy_decode()` performs the simplest autoregressive loop.

`generated_ids` starts as the prompt. Each loop calls the model, chooses one
token, appends it with `torch.cat`, and stops early if it chooses `<eos>`.

### `sampling.py`

This contains reusable transformations for random generation.

- `apply_temperature()` divides logits by temperature.
- `filter_top_k()` keeps exactly the best `k` candidates.
- `filter_top_p()` keeps enough candidates to reach probability mass `p`.
- `sample_next_token()` calls `torch.multinomial()` to make the random choice.
- the three `*_decode()` functions reuse one private generation loop.

Passing `seed=42` creates repeatable samples. Different seeds can produce
different valid continuations.

### `beam_search.py`

`Beam` stores one candidate's token IDs, cumulative log probability, and finish
state. At every step, `beam_search_decode()` expands unfinished candidates,
sorts them by score, and retains only `beam_width` candidates.

This implementation supports one prompt at a time to keep the learning code
clear. Production libraries batch and cache beam computations for speed.

### `generation_inference.py`

`generate()` is the shared public interface. It reads `config.strategy` and
dispatches to the correct implementation. It also:

- enables PyTorch inference mode;
- temporarily switches a PyTorch model to evaluation mode;
- restores the model's original mode;
- measures elapsed generation time.

The project already has `backend/ml/nlp/inference.py` from the earlier text
classification lab. This lab deliberately uses `generation_inference.py` so
that existing classifier code is not overwritten.

### `compare_decoding.py`

`ToyLanguageModel` is a fixed table of next-token scores. It is not trained and
does not download anything. This isolates the decoding behavior and lets all
five strategies use the exact same prompt and model.

The script measures:

- whether generation reached `<eos>`;
- unique-token ratio as a small diversity signal;
- repeated-bigram ratio as a repetition signal;
- average selected-token log probability under that strategy;
- inference time.

Text quality is subjective, so read the outputs yourself instead of inventing a
numeric "quality" score.

Hand-checked examples from the transition table are:

```text
Greedy: the cat sat on mat.
Beam:   the cat sat on mat.
```

A valid sampled path could instead be `the dog ran quickly.` or
`the bird sang.`. Those are illustrations, not measured results from your run;
the seed and configuration determine the actual sampled rows.

### `test_decoding_pipeline.py`

The tests check the behavior that matters:

- greedy chooses the largest score and stops at `<eos>`;
- low temperature sharpens the distribution;
- top-k keeps exactly `k` tokens;
- top-p keeps the smallest set that reaches `p`;
- a seed makes sampling repeatable;
- beam width 1 matches greedy;
- beam search can beat a locally greedy path;
- the shared API dispatches every strategy;
- invalid settings fail clearly.

## Run the lab

From the repository root with the virtual environment active:

```powershell
python -m pytest backend/tests/test_decoding_pipeline.py -q
python -m backend.ml.nlp.compare_decoding
```

Expected test count for this file: `13 passed`. Record the actual result from
your computer; do not claim it passed until you run it.

No Matplotlib dependency is needed for this lab.

## Use the shared inference API

```python
import torch

from backend.ml.nlp.decoding import DecodingConfig
from backend.ml.nlp.generation_inference import generate
from backend.ml.nlp.transformer_decoder import CausalLanguageModel

model = CausalLanguageModel(
    vocabulary_size=100,
    padding_id=0,
)

# In a real project, load trained weights before generation.
prompt_token_ids = torch.tensor([[1, 12, 27]], dtype=torch.long)

config = DecodingConfig(
    strategy="top_p",
    max_new_tokens=20,
    eos_token_id=2,
    top_p=0.90,
    temperature=0.8,
    seed=42,
)

result = generate(model, prompt_token_ids, config)
print(result.token_ids)
print(result.generated_token_ids)
```

Day 11's `CausalLanguageModel` returns logits with shape
`[batch, sequence, vocabulary]`, so it can be passed directly to `generate()`.
An untrained model emits meaningless text; train it or load a checkpoint before
evaluating output quality.

## Reuse the Day 12 encoder-decoder model

An encoder-decoder model needs source tokens as well as the generated target.
Encode the source once, then close over the reusable encoder output:

```python
encoder_output = seq2seq_model.encode(source_token_ids)

def target_logits(target_token_ids):
    logits, _, _ = seq2seq_model.decode(
        target_token_ids,
        encoder_output,
    )
    return logits

result = generate(target_logits, beginning_of_sentence_ids, config)
```

This is why the API accepts any logits-providing callable, not only one specific
model class.

## Strategy selection for the quiz

| Use case | Good starting strategy | Why |
|---|---|---|
| Machine translation | Beam search | Compares complete high-probability translations. |
| Creative writing | Top-p with temperature around 0.8–1.2 | Allows variety while removing an unlikely tail. |
| Code generation | Greedy or low-temperature top-p | Favors predictable syntax and reduces random errors. |
| Chatbots | Top-p with moderate temperature | Balances coherent answers and natural variation. |

These are starting points, not universal laws. Tune on task-specific evaluation
data and measure correctness, latency, safety, and repetition.

## Lab / quiz answers

### Explain greedy decoding in your own words

Greedy decoding asks the model for the next-token probabilities and always picks
the largest one. It repeats that decision until reaching the end token or the
generation limit.

### Compare beam search and greedy search

Greedy keeps one sequence and makes the best immediate choice. Beam search keeps
several sequences, so it can recover when the best immediate token leads to a
worse complete answer. Greedy is faster; beam search uses more computation and
memory.

### When would you increase temperature?

Increase temperature when outputs are too predictable, repetitive, or similar
and the task can tolerate more risk—for example, brainstorming or creative
writing. Do not increase it merely to improve factual accuracy.

### Compare top-k and top-p sampling

Top-k retains a fixed number of candidates. Top-p retains a variable number
whose combined probability reaches a chosen threshold. Top-p adapts when the
model is very confident or uncertain; top-k is simpler and more fixed.

## Git workflow

First make sure the Day 12 work you want to build on is committed. Then inspect
the current branch and create today's branch—the `-c` is important because it
creates a new branch:

```powershell
git status -sb
git switch -c feature/llm-decoding-strategies
```

If Git says the branch already exists, use:

```powershell
git switch feature/llm-decoding-strategies
```

Stage only today's files so earlier artifacts or checkpoints are not included:

```powershell
git add backend/ml/nlp/decoding.py
git add backend/ml/nlp/beam_search.py
git add backend/ml/nlp/sampling.py
git add backend/ml/nlp/generation_inference.py
git add backend/ml/nlp/compare_decoding.py
git add backend/ml/nlp/README_DECODING.md
git add backend/tests/test_decoding_pipeline.py
git status --short
git diff --cached --stat
```

After both commands in **Run the lab** succeed:

```powershell
git commit -m "feat(ml): implement configurable LLM decoding strategies"
git push -u origin HEAD
```

Do not commit model checkpoints, generated artifacts, `.env` files, secrets, or
the virtual environment.

## Definition of done

- [ ] all five methods run through the shared interface
- [ ] all decoding tests pass locally
- [ ] the comparison script prints five measured rows
- [ ] example output is reviewed for diversity and repetition
- [ ] actual results are added to this README
- [ ] only intended source and documentation files are staged
