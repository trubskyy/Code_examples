# Tiny LLM: train a GPT on a few books, then watch it think

A complete, small language model in about 400 lines of PyTorch. It uses the same
transformer architecture as GPT, scaled down so it trains on a laptop CPU in minutes.
It reads characters, not words, so it needs no tokenizer library.

| File | What it does |
|---|---|
| `download_books.py` | Fetches public-domain books from Project Gutenberg into `books/`. |
| `model.py` | The GPT: embeddings, causal self-attention, MLP blocks, sampling. |
| `train.py` | Trains on any `.txt` files and saves `checkpoint.pt`. |
| `chat.py` | Continues your prompt. `--visualize` / `--animate` / `--gif` draw the network. |
| `visualize.py` | Draws the forward pass for each predicted character. |

## Run it

```bash
pip install -r requirements.txt

python download_books.py                 # Alice in Wonderland, Sherlock Holmes, Pride and Prejudice
python train.py books/*.txt              # "small" preset, ~0.8M params, ~10-20 min on a CPU
python chat.py --prompt "Alice was"      # continue a prompt
python chat.py                           # interactive mode
```

Use any plain-text books you like: `python train.py my_book.txt other_book.txt`.

Training presets (`--preset`):

| Preset | Layers × heads × width | Context | Params | Time on a laptop CPU |
|---|---|---|---|---|
| `tiny` | 2 × 2 × 64 | 64 chars | ~0.1M | ~2 min (words appear, mostly nonsense) |
| `small` (default) | 4 × 4 × 128 | 128 chars | ~0.8M | ~10-20 min (real words, plausible sentences) |
| `large` | 6 × 6 × 384 | 256 chars | ~10.7M | hours on CPU, ~10 min on a GPU |

A CUDA GPU or Apple Silicon (MPS) is used automatically when present. Ctrl+C stops
training early; the best checkpoint so far is already saved.

## See the network work

```bash
python chat.py --prompt "Alice was" --visualize        # picture of the first prediction -> network.png
python chat.py --prompt "Alice was" --animate --length 30   # live window, redrawn for every character
python chat.py --prompt "Alice was" --gif run.gif --length 30  # save every step as an animated GIF
```

On a machine without a screen the picture is saved instead of shown.

Each picture shows one forward pass, the computation that produces one character:

- **Network flow** (top left). Columns are the stages: input characters, embedding,
  each transformer block, output. Rows are the last 14 characters of the prompt. A
  node's colour is how strongly that character's vector is activated at that layer.
  Red lines are attention into the last character: in each block, which earlier
  characters it reads information from. Thicker lines mean more attention. Grey
  lines are the same for the other characters.
- **Output** (top right). The probability the model gives each possible next
  character. The red bar is the one it sampled.
- **Attention maps** (bottom). Each block's full attention matrix, averaged over
  heads. Row = the character reading; column = the character being read. Everything
  above the diagonal is zero because a character cannot see the future.
- **MLP neurons**. The feed-forward neurons in each block for the last character.
  Red fired positive, blue negative.
- **Logit lens**. The model's top three guesses if you cut it off after each layer.
  Read top to bottom to see the prediction take shape as information flows up.

## How it works

1. **Tokenize.** Every distinct character in the books gets an id (about 80-100 of them).
2. **Embed.** Each id becomes a learned vector of 128 numbers, plus a vector for its position.
3. **Transformer blocks.** Each block does two things:
   - *Self-attention*: each character compares itself (query) with every earlier character
     (keys) and takes a weighted mix of their information (values). This is how
     the model knows that `q` is followed by `u`, or that a quote mark is still open.
   - *MLP*: a small two-layer network applied to each position, where the model
     stores patterns such as spellings and common words.
4. **Predict.** The final vector is compared with every character's embedding to
   give a score per character. Softmax turns the scores into probabilities.
5. **Train.** Show the model random 128-character windows from the books, ask it to
   predict each next character, and nudge every weight to make the right answer
   more likely (cross-entropy loss and AdamW).
6. **Generate.** Predict one character, sample it, append it, repeat. `--temperature`
   controls randomness and `--top_k` limits sampling to the k most likely characters.

A real LLM differs mainly in scale: subword tokens instead of characters, about
10,000× more parameters and data, and extra fine-tuning so that it follows
instructions. This model only continues text in the style of the books it read.
