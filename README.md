# LLM Design Config

## Usage

Requires `openai>=1.3`.

Run the baseline agent on specific tasks:

```bash
python main.py baseline --tasks 1 2 3 --outdir runs/raw
```

Run the buffered agent with a buffer size of four:

```bash
python main.py buffered --tasks 1 2 3 4 5 --buffer 4 --model gpt-4o
```

Complete five runs of task 1 with the buffered agent with a buffer size of four:

```bash
python main.py buffered --tasks 1 --buffer 4 --model gpt-4o --runs 5
```

Quick start:

```bash
python main.py buffered --tasks 1 --buffer 3 --model gpt-3.5-turbo --seed 123
# add --dry-run for offline
```

## Running Experiments

### Human-median length constraints

The buffered agent enforces per-scene limits derived from our human dataset:

| Scene | Median steps | Median words / step | Word-cap |
|-------|--------------|---------------------|----------|
| 1     | 15           | 14                  | 268      |
| …     | …            | …                   | …        |

Example command:

```bash
python main.py buffered --tasks 4 --buffer 3 --model gpt-4o
```

## Outputs

- runs/taskX_bufferY.txt now contains all generated thoughts; the model,
  however, only sees the last Y thoughts during generation.
