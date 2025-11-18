
# Fine-grained Crosslinguistic Generalisation Analysis

The `notebooks/lm_pcfg.py` file contains the core training and intervention logic of this project. It is used for training Language Model (like GPT-2) on synthetic data derived from a Probabilistic Context-Free Grammar (PCFG). <br>
This script performs **online, causal interventions** on the data, allowing to modify the grammatical structure of sentences on-the-fly (e.g., swapping Subject and Verb) to test whether a model learns abstract structural rules or merely memorizes surface statistics.

**Data** provided by the authors of ["Transparency at the Source"](https://aclanthology.org/2023.findings-emnlp.288/) could be downloaded from here:
https://drive.google.com/drive/folders/1Sw2U8hcaqH7Ynqzw5Agk4WAcXitUaVuF?usp=sharing


## Usage (see [description](#Descrpition) section for more details)

### Running the Baseline (Exp 0)

Train a model with a 50/50 split between two "dialects" that differ only by suffix.

```bash
CUDA_VISIBLE_DEVICES=0 python -m lm_pcfg \
    --experiment 0 \
    --p 0.5 \
    --my_filter "baseline_test"
```

### Running the Syntactic Swap (Exp 1)

Train a model where the `_en2` language forces the Verb Phrase (VP) to come before the Noun Phrase (NP):

```bash
CUDA_VISIBLE_DEVICES=0 python -m lm_pcfg \
    --experiment 1 \
    --p 0.8 \
    --rule "S -> NP VP" \
    --my_filter "foobar swap_S_nodes"

```

Or train a model with many swapped rules:

```bash
CUDA_VISIBLE_DEVICES=0 python -m lm_pcfg --my_filter "running experiment 1 with many swapped rules" --p 0.9 --experiment 1 --rule "S -> ATS DOT" --rule "PP -> IN NP" --rule "S -> NP VP" --rule "ATS -> NP VP" --rule "NP -> DT NN" --rule "ATS -> ATS VP" --rule "NP -> ATNP NN" --rule "VP -> ATVP PP" --rule "VP -> ATVP VP" --rule "NP -> NP PP" --rule "ATS -> ATS NP" --rule "S -> ATS APOSTROPHE" --rule "ATS -> ATS DOT" --rule "ATNP -> DT JJ" --rule "NP -> PRPDOLLAR NN" --rule "VP -> TO VP" --rule "VP -> VBD NP" --rule "SBAR -> IN S" --rule "VP -> VB NP" --rule "ATVP -> VBD NP" --rule "VP -> MD VP" --rule "ATS -> ATS S" --rule "VP -> VBD VP" --rule "ATS -> TICK NP" --rule "PP -> TO NP" --rule "VP -> ATVP S" --rule "ATVP -> VP CC" --rule "ATS -> ATS COMMA" --rule "VP -> VBD PP" --rule "NP -> ATNP NNS" --rule "VP -> ATVP SBAR" --rule "ATVP -> ATVP COMMA"

```

## Environment Setup

```bash
# 1. Check installed GPU driver and CUDA version compatibility
nvidia-smi

# 2. Set the system environment paths for CUDA 11.8, if the CUDA toolkit is not in the default system path
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 3. Create a new environment using Python 3.11
conda create --name venvcu118 python=3.11.10

# 4. Activate the new environment
conda activate venvcu118

# 5. Install PyTorch built for CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118

# 6. Install core libraries
pip install -r requirements.txt
```


## Required File Structure $${\color{orange}{\textsf{(! Might be not fully up to date)}}}$$


To run successfully, the script expects a specific directory structure and file types in `lm_training/corpora`:

### 1\. Raw Text Files (`.txt`)

Used for both Experiment 0 and 1. Contains the raw sentences.

  * `train.txt`, `dev.txt`, `test.txt`, `eval.txt`

### 2\. Parse Tree Files (`.nltk`)

**Crucial for Experiment 1.** These contain the parse trees corresponding *line-by-line* to the text files.

  * `train.nltk`, `dev.nltk`, `test.nltk`

### 3\. Vocabulary (`.json`)

Defines the base tokens. The script automatically augments this with `_en1` and `_en2` suffixes during initialization.

  * Path: `src/lm_training/vocab/added_tokens.json`

### 4\. PCFG Dictionary (`.pickle`)

Used for the final evaluation to compare Model probabilities vs. True PCFG probabilities.

  * File: `earleyx_pcfg_dict.pickle`

-----

# Description

## The Project Concept

The core goal is **Causal Disentanglement** in the context of work [**"Transparency at the Source"**](https://github.com/clclab/pcfg-lm).

1.  **The "White Box" (PCFG):** Use a mathematical grammar generator where the exact probability of every sentence is known (Ground Truth).
2.  **The "Black Box" (Neural Model):** Train an LLM on this data and compare its internal probability distribution against the Ground Truth.

By intervening on the grammar, the model is forced to distinguish between two languages that use the same words but different rules. If the model successfully learns both, it demonstrates **structural awareness** rather than just lexical memorization.


## Experimental Modes

The script supports two distinct experiments controlled via the `--experiment` flag.

### Experiment 0: The Lexical Baseline (Control)

**"Can the model handle two vocabularies that mean the same thing?"**

This is a memory stress test. It creates two "languages" that are structurally identical but use different vocabulary markers.

  * **Mechanism:**
      * **Language A (`_en1`):** Standard Grammar. (e.g., *"The\_en1 cat\_en1 sat\_en1"*)
      * **Language B (`_en2`):** Standard Grammar. (e.g., *"The\_en2 cat\_en2 sat\_en2"*)
  * **Goal:** Proves the model has the capacity to learn two sets of tokens without confusion.

### Experiment 1: The Syntactic Probe (Intervention)

**"Can the model handle two grammars that contradict each other?"**

This is the core reasoning test. It creates two languages that are lexically distinct **AND** structurally opposite.

  * **Mechanism:**
      * **Language A (`_en1`):** Standard Grammar. (e.g., *"The\_en1 cat\_en1 sat\_en1"*)
      * **Language B (`_en2`):** **Modified Grammar.** The script parses the sentence structure and physically swaps specific nodes (e.g., `S -> NP VP` becomes `S -> VP NP`). (e.g., *"Sat\_en2 the\_en2 cat\_en2"*)
  * **Goal:** Forces the model to learn an abstract rule: *"If I see `_en1`, use Rule A. If I see `_en2`, use Rule B."*

-----

## How It Works: "Online" Processing

For **Experiment 1**, this script does **not** rely on pre-generated "swapped" text files (which would require generating additional data for every hypothesis). Instead, it processes rules **online**:

1.  **Load:** Reads a raw sentence and its corresponding **NLTK Parse Tree**.
2.  **Target:** Uses Regex to identify specific grammar productions defined in the CLI (e.g., `S -> NP VP`).
3.  **Swap:** Physically rearranges the branches of the tree in memory.
4.  **Flatten:** Converts the modified tree back into a text string.
5.  **Train:** Feeds the modified sentence to the model immediately.

This allows for flexible hypothesis testing. It is possible to change the laws of grammar directly by changing a command-line argument.

