import io
import json
import os
import pickle
import random
import re
from typing import *
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForMaskedLM, BertTokenizer,
                          DataCollatorForLanguageModeling, PreTrainedTokenizer,
                          PreTrainedTokenizerFast, Trainer, TrainingArguments)

import wandb
import torch

import argparse

def preprocess_function(examples, p):
    processed = []
    for text in examples["text"]:
        suffix = "_en1" if random.random() < p else "_en2"
        tokens = text.split()
        processed_tokens = []
        for token in tokens:
            if re.search(r'[A-Za-z0-9]', token):
                processed_tokens.append(token + suffix)
            else:
                processed_tokens.append(token)
        processed.append(" ".join(processed_tokens))
    return {"text": processed}


def assign_language_suffix(example, index, total_length, p):
    # Calculate the threshold index for suffix _en1
    threshold = int(total_length * p)
    suffix = "_en1" if index < threshold else "_en2"
    tokens = example["text"].split()
    processed_tokens = [
        token + suffix if re.search(r'[A-Za-z0-9]', token) else token
        for token in tokens
    ]
    return {"text": " ".join(processed_tokens)}


def augment_vocab_with_suffixes(vocab, suffixes=('_en1', '_en2')):
    new_vocab = {}
    for token in list(vocab.keys()):
        if token.startswith('<') and token.endswith('>') and "<apostrophe>" not in token and "<cross>" not in token:
            new_vocab[token] = len(new_vocab)
            continue
        for suffix in suffixes:
            new_token = token + suffix
            if new_token not in new_vocab:
                new_vocab[new_token] = len(new_vocab)
    return new_vocab


# def create_tokenizer(
#     corpus: str, 
#     unk_token: str = '<unk>', 
#     pad_token: str = '<pad>', 
#     mask_token: str = '<mask>',
#     min_freq: int = 1,
# ):
#     vocab = create_vocab(corpus, unk_token=unk_token, pad_token=pad_token, mask_token=mask_token, min_freq=min_freq)
            
#     tokenizer = create_tf_tokenizer_from_vocab(vocab, unk_token=unk_token, pad_token=pad_token, mask_token=mask_token)
    
#     return tokenizer


# def create_vocab(
#     corpus: str, 
#     unk_token: str = '<unk>', 
#     pad_token: str = '<pad>', 
#     mask_token: str = '<mask>',
#     min_freq: int = 1,
# ):
#     with open(corpus) as f:
#         train = f.read().split('\n')

#     token_freqs = Counter()

#     for sen in train:
#         for w in sen.split():
#             token_freqs[w] += 1
            
#     vocab = {unk_token: 0, pad_token: 1, mask_token: 2}
    
#     for w, freq in token_freqs.most_common():
#         if freq >= min_freq:
#             vocab[w] = len(vocab)

#     return vocab


class CustomTokenizer(PreTrainedTokenizer):
    def __len__(self):
        return len(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    def save_vocabulary(self, *args, **kwargs):
        return BertTokenizer.save_vocabulary(self, *args, **kwargs)
    
    def _tokenize(self, sen: str):
        return sen.split(" ")
    
    def _convert_token_to_id(self, w: str):
        return self.vocab.get(w, self.vocab[self.unk_token])


def create_tf_tokenizer_from_vocab(
    vocab, 
    unk_token: str = '<unk>', 
    pad_token: str = '<pad>',
    mask_token: str | None = None,
    bos_token: str | None = None,
    eos_token: str | None = None,
):
    tokenizer = CustomTokenizer()

    tokenizer.added_tokens_encoder = vocab
    tokenizer.added_tokens_decoder = {idx: w for w, idx in vocab.items()}
    tokenizer.vocab = tokenizer.added_tokens_encoder
    tokenizer.ids_to_tokens = tokenizer.added_tokens_decoder
    
    tokenizer.unk_token = unk_token
    tokenizer.pad_token = pad_token
    # tokenizer.mask_token = mask_token
    if mask_token is not None:
        tokenizer.mask_token = mask_token

    if bos_token is not None:
        tokenizer.bos_token = bos_token
    if eos_token is not None:
        tokenizer.eos_token = eos_token
    return tokenizer



def tokenize_wrapper(tokenizer):
    def tokenize(batch):
        input_ids = []
        for text in batch["text"]:
            encoded = tokenizer(text)["input_ids"]
            tokens = [tokenizer.bos_token_id] + encoded + [tokenizer.eos_token_id]
            input_ids.append(tokens)
        return {"input_ids": input_ids}
    return tokenize


def load_data_experiment_0(  # only adds _en1 or _en2 to the evaluated data from the original corpora
    tokenizer: PreTrainedTokenizerFast,
    corpora_original_dir: str,

    p: float = 1.0,
    add_language_pseudo_suffixes: bool = False,

    train_size: Optional[int] = None,
    dev_size: Optional[int] = None,
    test_size: Optional[int] = None,

    train_file = 'train.txt',
    dev_file = 'dev.txt',
    test_file = 'test.txt',
    eval_file = 'eval.txt',
) -> DatasetDict:

    raw_train = load_dataset("text", data_files=os.path.join(corpora_original_dir, train_file))["train"]
    raw_dev = load_dataset("text", data_files=os.path.join(corpora_original_dir, dev_file))["train"]
    raw_test = load_dataset("text", data_files=os.path.join(corpora_original_dir, test_file))["train"]
    if eval_file is not None:
        raw_eval = load_dataset("text", data_files=os.path.join(corpora_original_dir, eval_file))["train"]
    else:
        raw_eval = None

    if train_size is not None:
        raw_train = raw_train.shuffle().select(range(train_size))
    if dev_size is not None:
        raw_dev = raw_dev.shuffle().select(range(dev_size))
    if test_size is not None:
        raw_test = raw_test.shuffle().select(range(test_size))

    dataset_dict = {
        "train": raw_train,
        "valid": raw_dev,
        "test": raw_test,
    }

    if raw_eval is not None:
        dataset_dict["eval"] = raw_eval

    raw_datasets = DatasetDict(dataset_dict)

    if add_language_pseudo_suffixes:
        for split in raw_datasets.keys():
            # shuffle again
            raw_datasets[split] = raw_datasets[split].map(
                lambda ex, idx: {"original_index": idx},
                with_indices=True,
                batched=False,
            )
            raw_datasets[split] = raw_datasets[split].shuffle()

            total_length = len(raw_datasets[split])
            raw_datasets[split] = raw_datasets[split].map(
                lambda ex, idx: assign_language_suffix(ex, idx, total_length, p),
                with_indices=True,
                batched=False,
            )

            # restore order
            raw_datasets[split] = raw_datasets[split].sort("original_index")

    # # second shuffle to randomize the effect of assigning _en1 and _en2...
    # # it's needed for all datasets regardles if they have pseudo_suffixes or not...
    # # e.g. to compare cross_entropy scores (datasets_lm["eval"] with _en1 & _en2) vs (datasets_pcfg["eval"])
    # for split in raw_datasets.keys():
    #     raw_datasets[split] = raw_datasets[split].shuffle(seed=42)

    tokenized_datasets = raw_datasets.map(
        tokenize_wrapper(tokenizer),
        batched=True,
    )

    return tokenized_datasets



def load_data_experiment_1(
    tokenizer: PreTrainedTokenizerFast,
    corpora_original: str,
    corpora_swapped: str,

    p: float,

    add_language_pseudo_suffixes: bool = False,  # not used actually

    train_size: Optional[int] = None,
    dev_size: Optional[int] = None,
    test_size: Optional[int] = None,

) -> DatasetDict:

    train_file = "train.txt"
    dev_file = "dev.txt"
    test_file = "test.txt"
    eval_file = "eval.txt"

    raw_train_orig = load_dataset("text", data_files=os.path.join(corpora_original, train_file))["train"]
    raw_train_swap = load_dataset("text", data_files=os.path.join(corpora_swapped, train_file))["train"]

    raw_dev_orig = load_dataset("text", data_files=os.path.join(corpora_original, dev_file))["train"]
    raw_dev_swap = load_dataset("text", data_files=os.path.join(corpora_swapped, dev_file))["train"]

    raw_test_orig = load_dataset("text", data_files=os.path.join(corpora_original, test_file))["train"]
    raw_test_swap = load_dataset("text", data_files=os.path.join(corpora_swapped, test_file))["train"]

    try:
        raw_eval_orig = load_dataset("text", data_files=os.path.join(corpora_original, eval_file))["train"]
        raw_eval_swap = load_dataset("text", data_files=os.path.join(corpora_swapped, eval_file))["train"]
        eval_exists = True
    except Exception:
        eval_exists = False

    if train_size is not None:
        raw_train_orig = raw_train_orig.shuffle().select(range(train_size))
        raw_train_swap = raw_train_swap.shuffle().select(range(train_size))
    if dev_size is not None:
        raw_dev_orig = raw_dev_orig.shuffle().select(range(dev_size))
        raw_dev_swap = raw_dev_swap.shuffle().select(range(dev_size))
    if test_size is not None:
        raw_test_orig = raw_test_orig.shuffle().select(range(test_size))
        raw_test_swap = raw_test_swap.shuffle().select(range(test_size))
    if eval_exists and test_size is not None:
        raw_eval_orig = raw_eval_orig.shuffle().select(range(test_size))
        raw_eval_swap = raw_eval_swap.shuffle().select(range(test_size))

    def combine_pairs(orig_texts, swap_texts, p):
        combined = []
        for i in range(min(len(orig_texts), len(swap_texts))):
            if random.random() < p:
                sentence = orig_texts[i]
                suffix = "_en1"
            else:
                sentence = swap_texts[i]
                suffix = "_en2"
            tokens = sentence.split()
            processed_tokens = []
            for token in tokens:
                if re.search(r"[A-Za-z0-9]", token):
                    processed_tokens.append(token + suffix)
                else:
                    processed_tokens.append(token)
            combined.append(" ".join(processed_tokens))
        return combined

    combined_train = combine_pairs(raw_train_orig["text"], raw_train_swap["text"], p)
    combined_dev = combine_pairs(raw_dev_orig["text"], raw_dev_swap["text"], p)
    combined_test = combine_pairs(raw_test_orig["text"], raw_test_swap["text"], p)
    if eval_exists:
        combined_eval = combine_pairs(raw_eval_orig["text"], raw_eval_swap["text"], p)
    else:
        combined_eval = None

    dataset_dict = {
        "train": Dataset.from_dict({"text": combined_train}),
        "valid": Dataset.from_dict({"text": combined_dev}),
        "test": Dataset.from_dict({"text": combined_test}),
    }
    if combined_eval is not None:
        dataset_dict["eval"] = Dataset.from_dict({"text": combined_eval})
    raw_datasets = DatasetDict(dataset_dict)

    tokenized_datasets = raw_datasets.map(
        tokenize_wrapper(tokenizer),
        batched=True,
    )

    return tokenized_datasets

def initialize_model(
    tokenizer: PreTrainedTokenizer, model_type: str, is_mlm: bool = True, **config
) -> Union[AutoModelForMaskedLM, AutoModelForCausalLM]:
    config = AutoConfig.from_pretrained(
        model_type,
        vocab_size=len(tokenizer.added_tokens_encoder),
        **config,
    )
    
    auto_model = AutoModelForMaskedLM if is_mlm else AutoModelForCausalLM

    model = auto_model.from_config(config)

    return model


def initialize_trainer(
    model: AutoModelForMaskedLM,
    tokenizer: PreTrainedTokenizerFast,
    data_collator: DataCollatorForLanguageModeling,
    datasets: DatasetDict,
    model_init = None,
    **config,
):
    args = TrainingArguments(**config)

    # if model_init is not None:
    #     model = None

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        model_init=model_init
    )

    return trainer


def ray_hp_space(trial):
    return {
        # "learning_rate": tune.loguniform(1e-5, 1e-3),
        # "per_device_train_batch_size": tune.choice([64, 128]),
        "num_hidden_layers": tune.grid_search([6, 8, 10]),
        "hidden_size": tune.grid_search([128, 256, 512]),
        "num_attention_heads": tune.grid_search([8, 16]),
    }


def model_init(trial):
    return initialize_model(
        tokenizer, 
        'phueb/BabyBERTa-1', 
        num_hidden_layers=trial['num_hidden_layers'], 
        intermediate_size=trial['hidden_size'],
        hidden_size=trial['hidden_size'],
        num_attention_heads=trial['num_attention_heads'],
    )



####### EVAL CODE:


def extract_pcfg_and_model_probs(corpus_lm, corpus_pcfg, pcfg_dict, model):
    pcfg_probs = []
    lm_probs = []

    for sentence, sentence_probs__pcfg in pcfg_dict.items():
        i = corpus_pcfg['text'].index(sentence)
    
        input_ids_list__lm = corpus_lm['input_ids'][i]
        sentence__lm = corpus_lm['text'][i]
    
        input_ids_list__lm = [token for token in input_ids_list__lm if token is not None]
    
        input_ids__lm = torch.tensor(input_ids_list__lm).unsqueeze(0)
    
        with torch.no_grad():
            probs = model(input_ids__lm[:, :-1]).logits.log_softmax(-1)[0]
    
            sentence_probs__lm = []
            for idx, prob_row in enumerate(probs, start=1):
                token_id = input_ids__lm[0, idx].item()
                
                token_prob = prob_row[token_id].item()
                sentence_probs__lm.append(token_prob)
    
    
        lm_probs.extend(sentence_probs__lm)
        pcfg_probs.extend([-x for x in sentence_probs__pcfg[1:]])  # start=1, use: -prob

    return np.array(lm_probs), np.array(pcfg_probs)


def plot_probs(
    lm_probs, 
    pcfg_probs, 
    model_name: str, 
    save_as: Optional[str] = None, 
    cmap = 'OrRd', 
    ymodel='PCFG', 
    ylim=(-20,0.1), 
    xlim=(-20,0.1), 
    do_scatter=False,
    mincnt=3,
    **plot_args,
):
    fig, ax = plt.subplots(figsize=(4,4))

    # sns.regplot(lm_probs, pcfg_probs, scatter_kws={'alpha':0.05, 's':2}, color='orange', line_kws={"color": "0.5", 'ls': '--', 'lw': 1})
    if do_scatter:
        ax.scatter(lm_probs, pcfg_probs, cmap=cmap, **plot_args)
    else:
        ax.hexbin(
            lm_probs, 
            pcfg_probs, 
            gridsize=40, 
            mincnt=mincnt,
            cmap=cmap,
            bins='log',
            **plot_args,
        )

    plt.xlabel(r'$log$ P$_{LM}(w)$', fontsize=14)
    plt.ylabel(r'$log$ P$_{PCFG}(w)$'.replace('PCFG', ymodel), fontsize=14)

    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)

    # Ticks
    plt.xticks(range(xlim[0],1,5), range(xlim[0],1,5))
    plt.yticks(range(xlim[0]+5,1,5), range(xlim[0]+5,1,5))

    ax.tick_params(
        axis='x', 
        which='major', 
        pad=-3,
        labelcolor="0.5",
    )
    ax.tick_params(
        axis='y', 
        which='major', 
        pad=-3,
        labelcolor="0.5",
    )

    # Grid
    ax.grid(color='0.7', linestyle='--', linewidth=0.5)

    # Spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.setp(ax.spines.values(), color="black", lw=.5)

    # Title
    corr, p = spearmanr(lm_probs, pcfg_probs)
    r2 = r2_score(lm_probs, pcfg_probs) * 100
    
    # plt.plot(np.linspace(xlim[0],0,10), pearsonr(lm_probs, pcfg_probs)[0] * np.linspace(xlim[0],0,10), c='0.2', linestyle='--', lw=2, alpha=0.5)
    plt.plot(np.linspace(xlim[0],0,10), np.linspace(xlim[0],0,10), c='0.2', linestyle='--', lw=2, alpha=0.5)
    plt.title(f"{model_name}\n" + f"$\\rho$: {corr*100:.1f}, $R^2$: {r2:.1f}", fontweight=600, color="royalblue", fontsize=15)
        
    if save_as is not None:
        plt.savefig(f"{save_as}.jpeg", bbox_inches="tight")
    
    plt.show()

    return fig


def log_plot(run, fig, step, label=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)

    # label = fig.axes[0].get_title()
    if label is None:
        label = "mydefaultlabel"

    run.log({label: wandb.Image(img)}, step=step)


def log_plot_as_artifact(run, fig, label=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)

    if label is None:
        label = fig.axes[0].get_title() or "untitled_plot"
    file_name = f"{label}.png"
    
    img.save(file_name)
    
    artifact = wandb.Artifact(label, type="image")
    artifact.add_file(file_name)
    
    run.log_artifact(artifact)



### MAIN LOOP

load_data__for_evaluation = load_data_experiment_0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--experiment", type=int, choices=range(3), required=True)
    args = parser.parse_args()

    print(f"p = {args.p}")


    with open('lm_training/vocab/added_tokens.json') as f:
        vocab = json.load(f)
    if "<BOS>" not in vocab:
        vocab["<BOS>"] = len(vocab)
    if "<EOS>" not in vocab:
        vocab["<EOS>"] = len(vocab)
    vocab = augment_vocab_with_suffixes(vocab, suffixes=('_en1', '_en2'))
    tokenizer = create_tf_tokenizer_from_vocab(vocab, unk_token='<unk>', pad_token='<pad>', mask_token=None, bos_token='<BOS>', eos_token='<EOS>')

    if args.experiment == 0:
        datasets = load_data_experiment_0(tokenizer, 'lm_training/corpora_11mil', p=args.p, add_language_pseudo_suffixes=True, eval_file=None)
    elif args.experiment == 1:
        datasets = load_data_experiment_1(
            tokenizer, corpora_original='lm_training/corpora', corpora_swapped="lm_training/corpora__swapped", p=args.p, add_language_pseudo_suffixes=True,
        )

    is_mlm = False

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=is_mlm)

    model = initialize_model(
        tokenizer, 
        'distilgpt2', #'microsoft/deberta-v3-base',  # 'phueb/BabyBERTa-1', 
        num_hidden_layers=8, 
        intermediate_size=256,
        hidden_size=256,
        num_attention_heads=8,
        is_mlm=is_mlm,
    )

    # model = initialize_model(
    #     tokenizer, 
    #     'facebook/opt-125m',
    #     num_hidden_layers=4, 
    #     intermediate_size=256,
    #     hidden_size=256,
    #     word_embed_proj_dim=256,
    #     ffn_dim=256,
    #     num_attention_heads=8,
    #     max_position_embeddings=32,
    #     is_mlm=is_mlm,
    # )

    # model.transformer.wte.weight = model.lm_head.weight

    print('#params', sum(param.numel() for param in model.parameters()))




    batch_size = 64
    lr = 5e-4

    # os.environ["WANDB_PROJECT"] = "pcfg-lm"
    d = {}
    d.update(vars(args))
    d.update({"basename": os.path.basename(__file__)})

    run = wandb.init(
        project="pcfg-lm",
        config=d,
    )  #, run_name=f"bsz_{batch_size}-lr_{lr}")
    # run.summary.update({"p": args.p})

    trainer = initialize_trainer(
        model, 
        # model_init=model_init,
        tokenizer,
        data_collator,
        datasets,
        output_dir='checkpoints',
        save_steps=10_000,
        eval_steps=100, 
        logging_steps=100,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,
        weight_decay=0.1,
        lr_scheduler_type='reduce_lr_on_plateau',
        warmup_steps=0,
        learning_rate=lr,
        num_train_epochs=1,
        fp16=True,
        max_grad_norm=0.5,
        group_by_length=True,
        auto_find_batch_size=False,
        do_eval=True,
        evaluation_strategy='steps',
        report_to="wandb",
    )

    trainer.train()

    trainer._save_checkpoint(trainer.model, None)



    corpora_path = "lm_training/corpora"
    final_checkpoint = f"checkpoints/checkpoint-{trainer.state.global_step}"

    automodel = AutoModelForCausalLM
    model = automodel.from_pretrained(f'{final_checkpoint}/')
    model.eval()

    with open(f'{final_checkpoint}/added_tokens.json') as f:
        vocab_lm = json.load(f)
    tokenizer_lm = create_tf_tokenizer_from_vocab(vocab)
    # with open('lm_training/vocab/added_tokens.json') as f:
    #     vocab = json.load(f)
    # tokenizer_pcfg = create_tf_tokenizer_from_vocab(vocab)

    with open("earleyx_pcfg_dict.pickle", "rb") as f:
        pcfg_dict = pickle.load(f)

    # _en1
    # datasets_lm might be different from the datsets_pcfg but we need to obtain exactly the same sentences for "eval" as in pcfg_dict
    # ... thus two different variables here.
    datasets_lm = load_data__for_evaluation(tokenizer_lm, corpora_path, p=1.0, add_language_pseudo_suffixes=True, train_size=0, dev_size=0, test_size=0)
    datasets_pcfg = load_data__for_evaluation(tokenizer_lm, 'lm_training/corpora', train_size=0, dev_size=0, test_size=0)
    lm_probs_en1, pcfg_probs = extract_pcfg_and_model_probs(corpus_lm=datasets_lm['eval'][:100], corpus_pcfg=datasets_pcfg['eval'][:100], pcfg_dict=pcfg_dict, model=model)
    fig1 = plot_probs(lm_probs_en1, pcfg_probs, "GPT2 EN1 $\\times$ PCFG", ylim=(-15,0.1), xlim=(-15,0.1), do_scatter=False, mincnt=1, save_as="en1_vs_pcfg")

    # _en2
    datasets_lm = load_data__for_evaluation(tokenizer_lm, corpora_path, p=0.0, add_language_pseudo_suffixes=True, train_size=0, dev_size=0, test_size=0)
    datasets_pcfg = load_data__for_evaluation(tokenizer_lm, 'lm_training/corpora', train_size=0, dev_size=0, test_size=0)
    lm_probs_en2, pcfg_probs = extract_pcfg_and_model_probs(corpus_lm=datasets_lm['eval'][:100], corpus_pcfg=datasets_pcfg['eval'][:100], pcfg_dict=pcfg_dict, model=model)
    fig2 = plot_probs(lm_probs_en2, pcfg_probs, "GPT2 EN2 $\\times$ PCFG", ylim=(-15,0.1), xlim=(-15,0.1), do_scatter=False, mincnt=1, save_as="en2_vs_pcfg")

    fig3 = plot_probs(lm_probs_en1, lm_probs_en2, "GPT2 EN1 $\\times$ GPT2 EN2", ylim=(-15,0.1), xlim=(-15,0.1), do_scatter=False, mincnt=1, save_as="en1_vs_en2")
    print()

    log_plot(wandb.run, fig1, step=trainer.state.global_step, label="gpt2_en1_vs_pcfg")
    log_plot(wandb.run, fig2, step=trainer.state.global_step, label="gpt2_en2_vs_pcfg")
    log_plot(wandb.run, fig3, step=trainer.state.global_step, label="gpt2_en1_vs_en2")
    # log_plot_as_artifact(wandb.run, fig3, label="en1_vs_en2")

    wandb.run.log({
            "cross_entropy_lm_en1": -np.mean(lm_probs_en1),
            "cross_entropy_lm_en2": -np.mean(lm_probs_en2),
            "cross_entropy_pcfg": -np.mean(pcfg_probs),
        },
        step=trainer.state.global_step,
    )

    print("cross_entropy_lm_en1:", -np.mean(lm_probs_en1))
    print("cross_entropy_lm_en2:", -np.mean(lm_probs_en2))
    print("cross_entropy_pcfg:", -np.mean(pcfg_probs))


if __name__ == "__main__":
    main()
