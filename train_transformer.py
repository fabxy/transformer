import torch
import torch.nn as nn
from torch import Tensor

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformer import Transformer
from tqdm import tqdm
import math

import plotext as plt

DEVICE = 'cpu'
# DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Set up data pipeline
# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)
    
# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor
    
# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# Get maximum tokens in training database
max_toks = max([len(token_transform[(SRC_LANGUAGE, TGT_LANGUAGE)[s]](sen)) for tup in Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)) for s, sen in enumerate(tup)])
max_toks += 2

# Set seed
torch.manual_seed(0)

# Define model
model_params = {
    'dvocin': len(vocab_transform[SRC_LANGUAGE]),
    'dvocout': len(vocab_transform[TGT_LANGUAGE]),
    'max_toks': max_toks,
    'dm': 512,
    'h': 8,
    'dff': 512,
    'nenc': 3,
    'ndec': 3,
    'stok': BOS_IDX,
    'etok': EOS_IDX,
    'ptok': PAD_IDX,
    'drop': 0.1,
}

model = Transformer(**model_params)
model = model.to(DEVICE)

# Define training
train_params = {
    'batch_size': 128,
    'beta1': 0.9,
    'beta2': 0.98,
    'eps': 1e-9,
    'lr': 1e-4,
    'init_steps': 4000,
    'steps': int(1e5),
    'ignore_idx': PAD_IDX,
    'eps_ls': 0.0,
    'epochs': 50
}
save_path = "second_model.pt"

optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'], betas=(train_params['beta1'], train_params['beta2']), eps=train_params['eps'])

# lr_lambda = lambda step: model_params['dm']**(-0.5) * min((step+1)**(-0.5), (step+1)**(-0.5) * train_params['init_steps']**(-1.5))
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

loss_fun = torch.nn.CrossEntropyLoss(ignore_index=train_params['ignore_idx'], label_smoothing=train_params['eps_ls'])

train_losses = []
val_losses = []
plot_losses = True
for epoch in range(train_params['epochs']):

    # Training
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=train_params['batch_size'], collate_fn=collate_fn)
    num_batches = math.ceil(multi30k.NUM_LINES['train'] / train_params['batch_size'])

    model.train()
    losses = []
    for src, tgt in (p := tqdm(train_dataloader, desc=f"Epoch {epoch}", total=num_batches)):
        
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        optimizer.zero_grad()

        probs = model(src, tgt[:-1])

        loss = loss_fun(torch.permute(probs, (1, 2, 0)), tgt[1:].T)
        loss.backward()

        optimizer.step()
        # scheduler.step()

        losses.append(loss.item())
        p.set_postfix(loss=loss.item())

    if plot_losses:
        plt.scatter(losses, marker='dot', color='black')
        plt.plotsize(plt.tw(),10)
        plt.xticks([])
        plt.xaxes(0,0)
        plt.yaxes(1,0)
        plt.clc()
        plt.show()
        plt.clear_figure()

    train_losses.append(losses)

    # Validation
    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=train_params['batch_size'], collate_fn=collate_fn)
    num_batches = math.ceil(multi30k.NUM_LINES['valid'] / train_params['batch_size'])

    losses = []
    for src, tgt in (p := tqdm(val_dataloader, desc=f"Validation", total=num_batches)):

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        probs = model(src, tgt[:-1])
        loss = loss_fun(torch.permute(probs, (1, 2, 0)), tgt[1:].T)

        losses.append(loss.item())
        p.set_postfix(loss=loss.item())

    val_loss = sum(losses) / num_batches
    val_losses.append(val_loss)
    print(f'Validation loss: {val_loss:.2f}')

    # Storing
    if save_path is not None:

        state_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'model_params': model_params,
            'train_params': train_params,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }

        torch.save(state_dict, save_path)