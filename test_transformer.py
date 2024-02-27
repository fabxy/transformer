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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get data
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

# Create data loader
train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
BATCH_SIZE = 128

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

train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
num_batches = math.ceil(multi30k.NUM_LINES['train'] / BATCH_SIZE)

# Set seed
torch.manual_seed(0)

# Define model
hyperparams = {
    'dvocin': len(vocab_transform[SRC_LANGUAGE]),
    'dvocout': len(vocab_transform[TGT_LANGUAGE]),
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

model = Transformer(**hyperparams)

# Set up tutorial model for comparison
from transformer_tutorial import Seq2SeqTransformer, create_mask

transformer = Seq2SeqTransformer(hyperparams['nenc'], hyperparams['ndec'], hyperparams['dm'], hyperparams['h'], hyperparams['dvocin'], hyperparams['dvocout'], hyperparams['dff'], hyperparams['drop'])
transformer = transformer.to(DEVICE)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# Copy weights
from transformer_tools import update_state_dict
model_state = model.state_dict()
transformer_state = transformer.state_dict()

model_state = update_state_dict(transformer_state, model_state)
model.load_state_dict(model_state)

# Define training
train_params = {
    'beta1': 0.9,
    'beta2': 0.98,
    'eps': 1e-9,
    'lr': 1e-4,
    'init_steps': 4000,
    'steps': int(1e5),
    'ignore_idx': PAD_IDX,
    'eps_ls': 0.0,
    'epochs': 18
}

optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'], betas=(train_params['beta1'], train_params['beta2']), eps=train_params['eps'])

# lr_lambda = lambda step: hyperparams['dm']**(-0.5) * min((step+1)**(-0.5), (step+1)**(-0.5) * train_params['init_steps']**(-1.5))
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

loss_fun = torch.nn.CrossEntropyLoss(ignore_index=train_params['ignore_idx'], label_smoothing=train_params['eps_ls'])

for epoch in range(train_params['epochs']):

    model.train()
    for src, tgt in (p := tqdm(train_dataloader, desc=f"Epoch {epoch}", total=num_batches)):

        optimizer.zero_grad()

        probs = model(src, tgt[:-1])
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt[:-1], pad_idx=PAD_IDX, device=DEVICE)
        probs_transformer = transformer(src, tgt[:-1], src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        loss = loss_fun(torch.permute(probs, (1, 2, 0)), tgt[1:].T)
        loss.backward()

        optimizer.step()
        # scheduler.step()

        p.set_postfix(loss=loss.item())