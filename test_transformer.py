import torch

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k

from transformer import Transformer

# Set up data pipeline
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
multi30k.URL["test"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt_task1_test2016.tar.gz"
multi30k.MD5["test"] = "876a95a689a2a20b243666951149fd42d9bfd57cbbf8cd2c79d3465451564dd2"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

def yield_tokens(data_iter, language):
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln), min_freq=1, specials=special_symbols, special_first=True)
    vocab_transform[ln].set_default_index(UNK_IDX)
    
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], vocab_transform[ln], tensor_transform)


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

# Load weights
load_path = "first_model.pt"
load_state = torch.load(load_path)

model_state = model.state_dict()
model_state.update(load_state)

model.load_state_dict(model_state)

# Translate test sentences
val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

max_len = 50

model.eval()
for src_seq, tgt_seq in val_iter:

    print(f"Translate: {src_seq}")

    src = text_transform[SRC_LANGUAGE](src_seq).unsqueeze(-1)
    tgt = torch.ones((1,1), dtype=src.dtype) * hyperparams['stok']

    while (len(tgt) < max_len) and (tgt[-1].item() != hyperparams['etok']):
        logits = model(src, tgt)
        probs = torch.softmax(logits, dim=-1)
        t = torch.argmax(probs, dim=-1)
        tgt = torch.cat((tgt, t[-1:,:]), dim=0)

    print(f"Translation: {' '.join(vocab_transform['en'].lookup_tokens(tgt[1:,:].reshape(-1).tolist())).replace('<bos>', '').replace('<eos>', '')}\n")