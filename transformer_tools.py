import torch

def update_state_dict(src_state, tgt_state):
    
    translation = {
        'linear': 'generator',
        'encoder': 'transformer.encoder.layers',
        'decoder': 'transformer.decoder.layers',
        'enc_norm': 'transformer.encoder.norm',
        'dec_norm': 'transformer.decoder.norm',
        'norms.0': 'norm1',
        'norms.1': 'norm2',
        'norms.2': 'norm3',
        'in_emb': 'src_tok_emb.embedding',
        'out_emb': 'tgt_tok_emb.embedding',
        'MLP.layers.0': 'linear1',
        'MLP.layers.2': 'linear2',
        'self_attention': 'self_attn',
        'attention': 'multihead_attn',
        'WO': 'out_proj',
    }

    W_names = ['WQ', 'WK', 'WV']
    num_heads = len([key for key in tgt_state.keys() if ('encoder.0.self_attention.WQ' in key) and ('weight' in key)])
    W_dict = {}
    for key in src_state:
        for var in ['weight', 'bias']:
            if f'in_proj_{var}' in key:
                W_weights = torch.split(src_state[key], len(src_state[key]) // len(W_names))
                
                for n, name in enumerate(W_names):
                    W_heads = torch.split(W_weights[n], len(W_weights[n]) // num_heads)

                    for h in range(num_heads):
                        W_dict[key.replace(f'in_proj_{var}', f'{name}.{h}.{var}')] = W_heads[h]

    src_state.update(W_dict)

    for key in tgt_state:
        transformer_key = key
        for org, repl in translation.items():
            transformer_key = transformer_key.replace(org, repl)

        # print(f"{transformer_key in src_state} - {key}: {transformer_key}")
        tgt_state[key] = src_state[transformer_key]

    return tgt_state