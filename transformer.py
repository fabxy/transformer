
import math
import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_size, out_size, hid_num, hid_size):
        super().__init__()

        if hid_num:
            layers = [nn.Linear(in_size, hid_size), nn.ReLU()]
            
            for _ in range(hid_num-1):
                layers.append(nn.Linear(hid_size, hid_size))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hid_size, out_size))

        else:
            layers = [nn.Linear(in_size, out_size)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class Attention(nn.Module):

    def __init__(self, mask=False):
        super().__init__()

        self.mask = mask

    def forward(self, Q, K, V, key_mask=None):

        # input dimensions: 
        # Q: nq x nbatch x dq 
        # K: nk x nbatch x dk (dk=dq)
        # V: nv x nbatch x dv (nv=nk)

        # dot product: nq x nbatch x nk
        y = torch.einsum('ijk, ljk -> ijl', Q, K)
        
        # scaling
        y /= math.sqrt(K.shape[2])

        # masking
        if self.mask:
            mask = torch.triu(torch.ones(y.shape[0], y.shape[-1], dtype=torch.bool), diagonal=1).unsqueeze(1).repeat(1,y.shape[1],1)
            y = y.masked_fill(mask == True, -torch.inf)
        if key_mask is not None:
            mask = key_mask.T.unsqueeze(0).repeat(y.shape[0], 1, 1)
            y = y.masked_fill(mask == True, -torch.inf)

        # softmax
        y = torch.softmax(y, dim=2)

        # obtain value
        y = torch.einsum('ijk, kjl -> ijl', y, V)

        return y


class MHA(nn.Module):

    def __init__(self, h, dm, dk=None, dv=None, mask=False):
        super().__init__()

        # dimensions: dm to dk to dv to h*dv to dm
        self.h = h

        if dk is None:
            dk = dm // h

        if dv is None:
            dv = dk

        # initialize weights
        self.WQ = nn.ModuleList([nn.Linear(dm, dk) for _ in range(self.h)])
        self.WK = nn.ModuleList([nn.Linear(dm, dk) for _ in range(self.h)])
        self.WV = nn.ModuleList([nn.Linear(dm, dv) for _ in range(self.h)])
        self.WO = nn.Linear(h*dv, dm)

        # initialize attention layer
        self.attention = Attention(mask=mask)

    def forward(self, Q, K=None, V=None, key_mask=None):

        if K is None:
            K = Q
        
        if V is None:
            V = K

        y = [self.attention(self.WQ[i](Q), self.WK[i](K), self.WV[i](V), key_mask=key_mask) for i in range(self.h)]

        y = torch.concat(y, dim=2)

        y = self.WO(y)

        return y
    

class AttentionBlock(nn.Module):

    def __init__(self, h, dm, dff, drop=0.0, decoder=False):
        super().__init__()

        self.dropout = nn.Dropout(p=drop)
        self.decoder = decoder

        self.self_attention = MHA(h, dm, mask=self.decoder)
        norms = [nn.LayerNorm(dm)]
        
        if self.decoder:
            self.attention = MHA(h, dm)
            norms.append(nn.LayerNorm(dm))

        self.MLP = MLP(dm, dm, 1, dff)
        norms.append(nn.LayerNorm(dm))
        self.norms = nn.ModuleList(norms)

    def forward(self, X, Y=None, mask1=None, mask2=None):

        # (masked) self-attention
        res = X
        X = self.dropout(self.self_attention(X, key_mask=mask1))
        X = self.norms[0](X + res)

        # attention
        if self.decoder:
            res = X
            X = self.dropout(self.attention(X, Y, key_mask=mask2))
            X = self.norms[1](X + res)

        # MLP
        nseq = X.shape[0]
        dm = X.shape[-1]
        res = X
        X = self.dropout(self.MLP(X.reshape(-1,dm)).reshape(nseq, -1, dm))
        X = self.norms[-1](X + res)

        return X
    

class Transformer(nn.Module):

    def __init__(self, dvocin, dvocout, dm, h, dff, nenc, ndec, stok=0, etok=1, ptok=2, drop=0.0):
        super().__init__()

        # store start, end and padding tokens
        self.stok = stok
        self.etok = etok
        self.ptok = ptok

        self.in_emb = nn.Embedding(dvocin, dm)
        self.out_emb = nn.Embedding(dvocout, dm)

        self.encoder = nn.ModuleList([AttentionBlock(h, dm, dff, drop) for _ in range(nenc)])
        self.decoder = nn.ModuleList([AttentionBlock(h, dm, dff, drop, decoder=True) for _ in range(ndec)])

        self.enc_norm = nn.LayerNorm(dm)
        self.dec_norm = nn.LayerNorm(dm)

        self.linear = nn.Linear(dm, dvocout)

        self.dropout = nn.Dropout(p=drop)

    def pos_enc(self, X):

        pos = torch.arange(X.shape[0]).repeat(X.shape[-1],1).T
        idx = torch.arange(X.shape[-1]).repeat(X.shape[0],1)

        res = torch.sin(pos / 10000**((idx-torch.remainder(idx,2))/X.shape[-1]) + torch.remainder(idx,2) * torch.pi/2)

        return res.unsqueeze(1).repeat(1, X.shape[1], 1)
        
    def forward(self, X, Y):

        # X: source sequence: ns x nbatch
        # Y: target sequence: nt x nbatch

        # get padding masks
        X_mask = (X == self.ptok)
        Y_mask = (Y == self.ptok)

        # embeddings
        X = self.in_emb(X) * math.sqrt(self.in_emb.weight.shape[1])
        Y = self.out_emb(Y) * math.sqrt(self.out_emb.weight.shape[1])

        # positional encodings
        X += self.pos_enc(X)
        Y += self.pos_enc(Y)

        # dropout
        X = self.dropout(X)
        Y = self.dropout(Y)

        # encoder
        for layer in self.encoder:
            X = layer(X, mask1=X_mask)
        X = self.enc_norm(X)

        # decoder
        for layer in self.decoder:
            Y = layer(Y, X, mask1=Y_mask, mask2=X_mask)
        Y = self.dec_norm(Y)
        
        # linear
        Y = self.linear(Y)

        # softmax
        # Y = torch.softmax(Y, dim=-1)

        return Y

