
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

    def forward(self, Q, K, V):

        # input dimensions: 
        # Q: nq x nbatch x dq 
        # K: nk x nbatch x dk (dk=dq)
        # V: nv x nbatch x dv (nv=nk)

        # dot product: nq x nbatch x nk
        y = torch.einsum('ijk, kjl -> ijl', Q, K.T)
        
        # scaling
        y /= math.sqrt(K.shape[2])

        # masking
        if self.mask:
            y -= torch.triu(torch.ones(y.shape[0], y.shape[-1]) * torch.inf, diagonal=1).unsqueeze(1).repeat(1,y.shape[1],1)

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
        self.WQ = nn.ParameterList([nn.Parameter(torch.Tensor(dm, dk).uniform_(-1/math.sqrt(dm), 1/math.sqrt(dm)), requires_grad=True) for _ in range(self.h)])
        self.WK = nn.ParameterList([nn.Parameter(torch.Tensor(dm, dk).uniform_(-1/math.sqrt(dm), 1/math.sqrt(dm)), requires_grad=True) for _ in range(self.h)])
        self.WV = nn.ParameterList([nn.Parameter(torch.Tensor(dm, dv).uniform_(-1/math.sqrt(dm), 1/math.sqrt(dm)), requires_grad=True) for _ in range(self.h)])
        self.WO = nn.Parameter(torch.Tensor(h*dv, dm).uniform_(-1/math.sqrt(h*dv), 1/math.sqrt(h*dv)), requires_grad=True)

        # initialize attention layer
        self.attention = Attention(mask=mask)

    def forward(self, Q, K=None, V=None):

        if K is None:
            K = Q
        
        if V is None:
            V = K

        y = [self.attention(torch.matmul(Q, self.WQ[i]), torch.matmul(K, self.WK[i]), torch.matmul(V, self.WV[i])) for i in range(self.h)]

        y = torch.concat(y, dim=2)

        y = torch.matmul(y, self.WO)

        return y
    

class AttentionBlock(nn.Module):

    def __init__(self, h, dm, dff, decoder=False):
        super().__init__()

        self.decoder = decoder

        self.self_attention = MHA(h, dm, mask=self.decoder)
        self.norms = [nn.LayerNorm(dm)]
        
        if self.decoder:
            self.attention = MHA(h, dm)
            self.norms.append(nn.LayerNorm(dm))

        self.MLP = MLP(dm, dm, 1, dff)
        self.norms.append(nn.LayerNorm(dm))

        # TODO: make sure that embedding is last dimension

    def forward(self, X):

        if self.decoder:
            X, Y = X

        # (masked) self-attention
        # TODO: do we need to create copies at some point?
        X += self.self_attention(X)
        X = self.norms[0](X)

        # attention
        if self.decoder:
            X += self.attention(X, Y)
            X = self.norms[1](X)

        # MLP
        nseq = X.shape[0]
        dm = X.shape[-1]
        X += self.MLP(X.reshape(-1,dm)).reshape(nseq, -1, dm)
        X = self.norms[-1](X)

        if self.decoder:
            return (X, Y)
        else:
            return X
    

class Transformer(nn.Module):

    def __init__(self, dvocin, dvocout, dm, h, dff, nenc, ndec, stok=0, etok=1, ptok=2):
        super().__init__()

        # store start, end and padding tokens
        self.stok = stok
        self.etok = etok
        self.ptok = ptok

        self.in_emb = nn.Embedding(dvocin, dm)
        self.out_emb = nn.Embedding(dvocout, dm)

        self.encoder = nn.Sequential(*[AttentionBlock(h, dm, dff)]*nenc)

        self.decoder = nn.Sequential(*[AttentionBlock(h, dm, dff, decoder=True)]*ndec)

    def pos_enc(self, X):

        pos = torch.arange(X.shape[0]).repeat(X.shape[-1],1).T
        idx = torch.arange(X.shape[-1]).repeat(X.shape[0],1)

        res = torch.sin(pos / 10000**(2*idx/X.shape[-1]) + torch.remainder(idx,2) * torch.pi/2)

        return res.unsqueeze(1).repeat(1, X.shape[1], 1)
        
    def forward(self, X):

        # input embedding
        X = self.in_emb(X)

        # positional encoding
        X += self.pos_enc(X)

        # encoder
        X = self.encoder(X)

        # output
        T = torch.ones((1,X.shape[1]), dtype=torch.int64) * self.stok

        # TODO: consider max length
        while not ((T[-1] == self.etok) | (T[-1] == self.ptok)).all():

            # output embedding
            Y = self.out_emb(T)

            # TODO: output embeddings are offset by one position

            # positional encoding
            Y += self.pos_enc(Y)

            # decoder
            Y, _ = self.decoder((Y, X))

            # linear
            # TODO: check: embedding layers, we multiply those weights by sqrt(dmodel)
            Y = torch.mm(Y[-1,:,:], self.out_emb.weight.T / math.sqrt(self.out_emb.weight.shape[1]))

            # softmax
            Y = torch.softmax(Y, dim=1)

            # greedy sampling
            t = torch.argmax(Y, dim=1)
            mask = (T[-1] == self.etok) | (T[-1] == self.ptok)
            t[mask] = self.ptok

            T = torch.cat((T, t.reshape(1,-1)), dim=0)

        return T
