from .core import Decoder
import numpy as np
import xarray as xr
import numpy.random as npr


DA = xr.DataArray

class FusionDecoder(Decoder):
    def __init__(self, codebook):
        """
        Args:
            codebook (DataArray): m, d
        """
        self.codebook = codebook
        self.m = codebook.sizes['m'] if codebook is not None else 0

    def decode(self, Q):
        return self.codebook[Q]

    def optimize(self, X, enc, γ, w):
        Q = enc(X) # each point -> quant val (single value for both dims)
        # codebook = mean of all points with the same quant val
        codebook = (X.groupby(Q)
                     .mean()
                     .rename(group='m'))


        if w is not None and γ > 0:
            Δ = -(codebook @ w)*w*(1 + 1e-3)
            codebook = xr.concat((codebook, codebook + Δ), dim='Δ')
            temp=codebook.sel(m=Q)
            mse = np.power(X - codebook.sel(m=Q),2).sum('d')
            zero_one = np.sign(X @ w) != np.sign(codebook.sel(m=Q) @ w)
            loss = (1-γ)*mse + γ*zero_one
            i = (loss.groupby(Q)
                     .mean()
                     .rename(group='m')
                     .argmin('Δ'))
            codebook = codebook.sel(Δ=i)

        uniqueQ = np.sort(np.unique(Q))
        delta = -1
        for i in np.arange(self.m):
            if not (i in uniqueQ):
                delta = delta + 1
                ind1 = i // enc.rates[0]
                ind2 = i % enc.rates[0]
                j = npr.randint(X.sizes['n'], size=1)
                sampled_codewords = X[j].rename(n='m')
                if ind1 in enc.encoders[0].q_indices.T:
                    sampled_codewords.T[0] = float(enc.encoders[0].protos.T[np.where(enc.encoders[0].q_indices.T==ind1)].mean())
                else:
                    sampled_codewords.T[0] = float(enc.encoders[0].protos.T[0])
                if ind2 in enc.encoders[1].q_indices.T:
                    sampled_codewords.T[1] = float(enc.encoders[1].protos.T[np.where(enc.encoders[1].q_indices.T==ind2)].mean())
                else:
                    sampled_codewords.T[1] = float(enc.encoders[1].protos.T[0])
                if i == 0:
                    codebook = xr.concat((sampled_codewords, codebook.drop('m')),
                                         dim='m')
                else:
                    if i == self.m:
                        codebook = xr.concat((codebook.drop('m'), sampled_codewords),
                                             dim='m')
                    else:
                        if delta==0:
                            codebookPrePlusNew = xr.concat((codebook[:i ].drop('m'), sampled_codewords),
                                                       dim='m')
                            codebook = xr.concat((codebookPrePlusNew, codebook[i :].drop('m')),
                                             dim='m')
                        else:
                            codebookPrePlusNew = xr.concat((codebook[:i ], sampled_codewords),
                                                       dim='m')
                            codebook = xr.concat((codebookPrePlusNew, codebook[i :]),
                                             dim='m')


        codebook = self.complete_codebook(codebook, X)
        return type(self)(codebook)

    def __repr__(self):
        return ('codebook:\t'
                + '\n\t\t'.join(self.codebook.__repr__().split('\n'))
                + '\n')

    def complete_codebook(self, codebook, X):
        Δm = self.m - codebook.sizes['m']
        if Δm > 0:
            i = npr.randint(X.sizes['n'], size=Δm)
            sampled_codewords = X[i].rename(n='m')
            codebook = xr.concat((sampled_codewords, codebook.drop('m')),
                                 dim='m')
            codebook = codebook.sel(m=npr.permutation(codebook.sizes['m']))
        return codebook


    @classmethod
    def init_from_encoder(cls, X, enc, γ, w):
        dec = cls(None).optimize(X, enc, γ, w) # optimize codebook inside dec
        dec.m = np.prod(enc.rates)
        return dec.optimize(X, enc, γ, w) # Why do we need to run optimize twice?
