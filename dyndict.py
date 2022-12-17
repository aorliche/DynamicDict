import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools

mseLoss = nn.MSELoss()

# Use make_dict() followed by fit_weights()

'''
LowRankCodes - dictionary components (rank-N matrices)
For now PSD (symmetric) components only
'''
class LowRankCodes(nn.Module):
    '''
    ranks: array of rank for each codebook matrix
    '''
    def __init__(self, ranks):
        super(LowRankCodes, self).__init__()
        self.As = []
        for rank in ranks:
            A = nn.Parameter(1e-2*torch.randn(rank,264).float().cuda())
            self.As.append(A)
        self.As = nn.ParameterList(self.As)

    '''
    Generate codebook
    '''
    def forward(self):
        book = []
        for A in self.As:
            AA = A.T@A
            book.append(AA)
        return torch.stack(book)
    
'''
LowRankWeights - weights for the LowRankCodes codebook entries
'''
class LowRankWeights(nn.Module):
    '''
    ncodes: number of pages in the codebook
    Xs: list of inputs to LowRankWeights of size [(nsubs, nrois, nt)...]
    subids: id of each subject in modlist (optional)
    '''
    def __init__(self, ncodes, Xs, subids=None):
        super(LowRankWeights, self).__init__()
        self.ncodes = ncodes
        self.modparams = []
        for mod in range(len(Xs)):
            nsubs = Xs[i][0]
            nt = Xs[i][-1]
            params = nn.Parameter(1e-2*torch.rand(nsubs, self.ncodes, nt).float().cuda())
            self.modparams.append(params)
        self.modparams = nn.ParameterList(self.modparams)
        self.subids = subids

    '''
    Get estimated instantaneous FC from book
    '''
    def forward(self, sub, book, mod):
        w = self.modparams[mod][sub]
        return torch.einsum('pt,pab->abt', F.leaky_relu(w), book)

def get_recon_loss(x, xhat):
    return mseLoss(xhat, x)

def get_smooth_loss_fc(xhat):
    before = xhat[:,:,:-1]
    after = xhat[:,:,1:]
    return torch.mean((before-after)**2)

def get_mag_loss(lrc):
    loss = [torch.mean((A-0.01)**2) for A in lrc.As]
    return sum(loss)/len(loss)

def get_sub_fc(subts):
    return torch.einsum('at,bt->abt',subts,subts)

def default_or_custom(kwargs, field, default):
    if field not in kwargs:
        kwargs[field] = default

def make_dict(Xs, ranks=400*[1], **kwargs):
    default_or_custom(kwargs, 'nbatch', 30)
    default_or_custom(kwargs, 'smooth_mult', 0)
    default_or_custom(kwargs, 'nepochs', 40)
    default_or_custom(kwargs, 'pperiod', 5)
    default_or_custom(kwargs, 'subids', None)
    default_or_custom(kwargs, 'lr', 1e-2)
    default_or_custom(kwargs, 'l2', 0)
    default_or_custom(kwargs, 'patience', 20)
    default_or_custom(kwargs, 'factor', 0.75)
    default_or_custom(kwargs, 'eps', 1e-7)
    default_or_custom(kwargs, 'verbose', False)

    nbatch = kwargs['nbatch']
    smooth_mult = kwargs['smooth_mult']
    ncodes = len(ranks)
    modlist = [dict(nsubs=X.shape[0], nt=X.shape[-1])]

    lrc = LowRankCodes(ranks)
    lrw = LowRankWeights(ncodes, Xs, kwargs['subids'])

    optim = torch.optim.Adam(
        itertools.chain(lrc.parameters(), lrw.parameters()), 
        lr=kwargs['lr'], 
        weight_decay=kwargs['l2'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, 
        patience=kwargs['patience'], 
        factor=kwargs['factor'], 
        eps=kwargs['eps'])

    for epoch in range(nepochs):
        for modidx in range(len(Xs)):
            ntrain = Xs[modidx].shape[0]
            for bstart in range(0,ntrain,nbatch):
                bend = = bstart+nbatch
                if bend > ntrain:
                    bend = ntrain
                optim.zero_grad()
                book = lrc()
                recon_loss = 0
                smooth_loss_fc = 0
                for subidx in range(bstart, bend):
                    xsub = get_sub_fc(Xs[modidx][subidx])
                    xhat = lrw(subidx, book, modidx)   
                    recon_loss += get_recon_loss(xsub, xhat)
                    smooth_loss_fc += smooth_mult*get_smooth_loss_fc(xhat)
                recon_loss /= (bend-bstart)
                smooth_loss_fc /= (bend-bstart)
                loss = recon_loss+smooth_loss_fc
                loss.backward()
                optim.step()
                sched.step(loss)

        if not kwargs['verbose']:
            continue
        if epoch % pperiod == 0 or epoch == nepochs-1:
            print(f'{epoch} {bstart} recon: {[float(ls)**0.5 for ls in [recon_loss, smooth_loss_fc]]} '
                f'lr: {sched._last_lr}')

    optim.zero_grad()
    if not kwargs['verbose']:
        print('Complete')

    return lrc, lrw       

def fit_weights(low_rank_codes, Xs, **kwargs):
    default_or_custom(kwargs, 'nepochs', 600)
    default_or_custom(kwargs, 'pperiod', 50)
    default_or_custom(kwargs, 'lr', 1e-1)
    default_or_custom(kwargs, 'l1', 0)
    default_or_custom(kwargs, 'l2', 1e-5)
    default_or_custom(kwargs, 'patience', 10)
    default_or_custom(kwargs, 'factor', 0.75)
    default_or_custom(kwargs, 'eps', 1e-7)
    default_or_custom(kwargs, 'verbose', False)

    book = low_rank_codes()
    A = book.reshape(book.shape[0], -1).permute(1,0).detach()
    AA = A.T@A
    ws = []

    for X in Xs:
        AB = []
        for sub in range(X.shape[0]):
            B = get_sub_fc(X[sub]).reshape(-1, X.shape[-1])
            AB.append(A.T@B)
        AB = torch.stack(AB)

        w = nn.Parameter(torch.rand(AB.shape[0],AA.shape[1],AB.shape[-1]).float().cuda())
        ws.append(w)

        optim = torch.optim.Adam(
            [w], 
            lr=kwargs['lr'], 
            weight_decay=kwargs['l2'])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 
            patience=kwargs['patience'], 
            factor=kwargs['factor'], 
            eps=kwargs['eps'])

        for epoch in range(nepochs):
            optim.zero_grad()
            ABhat = torch.matmul(AA.detach(),F.leaky_relu(w))
            pred_loss = mseLoss(ABhat, AB.detach())**0.5
            l1_loss = kwargs['l1']*torch.mean(torch.abs(w))
            loss = pred_loss+l1_loss
            loss.backward()
            optim.step()
            sched.step(loss)
            if not kwargs['verbose']:
                continue
            if epoch % pperiod == 0 or epoch == nepochs-1:
                print(f'{epoch} {[float(ls) for ls in [pred_loss, l1_loss]]} {sched._last_lr}')

        optim.zero_grad()
        if not kwargs['verbose']:
            print('Complete')

    return ws
