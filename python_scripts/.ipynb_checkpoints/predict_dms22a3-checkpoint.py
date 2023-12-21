import os, gc
import math
import random
from pathlib import Path
from typing import ClassVar
import json
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

from xpos_relative_position import XPOS


parser = argparse.ArgumentParser()
parser.add_argument("--bpp_path",
                    required=True,
                    type=str)
parser.add_argument("--test_path", 
                    required=True,
                    type=str)
parser.add_argument("--model_path", 
                    required=True,
                    type=str)
parser.add_argument("--react_preds_path", 
                    required=True,
                    type=str)
parser.add_argument("--out_path", 
                    required=True,
                    type=str)
parser.add_argument("--brackets", 
                    required=False,
                    default=[],
                    type=str, 
                    nargs='+')
parser.add_argument("--fold",  #solely for output naming
                    default=0,
                    type=int)
parser.add_argument("--batch_size",
                    default=128,
                    type=int)
parser.add_argument("--num_workers",
                    default=64,
                    type=int)
parser.add_argument("--device",
                    default=0,
                    type=int)
parser.add_argument("--pos_embedding",
                    choices=['xpos', 
                             'dyn',
                             'alibi'],
                    required=True)
parser.add_argument("--num_attn_layers",
                    default=12,
                    type=int)
parser.add_argument("--num_conv_layers",
                    default=12,
                    type=int)
parser.add_argument("--adj_ks",
                   required=True, 
                   type=int)
parser.add_argument("--not_slice",
                    action="store_true")
parser.add_argument("--pred_mode",
                    choices=["dms_2a3", "2a3_dms"],
                    required=True)

args = parser.parse_args()


class DynamicPositionBias(nn.Module):
    def __init__(self, dim, *, heads, depth, log_distance = False, norm = False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            nn.LayerNorm(dim) if norm else nn.Identity(),
            nn.SiLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else nn.Identity(),
                nn.SiLU()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert i == j
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device = device)
        context_arange = torch.arange(n, device = device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (n - 1)

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device = device).float()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias
    
def exists(val):
    return val is not None
def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, total_heads, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., -i:, -j:]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = 0)
        self.register_buffer('bias', bias, persistent = False)

        return self.bias
    
    
class MultiHeadSelfAttention(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 positional_embedding: str,
                 num_heads: int = None,
                # k_dim: int = None,
                # v_dim: int = None,
                 dropout: float = 0.10, 
                
                 bias: bool = True,
                 temperature: float = 1,
                ):
        super().__init__()
        
        assert positional_embedding in ("xpos", "dyn", "alibi")
        self.positional_embedding = positional_embedding
        self.hidden_dim = hidden_dim
        if num_heads == None:
            self.num_heads = 1
        else:
            self.num_heads = num_heads
        self.head_size = hidden_dim//self.num_heads
        self.dropout = dropout
        self.bias = bias
        self.temperature = temperature
        
        if self.positional_embedding == "dyn":
            self.dynpos = DynamicPositionBias(dim = hidden_dim//4,
                                              heads = num_heads, 
                                              depth = 2)
        elif self.positional_embedding == "alibi":
            alibi_heads = num_heads // 2 + (num_heads % 2 == 1)
            self.alibi = AlibiPositionalBias(alibi_heads, 
                                         self.num_heads)
        elif self.positional_embedding == "xpos":
            self.xpos = XPOS(self.head_size)
            

        assert hidden_dim == self.head_size*self.num_heads, "hidden_dim must be divisible by num_heads"
        
    
        self.dropout_layer = nn.Dropout(dropout)
        self.weights = nn.Parameter(
            torch.empty(self.hidden_dim, 3 * self.hidden_dim) #Q, K, V of equal sizes in given order
        )
        self.out_w = nn.Parameter(
            torch.empty(self.hidden_dim, self.hidden_dim) #Q, K, V of equal sizes in given order
        )
        if self.bias:
            self.out_bias = nn.Parameter(
                torch.empty(1,1,self.hidden_dim) #Q, K, V of equal sizes in given order
            )
            torch.nn.init.constant_(self.out_bias, 0.)
            self.in_bias = nn.Parameter(
                torch.empty(1,1, 3*self.hidden_dim) #Q, K, V of equal sizes in given order
            )
            torch.nn.init.constant_(self.in_bias, 0.)
        torch.nn.init.xavier_normal_(self.weights)
        torch.nn.init.xavier_normal_(self.out_w)
        self.gamma = nn.Parameter(torch.ones(self.num_heads).view(1, -1, 1, 1))

    def forward(self, x, adj, mask = None, same = True, return_attn_weights=False):
        b, l, h = x.shape
        x = x @ self.weights + self.in_bias # b, l, 3*hidden
        Q, K, V = x.view(b, l, self.num_heads, -1).permute(0,2,1,3).chunk(3, dim=3) # b, a, l, head
        
        if self.positional_embedding == "xpos":
            Q, K = self.xpos(Q), self.xpos(K, downscale=True)
        
        norm = self.head_size**0.5
        attention = (Q @ K.transpose(2,3)/self.temperature/norm)
        
        if self.positional_embedding == "dyn":
            i, j = map(lambda t: t.shape[-2], (Q, K))
            attn_bias = self.dynpos(i, j).unsqueeze(0)
            attention = attention + attn_bias
        elif self.positional_embedding == "alibi":
            i, j = map(lambda t: t.shape[-2], (Q, K))
            attn_bias = self.alibi(i, j).unsqueeze(0)
            attention = attention + attn_bias
        
        attention = attention + self.gamma * adj
        
        
        attention = attention.softmax(dim = -1) # b, a, l, l
        
        if mask is not None:
            attention = attention*mask.view(b,1,1,-1) 
        out = attention @ V  # b, a, l, head
        out = out.permute(0,2,1,3).flatten(2,3) # b, a, l, head -> b, l, (a, head) -> b, l, hidden
        if self.bias:
            out = out + self.out_bias
        if return_attn_weights:
            return out, attention
        else:
            return out  
        
        
class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 positional_embedding: str,
                 num_heads: int = None,
                 dropout: float = 0.10,
                 ffn_size: int = None,
                 activation: nn.Module = nn.GELU,
                 temperature: float = 1.,
                ):
        super().__init__()
        if num_heads is None:
            num_heads = 1
        if ffn_size is None:
            ffn_size = hidden_dim*4
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.mhsa = MultiHeadSelfAttention(hidden_dim=hidden_dim,
                                           num_heads=num_heads,
                                           positional_embedding=positional_embedding,
                                           dropout=dropout,
                                           bias=True,
                                           temperature=temperature,
                                          )
        self.dropout_layer = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_size),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_dim),
            nn.Dropout(dropout)
        )
        

    def forward(self, x, adj, mask = None, return_attn_weights = False):
        x_in = x
        if return_attn_weights:
            x, attn_w = self.mhsa(self.in_norm(x), adj=adj, mask=mask, return_attn_weights = True)
        else:
            x = self.mhsa(self.in_norm(x), adj=adj, mask=mask, return_attn_weights = False)
        x = self.dropout_layer(x) + x_in
        x = self.ffn(x) + x

        if return_attn_weights:
            return x, attn_w
        else:
            return x
        

class ResConv2dSimple(nn.Module):
    def __init__(self, 
                 in_c, 
                 out_c,
                 kernel_size=7
                ):  
        super().__init__()
        self.conv = nn.Sequential(
            # b c w h
            nn.Conv2d(in_c,
                      out_c, 
                      kernel_size=kernel_size, 
                      padding="same", 
                      bias=False),
            # b w h c#
            nn.BatchNorm2d(out_c), # maybe batchnorm
            nn.GELU(),
            # b c e 
        )
        
        if in_c == out_c:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_c,
                          out_c, 
                          kernel_size=1, 
                          bias=False)
            )

    def forward(self, x):
        # b e s 
        h = self.conv(x)
        x = self.res(x) + h
        return x
    
    
class AdjTransformerEncoder(nn.Module):
    def __init__(self,
                 positional_embedding: str,
                 dim: int  = 192,
                 head_size: int = 32,
                 dropout: float = 0.10,
                 dim_feedforward: int = 192 * 4,
                 activation: nn.Module = nn.GELU,
                 temperature: float = 1.,
                 num_layers: int = 12,
                 num_adj_convs: int =3,
                 ks: int = 3,
                ):
        super().__init__()
        print(f"Using kernel size {ks}")
        num_heads, rest = divmod(dim, head_size)
        assert rest == 0
        self.num_heads = num_heads
        
        self.layers = nn.Sequential(
            *[TransformerEncoderLayer(hidden_dim=dim,
                                     num_heads=num_heads,
                                     positional_embedding=positional_embedding,
                                     dropout=dropout,
                                     ffn_size=dim_feedforward,
                                     activation=activation,
                                     temperature=temperature,
                                    ) 
             for i in range(num_layers)]
        )
        self.conv_layers = nn.ModuleList()
        for i in range(num_adj_convs):
            self.conv_layers.append(ResConv2dSimple(in_c=1 if i == 0 else num_heads,
                                              out_c=num_heads,
                                              kernel_size=ks))
            
            
    def forward(self, x, adj, mask=None):
        # adj B S S 
        adj = torch.log(adj+1e-5)
        adj = adj.unsqueeze(1) # B 1 S S 
        
        for ind, mod in enumerate(self.layers):
            if ind < len(self.conv_layers):
                conv = self.conv_layers[ind]
                adj = conv(adj)
            x = mod(x, adj=adj, mask=mask)
            

        return x
    
    
class RNAdjNetBrk(nn.Module):
    def __init__(self,  
                 positional_embedding: str,
                 adj_ks: int,
                 not_slice: bool,
                 brk_names: list[str]  = None,
                 num_convs: int  = None,
                 dim=192, 
                 depth=12,
                 head_size=32,
                 brk_symbols=9,
                 ):
        super().__init__()
        self.slice_tokens = not not_slice 
        if not self.slice_tokens:
            print("Not removing unnecessary padding tokens. This can downgrade performance and slow the training")
        else:
            print("Removing unnecessary padding tokens")
            
        print(f"Using {positional_embedding} positional embedding")
        if num_convs is None:
            num_convs = depth
        print(f"Using {num_convs} conv layers")
        
        self.emb = nn.Embedding(4+3,dim) # 4 nucleotides + 3 tokens
        self.brk_names = brk_names
        print('Using', brk_names)
        
        self.transformer = AdjTransformerEncoder(
            num_layers=depth,
            num_adj_convs=num_convs,
            dim=dim,
            head_size=head_size,
            positional_embedding=positional_embedding,
            ks=adj_ks
        )
        
        self.proj_out = nn.Sequential(nn.Linear(dim, dim),
                                      nn.GELU(),
                                      nn.Linear(dim, 1))
        
        self.struct_embeds = nn.ModuleDict()
        
        if self.brk_names is not None:
        
            for method in self.brk_names:
                emb = nn.Embedding(brk_symbols+3, dim)
                self.struct_embeds[method] = emb
            self.struct_embeds_weights = torch.nn.Parameter(torch.ones(len(brk_names)))
            
        self.is_good_embed = nn.Embedding(2, dim)
        self.react_emb = nn.Conv1d(1, dim, kernel_size = 1)
        self.react_missing_emb = nn.Embedding(2, dim)
            
    def forward(self, x0):
        mask = x0['forward_mask']
        if self.slice_tokens:
            Lmax = mask.sum(-1).max()
            mask = mask[:,:Lmax]
            
        adj = x0['adj'] 
        if self.slice_tokens:
            adj = adj[:, :Lmax, :Lmax]      
        
        if self.slice_tokens:
            e = self.emb(x0['seq_int'][:, :Lmax])
        else:
            e = self.emb(x0['seq_int'])
        
        rmask = x0["react"].isnan().int()
        r_missing = self.react_missing_emb(rmask)[:,:Lmax]
        
        react = x0["react"].unsqueeze(1) #B 1 L
        react = react.clip(0,1).nan_to_num(-1.0)
        react = self.react_emb(react).permute(0,2,1) #B L E    
        react = react[:,:Lmax]
    
        x = e
        is_good = x0['is_good']
        e_is_good = self.is_good_embed(is_good) # B E
        e_is_good = e_is_good.unsqueeze(1) # B 1 E
        x = x + e_is_good + react + r_missing
        
        if self.brk_names is not None:
            for ind, method in enumerate(self.brk_names):
                st = x0[method]
                if self.slice_tokens:
                    st = st[:,:Lmax]
                st_embed = self.struct_embeds[method](st)
                x = x + st_embed * self.struct_embeds_weights[ind]
                
        x = self.transformer(x, adj, mask=mask)
        
        x = self.proj_out(x).squeeze(-1)
   
        return x


BPP_ROOT_DIR = Path(args.bpp_path)

def load_eterna(seq_id: str, maxL: int):
    path = BPP_ROOT_DIR / f"{seq_id}.npy"
    mat = np.load(path)
    dif = maxL - mat.shape[0]
    res = np.pad(mat, ((0, dif), (0, dif)))
    return res

class RNA_Dataset_Test(Dataset):
    def __init__(self, 
                 df,
                 seqid_to_react,
                 pred_mode = "dms_2a3",
                ):
        df['L'] = df.sequence.apply(len)
        self.pred_mode = pred_mode
        self.Lmax = df['L'].max()
        self.df = df
        self.react = seqid_to_react
        self.seq_map = {'A':0,'C':1,'G':2,'U':3, "START": 4, "END": 5, "EMPTY": 6}
        
        self.sid = df.sequence_id

    def __len__(self):
        return len(self.df)  
    
    def _process_seq(self, rawseq, shift):
        seq = [self.seq_map['EMPTY'] for i in range(shift)]
        seq.append(self.seq_map['START'])
        start_loc = len(seq) - 1
        seq.extend(self.seq_map[s] for s in rawseq)
        seq.append(self.seq_map['END'])
        end_loc = len(seq) - 1
        for i in range(len(seq), self.Lmax+2):
            seq.append(self.seq_map['EMPTY'])
            
        seq = np.array(seq)
        seq = torch.from_numpy(seq)
        
        return seq, start_loc, end_loc
    
    def __getitem__(self, idx):
        sid, id_min, id_max, seq = self.df.loc[idx, ['sequence_id', 'id_min', 'id_max', 'sequence']]
        real_seq_L = len(seq)
       
        shift = 0
        lbord = 1 + shift
        rbord = self.Lmax  + 1 - real_seq_L- shift
        
        seq_int, start_loc, end_loc = self._process_seq(seq, shift)
        mask = torch.zeros(self.Lmax + 2, dtype=torch.bool)
        mask[start_loc+1:end_loc] = True # not including START and END
        
        ids = np.arange(id_min,id_max+1)
        ids = np.pad(ids,(1,self.Lmax+1-real_seq_L), constant_values=-1)
      
        forward_mask = torch.zeros(self.Lmax + 2, dtype=torch.bool) # START, seq, END
        forward_mask[start_loc:end_loc+1] = True # including START and END
        
        
        react = self.react[sid].T
        react = react[~np.isnan(react[:, 0])]
        react = np.pad(react, ((lbord,
                                rbord),
                               (0,0)), constant_values=np.nan)
        
        react = torch.from_numpy(react)
         
        if self.pred_mode == "dms_2a3":
            x_react = react[:, 0].clip(0,1) #DMS NB! order changed in comparison to training code
            y_react = react
        elif self.pred_mode == "2a3_dms":
            x_react = react[:, 1].clip(0,1) #2A3
            y_react = react
        
        X = {'seq_int': seq_int,
             'mask': mask, 
             "forward_mask": forward_mask,
             'is_good': 1,
             'react': x_react.float()}
              
        adj = load_eterna(sid,  self.Lmax)[:real_seq_L, :real_seq_L]
        adj = np.pad(adj, ((lbord,rbord), (lbord, rbord)), constant_values=0)
        adj = torch.from_numpy(adj).float()
                
            
        X['adj'] = adj
        
        return X, \
               {'ids':ids,
                'react': y_react}
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = True # type: ignore
    
def val_to(x, device="cuda"):
    if isinstance(x, list):
        return [val_to(z) for z in x]
    return x.to(device)

def dict_to(x, device='cuda'):
    return {k: val_to(x[k], device) for k in x}

def to_device(x, device='cuda'):
    return tuple(dict_to(e,device) for e in x)

class DeviceDataLoader:
    def __init__(self, dataloader, device='cuda'):
        self.dataloader = dataloader
        self.device = device
    
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)
            
            
test = pd.read_csv(args.test_path)

ensemble_path = Path(args.react_preds_path)
test_pred = pd.read_parquet(ensemble_path)
test_pred.head()

seqid_map = {}
for i in test.itertuples():
    for idx in range(i.id_min, i.id_max + 1):
        seqid_map[idx] = i.sequence_id
        
test_pred["seq_id"] = test_pred.id.apply(lambda x: seqid_map[x])
Lmax = test.sequence.apply(len).max()

pred_data = {i: np.full((2,Lmax), fill_value=np.nan) for i in test.sequence_id}
coords_data = {i.sequence_id: (i.id_min, i.id_max) for i in test.itertuples()}


for data in tqdm(test_pred.itertuples(), total = len(test_pred)):
    seq_id = data.seq_id
    coords = coords_data[seq_id]
    pos_idx = data.id - coords[0]
    
    pred_data[seq_id][0][pos_idx] = data.reactivity_DMS_MaP
    pred_data[seq_id][1][pos_idx] = data.reactivity_2A3_MaP
    
id2seq = {i.sequence_id: i.sequence for i in test.itertuples()}

model = RNAdjNetBrk(positional_embedding=args.pos_embedding,
                         depth=args.num_attn_layers, 
                         num_convs=args.num_conv_layers,
                         adj_ks=args.adj_ks,
                         not_slice=args.not_slice
                   )

fold = args.fold
model_path = args.model_path
name = "_".join(model_path.split("/")[-2:]).removesuffix(".pth")
submit_path = Path(args.out_path) / f'submit_{name}.parquet'
print("Model path:", model_path)
print("Output path:", submit_path)

if Path(submit_path).exists():
    print("skipping")
    raise FileExistsError("Submit path should not lead to existing file.")
    
model.load_state_dict(torch.load(
    model_path, 
                                 map_location="cpu")['model'])
model = model.eval()
device = torch.device(f"cuda:{args.device}")

ds = RNA_Dataset_Test(test, pred_data, pred_mode=args.pred_mode)
dl = DeviceDataLoader(DataLoader(ds, 
                                 batch_size=args.batch_size, 
                                 shuffle=False,
                                 drop_last=False, 
                                 num_workers=args.num_workers, 
                                 persistent_workers=True), device)

ids, preds = [], []
model = model.to(device)
model = model.eval()
#gamma = 0.05
        

for x,y in tqdm(dl):
    with torch.no_grad():#torch.cuda.amp.autocast():
        p = model(x).clip(0,1)

    for idx, mask, pi in zip(y['ids'].cpu(), x['mask'].cpu(), p.cpu()):
        ids.append(idx[mask])
        preds.append(pi[mask[:pi.shape[0]]])
    

ids = torch.concat(ids)
preds = torch.concat(preds)


df = pd.DataFrame({'id':ids.numpy(), 'reactivity_2A3_MaP':preds[:].numpy()})
print(df.head())

os.makedirs(Path(args.out_path), exist_ok = True)
df.to_parquet(submit_path, index=False)


