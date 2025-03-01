import math
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import optim, nn
import torch.nn.functional as F
from dataloader import load_data, PAD
from torchinfo import summary
from tqdm import tqdm
import os

SEQ_MAX_LEN = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TokenEmbedding(nn.Embedding):
    # TODO: change nn.Embedding to DIY model
    """Input tokens embedding layer"""
    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size, d_model)

    
class PositionalEncoding(nn.Module):
    """Positional encoding layer, likes a token look-up table"""
    def __init__(self, d_model, max_len):
        super().__init__()
        max_len = SEQ_MAX_LEN if max_len is None else max_len
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        _2i = torch.arange(0, d_model, 2, dtype=torch.float)
        
        self.encoding[:, 0::2] = torch.sin(pos / 10000 ** (_2i / d_model))
        self.encoding[:, 1::2] = torch.cos(pos / 10000 ** (_2i / d_model))

    def forward(self, x):
        # [batch_size, seq_len]
        batch_size, seq_len = x.shape
        return self.encoding[:seq_len, :]
    

class TransformerEmbedding(nn.Module):
    """Embedding layer of Transformer model
    Args:
        vocab_size (int): size of the vocabulary
        d_model (int): dimension of the model
        max_len (int): maximum length of the input sequence
        dropout (float): dropout rate. Defaults to 0
    """
    def __init__(self, vocab_size, d_model, max_len, dropout=0.):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout, inplace=True)
    
    def forward(self, x):
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(x)
        out = self.dropout(token_emb + pos_emb)
        return out


class PositionwiseFeedForwardNet(nn.Module):
    """Feed forward layer"""
    def __init__(self, d_model, hidden_dim, dropout=0.):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # [batch_size, seq_len, d_model]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v, 
                mask=None # [batch_size, seq_len, seq_len]
                ):
        # [batch_size, n_head, seq_len, d_k]
        batch_size, n_head, seq_len, d_k = k.shape
        # 1. Q * K^T
        k_t = k.transpose(-1, -2)
        score = (q @ k_t) / math.sqrt(d_k)  # [batch_size, n_head, seq_len, seq_len]
        # 2. mask (opt.)
        if mask is not None:
            if len(mask.shape) != len(score.shape):
                mask = mask.unsqueeze(1)
                mask = mask.expand(-1, score.size(1), -1, -1)
            score = score.masked_fill(mask, -1e9)
        # 3. pass the softmax to range [0, 1]
        atten = self.softmax(score)
        # 4. score * V
        out = score @ v  # [batch_size, n_head, seq_len, d_k]
        return out, atten


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.attn = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        # 1. compute q, k, v
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # 2. split into multi-head
        q, k, v = self.split(q), self.split(k), self.split(v)
        # 3. do scaled dot-product attention
        # [batch_size, n_head, seq_len, d_k]
        out, attn_score = self.attn(q, k, v, mask=mask)
        # 4. concat and pass linear layer
        out = self.concat(out)  # [batch_size, seq_len, d_model]
        out = self.w_concat(out)
        return out, attn_score

    def split(self, tensor):
        """split tensor into multi-head by number of n_head"""
        # [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = tensor.shape
        d_tensor = d_model // self.n_head
        # [batch_size, n_head, seq_len, d_k]
        tensor = tensor.view(batch_size, self.n_head, seq_len, d_tensor)
        return tensor

    def concat(self, tensor):
        """concat multi-head tensor"""
        # [batch_size, n_head, seq_len, d_k]
        batch_size, n_head, seq_len, d_tensor = tensor.shape
        d_model = n_head * d_tensor
        # [batch_size, seq_len, d_model]
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return tensor
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden_dim, n_head, dropout=0.):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForwardNet(d_model=d_model, 
                                                hidden_dim=ffn_hidden_dim, 
                                                dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)  # [batch_size, d_model, sed_len]
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, src_mask=None):
        # x: [B, seq_len]
        # 1. compute self attention
        _x = x
        x, attn_score = self.enc_self_attn(x, x, x, mask=src_mask)  # [batch_size, seq_len, d_model]
        # 2. add & norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        # 3. feed forward
        _x = x
        x = self.ffn(x)  # [batch_size, seq_len, d_model]
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x, attn_score

def get_attn_pad_mask(seq_q, seq_k, pad_token=PAD, device=device):
    batch_size, len_q = seq_q.shape
    batch_size, len_k = seq_k.shape
    pad_attn_mask = seq_k.data.eq(pad_token).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k).to(device) # [batch_size, len_q, len_k] (len_q=len_k)
  
def get_attn_subsequent_mask(seq, device=device):
    # 返回一个上三角矩阵（对于前面的词不能与后面的词做计算）
    batch_size, seq_len = seq.shape
    attn_shape = [batch_size, seq_len, seq_len]
    subsequent_mask = torch.from_numpy(np.triu(np.ones(attn_shape)))
    return subsequent_mask.to(device)

class Encoder(nn.Module):
    """Encoder of Transformer model"""
    def __init__(self, vocab_size, d_model, 
                 max_len, ffn_hidden_dim, 
                 n_head, n_layers, dropout=0.):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size=vocab_size, 
                                        d_model=d_model, 
                                        max_len=max_len, 
                                        dropout=dropout)
        layers = []
        for _ in range(n_layers):
            layers.append(EncoderLayer(d_model=d_model, 
                                         ffn_hidden_dim=ffn_hidden_dim, 
                                         n_head=n_head, 
                                         dropout=dropout))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        # x: [B, seq_len]
        out = self.emb(x)
        enc_self_attn_mask = get_attn_pad_mask(x, x) # [batch_size, seq_len, seq_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_out, enc_self_attn = layer(out, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        # [B, seq_len, d_model]
        return enc_out, enc_self_attns
        
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden_dim, n_head, dropout=0.):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.enc_dec_attn = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        self.ffn = PositionwiseFeedForwardNet(d_model=d_model, 
                                                hidden_dim=ffn_hidden_dim, 
                                                dropout=dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, dec_in, enc_out, dec_self_attn_mask, dec_enc_attn_mask):
        # 1. compute decoder self attention
        # [B, max_len, d_model]
        _x = dec_in
        x, dec_self_attn_score = self.dec_self_attn(dec_in, dec_in, dec_in, mask=dec_self_attn_mask)
        # 2. add & norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. compute encoder-decoder cross attention
        _x = x
        x, enc_dec_attn_score = self.enc_dec_attn(q=x, k=enc_out, v=enc_out, mask=dec_enc_attn_mask)
        # 4. add & norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        
        # 5. pass positionwise feed forward network
        _x = x
        x = self.ffn(x)
        # 6. add & norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x, dec_self_attn_score, enc_dec_attn_score


class Decoder(nn.Module):
    """Decoder of Transformer model"""
    def __init__(self, vocab_size, 
                 max_len, d_model, 
                 ffn_hidden_dim,
                 n_head,
                 n_layers,
                 dropout=0.):
        super().__init__()
        self.emb = TransformerEmbedding(
            vocab_size=vocab_size, 
            d_model=d_model, 
            max_len=max_len, 
            dropout=dropout)

        layers = []
        for _ in range(n_layers):
            layers.append(DecoderLayer(d_model=d_model, 
                                            ffn_hidden_dim=ffn_hidden_dim, 
                                            n_head=n_head, 
                                            dropout=dropout))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, dec_in, enc_in, enc_out):
        # dec_in: [B, seq_len]
        # enc_in: [B, seq_len]
        # dec_in: [B, seq_len, d_model]
        dec_out = self.emb(dec_in)  # [B, seq_len, d_model]
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_in)  # [B, seq_len, seq_len]
        
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_in, dec_in)
        dec_self_attn_mask = torch.gt(dec_self_attn_pad_mask + dec_self_attn_subsequent_mask, 0).to(device)
        
        dec_enc_attn_mask = get_attn_pad_mask(dec_in, enc_in)
        
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_out, dec_self_attn, dec_enc_attn = layer(dec_out, enc_out, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_out, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self,
                 enc_vocab_size, dec_vocab_size, 
                 n_head, d_model, ffn_hidden_dim, 
                 n_layers, trg_vocab_size, max_len=None, dropout=0.):
        super().__init__()
        self.encoder = Encoder(vocab_size=enc_vocab_size, d_model=d_model, max_len=max_len, 
                               ffn_hidden_dim=ffn_hidden_dim, n_head=n_head, n_layers=n_layers, 
                               dropout=dropout)
        self.decoder = Decoder(vocab_size=dec_vocab_size, max_len=max_len, d_model=d_model, 
                               ffn_hidden_dim=ffn_hidden_dim, n_head=n_head, n_layers=n_layers, 
                               dropout=dropout)
        self.linear = nn.Linear(d_model, trg_vocab_size)
        
    def forward(self, enc_in, dec_in):  
        # enc_in: [B, seq_len]
        # dec_in: [B, seq_len]
        enc_out, enc_self_attns = self.encoder(enc_in)  # [B, seq_len, d_model]
        dec_out, dec_self_attns, dec_enc_attns = self.decoder(dec_in, enc_in, enc_out)  # [B, seq_len, d_model]
        dec_logits = self.linear(dec_out)  # [B, seq_len, trg_vocab_size]
        return dec_logits, (enc_self_attns, dec_self_attns, dec_enc_attns)
    

def visualize_attn_scores(attn_scores,  # [n_head, seq_len, seq_len] 
                          rows: int, 
                          cols: int, 
                          xlabel: str, 
                          ylabel: str,
                          titles: list = None,
                          figsize=(2.5, 2.5), 
                          cmap='Reds'):
    assert rows * cols >= len(attn_scores), 'Number of attention scores not match!'
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows))
        
    for i, attn_score in enumerate(attn_scores):
        row_idx = i // cols
        col_idx = i % cols
        ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]
        
        ax.matshow(attn_score, cmap=cmap)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if titles:
            ax.set_title(titles[i], fontsize=14)
    
    # 隐藏未使用的子图
    for j in range(len(attn_scores), rows * cols):
        row_idx = j // cols
        col_idx = j % cols
        ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
    
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks.

    Defined in :numref:`sec_seq2seq_decoder`"""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
    
def save_model(net, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(net.state_dict(), file_path)
    print(f'Save model checkpoint to: {file_path}')
    

def train_model(net, data_iter, lr, num_epochs, trg_vocab, device, save_dir='checkpoints'):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = MaskedSoftmaxCELoss()
    best_loss = float('inf')
    for epoch in range(num_epochs):
        loss_record = []
        pbar = tqdm(data_iter)
        for batch in pbar:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([trg_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)

            dec_input = torch.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input)
            l = criterion(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                loss = l.sum().detach().cpu().item()
                loss_record.append(loss)
                pbar.set_description(f'Train')
                pbar.set_postfix({'loss': loss})
                
        mean_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{num_epochs}] loss: {mean_loss}')
        if best_loss > mean_loss:
            save_model(net, os.path.join(save_dir, 'best_model.pth'))
        


if __name__ == "__main__":    
    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    lr = 1e-3
    dropout, batch_size, num_steps = 0.1, 64, 10
    
    train_iter, src_vocab, trg_vocab = load_data(batch_size=batch_size, num_steps=num_steps)
    
    model = Transformer(enc_vocab_size=len(src_vocab), 
                        dec_vocab_size=len(trg_vocab), 
                        n_head=n_heads, 
                        d_model=d_model, 
                        ffn_hidden_dim=d_ff, 
                        n_layers=n_layers, trg_vocab_size=len(trg_vocab),
                        dropout=dropout)
    model = model.to(device)
    train_model(model, train_iter, lr, num_epochs=50, trg_vocab=trg_vocab, device=device)
