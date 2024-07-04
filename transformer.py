import torch
import torch.nn as nn
from torch.nn import functional as F

if torch.backends.mps.is_available() and False:
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    mps_device = 'cpu'
    print ("MPS device not found.")

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# creating vocab from input text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(f"vocab size {vocab_size}")

# creating encoder and decoder functions
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda s: ''.join([itos[i] for i in s])

# tokenization and train/val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train = data[:n]
val = data[n:]

torch.manual_seed(1337)

# Hyperparameters
batch_size = 4
block_size = 8
train_iter = 5000
loss_iter = 10
n_embd = 32
lr=1e-3

# helper function to get batch
def get_batch(split):
    batch_data = train if split == 'train' else val
    idx = torch.randint(0, len(batch_data)-block_size, (batch_size,))
    xb = torch.stack([batch_data[i : i + block_size] for i in idx])
    yb = torch.stack([batch_data[i+1 : i + block_size+1] for i in idx])
    xb.to(mps_device)
    yb.to(mps_device)
    return xb, yb

# Visualize (nuermically) how block_size+1 behaves for indexing
# idx = torch.randint(0, len(train)-2, (100,))
# print(idx)
# print(train)
# print(train[7:7+3])

xb, yb = get_batch('train')
# for i in range(batch_size):
#     for j in range(block_size):
#         print(f"input {xb[i][:j+1]}, target {yb[i][j]}")


# create Bigram model
torch.manual_seed(1337)

@torch.no_grad()
def get_loss():
    m.eval()
    out = {}
    batch_data = ['train', 'val']
    for data in batch_data:
        loss_tensor = torch.zeros(loss_iter, device=mps_device)
        for iter in range(loss_iter):
            x, y = get_batch(data)
            logits, loss = m(x,y)
            loss_tensor[iter] = loss
        out[data] = loss_tensor.mean().item()
    m.train()
    return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
    
    def forward(self, x):
        B, T, C  = x.shape
        k = self.key(x) # B, T, 
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out

class SmallLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadedAttention(4, (n_embd//4))
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x, targets=None):
        x = x.to(mps_device)
        B, T = x.shape
        tok_emb = self.token_embedding_table(x) #B, T, C (C is n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=mps_device))
        x = tok_emb + pos_emb
        x = self.sa_heads(x)
        x = self.ffwd(x)
        logits = self.lm_head(x) #B, T, C (C is vocab_size)
        if targets is None:
            loss = None
        else:
            targets = targets.to(mps_device)
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, start, num_samples):
        start = start.to(mps_device)
        for _ in range(num_samples):
            modified_start = start[:, -block_size:] # crop the input till block size
            logits, loss = self(modified_start)
            logits = logits [:,-1,:] # look at just the last time channel for Bigram model
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            start = torch.cat((start, next_idx), dim=1)
        return ''.join(decode(start.tolist()[0]))
    
m = SmallLanguageModel()
m.to(mps_device)
logits, loss = m(xb, yb)
start = torch.zeros((1,1), dtype=torch.long, device=mps_device)
num_samples = 100
# expected initial loss = -ln(1/65)

optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

# training the model
for i in range(train_iter):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    if i%1000 == 0:
        print(get_loss())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(m.generate(start, 300))
