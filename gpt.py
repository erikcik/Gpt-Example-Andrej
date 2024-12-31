import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 32
block_size = 8
step_length = 10000
learning_rate = 1e-3
n_embed = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_iteration = 200
loss_interval = 300

text = open('input.txt', 'r', encoding='utf-8').read()

#tokenizer
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
print(f'We have vocab size of: {len(vocab)}')

stoi = {s : i for i, s in enumerate(vocab)}
itos = {i : s for s, i in stoi.items()}
encoder = lambda f: [stoi[i] for i in f]
decoder = lambda f: ''.join([itos[i] for i in f])

data = torch.tensor(encoder(text), dtype=torch.long)

#splitting dataset to training and validation
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:] 


def get_batch(label):
    data = train_data if label == 'train' else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix]) #creates 2d tensor from all of the lists of values we provide
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
    
xb, xy = get_batch('train')

@torch.no_grad()
def calculate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(loss_iteration)
        for i in range(loss_iteration):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False) 
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size)))) # if we want to assign variables different from module's predefined variables
        #we have to use this
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B, T, head_size
        q = self.query(x) # B, T, head_size

        wei = (q @ k.transpose(-2, -1)) * C**-0.5 #paper's dk value, -0.5 makes root square and divides  ### B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) ## B, T, T ##this T approach is more explicit.
        wei = F.softmax(wei, dim=-1) ## B, T, T
        v = self.value(x) ## B, T, head_size
        out = wei @ v ## B, T, T @ B, T, head_size ==> B, T, head_size
        return out
  
class MultiHead(nn.Module): # this layer can be thought as tokens have much more to talk about instead of just going through one iteration. so we run them parallel 
#loss decreased from 2.48 to 2.21 once implemented this logic with single attention head inside of it.
    def __init__(self, head_size, head_count):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(head_count)])
        self.proj = nn.Linear(n_embed, n_embed)
        
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out) ##residual pathway logic 
        return out  ##  (B,T,C)

class FeedForward(nn.Module): # think this layer as once tokens are understands each other, this layer gives tokens to think about what they have learned
    #it is also used for storing facts inside these neurons <== 3 blue 1 brown reference
    #tokens do not talk with each other at all in this layer, but they think individually about data they have learnt
    # 2.21 to 2.16 with this layer
    def __init__(self, n_embed):
        super().__init__()
        self.feed = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4), #times 4 is mentioned in the paper
            nn.ReLU(),
            nn.Linear(n_embed * 4 , n_embed), ##residual pathway logic
        )
    def forward(self, x):
        return self.feed(x)

class Block(nn.Module):
    ##2.16 to 2.044 ps:: i was looking wrong loss value so this is wrong :( rn it is 2.26 to val loss
    def __init__(self, n_embed, n_head):
        #n_embed number of embed dimensitons , n_head how many parallel attention blocks we want to run.
        super().__init__()
        head_size = n_embed // n_head
        self.sa_heads = MultiHead(head_size, n_head)
        self.ffl = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed) #just like batchnorm, paper suggest to normalize these output layers
        #BUT, commonly you apply layernorm before saheads and ffl but paper applies afterwards. we dont followed paper in this case
        #we dont want to compute any kind of running mean of normalization becuase we dont investigate single context's loss or overall loss
        #over training set, we just compute loss every 300 steps for another 10 different batch, see compute loss func. 
        #so running mean initiliazation is unneccesary for these normalizations.
        #took loss from 2.04 to 2.03
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        #this x plus is residual connection, instead of passing input x through sa heads and getting output from it, it also adds the initial input to it
        #to handle dissapearing gradients. we increase grad values by addition 1 to grad, but why does this make backpropogation optimal i have no idea
        #will look into this when finishing andrej's backprop ninja video. i didnt understood this part in video where it is 1;30;46
        ##BTW holy fuck, by applying this we decreased loss by 2.26 to 2.04 (these are real values) holy shit. 
        x = x + self.sa_heads(self.ln1(x)) 
        x = x + self.ffl(self.ln2(x))
        return x



class BiagramModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # self.sa_heads = MultiHead(head_size, n_embed//head_size ) # integer based division, resulting will be head sized concated tensors that each tensor gone through attention block,
        #this is kinda group convolutions
        #sa stands for self attention
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4)
        )
        self.ffwd = FeedForward(n_embed)
        self.lm_linear = nn.Linear(n_embed, vocab_size) #previously, embedding was (vocab_size, vocab_size), to convert logits back to vocab size, this layer introduced.
        #interesting result: when starting this and making this code only different from biagram model by putting this linear line made model much much more perform better.
        #and faster. Very interesting.
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C) => C is n embed
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B,T,C) 
        # x = self.sa_heads(x) # (B,T,C) ==>headsize is n embed
        x = self.blocks(x)# (B,T,C)
        # x = self.ffwd(x) # (B,T,C)
         
        logits = self.lm_linear(x) # (B, T, vocab_size)
        
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_tokens):
        # idx is (B, T) array of current context indicies wher B and T are 1 at start.
        for _ in range(max_tokens):
            idx_conc = idx[:, -block_size:] #we cannot put more than block size and should concancate once we get that limit
            #because token embedding table or position embeddding would run out when trying to generate embeddings for that tokens

            logits, loss = self(idx_conc)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = BiagramModel()
m = model.to(device) 

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(step_length):

    if iter % loss_interval == 0:
        losses = calculate_loss()
        print(f'step {iter} train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')

    xb, xy = get_batch('train')
    logits, loss = model(xb, xy)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    
print(decoder(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 200)[0].tolist())) 
