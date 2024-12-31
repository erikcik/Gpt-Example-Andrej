import torch
import torch.nn as nn
import torch.nn.functional as F

block_size = 8
batch_size = 4
step_length = 10000
loss_iteration = 200
loss_interval = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'

text = open('input.txt', 'r', encoding='utf-8').read()

#tokenizer
vocab_size = sorted(list(set(text)))
vocab_length = len(vocab_size)
print(f'We have vocab size of: {len(vocab_size)}')

stoi = {s : i for i, s in enumerate(vocab_size)}
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
print(f'inputs: {xb}') 
print(f'outputs: {xy}')

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

class BiagramModel(nn.Module):
    def __init__(self):
        
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_length, vocab_length) #from second lesson, we initiliez one hot representation to all indexes, but now we are doing it for embedding
        #the reason x dim and y dim should be vocab length is that for cross entropy loss, the compared values after softmax is targets indexes and logits embedding values.
        # and xdim is vocab length because the iterated indexes are within the interval of 0 and vocab length.
    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx) #B T C, it will generate batch size, block_size, vocab_size // assigning vocab sized embed to each index
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            logits, loss = self(idx)
            # getting last index of array and getting that time dimensions element
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            ## when you dont do dim 1 in here very strangely it closes down cursor, strange. but this line basically does concatting tensors on dim 1
            ## because they output as:: [[2]] [[23, 33]]
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = BiagramModel()
m = model.to(device) #moving parameters of model to device 

optimizer = torch.optim.Adam(model.parameters())

for iter in range(step_length):

    if iter % loss_interval == 0:
        losses = calculate_loss()
        print(f'step {iter} train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')



    xb, xy = get_batch('train')
    logits, loss = model(xb, xy)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    
print(loss.item())

print(decoder(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 200)[0].tolist())) #device is selected for creating initial matrix
#there is issue in init loss becuase we expect loss to be -ln(1/65) --> 4.174 but rn we get 4.588 which is we need to normalize init weights.




