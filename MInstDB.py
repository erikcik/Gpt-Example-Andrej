import tensorflow as tf
import torch 
import torch.nn.functional as F


mnist = tf.keras.datasets.mnist
(xdata, xdatalbl), (ydata, ydatalbl) = mnist.load_data()

ximg = torch.tensor(xdata).flatten(start_dim=1) / 255.0
yimg = torch.tensor(ydata).flatten(start_dim=1) / 255.0
xlab = torch.tensor(xdatalbl).long()
ylab = torch.tensor(ydatalbl)
numc = len(set([xlab[i].item() for i in range(len(xlab))]))
xlabenc = F.one_hot(xlab, num_classes=numc)

g = torch.Generator()
W = torch.randn(784, 10, requires_grad=True)


#the whole logic less go
for _ in range(100):
    #forward pass
    logits = ximg @ W
    prob = logits.exp()
    prob = prob / prob.sum(1, keepdim=True)
    loss = -prob[torch.arange(prob.shape[0]), xlab].log().mean()
    print(loss.item())

    #backward pass
    W.grad = None
    loss.backward()

    #grad descent
    W.data += -1 * W.grad

logits = yimg @ W
print(logits.shape)
prob = logits.exp()
prob = prob / prob.sum(1, keepdim=True)
accuracy = (prob.argmax(1) == ylab).float().mean()
print(accuracy.item())
#### this implementation of mine completely is gives accuracy of 0.81