# gumbel_sigmoid

A pytorch implementation of the gumbel sigmoid gate [paper]([https://duckduckgo.com](https://arxiv.org/pdf/1611.00712.pdf))

# example

```
logits = torch.randn(1000)
out = gumbel_sigmoid(logits, temp=0.6)
```
![gumbel softmax output density for normally distributed inputs (temperature = 0.6)](https://github.com/sirluk/gumbel_sigmoid/blob/main/example.jpg?raw=true)
