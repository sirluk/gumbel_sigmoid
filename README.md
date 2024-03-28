# gumbel_sigmoid

A pytorch implementation of the gumbel sigmoid gate [paper]([https://duckduckgo.com](https://arxiv.org/pdf/1611.00712.pdf))

# example

```
logits = torch.randn(1000)
out = gumbel_sigmoid(logits, temp=0.6)
```

