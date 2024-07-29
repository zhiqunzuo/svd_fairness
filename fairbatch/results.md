Equal Opportunity

Model:

```python
nn.Sequential(
    nn.Linear(3, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)
```

| | Accuracy | EO disparity|
|--|---|---|
|Original Model|0.884|0.122|
|FairBatch|0.867|0.036|

Select X1 and X2 in this way: 

```python
x1 = xz_test[(z_test == 0.0) & (y_test == 1.0)]
x2 = xz_test[(z_test == 1.0) & (y_test == 1.0)]

min_num = min(x1.size()[0], x2.size()[0])
x1 = x1[:min_num]
x2 = x2[:min_num]
```

| Output Norm | 1 layer | 2 layer | 3 layer |
|--|---|---|---|
|Original Model|92.31|100.48|30.38|
|FairBatch|91.54|94.03|26.89|

Select X1 and X2 in this way:

```python
x1 = torch.mean(xz_test[(z_test == 0.0) & (y_test == 1.0)], dim=0).unsqueeze(0)
x2 = torch.mean(xz_test[(z_test == 1.0) & (y_test == 1.0)], dim=0).unsqueeze(0)
```

| Output Norm | 1 layer | 2 layer | 3 layer |
|--|---|---|---|
|Original Model|3.35|3.81|0.842|
|FairBatch|3.35|3.44|0.079|

Select X1 and X2 in this way:

```python
def select_k_similar_pairs(x1, x2, k):
    distances = pairwise_euclidean_distance(x1, x2)

    flatten_distances = distances.flatten()

    top_k_indices = torch.topk(flatten_distances, k, largest=False)[1]

    selected_x1 = torch.zeros((k, x1.size()[1]))
    selected_x2 = torch.zeros((k, x2.size()[1]))

    for i in range(k):
        selected_x1[i] = x1[int(top_k_indices[i] // distances.size()[1])]
        selected_x2[i] = x2[int(top_k_indices[i] % distances.size()[1])]

    return selected_x1, selected_x2

x1 = xz_test[(z_test == 0.0) & (y_test == 1.0)]
x2 = xz_test[(z_test == 1.0) & (y_test == 1.0)]

x1, x2 = select_k_similar_pairs(x1, x2, k=100)
```

| Output Norm | 1 layer | 2 layer | 3 layer |
|--|---|---|---|
|Original Model|14.52|16.00|4.31|
|FairBatch|15.91|19.17|11.34|

If we choose k = 1,

| Output Norm | 1 layer | 2 layer | 3 layer |
|--|---|---|---|
|Original Model|||0.556|
|FairBatch|||1.296|

When we test the maximum pairs,

| Output Norm | 1 layer | 2 layer | 3 layer |
|--|---|---|---|
|Original Model|20.32|20.75|8.23|
|FairBatch|19.98|20.09|8.23|

With the mean vector and apply svd on the third layer, accuracy: 0.820, EO disparity: 0.038. (Need to adjust c value for every different seed.)

While select to suppress the largest pairs, accuracy: 0.821, EO disparity: 0.047. (Still the problem, performed very bad for some random seeds.)