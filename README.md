# Stellargraph cheatsheet

## Create graph

```python
import stellargraph as sg
```

```python

features = sg.IndexedArray(np.array([[0.2, 2.1], [0.2, 0.84], [0.6, 1], [1.2, 0.7]]), index=np.array([0,1,2,3]))

edges_dict = {
    "source": [0, 1, 2, 1],
    "target": [1, 2, 3, 3],
    "weight": [1, 1, 1.2, 1.5]
}

edges = pd.DataFrame(edges_dict)


```
```python
g = sg.StellarGraph(nodes=features, edges=edges)
```

## GraphSAGE


```python

# Node generator for GraphSAGE models. Specify numbers of samples per layer.
# If weighted=False, the samples are uniformly sampled from node neighbourhood.
generator = sg.mapper.GraphSAGENodeGenerator(g, num_samples=[20, 20, 20], batch_size=64, weighted=False)

# GraphSAGE layer.
gcn = sg.layer.GraphSAGE(
    layer_sizes=[8, 8, 8],
    activations=["relu", "relu", "relu"], 
    generator=generator,
    dropout=0.05
)
```
### Aggregators

