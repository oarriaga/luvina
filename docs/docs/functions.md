# Core functions

## Tokenization

```python
luvina.backend.tokenize(sentence, lowercase=True):
```
Just your basic tokenization operation.
 
### Example

```python
luv.tokenize('The cat sat on the mat.')
>>> ['the', 'cat', 'sat', 'on', 'the', 'mat', '.']
```


## Definitions

Get definitions from token using WordNet
```python
luvina.backend.get_definition(token):
```

### Example

```python
luv.get_definition('car')
>>> 'a motor vehicle with four wheels; usually propelled by an internal combustion engine'
```

## Stop words
Get a list of stop words (most common words).
```python
luvina.backend.get_stop_words(language='english'):
```

