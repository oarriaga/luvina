# Core functions

## Tokenization

```python
luvina.backend.tokenize(sentence, lowercase=True)
```
Just your basic tokenization operation.
 
```python
luv.tokenize('The cat sat on the mat.')
>>> ['the', 'cat', 'sat', 'on', 'the', 'mat', '.']
```


## Definitions

Get definitions from token using WordNet.
```python
luvina.backend.get_definition(token)
```

```python
luv.get_definition('car')
>>> 'a motor vehicle with four wheels; usually propelled by an internal combustion engine'
```

## Synonyms
```python
luvina.backend.get_synonyms(token)
```

Get synonyms of the given token using WordNet.

```python
luv.get_synonyms('wallet')
>>> ['billfold', 'notecase', 'pocketbook']
```

## Stop words
```python
luvina.backend.get_stop_words(language='english'):
```

Get a set of stop words (most common words) in the given language.

```python
list(luv.get_stop_words())[:5]
>>> ['doing', 'than', 'up', 'wouldn', 'this']
```

## Calculate WordNet similarity

```python
luvina.backend.calculate_wordnet_similarity(token_1, token_2):
```
Returns a score based on the shortest path between the senses in the is-a (hypernym/hypnoym) taxonomy.

```python
luv.calculate_wordnet_similarity('dog','cat')
>>> 0.8571428571428571
```

### Lemmatize

```python
luvina.backend.lemmatize(token, pos='n'):
```
Basic lemmatization using a given part-of-speech.

```python
luv.lemmatize('driving', pos='v')
>>> 'drive'
```

### Expanding contractions

```python
luvina.backend.expand_contractions(sentence):
```
Expand English contractions found in sentence/token:


```python
luv.expand_contractions("don't eat it")
>>> 'do not eat it'
```

### POS tagging

```python
luvina.backend.tag_pos(tokens):
```
POS tagging using a pre-trained neural network from NLTK.


```python
luv.tag_pos(luv.tokenize('a person is driving'))
>>> [('a', 'DT'), ('person', 'NN'), ('is', 'VBZ'), ('driving', 'VBG')]
```

### Levenshtein distance

```python
luvina.backend.calculate_levenshtein_distance(token_1, token_2):
```
Returns number of characters that need to be substituted, inserted or deleted from "token_1" to get "token_2".
 
```python
luv.calculate_levenshtein_distance('base','vase')
>>> 1
```


<!---
missing documentation from NLTK backend:
remove_repeated_characters
filter_tokens
make_ngrams
-->
