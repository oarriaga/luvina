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

## Lemmatize

```python
luvina.backend.lemmatize(token, pos='n'):
```
Basic lemmatization using a given part-of-speech.

```python
luv.lemmatize('driving', pos='v')
>>> 'drive'
```

## Expanding contractions

```python
luvina.backend.expand_contractions(sentence):
```
Expand English contractions found in sentence/token:


```python
luv.expand_contractions("don't eat it")
>>> 'do not eat it'
```

## POS tagging

```python
luvina.backend.tag_pos(tokens):
```
POS tagging using a pre-trained neural network from NLTK.


```python
luv.tag_pos(luv.tokenize('a person is driving'))
>>> [('a', 'DT'), ('person', 'NN'), ('is', 'VBZ'), ('driving', 'VBG')]
```

## Levenshtein distance

```python
luvina.backend.calculate_levenshtein_distance(token_1, token_2):
```
Returns number of characters that need to be substituted, inserted or deleted from "token_1" to get "token_2".
 
```python
luv.calculate_levenshtein_distance('base','vase')
>>> 1
```

## Check dictionary

```python
luvina.backend.in_dictionary(token)
```
Returns True if the given token is found in the enchant dictionary. Returns False otherwise.

```python
luv.in_dictionary('book')
>>> True
```

## Suggest words

```python
luvina.backend.suggest_words(token)
```
Returns a list of similar tokens.

```python
luv.suggest_words('better')[:5]
>>> ['better', 'netter', 'betters', 'bester', 'setter']
```

## Get vector

```python
luvina.backend.get_vector(token)
```

Get glove word vector representation of a token.

```python
luv.get_vector('space')
```

## Get vectors

```python
luvina.backend.get_vectors(tokens)
```

Returns a list of glove word vector representation of a list of tokens.

```python
luv.get_vectors(['I','was', 'here']) 
```


## Calculate norm

```python
luvina.backend.calculate_norm(vector)
```
Calculates the norm of a vector.


```python
luv.calculate_norm(luv.get_vector('mouse'))
>>> 7.2006297
```

## Calculate cosine similarity

```python
luvina.backend.calculate_cosine_similarity(vector_1, vector_2)
```
Calculates the cosine similarity between two vectors.

```python
vector_1 = luv.get_vector('cup')
vector_2 = luv.get_vector('plate')
luv.calculate_cosine_similarity(vector_1, vector_2)
>>> 0.4010278
```

## Get related words

```python
luvina.backend.get_related_words(token, max_num=3)
```

Use glove embeddings to get closer word vectors.

```python
luv.get_related_words('animal', 3)
>>> ['animals', 'pet', 'dog']
```


<!---
missing documentation from NLTK backend:
remove_repeated_characters
filter_tokens
make_ngrams

missing documentation from common backend
-->


