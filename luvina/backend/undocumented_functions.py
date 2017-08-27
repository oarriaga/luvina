def calculate_synonyms_similarities(synonyms):
    similarities = []
    for word_1, word_2 in synonyms.items():
        if len(word_2) == 0:
            continue
        elif len(word_2) >= 1:
            selected_word_2 = word_2[0]
        vector_1 = get_vector(word_1)
        vector_2 = get_vector(selected_word_2)
        if word_1 == selected_word_2:
            similarity = 1.
            print(word_1)
        elif calculate_norm(vector_1) == 0 or calculate_norm(vector_2) == 0:
            continue
        else:
            similarity = calculate_cosine_similarity(vector_1, vector_2)
        similarities.append(similarity)
    return similarities


def find_synonyms(reference_sentence, hypothesis_sentence, return_tokens=True):
    """
    Inputs: two lists of tokenized string sentences
    Returns: A dictionary with keys being the word in the reference_sentence
    and the values a list of the arguments where there is a synonym in the
    hypothesis_sentence.
    """
    synonyms_connections = OrderedDict()
    for reference_token in reference_sentence:
        reference_synonyms = list(set(get_synonyms(reference_token)))
        reference_synonyms = reference_synonyms + [reference_token]
        hypothesis_args = []
        for hypothesis_arg, hypothesis_token in enumerate(hypothesis_sentence):
            for reference_synonym in reference_synonyms:
                words_are_synonyms = hypothesis_token == reference_synonym
                words_are_the_same = hypothesis_token == reference_token
                if words_are_the_same:
                    if return_tokens:
                        hypothesis_args.append(hypothesis_token)
                    else:
                        hypothesis_args.append(hypothesis_arg)
                    break
                elif words_are_synonyms:
                    if return_tokens:
                        hypothesis_args.append(hypothesis_token)
                    else:
                        hypothesis_args.append(hypothesis_arg)

        synonyms_connections[reference_token] = hypothesis_args
    return synonyms_connections


def find_similarities(reference_sentence, hypothesis_sentence,
                      return_tokens=True):
    """
    Inputs: two lists of tokenized string sentences
    Returns: A dictionary with keys being the word in the reference_sentence
    and the values a list of the arguments where there is a synonym in the
    hypothesis_sentence.
    """
    synonyms_connections = OrderedDict()
    for reference_token in reference_sentence:
        reference_synonyms = list(set(get_synonyms(reference_token)))
        reference_synonyms = reference_synonyms + [reference_token]
        hypothesis_args = []
        for hypothesis_arg, hypothesis_token in enumerate(hypothesis_sentence):
            for reference_synonym in reference_synonyms:
                words_are_synonyms = hypothesis_token == reference_synonym
                words_are_the_same = hypothesis_token == reference_token
                if words_are_the_same:
                    if return_tokens:
                        hypothesis_args.append(hypothesis_token)
                    else:
                        hypothesis_args.append(hypothesis_arg)
                    break
                elif words_are_synonyms:
                    if return_tokens:
                        hypothesis_args.append(hypothesis_token)
                    else:
                        hypothesis_args.append(hypothesis_arg)

        synonyms_connections[reference_token] = hypothesis_args
    return synonyms_connections


def edit(hypothesis_sentence, token_connections,
         start_string='**', end_string='**'):
    edited_hypothesis = []
    hypothesis_highlights = list(chain(*list(token_connections.values())))
    for hypothesis_token in hypothesis_sentence:
        if hypothesis_token in hypothesis_highlights:
            hypothesis_token = start_string + hypothesis_token + end_string
        edited_hypothesis.append(hypothesis_token)
    edited_hypothesis = ' '.join(edited_hypothesis)
    return edited_hypothesis


