import numpy as np
from hmm import HMM
import pickle
import re

tagnames = {
    'CC': 'Coordinating conjunction',
    'CD': 'Cardinal number',
    'DT': 'Determiner',
    'EX': 'Existential there',
    'FW': 'Foreign word',
    'IN': 'Preposition or subordinating conjunction',
    'JJ': 'Adjective',
    'JJR': 'Adjective, comparative',
    'JJS': 'Adjective, superlative',
    'LS': 'List item marker',
    'MD': 'Modal',
    'NN': 'Noun, singular or mass',
    'NNS': 'Noun, plural',
    'NNP': 'Proper noun, singular',
    'NNPS': 'Proper noun, plural',
    'PDT': 'Predeterminer',
    'POS': 'Possessive ending',
    'PRP': 'Personal pronoun',
    'PRP$': 'Possessive pronoun',
    'RB': 'Adverb',
    'RBR': 'Adverb, comparative',
    'RBS': 'Adverb, superlative',
    'RP': 'Particle',
    'SYM': 'Symbol',
    'TO': 'to',
    'UH': 'Interjection',
    'VB': 'Verb, base form',
    'VBD': 'Verb, past tense',
    'VBG': 'Verb, gerund or present participle',
    'VBN': 'Verb, past participle',
    'VBP': 'Verb, non-3rd person singular present',
    'VBZ': 'Verb, 3rd person singular present',
    'WDT': 'Wh-determiner',
    'WP': 'Wh-pronoun',
    'WP$': 'Possessive wh-pronoun',
    'WRB': 'Wh-adverb',
    ':': ':',
    '.': '.',
    ';': ';',
    ',': ',',
    '?': '?',
    '!': '!',
    '(': '(',
    ')': ')'
}


def map_sentence(sentence):
    return [line.split(' ')[0:2] for line in sentence]


def tag_sentence(word_to_idx, idx_to_tag, model: HMM, sentence: str):
    sentence = re.sub(r"([.,:!?])", r" \1", sentence)
    sentence = re.sub(r'[^a-zA-Z0-9.,:;!?\s]', '', sentence)
    sentence = re.sub(r'\d+', r'<num>', sentence)
    words = sentence.lower().split(' ')
    words = [word_to_idx[word] for word in words if word != '']
    tagged = model.predict(np.array(words))
    return [idx_to_tag[tag] for tag in tagged]


def make_tagger(location, ds_location):
    with open('hmm.p', 'rb') as f:
        hmm = pickle.load(f)
    idx_to_word, word_to_idx, idx_to_tag, tag_to_idx, sentences = load(ds_location)

    return lambda s: tag_sentence(word_to_idx, idx_to_tag, hmm, s)


def load(filepath):
    with open(filepath, 'r') as f:
        contents: str = f.read().lower()
        contents = re.sub(r'\d+', r'<num>', contents)
        lines = contents.splitlines()
    sentences = [[]]
    for x in lines:
        if len(x.strip()) == 0:
            sentences.append([])
        else:
            sentences[-1].append(x)

    del lines, contents

    if len(sentences[-1]) == 0:
        sentences = sentences[:-1]

    sentences = [map_sentence(s) for s in sentences]

    words = set()
    tags = set()

    for sentence in sentences:
        for word, tag in sentence:
            words.add(word)
            tags.add(tag)

    idx_to_word = dict(enumerate(sorted(words)))
    word_to_idx = dict((word, idx) for idx, word in idx_to_word.items())
    idx_to_tag = dict(enumerate(sorted(tags)))
    tag_to_idx = dict((tag, idx) for idx, tag in idx_to_tag.items())

    return idx_to_word, word_to_idx, idx_to_tag, tag_to_idx, sentences


if __name__ == "__main__":
    # m = HMM(2, 3)
    # m.start_prob_ = np.array([2 / 3, 1 / 3])
    # m.emission_prob_ = np.array([[0.25, 0.25, 0.5], [0.75, 0.25, 0]])
    # m.transition_prob_ = np.array([[0.75, 0.25], [1 / 3, 2 / 3]])
    # print(m.predict(np.array([1, 2, 2])))
    # print(m.predict(np.array([0, 2, 2])))
    # print(m.predict(np.array([0, 0, 2])))
    # print(m.predict(np.array([0, 0, 0])))
    # print(m.predict(np.array([2, 2, 2])))

    idx_to_word, word_to_idx, idx_to_tag, tag_to_idx, sentences = load('train.txt')

    x = np.array([word_to_idx[word] for sentence in sentences for word, _ in sentence], dtype=np.int)
    y = np.array([tag_to_idx[tag] for sentence in sentences for _, tag in sentence], dtype=np.int)
    l = np.array([len(sentence) for sentence in sentences])

    model = HMM(len(idx_to_tag), len(idx_to_word))
    model.fit(x, y, l)

    with open('hmm.p', 'wb') as f:
        pickle.dump(model, f)
