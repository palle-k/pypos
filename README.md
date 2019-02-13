# PyPOS - Python POS tagger

    This is  a  project, which allows its  users to assign part of speech tags to words in a  sentence .
    dt   vbz dt nn     , wdt   vbz    prp$ nns   to vb     nn   in nn     nns  to nns   in dt nn       .
    
PyPOS uses Hidden Markov Models and Viterbi decoding to determine the most likely sequence of POS tags for a given sequence of words.

## Usage

### Training
```python
from tagger import PartOfSpeechTagger
tagger = PartOfSpeechTagger()
ds = tagger.load_pos_dataset('train.txt')
tagger.fit(ds).save('tagger.p')

```

### Tagging

```python
from tagger import PartOfSpeechTagger
tagger = PartOfSpeechTagger.load('tagger.p')

# Reproducing the results shown above:
sentence = 'This is a project, which allows its users to assign part of speech tags to words in a sentence.'
tokens = tagger.tokenize(sentence)
tags = tagger.tag(sentence, human_readable=False)
```
