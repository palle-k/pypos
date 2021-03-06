# PyPOS - Python Part-of-Speech tagger

    This is  a  project, which allows its  users to assign part of speech tags to words in a  sentence .
    dt   vbz dt nn     , wdt   vbz    prp$ nns   to vb     nn   in nn     nns  to nns   in dt nn       .
    
PyPOS uses Hidden Markov Models and Viterbi decoding to determine the most likely sequence of POS tags for a given sequence of words.

## Usage

### Installation
Requires Python 3.6 or higher

    pip3 install pypos

### Training
```python
from pypos import PartOfSpeechTagger, PartOfSpeechDataset
tagger = PartOfSpeechTagger()
ds = PartOfSpeechDataset.load('train.txt')
tagger.fit(ds).save('tagger.p')
```

### Tagging

```python
from pypos import PartOfSpeechTagger
tagger = PartOfSpeechTagger.load('tagger.p')

# Reproducing the results shown above:
sentence = 'This is a project, which allows its users to assign part of speech tags to words in a sentence.'
tokens = tagger.tokenize(sentence)
tags = tagger.tag(sentence, human_readable=False)
```
