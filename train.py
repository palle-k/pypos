from .tagger import PartOfSpeechTagger


if __name__ == "__main__":
    tagger = PartOfSpeechTagger()
    ds = tagger.load_pos_dataset('train.txt')
    tagger.fit(ds)

