import string

blank = '_'
alphabet = list(string.ascii_lowercase)
units = ["'", ' '] + alphabet + [blank]

class CharTokenizer:
    def __init__(self):
        self.unit2id = {k: v for v, k in enumerate(units)}
        self.id2unit = {v: k for v, k in enumerate(units)}


    def to_id(self, unit):
        return self.unit2id[unit]

    def to_unit(self, unit):
        return self.id2unit[unit]

    def text_to_ids(self, text):
        return [self.unit2id[char] for char in text]

    def ids_to_text(self, ids):
        return "".join([self.id2unit[i] for i in ids])

    def add_blank(self, sequence_ids):
        blank_id = self.unit2id[blank]
        out_sequence = []

        for ind in sequence_ids:
            out_sequence = out_sequence + [blank_id, ind]

        out_sequence.append(blank_id)

        return out_sequence

    def clean_text(self, text):
        return ''.join([char.lower().translate(str.maketrans('', '', string.punctuation)) for char in text])

    def get_vocab_size(self):
        return len(units)

def main():
    text_utt = 'my name is cat'
    tokenizer = CharTokenizer()
    ids = tokenizer.text_to_ids(text_utt)

    print(ids)
    print(tokenizer.add_blank(ids))
    print(tokenizer.clean_text('MuMu M!'))

    text_utt = tokenizer.clean_text(text_utt)
    utt_ids = tokenizer.text_to_ids(text_utt)
    utt_ids = tokenizer.add_blank(utt_ids)

    print(utt_ids)

if __name__ == '__main__':
    main()
