class GloveTokenizer(object):
    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as f:
            self.vocab = {word: idx+2 for idx, word in enumerate(f.read().strip().split('\n'))}
        # self.vocab['[PAD]'] = 0
        self.vocab['[UNK]'] = 1

    def __call__(self, words):
        return [self.vocab.get(w, 0) for w in words.split()]


class SimpleTokenizer(object):
    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as f:
            self.vocab = f.read().split('\n')
        self.word_to_id, self.id_to_word = {}, {}
        for idx, word in enumerate(self.vocab):
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

    def __call__(self, words):
        return [self.word_to_id.get(w, self.word_to_id['<unk>']) for w in words]
    
    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False, return_list=False):
        words = [self.id_to_word.get(int(id), '<unk>') for id in ids]
        if skip_special_tokens:
            ww = []
            for w in words:
                if w == '<bos>':
                    continue
                if w == '<eos>':
                    break
                ww.append(w)
            if return_list:
                return ww
            else:
                return ' '.join(ww)
        if return_list:
            return words
        else:
            return ' '.join(words)