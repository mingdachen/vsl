import os
import pickle

import numpy as np

from collections import Counter

from decorators import auto_init_args, lazy_execute
from config import UNK_WORD_IDX, UNK_WORD, UNK_CHAR_IDX, \
    UNK_CHAR


class data_holder:
    @auto_init_args
    def __init__(self, train, dev, test, unlabel,
                 tag_vocab, vocab, char_vocab):
        self.inv_vocab = {i: w for w, i in vocab.items()}
        self.inv_tag_vocab = {i: w for w, i in tag_vocab.items()}


class data_processor:
    def __init__(self, experiment):
        self.expe = experiment

    def process(self):
        fn = "vocab_" + str(self.expe.config.vocab_size)
        vocab_file = os.path.join(self.expe.config.vocab_file, fn)

        self.expe.log.info("loading data from {} ...".format(
            self.expe.config.data_file))
        with open(self.expe.config.data_file, "rb+") as infile:
            train_data, dev_data, test_data = pickle.load(infile)

        train_v_data = train_data[0]
        unlabeled_data = None

        if self.expe.config.use_unlabel:
            unlabeled_data = self._load_sent(self.expe.config.unlabel_file)
            train_v_data = unlabeled_data + train_data[0]

        W, vocab, char_vocab = \
            self._build_vocab_from_embedding(
                train_v_data, dev_data[0] + test_data[0],
                self.expe.config.embed_file,
                self.expe.config.vocab_size, self.expe.config.char_vocab_size,
                file_name=vocab_file)

        tag_vocab = self._load_tag(self.expe.config.tag_file)

        self.expe.log.info("tag vocab size: {}".format(len(tag_vocab)))

        train_data = self._label_to_idx(
            train_data[0], train_data[1], vocab, char_vocab, tag_vocab)
        dev_data = self._label_to_idx(
            dev_data[0], dev_data[1], vocab, char_vocab, tag_vocab)
        test_data = self._label_to_idx(
            test_data[0], test_data[1], vocab, char_vocab, tag_vocab)

        def cal_stats(data):
            unk_count = 0
            total_count = 0
            leng = []
            for sent in data:
                leng.append(len(sent))
                for w in sent:
                    if w == UNK_WORD_IDX:
                        unk_count += 1
                    total_count += 1
            return (unk_count, total_count, unk_count / total_count), \
                (len(leng), max(leng), min(leng), sum(leng) / len(leng))

        train_unk_stats, train_len_stats = cal_stats(train_data[0])
        self.expe.log.info("#train data: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}"
                           .format(*train_len_stats))

        self.expe.log.info("#unk in train sentences: {}"
                           .format(train_unk_stats))

        dev_unk_stats, dev_len_stats = cal_stats(dev_data[0])
        self.expe.log.info("#dev data: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}"
                           .format(*dev_len_stats))

        self.expe.log.info("#unk in dev sentences: {}"
                           .format(dev_unk_stats))

        test_unk_stats, test_len_stats = cal_stats(test_data[0])
        self.expe.log.info("#test data: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}"
                           .format(*test_len_stats))

        self.expe.log.info("#unk in test sentences: {}"
                           .format(test_unk_stats))

        if self.expe.config.use_unlabel:
            unlabeled_data = self._unlabel_to_idx(
                unlabeled_data, vocab, char_vocab)
            un_unk_stats, un_len_stats = cal_stats(unlabeled_data[0])
            self.expe.log.info("#unlabeled data: {}, max len: {}, "
                               "min len: {}, avg len: {:.2f}"
                               .format(*un_len_stats))

            self.expe.log.info("#unk in unlabeled sentences: {}"
                               .format(un_unk_stats))

        data = data_holder(
            train=train_data,
            dev=dev_data,
            test=test_data,
            unlabel=unlabeled_data,
            tag_vocab=tag_vocab,
            vocab=vocab,
            char_vocab=char_vocab)

        return data, W

    def _load_tag(self, path):
        self.expe.log.info("loading tags from " + path)
        tag = {}
        with open(path, 'r') as f:
            for (n, i) in enumerate(f):
                tag[i.strip()] = n
        return tag

    def _load_sent(self, path):
        self.expe.log.info("loading data from " + path)
        sents = []
        with open(path, "r+", encoding='utf-8') as df:
            for line in df:
                if line.strip():
                    words = line.strip("\n").split(" ")
                    sents.append(words)
        return sents

    def _label_to_idx(self, sentences, tags, vocab, char_vocab, tag_vocab):
        sentence_holder = []
        sent_char_holder = []
        tag_holder = []
        for sentence, tag in zip(sentences, tags):
            chars = []
            words = []
            for w in sentence:
                words.append(vocab.get(w, 0))
                chars.append([char_vocab.get(c, 0) for c in w])
            sentence_holder.append(words)
            sent_char_holder.append(chars)
            tag_holder.append([tag_vocab[t] for t in tag])
        self.expe.log.info("#sent: {}".format(len(sentence_holder)))
        self.expe.log.info("#word: {}".format(len(sum(sentence_holder, []))))
        self.expe.log.info("#tag: {}".format(len(sum(tag_holder, []))))
        return np.asarray(sentence_holder), np.asarray(sent_char_holder), \
            np.asarray(tag_holder)

    def _unlabel_to_idx(self, sentences, vocab, char_vocab):
        sentence_holder = []
        sent_char_holder = []
        for sentence in sentences:
            chars = []
            words = []
            for w in sentence:
                words.append(vocab.get(w, 0))
                chars.append([char_vocab.get(c, 0) for c in w])
            sentence_holder.append(words)
            sent_char_holder.append(chars)
        self.expe.log.info("#sent: {}".format(len(sentence_holder)))
        return np.asarray(sentence_holder), np.asarray(sent_char_holder)

    def _load_twitter_embedding(self, path):
        with open(path, 'r', encoding='utf8') as fp:
            # word_vectors: word --> vector
            word_vectors = {}
            for line in fp:
                line = line.strip("\n").split("\t")
                word_vectors[line[0]] = np.array(
                    list(map(float, line[1].split(" "))), dtype='float32')
        vocab_embed = word_vectors.keys()
        embed_dim = word_vectors[next(iter(vocab_embed))].shape[0]

        return word_vectors, vocab_embed, embed_dim

    def _load_glove_embedding(self, path):
        with open(path, 'r', encoding='utf8') as fp:
            # word_vectors: word --> vector
            word_vectors = {}
            for line in fp:
                line = line.strip("\n").split(" ")
                word_vectors[line[0]] = np.array(
                    list(map(float, line[1:])), dtype='float32')
        vocab_embed = word_vectors.keys()
        embed_dim = word_vectors[next(iter(vocab_embed))].shape[0]

        return word_vectors, vocab_embed, embed_dim

    def _load_ud_embedding(self, path):
        from gensim.models.keyedvectors import KeyedVectors
        word_vectors = KeyedVectors.load_word2vec_format(path, binary=True)
        vocab_embed = word_vectors.vocab.keys()
        embed_dim = word_vectors[next(iter(vocab_embed))].shape[0]
        return word_vectors, vocab_embed, embed_dim

    def _build_vocab_from_data(self, train_sents, devtest_sents):
        self.expe.log.info("vocab file not exist, start building")
        train_char_vocab = Counter()
        train_vocab = Counter()
        for sent in train_sents:
            for w in sent:
                train_vocab[w] += 1
                for c in w:
                    train_char_vocab[c] += 1
        devtest_vocab = Counter()
        for sent in devtest_sents:
            for w in sent:
                devtest_vocab[w] += 1

        return train_char_vocab, train_vocab, devtest_vocab

    @lazy_execute("_load_from_pickle")
    def _build_vocab_from_embedding(
            self, train_sents, devtest_sents, embed_file,
            vocab_size, char_vocab_size, file_name):
        self.expe.log.info("loading embedding file from {}".format(embed_file))
        if self.expe.config.embed_type.lower() == "glove":
            word_vectors, vocab_embed, embed_dim = \
                self._load_glove_embedding(embed_file)
        elif self.expe.config.embed_type.lower() == "twitter":
            word_vectors, vocab_embed, embed_dim = \
                self._load_twitter_embedding(embed_file)
        else:
            word_vectors, vocab_embed, embed_dim = \
                self._load_ud_embedding(embed_file)

        train_char_vocab, train_vocab, devtest_vocab = \
            self._build_vocab_from_data(train_sents, devtest_sents)

        word_vocab = train_vocab + devtest_vocab

        char_ls = train_char_vocab.most_common(char_vocab_size)
        self.expe.log.info('#Chars: {}'.format(len(char_ls)))
        for key in char_ls[:5]:
            self.expe.log.info(key)
        self.expe.log.info('...')
        for key in char_ls[-5:]:
            self.expe.log.info(key)
        char_vocab = {c[0]: index + 1 for (index, c) in enumerate(char_ls)}

        char_vocab[UNK_CHAR] = UNK_CHAR_IDX

        self.expe.log.info("char vocab size: {}".format(len(char_vocab)))

        vocab = {UNK_WORD: UNK_WORD_IDX}
        W = [np.random.uniform(-0.1, 0.1, size=(1, embed_dim))]
        n = 0
        for w, c in sorted(word_vocab.items(), key=lambda x: -x[1]):
            if w in vocab_embed:
                W.append(word_vectors[w][None, :])
                vocab[w] = n + 1
                n += 1
            elif w.lower() in vocab_embed:
                W.append(word_vectors[w.lower()][None, :])
                vocab[w] = n + 1
                n += 1
        W = np.concatenate(W, axis=0).astype('float32')

        self.expe.log.info(
            "{}/{} words are initialized with loaded embeddings."
            .format(n, len(vocab)))
        return W, vocab, char_vocab

    def _load_from_pickle(self, file_name):
        self.expe.log.info("loading from {}".format(file_name))
        with open(file_name, "rb") as fp:
            data = pickle.load(fp)
        return data


class minibatcher:
    @auto_init_args
    def __init__(self, word_data, char_data, label, batch_size, shuffle):
        self._reset()

    def __len__(self):
        return len(self.idx_pool)

    def _reset(self):
        self.pointer = 0
        idx_list = np.arange(len(self.word_data))
        if self.shuffle:
            np.random.shuffle(idx_list)
        self.idx_pool = [idx_list[i: i + self.batch_size]
                         for i in range(0, len(self.word_data),
                         self.batch_size)]

    def _pad(self, word_data, char_data, labels):
        max_word_len = max([len(sent) for sent in word_data])
        max_char_len = max([len(char) for sent in char_data
                           for char in sent])

        input_data = \
            np.zeros((len(word_data), max_word_len)).astype("float32")
        input_mask = \
            np.zeros((len(word_data), max_word_len)).astype("float32")
        input_char = \
            np.zeros(
                (len(word_data), max_word_len, max_char_len)).astype("float32")
        input_char_mask = \
            np.zeros(
                (len(word_data), max_word_len, max_char_len)).astype("float32")
        input_label = \
            np.zeros((len(word_data), max_word_len)).astype("float32")

        for i, (sent, chars, label) in enumerate(
                zip(word_data, char_data, labels)):
            input_data[i, :len(sent)] = \
                np.asarray(list(sent)).astype("float32")
            input_label[i, :len(label)] = \
                np.asarray(list(label)).astype("float32")
            input_mask[i, :len(sent)] = 1.

            for k, char in enumerate(chars):
                input_char[i, k, :len(char)] = \
                    np.asarray(char).astype("float32")
                input_char_mask[i, k, :len(char)] = 1

        return [input_data, input_mask, input_char,
                input_char_mask, input_label]

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.idx_pool):
            self._reset()
            raise StopIteration()

        idx = self.idx_pool[self.pointer]
        sents, chars, label = \
            self.word_data[idx], self.char_data[idx], self.label[idx]

        self.pointer += 1
        return self._pad(sents, chars, label) + [idx]
