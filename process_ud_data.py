import pickle
import argparse
import logging

from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser(
        description='Data Preprocessing for Universal Dependencies')
    parser.add_argument('--train', type=str, default=None,
                        help='train data path')
    parser.add_argument('--dev', type=str, default=None,
                        help='dev data path')
    parser.add_argument('--test', type=str, default=None,
                        help='test data path')
    parser.add_argument('--output', type=str, default=None,
                        help='output file name')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='labeled data ratio')
    args = parser.parse_args()
    return args


def load_data(data_file):
    logging.info("loading data from {} ...".format(data_file))
    sents = []
    tags = []
    sent = []
    tag = []
    with open(data_file, 'r', encoding="utf-8") as f:
        for line in f:
            if line[0] == "#":
                continue
            if line.strip():
                token = line.strip("\n").split("\t")
                word = token[1]
                t = token[3]
                sent.append(word)
                tag.append(t)
            else:
                sents.append(sent)
                tags.append(tag)
                sent = []
                tag = []
    return sents, tags


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M')
    args = get_args()
    logging.info("##### training data #####")
    all_sents, all_tags = load_data(args.train)
    logging.info("random splitting training data with ratio of {}..."
                 .format(args.ratio))
    train_sents, unlabel_sents, train_tags, unlabel_tags = \
        train_test_split(all_sents, all_tags,
                         train_size=args.ratio, shuffle=True)
    logging.info("#train sents: {}, #train words: {}, #train tags: {}"
                 .format(len(train_sents), len(sum(train_sents, [])),
                         len(sum(train_tags, []))))
    logging.info("#unlabeled sents: {}"
                 .format(len(unlabel_sents)))
    logging.info("##### dev data #####")
    dev_sents, dev_tags = load_data(args.dev)
    logging.info("#dev sents: {}, #dev words: {}, #dev tags: {}"
                 .format(len(dev_sents), len(sum(dev_sents, [])),
                         len(sum(dev_tags, []))))
    logging.info("##### test data #####")
    test_sents, test_tags = load_data(args.test)
    logging.info("#dev sents: {}, #dev words: {}, #dev tags: {}"
                 .format(len(test_sents), len(sum(test_sents, [])),
                         len(sum(test_tags, []))))
    output = "data" if args.output is None else args.output
    output += ".ud"

    tag_set = set(sum([sum(d, []) for d in [all_tags, dev_tags, test_tags]],
                  []))
    with open("ud_tagfile", "w+", encoding='utf-8') as fp:
        fp.write('\n'.join(sorted(list(tag_set))))
    dataset = {"train": [train_sents, train_tags],
               "unlabel": [unlabel_sents, unlabel_tags],
               "dev": [dev_sents, dev_tags],
               "test": [test_sents, test_tags]}
    pickle.dump(dataset, open(output, "wb+"), protocol=-1)
    logging.info("data saved to {}".format(output))
