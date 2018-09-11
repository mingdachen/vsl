import pickle
import argparse
import logging

from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser(
        description='Data Preprocessing for Twitter')
    parser.add_argument('--train', type=str, default=None,
                        help='train data path')
    parser.add_argument('--dev', type=str, default=None,
                        help='dev data path')
    parser.add_argument('--test', type=str, default=None,
                        help='test data path')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='training data ratio')
    args = parser.parse_args()
    return args


def process_file(data_file):
    logging.info("loading data from " + data_file + " ...")
    sents = []
    tags = []
    with open(data_file, 'r', encoding='utf-8') as df:
        for line in df.readlines():
            if line.strip():
                index = line.find('|||')
                if index == -1:
                    raise ValueError('Format Error')
                sent = line[: index - 1]
                tag = line[index + 4: -1]
                sents.append(sent.split(' '))
                tags.append(tag.split(' '))
    return sents, tags


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M')
    args = get_args()
    train = process_file(args.train)
    dev = process_file(args.dev)
    test = process_file(args.test)

    tag_set = set(sum([sum(d[1], []) for d in [train, dev, test]],
                  []))
    with open("twitter_tagfile", "w+", encoding='utf-8') as fp:
        fp.write('\n'.join(sorted(list(tag_set))))

    if args.ratio != 1:
        train_x, test_x, train_y, test_y = \
            train_test_split(train[0], train[1], test_size=args.ratio)
        train = [test_x, test_y]
        assert len(train_x) == len(train_y)

    logging.info("#train: {}".format(len(train[0])))
    logging.info("#dev: {}".format(len(dev[0])))
    logging.info("#test: {}".format(len(test[0])))

    pickle.dump(
        [train, dev, test],
        open("data/twitter{}.data".format(args.ratio), "wb+"), protocol=-1)
