import time
import logging
import argparse
import os
import pickle

import numpy as np

from config import get_parser
from data_utils import minibatcher
from decorators import auto_init_args


def register_exit_handler(exit_handler):
    import atexit
    import signal

    atexit.register(exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)
    signal.signal(signal.SIGINT, exit_handler)


def get_kl_temp(kl_anneal_rate, curr_iteration, max_temp):
    temp = np.exp(kl_anneal_rate * curr_iteration) - 1.
    return float(np.minimum(temp, max_temp))


class tracker:
    @auto_init_args
    def __init__(self, names):
        assert len(names) > 0
        self.reset()

    def __getitem__(self, name):
        return self.values.get(name, 0) / self.counter if self.counter else 0

    def __len__(self):
        return len(self.names)

    def reset(self):
        self.values = dict({name: 0. for name in self.names})
        self.counter = 0
        self.create_time = time.time()

    def update(self, named_values, count):
        """
        named_values: dictionary with each item as name: value
        """
        self.counter += count
        for name, value in named_values.items():
            self.values[name] += value.data.cpu().numpy()[0] * count

    def summarize(self, output=""):
        if output:
            output += ", "
        for name in self.names:
            output += "{}: {:.3f}, ".format(
                name, self.values[name] / self.counter if self.counter else 0)
        output += "elapsed time: {:.1f}(s)".format(
            time.time() - self.create_time)
        return output

    @property
    def stats(self):
        return {n: v / self.counter if self.counter else 0
                for n, v in self.values.items()}


class experiment:
    @auto_init_args
    def __init__(self, config, experiments_prefix, logfile_name="log"):
        """Create a new Experiment instance.

        Modified based on: https://github.com/ex4sperans/mag

        Args:
            logfile_name: str, naming for log file. This can be useful to
                separate logs for different runs on the same experiment
            experiments_prefix: str, a prefix to the path where
                experiment will be saved
        """

        # get all defaults
        all_defaults = {}
        for key in vars(config):
            all_defaults[key] = get_parser().get_default(key)

        self.default_config = all_defaults

        if not config.debug:
            if os.path.isdir(self.experiment_dir):
                raise ValueError("log exists: {}".format(self.experiment_dir))

            print(config)
            self._makedir()

        self._make_misc_dir()

    def _makedir(self):
        os.makedirs(self.experiment_dir, exist_ok=False)

    def _make_misc_dir(self):
        os.makedirs(self.config.prior_file, exist_ok=True)
        os.makedirs(self.config.vocab_file, exist_ok=True)

    @property
    def experiment_dir(self):
        if self.config.debug:
            return "./"
        else:
            # get namespace for each group of args
            arg_g = dict()
            for group in get_parser()._action_groups:
                group_d = {a.dest: self.default_config.get(a.dest, None)
                           for a in group._group_actions}
                arg_g[group.title] = argparse.Namespace(**group_d)

            # skip default value
            identifier = ""
            for key, value in sorted(vars(arg_g["configs"]).items()):
                if getattr(self.config, key) != value:
                    identifier += key + str(getattr(self.config, key))
            return os.path.join(self.experiments_prefix, identifier)

    @property
    def log_file(self):
        return os.path.join(self.experiment_dir, self.logfile_name)

    def register_directory(self, dirname):
        directory = os.path.join(self.experiment_dir, dirname)
        os.makedirs(directory, exist_ok=True)
        setattr(self, dirname, directory)

    def _register_existing_directories(self):
        for item in os.listdir(self.experiment_dir):
            fullpath = os.path.join(self.experiment_dir, item)
            if os.path.isdir(fullpath):
                setattr(self, item, fullpath)

    def __enter__(self):

        if self.config.debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%m-%d %H:%M')
        else:
            print("log saving to", self.log_file)
            logging.basicConfig(
                filename=self.log_file,
                filemode='w+', level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%m-%d %H:%M')

        self.log = logging.getLogger()
        return self

    def __exit__(self, *args):
        logging.shutdown()


class prior_buffer:
    def __init__(self, inputs, dim, freq, name, experiment, init_path=None):
        self.dim = dim
        self.freq = freq
        self.expe = experiment
        if init_path is not None:
            self.path = os.path.join(init_path, name + "_" + str(dim))
        else:
            self.path = init_path
        if self.path is None or not os.path.isfile(self.path):
            self.buffer = np.asarray(
                [np.zeros((len(r), dim)).astype('float32') for r in inputs])
        elif self.path is not None and os.path.isfile(self.path):
            self.buffer = self.load()
        else:
            raise ValueError(
                "invalid initial path for prior buffer: {}".format(init_path))
        self.count = [0] * len(inputs)

    def __len__(self):
        return len(self.buffer)

    def update_buffer(self, ixs, post, seq_len):
        """
        Args:
        ixs: list of index
        post: batch size x batch length x dim
        seq_len: batch size
        """
        for p, i, l in zip(post.data.cpu().numpy(), ixs, seq_len):
            new_i = i % len(self)
            if self.count[new_i] % self.freq == 0:
                assert len(self.buffer[new_i]) == l
                self.buffer[new_i] = p[:int(l), :]
            self.count[new_i] += 1

    def __getitem__(self, ixs):
        get_buffer = self.buffer[ixs]
        max_len = np.max([len(b) for b in get_buffer])
        batch_size = len(ixs)

        pad_buffer = np.zeros((batch_size, max_len, self.dim)) \
            .astype("float32")
        for i, b in enumerate(get_buffer):
            pad_buffer[i, :len(b), :] = b
        return pad_buffer

    def save(self):
        pickle.dump(self.buffer, open(self.path, "wb+"), protocol=-1)
        self.expe.log.info("prior saved to: {}".format(self.path))

    def load(self):
        with open(self.path, "rb+") as infile:
            priors = pickle.load(infile)
        self.expe.log.info("prior loaded from: {}".format(self.path))
        return priors


class accuracy_reporter:
    def __init__(self):
        self.right_count = 0
        self.instance_count = 0

    def update(self, pred, label, mask):
        self.right_count += ((pred == label) * mask).sum()
        self.instance_count += mask.sum()

    def report(self):
        acc = self.right_count / self.instance_count \
            if self.instance_count else 0.0
        return {"acc": acc, "f1": 0., "prec": 0., "rec": 0.}, acc


class f1_reporter:
    """
    modified based on: https://github.com/kimiyoung/transfer/blob/master/ner_span.py
    """
    def __init__(self, inv_tag_vocab):
        self.inv_tag_vocab = inv_tag_vocab
        self.instance_count = 0
        self.right_count = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    @staticmethod
    def extract_ent(y, m, inv_tag_vocab):
        def label_decode(label):
            if label == 'O':
                return 'O', 'O'
            return tuple(label.split('-'))

        def new_match(y_prev, y_next):
            l_prev, l_next = inv_tag_vocab[y_prev], inv_tag_vocab[y_next]
            c1_prev, c2_prev = label_decode(l_prev)
            c1_next, c2_next = label_decode(l_next)
            if c2_prev != c2_next:
                return False
            if c1_next not in ['I', 'E']:
                return False
            return True

        ret = set()
        i = 0
        while i < y.shape[0]:
            if m[i] == 0:
                i += 1
                continue
            c1, c2 = label_decode(inv_tag_vocab[y[i]])
            if c1 in ['O', 'I', 'E']:
                i += 1
                continue
            if c1 == 'S':
                ret.add((i, i + 1, c2))
                i += 1
                continue
            j = i + 1
            if j == y.shape[0]:
                break
            end = False
            while m[j] != 0 and not end and new_match(y[i], y[j]):
                ic1, ic2 = label_decode(inv_tag_vocab[y[j]])
                if ic1 == 'E':
                    end = True
                    break
                j += 1
            if not end:
                i += 1
                continue
            ret.add((i, j, c2))
            i = j
        return ret

    def update(self, pred, label, mask):
        pred, label, mask = pred.flatten(), label.flatten(), mask.flatten()
        self.right_count += ((label == pred) * mask).sum()
        self.instance_count += mask.sum()

        p_ent = f1_reporter.extract_ent(pred, mask, self.inv_tag_vocab)
        y_ent = f1_reporter.extract_ent(label, mask, self.inv_tag_vocab)

        for ent in p_ent:
            if ent in y_ent:
                self.tp += 1
            else:
                self.fp += 1
        for ent in y_ent:
            if ent not in p_ent:
                self.fn += 1

    def report(self):
        acc = self.right_count / self.instance_count \
            if self.instance_count else 0.0
        prec = 1.0 * self.tp / (self.tp + self.fp) \
            if self.tp + self.fp > 0 else 0.0
        recall = 1.0 * self.tp / (self.tp + self.fn) \
            if self.tp + self.fn > 0 else 0.0
        f1 = 2.0 * prec * recall / (prec + recall) \
            if prec + recall > 0 else 0.0
        return {"acc": acc, "f1": f1, "prec": prec, "rec": recall}, f1


class evaluator:
    @auto_init_args
    def __init__(self, inv_tag_vocab, model, experiment):
        self.expe = experiment

    def evaluate(self, data):
        self.model.eval()
        eval_stats = tracker(["log_loss"])
        if self.expe.config.f1_score:
            reporter = f1_reporter(self.inv_tag_vocab)
        else:
            reporter = accuracy_reporter()
        for data, mask, char, char_mask, label, _ in \
            minibatcher(
                word_data=data[0],
                char_data=data[1],
                label=data[2],
                batch_size=100,
                shuffle=False):
            outputs = self.model(data, mask, char, char_mask,
                                 label, None, None, 1.0)
            pred, log_loss = outputs[-1], outputs[1]
            reporter.update(pred, label, mask)

            eval_stats.update(
                {"log_loss": log_loss}, mask.sum())
        perf, res = reporter.report()
        summary = eval_stats.summarize(
            ", ".join([x[0] + ": {:.5f}".format(x[1])
                      for x in sorted(perf.items())]))
        self.expe.log.info(summary)
        return perf, res
