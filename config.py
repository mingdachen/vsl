import argparse


UNK_WORD_IDX = 0
UNK_WORD = "UUUNKKK"
UNK_CHAR_IDX = 0
UNK_CHAR = "UUUNKKK"


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_parser():
    parser = argparse.ArgumentParser(
        description='Variational Sequential Labelers \
        for Semi-supervised learning')
    parser.register('type', 'bool', str2bool)

    basic_group = parser.add_argument_group('basics')
    basic_group.add_argument('--debug', type="bool", default=False,
                             help='whether to activate debug mode \
                             (default: False)')
    basic_group.add_argument('--model', type=str, default='g',
                             choices=['g', 'flat', 'hier'],
                             help='type of model (default: g)')
    basic_group.add_argument('--random_seed', type=int, default=0,
                             help='Random seed (default: 0)')

    data = parser.add_argument_group('data')
    data.add_argument('--prefix', type=str, default=None,
                      help='save file prefix (default: None)')
    data.add_argument('--data_file', type=str, default=None,
                      help='path to training data file (default: None)')
    data.add_argument('--unlabel_file', type=str, default=None,
                      help='path to unlabeled file (default: None)')
    data.add_argument('--vocab_file', type=str, default=None,
                      help='path to vocab file (default: None)')
    data.add_argument('--tag_file', type=str, default=None,
                      help='path to tag file (default: None)')
    data.add_argument('--embed_file', type=str, default=None,
                      help='path to embedding file (default: None)')
    data.add_argument('--use_unlabel', type="bool", default=None,
                      help='whether to use unlabeled data (default: None)')
    data.add_argument('--prior_file', type=str, default=None,
                      help='path to saved prior file (default: None)')

    data.add_argument('--embed_type', type=str, default='twitter',
                      choices=['glove', 'twitter', 'ud'],
                      help='types of embedding file (default: twitter)')

    config = parser.add_argument_group('configs')
    config.add_argument('-edim', '--embed_dim',
                        dest='edim', type=int, default=100,
                        help='embedding dimension (default: 100)')
    config.add_argument('-rtype', '--rnn_type',
                        dest='rtype', type=str, default='gru',
                        choices=['gru', 'lstm', 'rnn'],
                        help='types of optimizer: gru (default), lstm, rnn')
    config.add_argument('-tw', '--tie_weights',
                        dest='tw', type='bool', default=True,
                        help='whether to tie weights (default: True)')

    # Character level model detail
    config.add_argument('-cdim', '--char_embed_dim',
                        dest='cdim', type=int, default=15,
                        help='character embedding dimension (default: 15)')
    config.add_argument('-chsize', '--char_hidden_size',
                        dest='chsize', type=int, default=15,
                        help='character rnn hidden size (default: 15)')

    # Latent variable specs
    config.add_argument('-zsize', '--latent_z_size',
                        dest='zsize', type=int, default=100,
                        help='dimension of latent variable (default: 100)')
    config.add_argument('-ysize', '--latent_y_size',
                        dest='ysize', type=int, default=25,
                        help='dimension of latent variable (default: 25)')
    config.add_argument('-rsize', '--rnn_size',
                        dest='rsize', type=int, default=100,
                        help='dimension of recurrent nnet (default: 100)')
    config.add_argument('-mhsize', '--mlp_hidden_size',
                        dest='mhsize', type=int, default=100,
                        help='hidden dimension of feedforward nnet \
                        (default: 100)')
    config.add_argument('-mlayer', '--mlp_layer',
                        dest='mlayer', type=int, default=2,
                        help='number of layers of feedforward nnet \
                        (default: 2)')
    config.add_argument('-xvar', '--latent_x_logvar',
                        dest='xvar', type=int, default=1e-3,
                        help='log varaicne of latent variable x \
                        (default: 1e-3)')

    # KL annealing
    config.add_argument('-klr', '--kl_anneal_rate',
                        dest='klr', type=float, default=1e-3,
                        help='annealing rate (default: 1e-3)')
    # Loss specs
    config.add_argument('-ur', '--unlabel_ratio',
                        dest='ur', type=float, default=0.1,
                        help='unlabeled loss ratio (default: 0.1)')
    config.add_argument('-ufl', '--update_freq_label',
                        dest='ufl', type=int, default=1,
                        help='frequency of updating prior for labeled data \
                        (default: 1)')
    config.add_argument('-ufu', '--update_freq_unlabel',
                        dest='ufu', type=int, default=1,
                        help='frequency of updating prior for unlabeled data \
                        (default: 1)')

    train = parser.add_argument_group('training')
    train.add_argument('--opt', type=str, default='adam',
                       choices=['adam', 'sgd', 'rmsprop'],
                       help='types of optimizer: adam (default), \
                       sgd, rmsprop')
    train.add_argument('--n_iter', type=int, default=30000,
                       help='number of iteration (default: 30000)')
    train.add_argument('--batch_size', type=int, default=10,
                       help='labeled data batch size (default: 10)')
    train.add_argument('--unlabel_batch_size', type=int, default=10,
                       help='unlabeled data batch size (default: 10)')
    train.add_argument('--vocab_size', type=int, default=50000,
                       help='maximum number of words in vocabulary \
                       (default: 50000)')
    config.add_argument('--char_vocab_size', type=int, default=300,
                        help='character vocabulary size (default: 300)')
    train.add_argument('--train_emb', type="bool", default=False,
                       help='whether to train word embedding (default: False)')
    train.add_argument('--save_prior', type="bool", default=False,
                       help='whether to save trained prior (default: False)')
    train.add_argument('-lr', '--learning_rate',
                       dest='lr',
                       type=float, default=1e-3,
                       help='learning rate (default: 1e-3)')
    train.add_argument('--l2', type=float, default=0,
                       help='weight decay rate (default: 0)')
    train.add_argument('--grad_clip', type=float, default=10.,
                       help='gradient clipping (default: 10)')
    train.add_argument('--f1_score', type="bool", default=False,
                       help='whether to report F1 score (default: False)')

    misc = parser.add_argument_group('misc')
    misc.add_argument('--print_every', type=int, default=10,
                      help='print training details after \
                      this number of iterations (default: 10)')
    misc.add_argument('--eval_every', type=int, default=100,
                      help='evaluate model after \
                      this number of iterations (default: 100)')
    misc.add_argument('--summarize', type="bool", default=False,
                      help='whether to summarize training stats\
                      (default: False)')
    return parser
