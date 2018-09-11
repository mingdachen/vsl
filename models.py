import os
import torch
import model_utils

import torch.nn as nn
import stochastic_layers as sl
import torch.nn.functional as F

from decorators import auto_init_pytorch
from torch.autograd import Variable


class base(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, embed_dim,
                 embed_init, n_tags, experiment):
        super(base, self).__init__()
        self.expe = experiment
        self.use_cuda = self.expe.config.use_cuda
        self.char_encoder = model_utils.char_rnn(
            rnn_type=self.expe.config.rtype,
            vocab_size=char_vocab_size,
            embed_dim=self.expe.config.cdim,
            hidden_size=self.expe.config.chsize)

        self.word_embed = nn.Embedding(word_vocab_size, embed_dim)

        if embed_init is not None:
            self.word_embed.weight.data.copy_(torch.from_numpy(embed_init))
            self.expe.log.info("Initialized with pretrained word embedding")
        if not self.expe.config.train_emb:
            self.word_embed.weight.requires_grad = False
            self.expe.log.info("Word Embedding not trainable")

        self.word_encoder = model_utils.get_rnn(self.expe.config.rtype)(
            input_size=(embed_dim + 2 * self.expe.config.chsize),
            hidden_size=self.expe.config.rsize,
            bidirectional=True,
            batch_first=True)

        self.x2token = nn.Linear(
            embed_dim, word_vocab_size, bias=False)

        if self.expe.config.tw:
            self.x2token.weight = self.word_embed.weight

    def get_input_vecs(self, data, mask, char, char_mask):
        word_emb = self.word_embed(data.long())

        char_input = self.char_encoder(char, char_mask, mask)
        data_emb = torch.cat([char_input, word_emb], dim=-1)

        return data_emb

    def to_var(self, inputs):
        if self.use_cuda:
            if isinstance(inputs, Variable):
                inputs = inputs.cuda()
                inputs.volatile = self.volatile
                return inputs
            else:
                if not torch.is_tensor(inputs):
                    inputs = torch.from_numpy(inputs)
                return Variable(inputs.cuda(), volatile=self.volatile)
        else:
            if isinstance(inputs, Variable):
                inputs = inputs.cpu()
                inputs.volatile = self.volatile
                return inputs
            else:
                if not torch.is_tensor(inputs):
                    inputs = torch.from_numpy(inputs)
                return Variable(inputs, volatile=self.volatile)

    def to_vars(self, *inputs):
        return [self.to_var(inputs_) if inputs_ is not None and inputs_.size
                else None for inputs_ in inputs]

    def optimize(self, loss):
        self.opt.zero_grad()
        loss.backward()
        if self.expe.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm(
                self.parameters(), self.expe.config.grad_clip)
        self.opt.step()

    def init_optimizer(self, opt_type, learning_rate, weight_decay):
        if opt_type.lower() == "adam":
            optimizer = torch.optim.Adam
        elif opt_type.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop
        elif opt_type.lower() == "sgd":
            optimizer = torch.optim.SGD
        else:
            raise NotImplementedError("invalid optimizer: {}".format(opt_type))

        opt = optimizer(
            params=filter(
                lambda p: p.requires_grad, self.parameters()
            ),
            lr=learning_rate,
            weight_decay=weight_decay)
        return opt

    def save(self, dev_perf, test_perf, iteration):
        save_path = os.path.join(self.expe.experiment_dir, "model.ckpt")
        checkpoint = {
            "dev_perf": dev_perf,
            "test_perf": test_perf,
            "iteration": iteration,
            "state_dict": self.state_dict(),
            "config": self.expe.config
        }
        torch.save(checkpoint, save_path)
        self.expe.log.info("model saved to {}".format(save_path))

    def load(self, checkpointed_state_dict=None):
        if checkpointed_state_dict is None:
            save_path = os.path.join(self.expe.experiment_dir, "model.ckpt")
            checkpoint = torch.load(save_path,
                                    map_location=lambda storage,
                                    loc: storage)
            self.load_state_dict(checkpoint['state_dict'])
            self.expe.log.info("model loaded from {}".format(save_path))
        else:
            self.load_state_dict(checkpointed_state_dict)
            self.expe.log.info("model loaded!")

    @property
    def volatile(self):
        return not self.training

    @property
    def sampling(self):
        return self.training


class vsl_g(base):
    @auto_init_pytorch
    def __init__(self, word_vocab_size, char_vocab_size, embed_dim,
                 embed_init, n_tags, experiment):
        super(vsl_g, self).__init__(
            word_vocab_size, char_vocab_size, embed_dim, embed_init,
            n_tags, experiment)
        assert self.expe.config.model.lower() == "g"
        self.to_latent_variable = sl.gaussian_layer(
            input_size=2 * self.expe.config.rsize,
            latent_z_size=self.expe.config.zsize)

        self.classifier = nn.Linear(self.expe.config.zsize, n_tags)

        self.z2x = model_utils.get_mlp_layer(
            input_size=self.expe.config.zsize,
            hidden_size=self.expe.config.mhsize,
            output_size=embed_dim,
            n_layer=self.expe.config.mlayer)

    def forward(
            self, data, mask, char, char_mask, label,
            prior_mean, prior_logvar, kl_temp):
        data, mask, char, char_mask, label, prior_mean, prior_logvar = \
            self.to_vars(data, mask, char, char_mask, label,
                         prior_mean, prior_logvar)

        batch_size, batch_len = data.size()
        input_vecs = self.get_input_vecs(data, mask, char, char_mask)
        hidden_vecs, _, _ = model_utils.get_rnn_output(
            input_vecs, mask, self.word_encoder)

        z, mean_qs, logvar_qs = \
            self.to_latent_variable(hidden_vecs, mask, self.sampling)

        mean_x = self.z2x(z)

        x = model_utils.gaussian(
            mean_x, Variable(mean_x.data.new(1).fill_(self.expe.config.xvar)))

        x_pred = self.x2token(x)

        if label is None:
            sup_loss = class_logits = None
        else:
            class_logits = self.classifier(z)
            sup_loss = F.cross_entropy(
                class_logits.view(batch_size * batch_len, -1),
                label.view(-1).long(),
                reduce=False).view_as(data) * mask
            sup_loss = sup_loss.sum(-1) / mask.sum(-1)

        log_loss = F.cross_entropy(
            x_pred.view(batch_size * batch_len, -1),
            data.view(-1).long(),
            reduce=False).view_as(data) * mask
        log_loss = log_loss.sum(-1) / mask.sum(-1)

        if prior_mean is not None and prior_logvar is not None:
            kl_div = model_utils.compute_KL_div(
                mean_qs, logvar_qs, prior_mean, prior_logvar)

            kl_div = (kl_div * mask.unsqueeze(-1)).sum(-1)
            kl_div = kl_div.sum(-1) / mask.sum(-1)

            loss = log_loss + kl_temp * kl_div
        else:
            kl_div = None
            loss = log_loss

        if sup_loss is not None:
            loss = loss + sup_loss

        return loss.mean(), log_loss.mean(), \
            kl_div.mean() if kl_div is not None else None, \
            sup_loss.mean() if sup_loss is not None else None, \
            mean_qs, logvar_qs, \
            class_logits.data.cpu().numpy().argmax(-1) \
            if class_logits is not None else None


class vsl_gg(base):
    @auto_init_pytorch
    def __init__(self, word_vocab_size, char_vocab_size, embed_dim,
                 embed_init, n_tags, experiment):
        super(vsl_gg, self).__init__(
            word_vocab_size, char_vocab_size, embed_dim, embed_init,
            n_tags, experiment)
        if self.expe.config.model.lower() == "flat":
            self.to_latent_variable = sl.gaussian_flat_layer(
                input_size=2 * self.expe.config.rsize,
                latent_z_size=self.expe.config.zsize,
                latent_y_size=self.expe.config.ysize)
            yzsize = self.expe.config.zsize + self.expe.config.ysize
        elif self.expe.config.model.lower() == "hier":
            self.to_latent_variable = sl.gaussian_hier_layer(
                input_size=2 * self.expe.config.rsize,
                latent_z_size=self.expe.config.zsize,
                latent_y_size=self.expe.config.ysize)
            yzsize = self.expe.config.zsize
        else:
            raise ValueError(
                "invalid model type: {}".format(self.expe.config.model))

        self.classifier = nn.Linear(self.expe.config.ysize, n_tags)

        self.yz2x = model_utils.get_mlp_layer(
            input_size=yzsize,
            hidden_size=self.expe.config.mhsize,
            output_size=embed_dim,
            n_layer=self.expe.config.mlayer)

    def forward(
            self, data, mask, char, char_mask, label,
            prior_mean, prior_logvar, kl_temp):
        if prior_mean is not None:
            prior_mean1, prior_mean2 = prior_mean
            prior_logvar1, prior_logvar2 = prior_logvar
        else:
            prior_mean1 = prior_mean2 = prior_logvar1 = prior_logvar2 = None

        data, mask, char, char_mask, label, prior_mean1, \
            prior_mean2, prior_logvar1, prior_logvar2 = \
            self.to_vars(data, mask, char, char_mask, label,
                         prior_mean1, prior_mean2,
                         prior_logvar1, prior_logvar2)

        batch_size, batch_len = data.size()
        input_vecs = self.get_input_vecs(data, mask, char, char_mask)
        hidden_vecs, _, _ = model_utils.get_rnn_output(
            input_vecs, mask, self.word_encoder)

        z, y, mean_qs, logvar_qs, mean2_qs, logvar2_qs = \
            self.to_latent_variable(hidden_vecs, mask, self.sampling)

        if self.expe.config.model.lower() == "flat":
            yz = torch.cat([z, y], dim=-1)
        elif self.expe.config.model.lower() == "hier":
            yz = z

        mean_x = self.yz2x(yz)

        x = model_utils.gaussian(
            mean_x, Variable(mean_x.data.new(1).fill_(self.expe.config.xvar)))

        x_pred = self.x2token(x)

        if label is None:
            sup_loss = class_logits = None
        else:
            class_logits = self.classifier(y)
            sup_loss = F.cross_entropy(
                class_logits.view(batch_size * batch_len, -1),
                label.view(-1).long(),
                reduce=False).view_as(data) * mask
            sup_loss = sup_loss.sum(-1) / mask.sum(-1)

        log_loss = F.cross_entropy(
            x_pred.view(batch_size * batch_len, -1),
            data.view(-1).long(),
            reduce=False).view_as(data) * mask
        log_loss = log_loss.sum(-1) / mask.sum(-1)

        if prior_mean is not None:
            kl_div1 = model_utils.compute_KL_div(
                mean_qs, logvar_qs, prior_mean1, prior_logvar1)
            kl_div2 = model_utils.compute_KL_div(
                mean2_qs, logvar2_qs, prior_mean2, prior_logvar2)

            kl_div = (kl_div1 * mask.unsqueeze(-1)).sum(-1) + \
                (kl_div2 * mask.unsqueeze(-1)).sum(-1)
            kl_div = kl_div.sum(-1) / mask.sum(-1)

            loss = log_loss + kl_temp * kl_div
        else:
            kl_div = None
            loss = log_loss

        if sup_loss is not None:
            loss = loss + sup_loss

        return loss.mean(), log_loss.mean(), \
            kl_div.mean() if kl_div is not None else None, \
            sup_loss.mean() if sup_loss is not None else None, \
            mean_qs, logvar_qs, mean2_qs, logvar2_qs, \
            class_logits.data.cpu().numpy().argmax(-1) \
            if class_logits is not None else None
