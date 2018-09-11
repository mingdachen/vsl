import inspect
import pickle
import os


def auto_init_args(init):
    def new_init(self, *args, **kwargs):
        arg_dict = inspect.signature(init).parameters
        arg_names = list(arg_dict.keys())[1:]  # skip self
        proc_names = set()
        for name, arg in zip(arg_names, args):
            setattr(self, name, arg)
            proc_names.add(name)
        for name, arg in kwargs.items():
            setattr(self, name, arg)
            proc_names.add(name)
        remain_names = set(arg_names) - proc_names
        if len(remain_names):
            for name in remain_names:
                setattr(self, name, arg_dict[name].default)
        init(self, *args, **kwargs)

    return new_init


def auto_init_pytorch(init):
    def new_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.opt = self.init_optimizer(
            self.expe.config.opt,
            self.expe.config.lr,
            self.expe.config.l2)

        if self.use_cuda:
            self.cuda()
            self.expe.log.info("transferred model to gpu")

    return new_init


class lazy_execute:
    @auto_init_args
    def __init__(self, func_name):
        pass

    def __call__(self, fn):
        func_name = self.func_name

        def new_fn(self, *args, **kwargs):
            if os.path.isfile(kwargs['file_name']):
                return getattr(self, func_name)(kwargs['file_name'])
            else:
                data = fn(self, *args, **kwargs)

                self.expe.log.info("saving to {}"
                                   .format(kwargs['file_name']))
                with open(kwargs['file_name'], "wb+") as fp:
                    pickle.dump(data, fp, protocol=-1)
                return data
        return new_fn
