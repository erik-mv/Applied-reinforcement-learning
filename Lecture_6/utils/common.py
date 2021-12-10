import copy
import random
import torch
import numpy as np
import random


def hyperparameter_random_sample(config, name2variants, count):
    lens = [len(variants) for _, variants in name2variants]
    param_ids = set()
    while len(param_ids) < count:
        sampled = tuple([random.randint(0, i - 1) for i in lens])
        param_ids.add(sampled)

    configs = []
    for index, ids in enumerate(param_ids):
        cfg = copy.deepcopy(config)
        for i, id in enumerate(ids):
            name, variants = name2variants[i]
            pre_names = list(name)[:-1]
            name = name[-1]

            # Find hyperparameter in config
            f = cfg
            for key in pre_names:
                if key not in f:
                    f[key] = {}
                f = f[key]

            # Set it
            f[name] = variants[id]
        cfg["index"] = index
        configs.append(cfg)
    return configs


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def soft_update(target, source, tau):
    for param, target_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
