import torch

def l2_loss(input, target):
    loss = torch.square(target - input)
    loss = torch.mean(loss)
    return loss


def features_mask(features, domains, climate):
    mask = (domains == climate).nonzero()
    features[(domains == climate).nonzero()[0]] = torch.zeros_like(features[0])
    return features