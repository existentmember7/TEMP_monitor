import torch
from torch.autograd import Variable


class ExtractFeature(object):
    """A function to be called in the val/test set, to extract features.
    Args:
      TVT: A callable to transfer images to specific device.
    """

    def __init__(self, model, TVT):
        self.model = model
        self.TVT = TVT

    def __call__(self, ims):
        old_train_eval_model = self.model.training
        # Set eval mode.
        # Force all BN layers to use global mean and variance, also disable
        # dropout.
        self.model.eval()
        ims = Variable(self.TVT(torch.from_numpy(ims).float()))
        global_feat, local_feat = self.model(ims)[:2]
        global_feat = global_feat.data.cpu().numpy()
        local_feat = local_feat.data.cpu().numpy()
        # Restore the model to its old train/eval mode.
        self.model.train(old_train_eval_model)
        return global_feat, local_feat