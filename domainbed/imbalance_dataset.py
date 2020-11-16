
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
import pandas as pd
import datetime
from domainbed.lib.imbalance import *
from domainbed.datasets import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Imbalance
    "ImbalanceDomainNet"
]


class ImbalanceDomainNet(DomainNet):
    CHECKPOINT_FREQ = 10
    # N_STEPS = 1 # debug
    N_STEPS = 500
    ENVIRONMENTS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    ORIGINDATA = 'DomainNet'

    def __init__(self, root, trainorval,running_targets, hparams):
        # root is default dg dataset path.
        # target_domains is target domains list by number of sorted domain list
        super().__init__(root, running_targets + hparams['targets_fix'], hparams)
        params = (hparams['imb_data_root'],hparams['dataset_version'], self.ORIGINDATA, hparams['numcls'], hparams['testrate'], hparams['valrate'],
        hparams['targets_fix'])
        for x in running_targets + [hparams['minor_domain']]:
            assert (x in [i for i in range(len(self.ENVIRONMENTS))]), 'target of minor domain index does not correct'
            assert (x not in hparams['targets_fix']), 'running target or minor domain is in fixed target domain.'
        assert (hparams['minor_domain'] not in running_targets), 'minor_domain in running target domain!!'

        self.imbtrain_csv_path = get_imb_csvpath_bysetting(params, running_targets, 'imbtrain', minor=hparams['minor_domain'],imbrate=hparams['imbrate'],clsordom=hparams['clsordom'])
        self.val_csv_path = get_imb_csvpath_bysetting(params, running_targets, 'val', minor=hparams['minor_domain'],imbrate=hparams['imbrate'],clsordom=hparams['clsordom'])

        if not os.path.isfile(self.val_csv_path):  # if val csv does not exist for this setting.
            print(self.val_csv_path, ' does not exist, so make the val.csv, val.csv file')
            make_trteval_csv_bysetting(params,root,running_targets)

        if not os.path.isfile(self.imbtrain_csv_path):  # if imbtrain csv does not exist for this setting.
            print(self.imbtrain_csv_path, ' does not exist, so make the trainval.csv, test.csv file')
            make_imbtrain_csv_bysetting(params,running_targets,hparams['minor_domain'],hparams['imbrate'],hparams['clsordom'])

        # read the imb & val csv.
        if trainorval == 'train':
            temp_df = pd.read_csv(self.imbtrain_csv_path)
        elif trainorval == 'val':
            temp_df = pd.read_csv(self.val_csv_path)

        # changing mother's self.datasets instances.
        for idx, domain in enumerate(self.ENVIRONMENTS):
            dom_df = temp_df[temp_df['dom'] == domain]
            self.datasets[idx].classes = dom_df.groupby('cls').count().index.to_list()
            self.datasets[idx].class_to_idx = {k: v for v, k in enumerate(self.datasets[idx].classes)}
            self.datasets[idx].imgs = [(p, self.datasets[idx].class_to_idx[c]) for d, c, p in dom_df.values.tolist()]
            self.datasets[idx].samples = list.copy(self.datasets[idx].imgs)
            self.datasets[idx].targets = [c_id for p, c_id in self.datasets[idx].imgs]

        # target domain을 self.dataset에서 없애기.
        tempdatasetlist = []
        tempdomnamelist =[]
        for idx, domain in enumerate(self.datasets):
            if len(domain)>0:
                tempdatasetlist.append(domain)
                tempdomnamelist.append(self.ENVIRONMENTS[idx])
        self.datasets = tempdatasetlist
        self.ENVIRONMENTS = tempdomnamelist
        self.num_classes = len(self.datasets[-1].classes)

        del temp_df,dom_df
