import torch
import pylib as py
import tqdm
import numpy as np
import torchvision
from torchvision.transforms import transforms
import utils
from PIL import Image
import os
from generate_features import ImageLoader
import re

py.arg('--experiment_name', default='AttGAN_128_MIT_STATES_FINAL')
args_ = py.args()


# output_dir
output_dir = py.join('./output', args_.experiment_name)

# save settings
# args = py.args_from_yaml(py.join(output_dir, 'settings.yml'))
# args.__dict__.update(args_.__dict__)



class GenerateFeatures():
    def __init__(self):
        self.save_dir_eval=  py.join(output_dir,'eval_testing_2')
        self.save_dir_reconstructed = py.join(output_dir,'eval_testing_reconstructed_2')
        self.out_file_eval = py.join(output_dir,'eval_testing_2.t7')
        self.out_file_reconstructed = py.join(output_dir, 'eval_testing_reconstructed_2.t7')

        self.loader = ImageLoader(None)
        self.transform = utils.imagenet_transform()

    def get_files(self):
        #save_dir_eval = './data/mit-states-original/test_imgs_compose_AE'
        test_images_generated = os.listdir(self.save_dir_eval)
        test_images_generated.sort(key=lambda f: int(re.sub('\D', '', f)))

        test_imgages_full_path = [py.join(self.save_dir_eval, img) for img in test_images_generated]
        return test_imgages_full_path

    def generate_features(self, feat_extractor):
       # self.out_file = './data/mit-states-original/features_test_compose_AE.t7'
        data = self.get_files()
        # data = self.train_data+self.test_data<
        # transform = data_utils.imagenet_transform('test', transform_type)

        if feat_extractor is None:
            feat_extractor = torchvision.models.resnet18(pretrained=True)
            feat_extractor.fc = torch.nn.Sequential()
        feat_extractor.eval().cpu()

        image_feats = []
        image_files = []
        for chunk in tqdm.tqdm(utils.chunks(data, 4), total=len(data) // 4):
            files = chunk
            imgs = list(map(self.loader, files))
            imgs = list(map(self.transform, imgs))
            feats = feat_extractor(torch.stack(imgs, 0).cpu())
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, 0)
        print('features for %d images generated' % (len(image_files)))

        torch.save({'features': image_feats, 'files': image_files}, self.out_file_eval)



features = GenerateFeatures()
features.generate_features(None)