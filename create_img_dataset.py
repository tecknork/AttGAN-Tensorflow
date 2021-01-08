import os
from data import MitStatesDataSet
import torchvision
import torch
from collections import defaultdict
import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tflib as tl
import tqdm
from generate_features import Features
import data
import module
import utils
from generate_features import  ImageLoader
from operator import add
import re



sess = tl.session()
sess.__enter__()  # make default

# output_dir

# save settings
# args = py.args_from_yaml(py.join(output_dir, 'settings.yml'))
# args.__dict__.update(args_.__dict__)

save_dir_eval ='./data/mit-states-original/test_imgs_compose_AE'
#save_dir_reconstructed = py.join(output_dir,'eval_testing_reconstructed_2')
py.mkdir(save_dir_eval)
# others

data_set = MitStatesDataSet(training=False)
test_data = data_set.test_data
img_deck,len_img_deck = data_set.get_image_dataset(test_data)
test_iter = img_deck.make_one_shot_iterator()

test_next = test_iter.get_next()
cnt = 0;
for _ in tqdm.trange(len_img_deck):
    xa,_,_ = sess.run(test_next)
    x_opt_list = [xa]
    sample = np.transpose(x_opt_list, (1, 2, 0, 3, 4))
    sample = np.reshape(sample, (sample.shape[0], -1, sample.shape[2] * sample.shape[3], sample.shape[4]))
    for _, s in enumerate(sample):
        cnt += 1
        # modified img with a attribute with b
        im.imwrite(s, '%s/%d.jpg' % (save_dir_eval, cnt))

print(len(test_data))
print(len_img_deck)




class GenerateFeatures():
    def __init__(self):
        self.loader = ImageLoader(None)
        self.transform = utils.imagenet_transform()

    def get_files(self):
        save_dir_eval = './data/mit-states-original/test_imgs_compose_AE'
        test_images_generated = os.listdir(save_dir_eval)
        test_imgages_full_path = [py.join(save_dir_eval, img) for img in test_images_generated]
        test_imgages_full_path.sort(key=lambda f: int(re.sub('\D', '', f)))
        return test_imgages_full_path

    def generate_features(self, feat_extractor):
        self.out_file = './data/mit-states-original/features_test_compose_AE.t7'
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

        torch.save({'features': image_feats, 'files': image_files}, self.out_file)



features = GenerateFeatures()
features.generate_features(None)