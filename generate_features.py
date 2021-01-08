import torch
import pylib as py
import tqdm
import numpy as np
import torchvision
from torchvision.transforms import transforms
import utils
from PIL import Image

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        str_types = [str]
        try:
            str_types.append(unicode)
        except NameError:
            pass

        if type(img) in str_types:
            f = '%s'%( img)
            img = Image.open(f).convert('RGB')
        elif type(img) in [list, tuple]:
            f = '%s'%(img[0])
            x,y,w,h = img[1:]  # bbox
            img = Image.open(f).convert('RGB')
            img = img.crop((x, y, x+w, y+h))
        else:
            raise NotImplementedError(str(type(img)))
        return img

class Features():

    def __init__(self):
        self.root = "./data/mit-states-original"
        self.img_path = "./data/mit-states-original/images"
        self.file_path = "./data/mit-states-original/features_new.t7" #all images
        self.eval_img_path = "./data/mit-states-original/"
        self.transform = utils.imagenet_transform()
        self.feat_extractor = torchvision.models.resnet18(pretrained=True)
        self.feat_extractor.fc = torch.nn.Sequential()
        self.feat_extractor.eval().cpu()

    def get_dataset_features(self,dataset):
        activation_data = torch.load(self.file_path)
        activations = []
        self.activation_dict = dict(zip(activation_data['files'], activation_data['features']))
        self.feat_dim = activation_data['features'].size(1)
        print(self.feat_dim)
        for data in dataset:
            activations.append(self.activation_dict[data[0]].cpu().detach().numpy())

        return np.array(activations)
       # return [activation.cpu().detach().numpy() for activation in activations]

    def get_dataset_features_V2(self):
        path = "./data/mit-states-original/features_test_compose_AE.t7" #all images
        activation_data = torch.load(path)
        activations = []
        #self.activation_dict = dict(zip(activation_data['files'], activation_data['features']))
        #self.feat_dim = activation_data['features'].size(1)
        #print(self.feat_dim)
        for data in activation_data:
            activations.append(data.cpu().detach().numpy())

        return np.array(activations)
       # return [activation.cpu().detach().numpy() for activation in activations]



    def generate_features_for_imgs(self,data):

        loader = ImageLoader(None)
        image_feats = []

        #for chunk in tqdm.tqdm(utils.chunks(data, 32), total=len(data) // 32):
           # files = zip(*chunk)
        imgs = list(map(loader, data))
        imgs = list(map(self.transform, imgs))
        feats = self.feat_extractor(torch.stack(imgs, 0).cpu())
        image_feats.append(feats.data.cpu())
        image_feats = torch.cat(image_feats, 0)
        return image_feats.cpu().detach().numpy()


    # def generate_features_for_imgs(self,files):
    #
    #     loader = ImageLoader(None)
    #     image_feats = []
    #
    #     #for chunk in tqdm.tqdm(utils.chunks(data, 32), total=len(data) // 32):
    #    # files = zip(*files)
    #     imgs = list(map(loader, files))
    #     imgs = list(map(self.transform, imgs))
    #     feats = self.feat_extractor(torch.stack(imgs, 0).cpu())
    #     image_feats.append(feats.data.cpu())
    #     image_feats = torch.cat(image_feats, 0)
    #     return image_feats



class GenerateFeatures():

    def __init__(self):
        self.root = "./data/mit-states-original"
        self.img_path = "./data/mit-states-original/images"
        self.split = "/compositional-split"
        self.out_file = py.join(self.root,"features_new.t7")
        self.attrs, self.objs, self.pairs, self.train_pairs, self.test_pairs = self.parse_split()
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        self.loader= ImageLoader(self.img_path)
        self.transform = utils.imagenet_transform()
        self.data = self.get_data()

    def parse_split(self):

        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs('%s/%s/train_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs('%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(list(set(tr_attrs + ts_attrs))), sorted(list(set(tr_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, ts_pairs

    def get_data(self):
            data = torch.load(self.root + '/metadata.t7')

            split_data = []
            for instance in data:

                image, attr, obj = instance['image'], instance['attr'], instance['obj']

                if attr == 'NA' or (attr, obj) not in self.pairs:
                    # ignore instances with unlabeled attributes
                    # ignore instances that are not in current split
                    continue

                data_i = [py.join(self.img_path,image),attr, obj, self.attr2idx[attr], self.obj2idx[obj]]
                split_data.append(data_i)

            return split_data



    def generate_features(self, feat_extractor):
            data = self.data
            # data = self.train_data+self.test_data
            # transform = data_utils.imagenet_transform('test', transform_type)

            if feat_extractor is None:
                feat_extractor = torchvision.models.resnet18(pretrained=True)
                feat_extractor.fc = torch.nn.Sequential()
            feat_extractor.eval().cpu()

            image_feats = []
            image_files = []
            for chunk in tqdm.tqdm(utils.chunks(data, 4), total=len(data) // 4):
                files, attrs, objs,_,_ = zip(*chunk)
                imgs = list(map(self.loader, files))
                imgs = list(map(self.transform, imgs))
                feats = feat_extractor(torch.stack(imgs, 0).cpu())
                image_feats.append(feats.data.cpu())
                image_files += files
            image_feats = torch.cat(image_feats, 0)
            print('features for %d images generated' % (len(image_files)))

            torch.save({'features': image_feats, 'files': image_files}, self.out_file)





if __name__ == '__main__':
   features = GenerateFeatures()
   features.generate_features(None)