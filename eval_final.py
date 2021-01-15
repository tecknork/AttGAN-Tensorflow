import os
import tflib as tl
from data import AoCelvrDataSet
import torch
import torchvision
import numpy as np
import pylib as py
import tqdm
import imlib as im
from PIL import Image
import utils
import os
from generate_features import ImageLoader
import re
from generate_features import Features

py.arg('--experiment_name', default='AttGAN_128_Ao_Clevr_1')
py.arg('--generate_query_features',  default=False, action='store_true')
py.arg('--dataset', choices=['ao', 'mit'], default='ao')
py.arg('--generate_test_features',  default=False, action='store_true')
py.arg('--generate_test_imgs',  default=False, action='store_true')
py.arg('--evaluate',  default=True, action='store_true')
args_ = py.args()

output_dir = py.join('./output', args_.experiment_name)

out_file_eval = py.join(output_dir, 'eval_testing_2.t7')
out_file_reconstructed = py.join(output_dir, 'eval_testing_reconstructed_2.t7')

if args_.dataset == "ao":
    test_img_feature = "./data/ao_clevr/test_features.t7"
    test_img_path = "./data/ao_clevr/images"
else:
    test_img_feature = "./data/mit-states-original/features_test_compose_AE.t7"
    test_img_path = "./data/mit-states-original/images"



class QueryGenerateFeatures():
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


    def get_query_features(self):
            # path = "./data/mit-states-original/features_test_compose_AE.t7"  # all images
            activation_data = torch.load( self.out_file_eval)
            activations = []
            # self.activation_dict = dict(zip(activation_data['files'], activation_data['features']))
            # self.feat_dim = activation_data['features'].size(1)
            # print(self.feat_dim)
            for data in activation_data['features']:
                activations.append(data.cpu().detach().numpy())

            return np.array(activations)

class TestImgGenerateFeatures():
    def __init__(self):
        self.test_features= "./data/ao_clevr/features_test_UV_random__comp_seed_2000__seen_seed_0__train.t7"
        self.test_img_path = "./data/ao_clevr/test_img__UV_random__comp_seed_2000__seen_seed_0__train"

        self.loader = ImageLoader(None)
        self.transform = utils.imagenet_transform()

    def get_files(self):
        #save_dir_eval = './data/mit-states-original/test_imgs_compose_AE'
        test_images_generated = os.listdir(self.test_img_path)
        test_images_generated.sort(key=lambda f: int(re.sub('\D', '', f)))

        test_imgages_full_path = [py.join(self.test_img_path, img) for img in test_images_generated]
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

        torch.save({'features': image_feats, 'files': image_files}, self.test_features)

    def get_test_img_features(self):
            # path = "./data/mit-states-original/features_test_compose_AE.t7"  # all images
            activation_data = torch.load(self.test_features)
            activations = []
            # self.activation_dict = dict(zip(activation_data['files'], activation_data['features']))
            # self.feat_dim = activation_data['features'].size(1)
            # print(self.feat_dim)
            for data in activation_data['features']:
                activations.append(data.cpu().detach().numpy())

            return np.array(activations)

    def generate_imgs(self):
        py.mkdir(self.test_img_path)
        sess = tl.session()
        sess.__enter__()  # make default
        dataset = AoCelvrDataSet()
        self.test_data = dataset.test_data
        img_deck, len_img_deck = dataset.get_image_dataset(self.test_data)
        test_iter = img_deck.make_one_shot_iterator()

        test_next = test_iter.get_next()
        cnt = 0;
        for _ in tqdm.trange(len_img_deck):
            xa, _, _ = sess.run(test_next)
            x_opt_list = [xa]
            sample = np.transpose(x_opt_list, (1, 2, 0, 3, 4))
            sample = np.reshape(sample, (sample.shape[0], -1, sample.shape[2] * sample.shape[3], sample.shape[4]))
            for _, s in enumerate(sample):
                # modified img with a attribute with b
                im.imwrite(s, '%s/%d.jpg' % (self.test_img_path, cnt))
                cnt += 1

def generate_query_features():
    features = QueryGenerateFeatures()
    features.generate_features(None)

def generate_test_img_features():
    features = TestImgGenerateFeatures()
    features.generate_features(None)

def generate_test_imgs():
    features = TestImgGenerateFeatures()
    features.generate_imgs()


class EvaluateNN():

    def __init__(self):
        dataset = AoCelvrDataSet()
        self.test_dataset_query = dataset.get_data(False)
        self.test_data = dataset.test_data
        # attr,obj
        self.target_labels_for_each_query = [(data[5], data[7]) for data in self.test_dataset_query]
        ### test data set features ####
        feature_extractor = TestImgGenerateFeatures()
        self.img_features = feature_extractor.get_test_img_features()  # TODO
        ### query_features ####
        query_features = QueryGenerateFeatures()
        self.test_query_img_features = query_features.get_query_features()

        for i in range(self.test_query_img_features.shape[0]):
            self.test_query_img_features[i, :] /= np.linalg.norm(self.test_query_img_features[i, :])
        for i in range(self.img_features.shape[0]):
            self.img_features[i, :] /= np.linalg.norm(self.img_features[i, :])

    def get_ground_label_for_image_ids(self, image_ids):
        lables_for_batch = []
        for image_id in image_ids:
            image_data = self.test_data[image_id]
            # attr,obj
            lables_for_batch.append((image_data[3], image_data[4]))
        return lables_for_batch


    def evaluate(self):
        nn_result_labels_all = []
        for chunk in tqdm.tqdm(utils.chunks(self.test_query_img_features, 1024), total=len(self.test_query_img_features) // 1024):

            sims = chunk.dot(self.img_features.T)
            # for i, t in enumerate(test_queries):
            #     sims[i, t['source_img_id']] = -10e10  # remove query image
            nn_result_labels = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]
            nn_result_labels_all.extend(nn_result_labels)

        nn_result_labels = nn_result_labels_all
        print(len(nn_result_labels))
        print(len(nn_result_labels[0]))
        # print(nn_result[0])

        nn_result_labels = [self.get_ground_label_for_image_ids(data) for data in nn_result_labels]

        # compute recalls
        out = []
        # nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
        for k in [1, 5, 10, 50, 100]:
            r = 0.0
            for target_query, nns in zip(self.target_labels_for_each_query, nn_result_labels):
                if target_query in nns[:k]:
                    r += 1
            r /= len(nn_result_labels)
            out += [('recall_top' + str(k) + '_correct_composition', r)]

            r = 0.0
            for target_query, nns in zip(self.target_labels_for_each_query, nn_result_labels):
                if target_query[0] in [x[0] for x in nns[:k]]:
                    r += 1
            r /= len(nn_result_labels)
            out += [('recall_top' + str(k) + '_correct_adj', r)]

            r = 0.0
            r = 0.0
            for target_query, nns in zip(self.target_labels_for_each_query, nn_result_labels):
                if target_query[1] in [x[1] for x in nns[:k]]:
                    r += 1
            r /= len(nn_result_labels)
            out += [('recall_top' + str(k) + '_correct_noun', r)]

        print(out)

if args_.generate_test_imgs:
    print("Generate test images")
    generate_test_imgs()


if args_.generate_test_features:
    print("Generating Test Features")
    generate_test_img_features()

if args_.generate_query_features:
    print("Generating Query Features")
    generate_query_features()

if args_.evaluate:
    print("Evaluating")
    eval = EvaluateNN()
    eval.evaluate()

print("Finished")