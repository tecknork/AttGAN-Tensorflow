import numpy as np
import pylib as py
import torch, torchvision
import tensorflow as tf
import tflib as tl
from utils import pad

# ATT_ID = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
#           'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
#           'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
#           'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
#           'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
#           'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
#           'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
#           'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
#           'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
#           'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
#           'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
#           'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
#           'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}


ATT_ID = {'ancient': 0, 'barren': 1, 'bent': 2, 'blunt': 3, 'bright': 4, 'broken': 5, 'browned': 6,
          'brushed': 7, 'burnt': 8, 'caramelized': 9, 'chipped': 10, 'clean': 11, 'clear': 12,
          'closed': 13, 'cloudy': 14, 'cluttered': 15, 'coiled': 16, 'cooked': 17, 'cored': 18,
          'cracked': 19, 'creased': 20, 'crinkled': 21, 'crumpled': 22, 'crushed': 23, 'curved': 24,
          'cut': 25, 'damp': 26, 'dark': 27, 'deflated': 28, 'dented': 29, 'diced': 30, 'dirty': 31,
          'draped': 32, 'dry': 33, 'dull': 34, 'empty': 35, 'engraved': 36, 'eroded': 37, 'fallen': 38,
          'filled': 39, 'foggy': 40, 'folded': 41, 'frayed': 42, 'fresh': 43, 'frozen': 44, 'full': 45,
          'grimy': 46, 'heavy': 47, 'huge': 48, 'inflated': 49, 'large': 50, 'lightweight': 51, 'loose': 52,
          'mashed': 53, 'melted': 54, 'modern': 55, 'moldy': 56, 'molten': 57, 'mossy': 58, 'muddy': 59,
          'murky': 60, 'narrow': 61, 'new': 62, 'old': 63, 'open': 64, 'painted': 65, 'peeled': 66,
          'pierced': 67, 'pressed': 68, 'pureed': 69, 'raw': 70, 'ripe': 71, 'ripped': 72, 'rough': 73,
          'ruffled': 74, 'runny': 75, 'rusty': 76, 'scratched': 77, 'sharp': 78, 'shattered': 79, 'shiny': 80,
          'short': 81, 'sliced': 82, 'small': 83, 'smooth': 84, 'spilled': 85, 'splintered': 86, 'squished': 87,
          'standing': 88, 'steaming': 89, 'straight': 90, 'sunny': 91, 'tall': 92, 'thawed': 93, 'thick': 94,
          'thin': 95, 'tight': 96, 'tiny': 97, 'toppled': 98, 'torn': 99, 'unpainted': 100, 'unripe': 101,
          'upright': 102, 'verdant': 103, 'viscous': 104, 'weathered': 105, 'wet': 106, 'whipped': 107,
          'wide': 108, 'wilted': 109, 'windblown': 110, 'winding': 111, 'worn': 112, 'wrinkled': 113,
          'young': 114}

ATT_ID = {'Canvas': 0, 'Cotton': 1, 'Faux.Fur': 2, 'Faux.Leather': 3, 'Full.grain.leather': 4, 'Hair.Calf': 5, 'Leather': 6,
          'Nubuck': 7, 'Nylon': 8, 'Patent.Leather': 9,
          'Rubber': 10, 'Satin': 11, 'Sheepskin': 12,
          'Suede': 13, 'Synthetic': 14, 'Wool': 15}

ID_ATT = {v: k for k, v in ATT_ID.items()}


def make_celeba_dataset(img_dir,
                        label_path,
                        att_names,
                        batch_size,
                        load_size=286,
                        crop_size=256,
                        training=True,
                        drop_remainder=True,
                        shuffle=True,
                        repeat=1):
    img_names = np.genfromtxt(label_path, dtype=str, usecols=0)
    img_paths = np.array([py.join(img_dir, img_name) for img_name in img_names])
    labels = np.genfromtxt(label_path, dtype=float, usecols=range(1, 116))
    labels = labels[:, np.array([ATT_ID[att_name] for att_name in att_names])]
    labels_b = np.genfromtxt(label_path, dtype=float, usecols=range(116, 231))
    labels_b = labels_b[:, np.array([ATT_ID[att_name] for att_name in att_names])]

    if shuffle:
        idx = np.random.permutation(len(img_paths))
        img_paths = img_paths[idx]
        labels = labels[idx]
        labels_b = labels_b[idx]

    if training:
        def map_fn_(img, label,label_b):
            img = tf.image.resize(img, [load_size, load_size])
            # img = tl.random_rotate(img, 5)
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_crop(img, [crop_size, crop_size, 3])
            # img = tl.color_jitter(img, 25, 0.2, 0.2, 0.1)
            # img = tl.random_grayscale(img, p=0.3)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            #label = (label + 1) // 2
            return img, label,label_b
    else:
        def map_fn_(img, label,label_b):
            img = tf.image.resize(img, [load_size, load_size])
            img = tl.center_crop(img, size=crop_size)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            #label = (label + 1) // 2
            return img, label,label_b

    dataset = tl.disk_image_batch_dataset(img_paths,
                                          batch_size,
                                          labels=labels,
                                          labels_b=labels_b,
                                          drop_remainder=drop_remainder,
                                          map_fn=map_fn_,
                                          shuffle=shuffle,
                                          repeat=repeat)

    if drop_remainder:
        len_dataset = len(img_paths) // batch_size
    else:
        len_dataset = int(np.ceil(len(img_paths) / batch_size))

    return dataset, len_dataset


def check_attribute_conflict(att_batch, att_name, att_names):
    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value

    idx = att_names.index(att_name)

    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[idx] == 1:
            _set(att, 0, 'Bangs')
        elif att_name == 'Bangs' and att[idx] == 1:
            _set(att, 0, 'Bald')
            _set(att, 0, 'Receding_Hairline')
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[idx] == 1:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name:
                    _set(att, 0, n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[idx] == 1:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name:
                    _set(att, 0, n)
        # elif att_name in ['Mustache', 'No_Beard'] and att[idx] == 1:  # enable this part help to learn `Mustache`
        #     for n in ['Mustache', 'No_Beard']:
        #         if n != att_name:
        #             _set(att, 0, n)

    return att_batch

def make_mitstates_dataset(img_dir,
                        label_path,
                        att_names,
                        batch_size,
                        load_size=286,
                        crop_size=256,
                        training=True,
                        drop_remainder=True,
                        shuffle=True,
                        repeat=1):



        mit_states = MitStatesDataSet(training)
        traindata = mit_states.get_data()

        #img_deck,len_img_deck = mit_states.get_images(training) #images_dataset
        img_paths = np.array([data[0] for data in traindata])
        labels = np.array([data[3] for data in traindata]) # attr
        labels_b = np.array([data[5] for data in traindata]) # neg_attr
        attr = np.array([data[1] for data in traindata]) # attr _ name
        obj = np.array([data[2] for data in traindata])  # obj _ name
        obj_id = np.array([data[4] for data in traindata])
        neg_attr= np.array([data[6] for data in traindata])
        if training:
            neg_img = np.array([data[7] for data in traindata])
        else:
            neg_img = None
        #labels_b = pad(labels_ba,-1)
        #img_names = np.genfromtxt(label_path, dtype=str, usecols=0)
        #img_paths = np.array([py.join(img_dir, img_name) for img_name in img_names])
        #labels = np.genfromtxt(label_path, dtype=float, usecols=range(1, 116))
        #labels = labels[:, np.array([ATT_ID[att_name] for att_name in att_names])]
        #labels_b = np.genfromtxt(label_path, dtype=float, usecols=range(116, 231))
        #labels_b = labels_b[:, np.array([ATT_ID[att_name] for att_name in att_names])]

        # no shuffle TODO in MITdatasetclass

        if shuffle:
            idx = np.random.permutation(len(img_paths))
            img_paths = img_paths[idx]
            labels = labels[idx]
            labels_b = labels_b[idx]
            attr= attr[idx]
            obj = obj[idx]
            obj_id = obj_id[idx]
            neg_img = neg_img[idx]
            neg_attr = neg_attr[idx]
        if training:

            def map_fn_(img,neg_img, label,label_b,attr,obj,obj_id,neg_attr):
                img = tf.image.resize(img, [136, 102])
                # img = tl.random_rotate(img, 5)
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_crop(img, [crop_size, crop_size, 3])
                # img = tl.color_jitter(img, 25, 0.2, 0.2, 0.1)
                # img = tl.random_grayscale(img, p=0.3)
                img = tf.clip_by_value(img, 0, 255) / 127.5 - 1

                neg_img = tf.image.resize(neg_img, [136, 102])
                # img = tl.random_rotate(img, 5)
                neg_img = tf.image.random_flip_left_right(neg_img)
                neg_img = tf.image.random_crop(neg_img, [crop_size, crop_size, 3])
                # img = tl.color_jitter(img, 25, 0.2, 0.2, 0.1)
                # img = tl.random_grayscale(img, p=0.3)
                neg_img = tf.clip_by_value(neg_img, 0, 255) / 127.5 - 1
                #label = (label + 1) // 2

                return img, label,label_b,attr,obj,obj_id,neg_attr,neg_img
        else:
            def map_fn_(img, label,label_b,attr,obj,obj_id,neg_attr):
                img = tf.image.resize(img, [136, 102])
                img = tl.center_crop(img, size=crop_size)
                img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
                #label = (label + 1) // 2

                return img, label,label_b,attr,obj,obj_id,neg_attr

        dataset = tl.disk_image_batch_dataset(img_paths,
                                              batch_size,
                                              labels=labels,
                                              labels_b=labels_b,
                                              attr= attr,
                                              neg_attr = neg_attr,
                                              obj = obj,
                                              obj_id = obj_id,
                                              neg_img = neg_img,
                                              drop_remainder=drop_remainder,
                                              map_fn=map_fn_,
                                              shuffle=shuffle,
                                              repeat=repeat)

        if drop_remainder:
            len_dataset = len(img_paths) // batch_size
        else:
            len_dataset = int(np.ceil(len(img_paths) / batch_size))

        return dataset, len_dataset #,img_deck,len_img_deck


class MitStatesDataSet():

    def __init__(self,training=True):
        self.root = "./data/ut-zap50k-original"
        self.img_path = "./data/ut-zap50k-original/images"
        self.split = "/compositional-split"
        self.attrs, self.objs, self.pairs, self.train_pairs, self.test_pairs = self.parse_split()
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        self.train_data, self.test_data, self.test_data_query = self.get_split_info()

        self.data = self.train_data if training  else self.test_data_query  # list of [img_name, attr, obj, attr_id, obj_id, feat]

        self.obj_affordance_mask = []
        for _obj in self.objs:
            candidates = [attr for (_, attr, obj, _, _) in self.train_data + self.test_data if obj == _obj]
            affordance = set(candidates)
            mask = [1 if x in affordance else 0 for x in self.attrs]
            self.obj_affordance_mask.append(mask)

        # negative image pool
        samples_grouped_by_obj = [[] for _ in range(len(self.objs))]
        for i, x in enumerate(self.train_data):
            samples_grouped_by_obj[x[4]].append(i)

        self.neg_pool = []  # [obj_id][attr_id] => list of sample id
        for obj_id in range(len(self.objs)):
            self.neg_pool.append([])
            for attr_id in range(len(self.attrs)):
                self.neg_pool[obj_id].append(
                    [i for i in samples_grouped_by_obj[obj_id] if
                     self.train_data[i][3] != attr_id]
                )

        self.query_data = []
        obj_attr_ids = self.noun2adjs_id_dataset(self.data)
        for i,datas in enumerate(self.data):
            if training:
                size = len(obj_attr_ids[datas[2]])
                #for attr_id in np.random.choice(obj_attr_ids[datas[2]], np.random.choice(size), replace=False):
                for attr_id in obj_attr_ids[datas[2]]:
                    if attr_id != self.data[i][3]:
                        query_d = datas
                        query_d = query_d + [attr_id] + [self.attrs[attr_id]] + [self.sample_negative(attr_id,self.data[i][4])]
                        self.query_data.append(query_d)

        self.data = self.query_data if training else self.test_data_query  # list of [img_name, attr, obj, attr_id, obj_id, feat]

    def get_data(self):

            return self.data

    def get_images(self,training=True):
        if training:
            return self.get_image_dataset_new(self.train_data)
        else:
            return self.get_image_dataset_new(self.test_data)

    def find_img_with_attr_obj(self,attr_id,obj_id):
        for i in self.data:
            if i[3] == attr_id and i[4] == obj_id:
                return i[0]


    def noun2adjs_id_dataset(self,data):
        noun2adjs = {}
        for i, img in enumerate(data):
            adj = img[3]
            noun = img[2]
            if noun not in noun2adjs.keys():
                noun2adjs[noun] = []
            if adj not in noun2adjs[noun]:
                noun2adjs[noun].append(adj)
        # for noun, adjs in noun2adjs.items():
        #     assert len(adjs) >= 2
        return noun2adjs

    def get_split_info(self):
            data = torch.load(self.root + '/metadata.t7')
            train_pair_set = set(self.train_pairs)
            test_pair_set = set(self.test_pairs)
            train_data, test_data = [], []

            for instance in data:

                image, attr, obj = instance['image'], instance['attr'], instance['obj']

                if attr == 'NA' or (attr, obj) not in self.pairs:
                    # ignore instances with unlabeled attributes
                    # ignore instances that are not in current split
                    continue

                data_i = [py.join(self.img_path,image),attr, obj, self.attr2idx[attr], self.obj2idx[obj]]
                # data_i = [image, attr, obj, self.attr2idx[attr], self.obj2idx[obj], self.activation_dict[image],
                # np.eye(len(self.attrs))[self.attr2idx[attr]]]

                # test_nouns = [
                #     u'armor', u'bracelet', u'bush', u'camera', u'candy', u'castle',
                #     u'ceramic', u'cheese', u'clock', u'clothes', u'coffee', u'fan', u'fig',
                #     u'fish', u'foam', u'forest', u'fruit', u'furniture', u'garden', u'gate',
                #     u'glass', u'horse', u'island', u'laptop', u'lead', u'lightning',
                #     u'mirror', u'orange', u'paint', u'persimmon', u'plastic', u'plate',
                #     u'potato', u'road', u'rubber', u'sand', u'shell', u'sky', u'smoke',
                #     u'steel', u'stream', u'table', u'tea', u'tomato', u'vacuum', u'wax',
                #     u'wheel', u'window', u'wool'
                # ]

                # train_nouns= [
                #     u'armor',
                # ]

                #
                # train_attr = ['ancient', 'barren', 'bent', 'blunt', 'bright', 'broken', 'browned', 'brushed',
                #               'burnt', 'caramelized', 'chipped', 'clean', 'clear', 'closed', 'cloudy', 'cluttered',
                #               'coiled', 'cooked', 'cored', 'cracked', 'creased', 'crinkled', 'crumpled', 'crushed',
                #               'curved', 'cut', 'damp', 'dark', 'deflated', 'dented', 'diced', 'dirty', 'draped',
                #               'dry', 'dull', 'empty', 'engraved', 'eroded', 'fallen', 'filled', 'foggy', 'folded',
                #               'frayed', 'fresh', 'frozen', 'full', 'grimy', 'heavy', 'huge', 'inflated', 'large',
                #               'lightweight', 'loose', 'mashed', 'melted', 'modern', 'moldy', 'molten', 'mossy',
                #               'muddy', 'murky', 'narrow', 'new', 'old', 'open', 'painted', 'peeled', 'pierced',
                #               'pressed', 'pureed', 'raw', 'ripe', 'ripped', 'rough', 'ruffled', 'runny', 'rusty',
                #               'scratched', 'sharp', 'shattered', 'shiny', 'short', 'sliced', 'small', 'smooth',
                #               'spilled', 'splintered', 'squished', 'standing', 'steaming', 'straight', 'sunny', 'tall',
                #               'thawed', 'thick', 'thin', 'tight', 'tiny', 'toppled', 'torn', 'unpainted', 'unripe',
                #               'upright',
                #               'verdant', 'viscous', 'weathered', 'wet', 'whipped', 'wide', 'wilted', 'windblown',
                #               'winding',
                #               'worn', 'wrinkled', 'young']
                #
                # train_attr = ['ancient', 'modern', 'moldy', 'blunt', 'bent', 'broken', 'peeled', 'rusty',
                #               'burnt', 'sliced', 'muddy', 'murky', 'mossy']

                train_attr = ['Canvas', 'Cotton', 'Faux.Fur', 'Faux.Leather', 'Full.grain.leather', 'Hair.Calf', 'Leather',
                     'Nubuck', 'Nylon', 'Patent.Leather', 'Rubber', 'Satin', 'Sheepskin', 'Suede', 'Synthetic', 'Wool']

                if (attr, obj) in test_pair_set:
                    if attr in train_attr:
                        test_data.append(data_i)
                else:
                    if attr in train_attr:
                        train_data.append(data_i)

            #print(train_data)
                    # negative image pool

                # test_data_query = []
                # for obj_neg_pool in self.neg_pool_test:
                #     for obj_id

            def noun2adjs_dataset(data):
                noun2adjs = {}
                for i, img in enumerate(data):
                    adj = img[1]
                    noun = img[2]
                    if noun not in noun2adjs.keys():
                        noun2adjs[noun] = []
                    if adj not in noun2adjs[noun]:
                        noun2adjs[noun].append(adj)
                # for noun, adjs in noun2adjs.items():
                #     assert len(adjs) >= 2
                return noun2adjs



            def noun2adjs_I_dataset(data):
                noun2adjs = {}
                noun2adjs_eye = {}
                for i, img in enumerate(data):
                    adj = img[1]
                    adj_eye = np.eye(len(self.attrs))[self.attr2idx[adj]]
                    noun = img[2]
                    if noun not in noun2adjs.keys():
                        noun2adjs[noun] = []
                        noun2adjs_eye[noun] = []
                    if adj not in noun2adjs[noun]:
                        noun2adjs[noun].append(adj)
                        if not len(noun2adjs_eye[noun]):
                            noun2adjs_eye[noun] = adj_eye
                        else:
                            current_eye = noun2adjs_eye[noun]
                            sum_eye = [sum(x) for x in zip(current_eye, adj_eye)]
                            noun2adjs_eye[noun] = sum_eye

                # for noun, adjs in noun2adjs.items():
                #     assert len(adjs) >= 2
                return noun2adjs_eye

            test_data_query = []
            noun2adjs_test_dataset = noun2adjs_dataset(test_data)
            for idx, data in enumerate(test_data):
                attr = data[1]
                obj = data[2]
                for target_adj in noun2adjs_test_dataset[obj]:
                    if target_adj != attr:
                        query = data + [self.attr2idx[target_adj], target_adj, self.obj2idx[obj], obj]
                        test_data_query.append(query)


            # f = open("train_composeAE_AttGAN.txt", "w")
            # noun_eye = noun2adjs_I_dataset(train_data)
            # for i in train_data:
            #     hot_enc = " ".join(str(x) for x in i[6])
            #     hot_b_enc = " ".join(str(x) for x in noun_eye[i[2]])
            #     f.write('%s %s %s\n' % (i[0], hot_enc, hot_b_enc))
            # f.close()

            # f = open("test_composeAE_AttGAN.txt", "w")
            # noun_eye = noun2adjs_I_dataset(test_data)
            # for i in test_data:
            #     hot_enc = " ".join(str(x) for x in i[6])
            #     hot_b_enc = " ".join(str(x) for x in noun_eye[i[2]])
            #     f.write('%s %s %s\n' % (i[0], hot_enc, hot_b_enc))
            # f.close()

            return train_data, test_data, test_data_query

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

    def sample_negative(self, attr_id, obj_id):
            id = np.random.choice(self.neg_pool[obj_id][attr_id])
            return self.data[id][0]

    def get_image_dataset(self,dataset):
          images_path = [data[0] for data in dataset]
          attr = [data[3] for data in dataset]
          obj = [data[4] for data in dataset]

          dataset = tf.data.Dataset.from_tensor_slices((images_path,attr,obj))
          import multiprocessing
          n_map_threads = multiprocessing.cpu_count()

          def map_fn_(path,attr,obj):
              load_size =256
              crop_size =256
              img = tf.io.read_file(path)
              img = tf.image.decode_png(img, 3)
              img = tf.image.resize(img, [load_size, load_size])
              img = tl.center_crop(img, size=crop_size)
              img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
              # label = (label + 1) // 2
              return img,attr,obj

          dataset = dataset.map(map_fn_, num_parallel_calls=n_map_threads)
          dataset = dataset.batch(2)
          return dataset ,len(images_path)//2

    def get_image_dataset_new(self, dataset):
        images_path = [data[0] for data in dataset]
        attr = [data[3] for data in dataset]
        obj = [data[4] for data in dataset]
        load_size = 256
        crop_size = 256
        img_list = []
        for data in dataset:
            img = tf.io.read_file(data[0])
            img = tf.image.decode_png(img, 3)
            img = tf.image.resize(img, [load_size, load_size])
            img = tl.center_crop(img, size=crop_size)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            img_list.append(img)

        return tf.stack(img_list) , len(images_path)
        # dataset = tf.data.Dataset.from_tensor_slices((images_path, attr, obj))
        # import multiprocessing
        # n_map_threads = multiprocessing.cpu_count()
        #
        # def map_fn_(path, attr, obj):
        #     load_size = 256
        #     crop_size = 256
        #     img = tf.io.read_file(path)
        #     img = tf.image.decode_png(img, 3)
        #     img = tf.image.resize(img, [load_size, load_size])
        #     img = tl.center_crop(img, size=crop_size)
        #     img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
        #     # label = (label + 1) // 2
        #     return img, attr, obj
        #
        # dataset = dataset.map(map_fn_, num_parallel_calls=n_map_threads)
        # dataset = dataset.batch(2)
        # return dataset, len(images_path) // 2