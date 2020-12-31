import numpy as np
import pylib as py
import tensorflow as tf
import tflib as tl


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
