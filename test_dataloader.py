import traceback

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tflib as tl
import tfprob
import tqdm
import utils
import data
import module

# default_att_names = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
#                      'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
default_att_names = ['ancient', 'barren', 'bent', 'blunt', 'bright', 'broken', 'browned', 'brushed',
                           'burnt', 'caramelized', 'chipped', 'clean', 'clear', 'closed', 'cloudy', 'cluttered', 'coiled',
                           'cooked', 'cored', 'cracked', 'creased', 'crinkled', 'crumpled', 'crushed', 'curved', 'cut',
                           'damp', 'dark', 'deflated', 'dented', 'diced', 'dirty', 'draped', 'dry', 'dull', 'empty',
                           'engraved', 'eroded', 'fallen', 'filled', 'foggy', 'folded', 'frayed', 'fresh', 'frozen',
                           'full', 'grimy', 'heavy', 'huge', 'inflated', 'large', 'lightweight', 'loose', 'mashed',
                           'melted', 'modern', 'moldy', 'molten', 'mossy', 'muddy', 'murky', 'narrow', 'new', 'old',
                           'open', 'painted', 'peeled', 'pierced', 'pressed', 'pureed', 'raw', 'ripe', 'ripped', 'rough',
                           'ruffled', 'runny', 'rusty', 'scratched', 'sharp', 'shattered', 'shiny', 'short', 'sliced',
                           'small', 'smooth', 'spilled', 'splintered', 'squished', 'standing', 'steaming', 'straight',
                           'sunny', 'tall', 'thawed', 'thick', 'thin', 'tight', 'tiny', 'toppled', 'torn', 'unpainted',
                           'unripe', 'upright', 'verdant', 'viscous', 'weathered', 'wet', 'whipped',
                           'wide', 'wilted', 'windblown', 'winding', 'worn', 'wrinkled', 'young']


py.arg('--att_names', choices=data.ATT_ID.keys(), nargs='+', default=default_att_names)

py.arg('--img_dir', default='./data/mit-states-original/images')
py.arg('--train_label_path', default='./data/mit-states-original/train_composeAE_AttGAN.txt')
py.arg('--val_label_path', default='./data/mit-states-original/val_label.txt')
py.arg('--load_size', type=int, default=256)
py.arg('--crop_size', type=int, default=256)

py.arg('--n_epochs', type=int, default=60)
py.arg('--epoch_start_decay', type=int, default=30)
py.arg('--batch_size', type=int, default=2)
py.arg('--learning_rate', type=float, default=2e-4)
py.arg('--beta_1', type=float, default=0.5)

py.arg('--model', default='model_256', choices=['model_128', 'model_256', 'model_384'])

py.arg('--n_d', type=int, default=5)  # # d updates per g update
py.arg('--adversarial_loss_mode', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'], default='wgan')
py.arg('--gradient_penalty_mode', choices=['none', '1-gp', '0-gp', 'lp'], default='1-gp')
py.arg('--gradient_penalty_sample_mode', choices=['line', 'real', 'fake', 'dragan'], default='line')
py.arg('--d_gradient_penalty_weight', type=float, default=10.0)
py.arg('--d_attribute_loss_weight', type=float, default=1.0)
py.arg('--g_attribute_loss_weight', type=float, default=10.0)
py.arg('--g_reconstruction_loss_weight', type=float, default=100.0)
py.arg('--weight_decay', type=float, default=0.0)

py.arg('--n_samples', type=int, default=12)
py.arg('--test_int', type=float, default=2.0)

py.arg('--experiment_name', default='default')
args = py.args()

# output_dir
output_dir = py.join('output', args.experiment_name)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# others
n_atts = len(args.att_names)


sess = tl.session()
sess.__enter__()  # make default

# get the next item
#while True:
with tf.Session() as sess:
        train_dataset, len_train_dataset,img_deck,_ = data.make_mitstates_dataset(args.img_dir, args.train_label_path, args.att_names,
                                                                       args.batch_size,
                                                                       load_size=args.load_size, crop_size=args.crop_size,
                                                                       training=True, shuffle=True, repeat=None)
        print(len_train_dataset)

        # val_dataset, len_val_dataset,test_img_deck,len_test_img_deck = data.make_mitstates_dataset(args.img_dir, args.train_label_path,
        #                                                                args.att_names,
        #                                                                args.batch_size,
        #                                                                load_size=args.load_size,
        #                                                                crop_size=args.crop_size,
        #                                                                training=False, shuffle=False, repeat=None)
        # print(len_val_dataset)
        # print(test_img_deck)
        # # print(train_dataset)
        #
        # train_iter = train_dataset.make_one_shot_iterator()
        # xa, a_x ,b ,attr, obj, obj_id,neg_attr, xb_ref = train_iter.get_next()
        #
        #
        # print(sess.run([a_x,b,attr,obj,obj_id,neg_attr,xb_ref]))
        #
        # val_iter = val_dataset.make_one_shot_iterator()
        # xa, a_x ,b ,attr, obj, obj_id,neg_attr = val_iter.get_next()



        print(img_deck)
        tile_deck = utils.tile_tensor(img_deck, 0, args.n_samples)
        print(tile_deck)
        #
        # test_img_deck = test_img_deck.batch(2)
        # test_img_iter = test_img_deck.make_one_shot_iterator()
        # train_iter = test_img_iter.get_next()
        # flatten_images = []
        # for _ in tqdm.trange(len_test_img_deck//2):
        #
        #     imgs, attr, obj = train_iter
        #     imgs_a = tf.reshape(imgs,[-1])
        #     #print(imgs)
        #    # print(imgs_a)
        #     flatten_images.append(imgs_a)
        #     #print(sess.run([imgs, imgs_a]))
        # print(flatten_images)
        #print(sess.run([flatten_images]))

# print(sess.run(train_iter))

        # a = tf.one_hot(a_x, depth=n_atts)
        # print(xa)
        # print(a_x)
        # print(b)
        #
        # Tensor("IteratorGetNext:0", shape=(2, 256, 256, 3), dtype=float32)
        # Tensor("IteratorGetNext:1", shape=(2,), dtype=int64)
        # Tensor("IteratorGetNext:2", shape=(2,), dtype=int64)
        # b_ = b-a
        # one = tf.constant(1.0,dtype=tf.float64)
        # attribute_switch_indices = tf.where(tf.equal(one,b_))
        # result = tf.squeeze(attribute_switch_indices)

       # print(sess.run([a_x,b,attr,obj,obj_id,neg_attr]))

        #
        # train_iter = train_dataset.make_one_shot_iterator()
        # xa, a_x, b = train_iter.get_next()
        # a = tf.one_hot(a_x, depth=n_atts)

        # b_ = b-a
        # one = tf.constant(1.0,dtype=tf.float64)
        # attribute_switch_indices = tf.where(tf.equal(one,b_))
        # result = tf.squeeze(attribute_switch_indices)

#        print(sess.run([a, b]))

       #  print(sess.run(a))# do something with element
       #  print(sess.run(b))
       #  print(sess.run(b_))
       # # # print(sess.run(one))
       #  print(sess.run(attribute_switch_indices))
       #  print(sess.run(result))
        # print(sess.run(tf.where(tf.equal(one,a))))
        # print(sess.run(tf.where(tf.equal(one, b))))


































#
#
#
# import traceback
#
# import imlib as im
# import numpy as np
# import pylib as py
# import tensorflow as tf
# import tflib as tl
# import tfprob
# import tqdm
#
# import data
# import module
#
# default_att_names = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
#                      'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
# # default_att_names = ['ancient', 'barren', 'bent', 'blunt', 'bright', 'broken', 'browned', 'brushed',
# #                            'burnt', 'caramelized', 'chipped', 'clean', 'clear', 'closed', 'cloudy', 'cluttered', 'coiled',
# #                            'cooked', 'cored', 'cracked', 'creased', 'crinkled', 'crumpled', 'crushed', 'curved', 'cut',
# #                            'damp', 'dark', 'deflated', 'dented', 'diced', 'dirty', 'draped', 'dry', 'dull', 'empty',
# #                            'engraved', 'eroded', 'fallen', 'filled', 'foggy', 'folded', 'frayed', 'fresh', 'frozen',
# #                            'full', 'grimy', 'heavy', 'huge', 'inflated', 'large', 'lightweight', 'loose', 'mashed',
# #                            'melted', 'modern', 'moldy', 'molten', 'mossy', 'muddy', 'murky', 'narrow', 'new', 'old',
# #                            'open', 'painted', 'peeled', 'pierced', 'pressed', 'pureed', 'raw', 'ripe', 'ripped', 'rough',
# #                            'ruffled', 'runny', 'rusty', 'scratched', 'sharp', 'shattered', 'shiny', 'short', 'sliced',
# #                            'small', 'smooth', 'spilled', 'splintered', 'squished', 'standing', 'steaming', 'straight',
# #                            'sunny', 'tall', 'thawed', 'thick', 'thin', 'tight', 'tiny', 'toppled', 'torn', 'unpainted',
# #                            'unripe', 'upright', 'verdant', 'viscous', 'weathered', 'wet', 'whipped',
# #                            'wide', 'wilted', 'windblown', 'winding', 'worn', 'wrinkled', 'young']
#
#
# py.arg('--att_names', choices=data.ATT_ID.keys(), nargs='+', default=default_att_names)
#
# py.arg('--img_dir', default='./data/CelebAMask-HQ/CelebA-HQ-img')
# py.arg('--train_label_path', default='./data/CelebAMask-HQ/train_label.txt')
# py.arg('--val_label_path', default='./data/CelebAMask-HQ/val_label.txt ')
# py.arg('--load_size', type=int, default=256)
# py.arg('--crop_size', type=int, default=256)
#
# py.arg('--n_epochs', type=int, default=60)
# py.arg('--epoch_start_decay', type=int, default=30)
# py.arg('--batch_size', type=int, default=1)
# py.arg('--learning_rate', type=float, default=2e-4)
# py.arg('--beta_1', type=float, default=0.5)
#
# py.arg('--model', default='model_256', choices=['model_128', 'model_256', 'model_384'])
#
# py.arg('--n_d', type=int, default=5)  # # d updates per g update
# py.arg('--adversarial_loss_mode', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'], default='wgan')
# py.arg('--gradient_penalty_mode', choices=['none', '1-gp', '0-gp', 'lp'], default='1-gp')
# py.arg('--gradient_penalty_sample_mode', choices=['line', 'real', 'fake', 'dragan'], default='line')
# py.arg('--d_gradient_penalty_weight', type=float, default=10.0)
# py.arg('--d_attribute_loss_weight', type=float, default=1.0)
# py.arg('--g_attribute_loss_weight', type=float, default=10.0)
# py.arg('--g_reconstruction_loss_weight', type=float, default=100.0)
# py.arg('--weight_decay', type=float, default=0.0)
#
# py.arg('--n_samples', type=int, default=12)
# py.arg('--test_int', type=float, default=2.0)
#
# py.arg('--experiment_name', default='default')
# args = py.args()
#
# # output_dir
# output_dir = py.join('output', args.experiment_name)
# py.mkdir(output_dir)
#
# # save settings
# py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)
#
# # others
# n_atts = len(args.att_names)
#
#
# train_dataset, len_train_dataset = data.make_celeba_dataset(args.img_dir, args.train_label_path, args.att_names, args.batch_size,
#                                                             load_size=args.load_size, crop_size=args.crop_size,
#                                                             training=True, shuffle=False, repeat=None)
# print(len_train_dataset)
# print(train_dataset)
#
# train_iter = train_dataset.make_one_shot_iterator()
# sess = tl.session()
# sess.__enter__()  # make default
#
# # get the next item
#
# with tf.Session() as sess:
#
#     xa, a = train_iter.get_next()
#     b = tf.random_shuffle(a)
#     b_ = b * 2 - 1
#     a_ = a * 2 - 1
#    # # print(sess.run(xa))
#    #  print(sess.run(a))# do something with element
#    #  print(sess.run(b))
#    #  print(sess.run(b_))
#    # # print(sess.run(one))
#     #print(sess.run(xa))
#     #print(sess.run(a))
#    # print(sess.run(b))
#    # print(sess.run(tf.random_shuffle([1,2,3,4,5])))
#     #print(sess.run(b_))
#     #print(sess.run([a,a_,b,b_]))
#     #print(sess.run(a))
#
#     xa_ipt, a_ipt = train_iter.get_next()
#
#     b_ipt_list = [a_ipt]  # the first is for reconstruction
#     for i in range(n_atts):
#         tmp = a_ipt.eval()
#         tmp[:, i] = 1 - tmp[:, i]  # inverse attribute
#         #tmp = data.check_attribute_conflict(tmp, args.att_names[i], args.att_names)
#         #print(tmp)
#         b_ipt_list.append(tmp)
#
#     print(b_ipt_list)
#     # x_opt_list = [xa_ipt]
#     # for i, b_ipt in enumerate(b_ipt_list):
#     #     b__ipt = (b_ipt * 2 - 1).astype(np.float32)  # !!!
#     #     if i > 0:  # i == 0 is for reconstruction
#     #         b__ipt[..., i - 1] = b__ipt[..., i - 1] * args.test_int
#     #     x_opt = sess.run(x, feed_dict={xa: xa_ipt, b_: b__ipt})
#     #     x_opt_list.append(x_opt)
#     # sample = np.transpose(x_opt_list, (1, 2, 0, 3, 4))
#     # sample = np.reshape(sample, (-1, sample.shape[2] * sample.shape[3], sample.shape[4]))
#
#     print(sess.run([a_ipt]))
#
#     # print(sess.run(tf.where(tf.equal(one,a))))
#     # print(sess.run(tf.where(tf.equal(one, b))))
#
# # array([[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]]),\
# # array([[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]]),\
# # array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]]),\
# # array([[0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]]),\
# # array([[0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1]]),\
# # array([[0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]]),\
# # array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]]),\
# # array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]]),\
# # array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1]]),\
# # array([[0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1]]),\
# # array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]), \
# # array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0]]),
# # array([[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]])]
# #
# #
# # [array([[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1]])]
#
#
#
# array([[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]]),
# array([[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]]),
# array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]]),
# array([[0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]]),
# array([[0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1]]),
# array([[0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]]),
# array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]]),
# array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]]),
# array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1]]),
# array([[0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1]]),
# array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]),
# array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0]]),
# array([[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]])]
#
# [array([[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1]])]
