import traceback

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tflib as tl
import tfprob
import tqdm

import data
import module

default_att_names = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
                     'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
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
py.arg('--batch_size', type=int, default=10)
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


train_dataset, len_train_dataset = data.make_celeba_dataset(args.img_dir, args.train_label_path, args.att_names, args.batch_size,
                                                            load_size=args.load_size, crop_size=args.crop_size,
                                                            training=True, shuffle=False, repeat=None)
print(len_train_dataset)
print(train_dataset)

train_iter = train_dataset.make_one_shot_iterator()
sess = tl.session()
sess.__enter__()  # make default

# get the next item

with tf.Session() as sess:

    xa, a ,b = train_iter.get_next()
    b_ = b-a
    one = tf.constant(1.0,dtype=tf.float64)
    attribute_switch_indices = tf.where(tf.equal(one,b_))
    result = tf.squeeze(attribute_switch_indices)

   # # print(sess.run(xa))
   #  print(sess.run(a))# do something with element
   #  print(sess.run(b))
   #  print(sess.run(b_))
   # # print(sess.run(one))
    print(sess.run(attribute_switch_indices))
    print(sess.run(result))
    # print(sess.run(tf.where(tf.equal(one,a))))
    # print(sess.run(tf.where(tf.equal(one, b))))
