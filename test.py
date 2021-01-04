import os

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tflib as tl
import tqdm

import data
import module


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--img_dir', default='./data/img_celeba/aligned/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)_jpg/data')
py.arg('--test_label_path', default='./data/img_celeba/test_label.txt')
py.arg('--test_int', type=float, default=2)
py.arg('--load_size', type=int, default=256)
py.arg('--crop_size', type=int, default=256)
py.arg('--n_samples', type=int, default=32)

py.arg('--experiment_name', default='default')
args_ = py.args()

# output_dir
output_dir = py.join('output', args_.experiment_name)

# save settings
args = py.args_from_yaml(py.join(output_dir, 'settings.yml'))
args.__dict__.update(args_.__dict__)

# others
n_atts = len(args.att_names)

sess = tl.session()
sess.__enter__()  # make default


# ==============================================================================
# =                               data and model                               =
# ==============================================================================

# data
test_dataset, len_test_dataset,test_img_database,len_test_img_database = data.make_mitstates_dataset(args.img_dir, args.test_label_path, args.att_names, args.n_samples,
                                                          load_size=args.load_size, crop_size=args.crop_size,
                                                          training=False, drop_remainder=False, shuffle=False, repeat=None)
test_iter = test_dataset.make_one_shot_iterator()
test_img_iter = test_img_database.make_one_shot_iterator()


# ==============================================================================
# =                                   graph                                    =
# ==============================================================================

def sample_graph():
    # ======================================
    # =               graph                =
    # ======================================

    test_next = test_iter.get_next()
    test_img_next = test_img_iter.get_next()

    if not os.path.exists(py.join(output_dir, 'generator.pb')):
        # model
        Genc, Gdec, _ = module.get_model(args.model, n_atts, weight_decay=args.weight_decay)

        # placeholders & inputs
        xa = tf.placeholder(tf.float32, shape=[None, args.crop_size, args.crop_size, 3])
        b_ = tf.placeholder(tf.float32, shape=[None, n_atts])

        # sample graph
        x = Gdec(Genc(xa, training=False), b_, training=False)

        # all image embeddings
        x_all = tf.placeholder(tf.float32, shape=[None, args.crop_size, args.crop_size, 3])

        z_all = Genc(x_all,training=False)

        x_r_all = Genc(x,training=False)

    else:
        # load freezed model
        with tf.gfile.GFile(py.join(output_dir, 'generator.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='generator')

        # placeholders & inputs
        xa = sess.graph.get_tensor_by_name('generator/xa:0')
        b_ = sess.graph.get_tensor_by_name('generator/b_:0')

        # sample graph
        x = sess.graph.get_tensor_by_name('generator/xb:0')

    # ======================================
    # =            run function            =
    # ======================================

    save_dir = './output/%s/samples_testing_%s' % (args.experiment_name, '{:g}'.format(args.test_int))
    py.mkdir(save_dir)


    def run():
        cnt = 0

        z_alls = []
        for _ in tqdm.trange(len_test_img_database):
            x_d , attr, obj = sess.run(test_img_next)

            z = sess.run(z_all,feed_dict={x_all:x_d})
            z_alls.append(z)

        print(z_alls)
        x_r_list= []
        for _ in tqdm.trange(len_test_dataset):
            # data for sampling
           # xa_ipt, a_ipt ,b_a_ipt= sess.run(test_next)
            xa_ipt, a_ipt, b_a_ipt, attr, obj, o, neg_attr = sess.run(test_next)

            a_ipt = tf.one_hot(a_ipt, depth=n_atts)
            b_a_ipt = tf.one_hot(b_a_ipt, depth=n_atts)
            b_ipt_list = [a_ipt,b_a_ipt]  # the first is for reconstruction
            # for attr in b_a_ipt:
            #     if attr != -1:
            #         tmp = tf.one_hot(attr, depth=n_atts)
            #         b_ipt_list.append(tmp)
            # x_opt_list = [xa_ipt]
            # for i, b_ipt in enumerate(b_ipt_list):
            #     b__ipt = b_ipt * 2 - 1
            #     x_opt = sess.run(x, feed_dict={xa: xa_ipt, b_: b__ipt.eval()})
            #     x_opt_list.append(x_opt)

            b__ipt = b_a_ipt * 2 - 1
            x_r = sess.run(x_r_all, feed_dict={xa: xa_ipt, b_: b__ipt.eval()})
            x_r_list.append(x_r)




        print(x_r_list)
            # sample = np.transpose(x_opt_list, (1, 2, 0, 3, 4))
            # sample = np.reshape(sample, (sample.shape[0], -1, sample.shape[2] * sample.shape[3], sample.shape[4]))
            #
            # for i,s in enumerate(sample):
            #     cnt += 1
            #     im.imwrite(s, '%s/%d-posAttr-%s-negAttr-%s-obj-%s.jpg' % (save_dir, cnt,attr[i],neg_attr[i],obj[i]))

    return run



sample = sample_graph()


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# checkpoint
if not os.path.exists(py.join(output_dir, 'generator.pb')):
    checkpoint = tl.Checkpoint(
        {v.name: v for v in tf.global_variables()},
        py.join(output_dir, 'checkpoints'),
        max_to_keep=1
    )
    checkpoint.restore().run_restore_ops()

sample()

sess.close()
