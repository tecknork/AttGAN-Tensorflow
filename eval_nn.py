import os
from data import MitStatesDataSet
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



py.arg('--experiment_name', default='AttGAN_128_UT_ZAPPOS_Eval_small_1')
args_ = py.args()


# output_dir
output_dir = py.join('./output', args_.experiment_name)

# save settings
# args = py.args_from_yaml(py.join(output_dir, 'settings.yml'))
# args.__dict__.update(args_.__dict__)

save_dir_eval = py.join(output_dir,'eval_testing_2')
save_dir_reconstructed = py.join(output_dir,'eval_testing_reconstructed_2')

# others
#n_atts = len(args.att_names)

sess = tl.session()
sess.__enter__()  # make default


def get_ground_label_for_image_ids(dataset,image_ids):
    lables_for_batch = []
    for image_id in image_ids:
        image_data = dataset[image_id]
        #attr,obj
        lables_for_batch.append((image_data[3], image_data[4]))
    return lables_for_batch

mit_states = MitStatesDataSet(training=False)
test_dataset_query = mit_states.get_data()
img_deck ,_= mit_states.get_images(training=False)
#attr,obj
target_labels_for_each_query = [(data[5],data[7]) for data in test_dataset_query]
#
feature_extractor = Features()
img_features = feature_extractor.get_dataset_features(img_deck)
tf_img_features = tf.constant(img_features)
print(tf_img_features)
test_images_generated = os.listdir(save_dir_eval)
test_imgages_full_path = [py.join(save_dir_eval,img) for img in test_images_generated]
print(len(test_imgages_full_path))
#
top_nn_labels_per_query = []
top_nn_per_query = []
for chunk in tqdm.tqdm(utils.chunks(test_imgages_full_path, 16), total=len(test_imgages_full_path) // 16):
    features = feature_extractor.generate_features_for_imgs(data=chunk)
    tf_features= tf.constant(features)
    tile_image_emb = utils.tile_tensor(tf_img_features, 0, len(chunk))
    #print(tile_image_emb.get_shape()) #(65*?,300)
    repeat_img_feat = utils.repeat_tensor(tf_features, 0, len(img_features))
   # print(repeat_img_feat.get_shape())
    dis = tf.negative(tf.norm(repeat_img_feat - tile_image_emb, axis=-1))
    # print(dis.get_shape())
    # dis_per_image =  tf.map_fn(fn=lambda k: dis[...,k],
    #                 elems=tf.range(batchsize),
    #                 dtype=tf.float32)
    dis_per_image = tf.split(dis,  len(chunk))
    values, indices = tf.nn.top_k(dis_per_image, k=100, sorted=False)
    top_nn =  sess.run(indices)
    top_nn_per_query.extend([ data for data in top_nn])
    top_nn_labels_per_query.extend([ get_ground_label_for_image_ids(img_deck,data) for data in top_nn])
#print(top_nn_per_query)



recall_k = defaultdict(list)
for k in [1, 5, 10, 50, 100]:
                        r = 0.0
                        r_a = 0.0
                        r_o = 0.0
                        for query_result_image_ids,query_result_image_labels,query_target_labels in zip(top_nn_per_query,top_nn_labels_per_query,target_labels_for_each_query):
                            if query_target_labels in query_result_image_labels[:k]:
                                r +=1
                            if query_target_labels[0] in [x[0] for x in query_result_image_labels[:k]]:
                                r_a +=1
                            if query_target_labels[1] in [x[1] for x in query_result_image_labels[:k]]:
                                r_o +=1
                        r /= len(target_labels_for_each_query)
                        r_a /= len(target_labels_for_each_query)
                        r_o /= len(target_labels_for_each_query)
                        recall_k[k].append([r,r_a,r_o])
                        print("k:%d recall_compositon:%s recall_attribue:%s recall_object:%s" %(k,r,r_a,r_o))
