import os
from data import MitStatesDataSet

import numpy as np
import pylib as py

from generate_features import Features

py.arg('--experiment_name', default='AttGAN_128_MIT_STATES_FINAL')
args_ = py.args()

output_dir = py.join('./output', args_.experiment_name)

out_file_eval = py.join(output_dir, 'eval_testing_2.t7')
out_file_reconstructed = py.join(output_dir, 'eval_testing_reconstructed_2.t7')





mit_states = MitStatesDataSet(training=False)
test_dataset_query = mit_states.get_data()
img_deck ,_= mit_states.get_images(training=False)
#attr,obj
target_labels_for_each_query = [(data[5],data[7]) for data in test_dataset_query]



feature_extractor = Features()
img_features = feature_extractor.get_dataset_features_V2()
#tf_img_features = np.concatenate(img_features)

print(img_features.shape)

def get_ground_label_for_image_ids(image_ids):
    lables_for_batch = []
    for image_id in image_ids:
        image_data = img_deck[image_id]
        #attr,obj
        lables_for_batch.append((image_data[3], image_data[4]))
    return lables_for_batch

test_query_img_features = feature_extractor.get_dataset_features_V3(out_file_eval)
#test_query_img_features = test_query_img_features[0:1000]
#tf_test_query_img_features = np.concatenate(test_dataset_query)
print(test_query_img_features.shape)

for i in range(test_query_img_features.shape[0]):
    test_query_img_features[i, :] /= np.linalg.norm(test_query_img_features[i, :])
for i in range(img_features.shape[0]):
    img_features[i, :] /= np.linalg.norm(img_features[i, :])

print(test_query_img_features.shape)
print(img_features.shape)

sims = test_query_img_features.dot(img_features.T)
# for i, t in enumerate(test_queries):
#     sims[i, t['source_img_id']] = -10e10  # remove query image
nn_result_labels = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]
print(len(nn_result_labels))
print(len(nn_result_labels[0]))
# print(nn_result[0])

nn_result_labels = [get_ground_label_for_image_ids(data) for data in nn_result_labels]

# compute recalls
out = []
#nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
for k in [1, 5, 10, 50, 100]:
    r = 0.0
    for target_query, nns in zip(target_labels_for_each_query, nn_result_labels):
        if target_query in nns[:k]:
            r += 1
    r /= len(nn_result_labels)
    out += [('recall_top' + str(k) + '_correct_composition', r)]


    r = 0.0
    for target_query, nns in zip(target_labels_for_each_query,nn_result_labels):
            if target_query[0] in [x[0] for x in nns[:k]]:
                r += 1
    r /= len(nn_result_labels)
    out += [('recall_top' + str(k) + '_correct_adj', r)]

    r = 0.0
    r = 0.0
    for target_query, nns in zip(target_labels_for_each_query, nn_result_labels):
            if target_query[1] in [x[1] for x in nns[:k]]:
                r += 1
    r /= len(nn_result_labels)
    out += [('recall_top' + str(k) + '_correct_noun', r)]

print(out)


#
# for i in range(tf_img_features.shape[0]):
#     tf_img_features[i, :] /= np.linalg.norm(tf_img_features[i, :])
# for i in range(tf_test_query_img_features.shape[0]):
#     tf_test_query_img_features[i, :] /= np.linalg.norm(tf_test_query_img_features[i, :])
# #test_query_img_features= test_query_img_features[1:1000]
# #tf_img_features = tf.constant(img_features)
#
# sims = tf_test_query_img_features.dot(tf_img_features.T)
# nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]
#
# top_nn_labels_per_query = []
# top_nn_per_query = []
# tile_image_emb = utils.tile_tensor(tf_img_features, 0, 32)
#
#
#
# def calculate_result_at_each_epoch(k,top_nn_per_query, top_nn_labels_per_query, target_labels_for_each_query):
#
#                         r = 0.0
#                         r_a = 0.0
#                         r_o = 0.0
#                         for query_result_image_ids,query_result_image_labels,query_target_labels in zip(top_nn_per_query,top_nn_labels_per_query,target_labels_for_each_query):
#                             if query_target_labels in query_result_image_labels[:k]:
#                                 r +=1
#                             if query_target_labels[0] in [x[0] for x in query_result_image_labels[:k]]:
#                                 r_a +=1
#                             if query_target_labels[1] in [x[1] for x in query_result_image_labels[:k]]:
#                                 r_o +=1
#                         # r /= len(target_labels_for_each_query)
#                         # r_a /= len(target_labels_for_each_query)
#                         # r_o /= len(target_labels_for_each_query)
#
#                        # print("k:%d recall_compositon:%s recall_attribue:%s recall_object:%s" %(k,r,r_a,r_o))
#                         return [r, r_a, r_o]
#
#
# k_1 = [0,0,0]
# k_5 = [0,0,0]
# k_10 = [0,0,0]
# k_50 = [0,0,0]
# k_100 = [0,0,0]
# current_start = 0
# for chunk in tqdm.tqdm(utils.chunks(test_query_img_features, 32), total=len(test_query_img_features) // 32):
#     target_labels_for_current_batch = target_labels_for_each_query[current_start:current_start+len(chunk)]
#    # features = feature_extractor.generate_features_for_imgs(data=chunk)
#
#     tf_features= tf.constant(chunk)
#     if len(chunk) != 32:
#         tile_image_emb = utils.tile_tensor(tf_img_features, 0, len(chunk))
#     #print(tile_image_emb.get_shape()) #(65*?,300)
#     repeat_img_feat = utils.repeat_tensor(tf_features, 0, len(img_features))
#    # print(repeat_img_feat.get_shape())
#     dis = tf.negative(tf.norm(repeat_img_feat - tile_image_emb, axis=-1))
#     # print(dis.get_shape())
#     # dis_per_image =  tf.map_fn(fn=lambda k: dis[...,k],
#     #                 elems=tf.range(batchsize),
#     #                 dtype=tf.float32)
#     dis_per_image = tf.split(dis,  len(chunk))
#     values, indices = tf.nn.top_k(dis_per_image, k=100, sorted=False)
#     top_nn =  sess.run(indices)
#     top_nn_per_batch = [ data for data in top_nn]
#     top_nn_labels_per_batch=[ get_ground_label_for_image_ids(img_deck,data) for data in top_nn]
#     current_start = current_start + len(chunk)
#     k_1 = [sum(x) for x in zip(k_1,calculate_result_at_each_epoch(1,top_nn_per_batch,top_nn_labels_per_batch,target_labels_for_current_batch))]
#
#     k_5 = [sum(x) for x in zip(k_5,calculate_result_at_each_epoch(5,top_nn_per_batch,top_nn_labels_per_batch,target_labels_for_current_batch))]
#
#     k_10 = [sum(x) for x in zip( k_10, calculate_result_at_each_epoch(10, top_nn_per_batch, top_nn_labels_per_batch,
#                                                             target_labels_for_current_batch))]
#     k_50 = [sum(x) for x in zip(k_50, calculate_result_at_each_epoch(50, top_nn_per_batch, top_nn_labels_per_batch,
#                                                         target_labels_for_current_batch))]
#     k_100 = [sum(x) for x in zip(k_100, calculate_result_at_each_epoch(100, top_nn_per_batch, top_nn_labels_per_batch,
#                                                         target_labels_for_current_batch))]
#
# k_1 =  [x/len(target_labels_for_each_query) for x in k_1]
# k_5 =  [x/len(target_labels_for_each_query) for x in k_5]
# k_10 =  [x/len(target_labels_for_each_query) for x in k_10]
# k_50 =  [x/len(target_labels_for_each_query) for x in k_50]
# k_100 =  [x/len(target_labels_for_each_query) for x in k_100]
#
#
#
# print("k:%d recall_compositon:%s recall_attribue:%s recall_object:%s" % (0, k_1[0], k_1[1], k_1[2]))
# print("k:%d recall_compositon:%s recall_attribue:%s recall_object:%s" % (5, k_5[0], k_5[1], k_5[2]))
# print("k:%d recall_compositon:%s recall_attribue:%s recall_object:%s" % (10, k_10[0], k_10[1], k_10[2]))
# print("k:%d recall_compositon:%s recall_attribue:%s recall_object:%s" % (50, k_50[0], k_50[1], k_50[2]))
# print("k:%d recall_compositon:%s recall_attribue:%s recall_object:%s" % (100, k_100[0], k_100[1], k_100[2]))
#
# #print(top_nn_per_query)
#
