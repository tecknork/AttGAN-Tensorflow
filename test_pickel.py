import pickle


filename = './data/ao_clevr/metadata_pickles/metadata_ao_clevr__UV_random__comp_seed_2000__seen_seed_0__train.pkl'
infile = open(filename,'rb')
new_dict = pickle.load(infile)
#print(new_dict)
for key in new_dict:
  print(key)

print(len(new_dict['test_data']))
print(len(new_dict['train_data']))
print(new_dict['pair2idx'])
print(new_dict['attrs'])
print(new_dict['objs'])
print(len(new_dict['all_open_pairs']))
print(new_dict['all_open_pairs'])
infile.close()
#
#test 42394
#train 94966