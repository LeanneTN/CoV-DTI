# -*- coding: utf-8 -*-
"""
@author: fangping wan
"""
import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import KFold, train_test_split, StratifiedKFold
import sys
from optparse import OptionParser
import utils
from utils import *

parser = OptionParser()
parser.add_option("-d", "--d", default=512, help="The embedding dimension d")
parser.add_option("-n","--n",default=1, help="global norm to be clipped")
parser.add_option("-k","--k",default=256, help="The dimension of project matrices k")
parser.add_option("-r","--r",default = 1, help="Repetition times")
parser.add_option("-l","--l",default = 0.0,help="l2 coefficient")
parser.add_option("-p","--p",default = 25,help="period to evaluate")
parser.add_option("-e","--e",default = 5000,help="epoch num")
(opts, args) = parser.parse_args()


#load network
drug_chemical = np.load('../data/Drug_simi_net.npy')
print np.shape(drug_chemical)
protein_protein = np.load('../data/PPI_net.npy')
print np.shape(protein_protein)
protein_sequence = np.load('../data/new_all_human_seq.npy')
print np.shape(protein_sequence)
virus_sequence = np.load('../data/all_seq_virus.npy')
print np.shape(virus_sequence)

virus_protein = np.load('../data/VHI_net.npy')
print np.shape(virus_protein)
protein_virus = virus_protein.T


#normalize network for mean pooling aggregation
drug_chemical_normalize = row_normalize(drug_chemical,True)
protein_protein_normalize = row_normalize(protein_protein,True)
protein_sequence_normalize = row_normalize(protein_sequence,True)
virus_sequence_normalize = row_normalize(virus_sequence,True)
virus_protein_normalize = row_normalize(virus_protein,False)
protein_virus_normalize = row_normalize(protein_virus,False)

#define computation graph
num_drug = len(drug_chemical)
num_protein = len(protein_protein_normalize)
num_virus = len(virus_sequence_normalize)


dim_drug = int(opts.d)
dim_protein = int(opts.d)
dim_virus = int(opts.d)

dim_pred = int(opts.k)
dim_pass = int(opts.d)

class Model(object):
    def __init__(self):
        self._build_model()
    
    def _build_model(self):
        #inputs

        self.drug_chemical = tf.placeholder(tf.float32, [num_drug, num_drug])
        self.drug_chemical_normalize = tf.placeholder(tf.float32, [num_drug, num_drug])


        self.protein_protein = tf.placeholder(tf.float32, [num_protein, num_protein])
        self.protein_protein_normalize = tf.placeholder(tf.float32, [num_protein, num_protein])
        self.protein_protein_pos_idx = tf.placeholder(tf.int32, [None, 2])
        self.protein_protein_neg_idx = tf.placeholder(tf.int32, [None, 2])

        self.protein_sequence = tf.placeholder(tf.float32, [num_protein, num_protein])
        self.protein_sequence_normalize = tf.placeholder(tf.float32, [num_protein, num_protein])


        self.virus_sequence = tf.placeholder(tf.float32, [num_virus, num_virus])
        self.virus_sequence_normalize = tf.placeholder(tf.float32, [num_virus, num_virus])


        self.virus_protein = tf.placeholder(tf.float32, [num_virus, num_protein])
        self.virus_protein_normalize = tf.placeholder(tf.float32, [num_virus, num_protein])
        self.virus_protein_pos_idx = tf.placeholder(tf.int32, [None, 2])
        self.virus_protein_neg_idx = tf.placeholder(tf.int32, [None, 2])

        self.protein_virus = tf.placeholder(tf.float32, [num_protein, num_virus])
        self.protein_virus_normalize = tf.placeholder(tf.float32, [num_protein, num_virus])


        self.drug_protein = tf.placeholder(tf.float32, [num_drug, num_virus + num_protein])
        self.drug_protein_normalize = tf.placeholder(tf.float32, [num_drug, num_virus + num_protein])
        self.drug_protein_pos_idx = tf.placeholder(tf.int32, [None, 2])
        self.drug_protein_neg_idx = tf.placeholder(tf.int32, [None, 2])


        self.protein_drug = tf.placeholder(tf.float32, [num_protein, num_drug])
        self.protein_drug_normalize = tf.placeholder(tf.float32, [num_protein, num_drug])

        self.virus_drug = tf.placeholder(tf.float32, [num_virus, num_drug])
        self.virus_drug_normalize = tf.placeholder(tf.float32, [num_virus, num_drug])


        self.drug_protein_mask = tf.placeholder(tf.float32, [num_drug, num_virus + num_protein])



        #features
        self.drug_embedding = weight_variable([num_drug,dim_drug])
        self.protein_embedding = weight_variable([num_protein,dim_protein])
        self.virus_embedding = weight_variable([num_virus,dim_virus])

        tf.add_to_collection('loss_reg', tf.contrib.layers.l2_regularizer(1.0)(self.drug_embedding))
        tf.add_to_collection('loss_reg', tf.contrib.layers.l2_regularizer(1.0)(self.protein_embedding))
        tf.add_to_collection('loss_reg', tf.contrib.layers.l2_regularizer(1.0)(self.virus_embedding))

        #feature passing weights (maybe different types of nodes can use different weights)
        W0 = weight_variable([dim_pass+dim_drug, dim_drug])
        b0 = bias_variable([dim_drug])
        tf.add_to_collection('loss_reg', tf.contrib.layers.l2_regularizer(1.0)(W0))

        W1 = weight_variable([dim_pass+dim_drug, dim_drug])
        b1 = bias_variable([dim_drug])
        tf.add_to_collection('loss_reg', tf.contrib.layers.l2_regularizer(1.0)(W1))


        combined_emb = tf.concat([self.virus_embedding, self.protein_embedding], axis=0)

        #passing 1 times (can be easily extended to multiple passes)
        drug_vector1 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(
            tf.concat([
            tf.matmul(self.drug_chemical_normalize, a_layer(self.drug_embedding, dim_pass)) + \
            tf.matmul(self.drug_protein_normalize, a_layer(combined_emb, dim_pass)), \
            self.drug_embedding], axis=1), W0)+b0+self.drug_embedding),dim=1)

        protein_vector1 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(
            tf.concat([
            tf.matmul(self.protein_protein_normalize, a_layer(self.protein_embedding, dim_pass)) + \
            tf.matmul(self.protein_sequence_normalize, a_layer(self.protein_embedding, dim_pass)) + \
            tf.matmul(self.protein_virus_normalize, a_layer(self.virus_embedding, dim_pass)) + \
            tf.matmul(self.protein_drug_normalize, a_layer(self.drug_embedding, dim_pass)), \
            self.protein_embedding], axis=1), W0)+b0+self.protein_embedding),dim=1)


        virus_vector1 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(
            tf.concat([
            tf.matmul(self.virus_sequence_normalize, a_layer(self.virus_embedding, dim_pass)) + \
            tf.matmul(self.virus_protein_normalize, a_layer(self.protein_embedding, dim_pass)) + \
            tf.matmul(self.virus_drug_normalize, a_layer(self.drug_embedding, dim_pass)), \
            self.virus_embedding], axis=1), W0)+b0+self.virus_embedding),dim=1)


        self.drug_representation_ = drug_vector1
        self.protein_representation_ = protein_vector1
        self.virus_representation_ = virus_vector1

        combined_emb_ = tf.concat([self.protein_representation_, self.virus_representation_], axis=0)

        #passing 2 times (can be easily extended to multiple passes)

        drug_vector2 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(
            tf.concat([
            tf.matmul(self.drug_chemical_normalize, a_layer(self.drug_representation_, dim_pass)) + \
            tf.matmul(self.drug_protein_normalize, a_layer(combined_emb_, dim_pass)), \
            self.drug_representation_], axis=1), W1)+b1+self.drug_representation_),dim=1)

        protein_vector2 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(
            tf.concat([
            tf.matmul(self.protein_protein_normalize, a_layer(self.protein_representation_, dim_pass)) + \
            tf.matmul(self.protein_sequence_normalize, a_layer(self.protein_representation_, dim_pass)) + \
            tf.matmul(self.protein_virus_normalize, a_layer(self.virus_representation_, dim_pass)) + \
            tf.matmul(self.protein_drug_normalize, a_layer(self.drug_representation_, dim_pass)), \
            self.protein_representation_], axis=1), W1)+b1+self.protein_representation_),dim=1)

        virus_vector2 = tf.nn.l2_normalize(tf.nn.relu(tf.matmul(
            tf.concat([
            tf.matmul(self.virus_sequence_normalize, a_layer(self.virus_representation_, dim_pass)) + \
            tf.matmul(self.virus_protein_normalize, a_layer(self.protein_representation_, dim_pass)) + \
            tf.matmul(self.virus_drug_normalize, a_layer(self.drug_representation_, dim_pass)), \
            self.virus_representation_], axis=1), W1)+b1+self.virus_representation_),dim=1)



        self.drug_representation = drug_vector2
        self.protein_representation = protein_vector2
        self.virus_representation = virus_vector2



        combined_emb2 = tf.concat([self.virus_representation, self.protein_representation], axis=0)
        #reconstructing networks

        self.drug_chemical_reconstruct = bi_layer(self.drug_representation,self.drug_representation, sym=True, dim_pred=dim_pred)
        self.drug_chemical_reconstruct_loss = square_loss(self.drug_chemical_reconstruct, self.drug_chemical)


        self.protein_protein_reconstruct = bi_layer(self.protein_representation,self.protein_representation, sym=True, dim_pred=dim_pred)
        self.protein_protein_reconstruct_loss = rank_loss(self.protein_protein_reconstruct, self.protein_protein_pos_idx, self.protein_protein_neg_idx)

        self.protein_sequence_reconstruct = bi_layer(self.protein_representation,self.protein_representation, sym=True, dim_pred=dim_pred)
        self.protein_sequence_reconstruct_loss = square_loss(self.protein_sequence_reconstruct, self.protein_sequence)


        self.virus_sequence_reconstruct = bi_layer(self.virus_representation,self.virus_representation, sym=True, dim_pred=dim_pred)
        self.virus_sequence_reconstruct_loss = square_loss(self.virus_sequence_reconstruct, self.virus_sequence)

        self.virus_protein_reconstruct = bi_layer(self.virus_representation,self.protein_representation, sym=False, dim_pred=dim_pred)
        self.virus_protein_reconstruct_loss = rank_loss(self.virus_protein_reconstruct, self.virus_protein_pos_idx, self.virus_protein_neg_idx)


        self.drug_protein_reconstruct = bi_layer(self.drug_representation, combined_emb2, sym=False, dim_pred=dim_pred)
        self.drug_protein_reconstruct_loss = rank_loss(self.drug_protein_reconstruct, self.drug_protein_pos_idx, self.drug_protein_neg_idx)

        self.l2_loss = tf.add_n(tf.get_collection("loss_reg"))

        self.loss = self.drug_protein_reconstruct_loss  + 1.0*(self.drug_chemical_reconstruct_loss+
                                            self.protein_protein_reconstruct_loss+self.protein_sequence_reconstruct_loss+
                                            self.virus_sequence_reconstruct_loss+self.virus_protein_reconstruct_loss) + float(opts.l)*self.l2_loss



graph = tf.get_default_graph()
with graph.as_default():
    model = Model()
    learning_rate = tf.placeholder(tf.float32, [])
    total_loss = model.loss
    dti_loss = model.drug_protein_reconstruct_loss

    optimize = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimize.compute_gradients(total_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, float(opts.n))
    optimizer = optimize.apply_gradients(zip(gradients, variables))
    saver = tf.train.Saver()

    DTI_pred = model.drug_protein_reconstruct


def train_and_evaluate(DTItrain, DTIvalid, DTItest, graph, verbose=True, num_steps = 5000):

    DTI_test_mask = np.zeros((num_drug,num_virus + num_protein))
    for ele in DTItest:
        DTI_test_mask[ele[0],ele[1]] = 1

    drug_protein = np.zeros((num_drug,num_virus + num_protein))
    DTI_mask = np.zeros((num_drug,num_virus + num_protein))
    for ele in DTItrain:
        drug_protein[ele[0],ele[1]] = ele[2]
        DTI_mask[ele[0],ele[1]] = 1
    protein_drug = drug_protein.T[num_virus:]
    virus_drug = drug_protein.T[:num_virus]

    aux_matrix_dti = np.zeros((num_drug,num_virus + num_protein))    
    for ele in DTItrain:
        aux_matrix_dti[ele[0],ele[1]] = ele[2]
    for ele in DTIvalid:
        aux_matrix_dti[ele[0],ele[1]] = -1
    for ele in DTItest:
        aux_matrix_dti[ele[0],ele[1]] = -1

    drug_protein_normalize = row_normalize(drug_protein,False)
    protein_drug_normalize = row_normalize(protein_drug,False)
    virus_drug_normalize = row_normalize(virus_drug,False)


    lr = 0.001
    best_valid_aupr = 0
    best_pred = 0
    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        for i in range(num_steps):


            protein_protein_pos_idx, protein_protein_neg_idx = sample(protein_protein)
            virus_protein_pos_idx, virus_protein_neg_idx = sample(virus_protein)
            drug_protein_pos_idx, drug_protein_neg_idx = sample(aux_matrix_dti)

            _, tloss, dtiloss, results_dti = sess.run([optimizer,total_loss,dti_loss, DTI_pred], \
                                        feed_dict={
                                        model.drug_chemical:drug_chemical, model.drug_chemical_normalize:drug_chemical_normalize,\
                                        model.protein_protein:protein_protein, model.protein_protein_normalize:protein_protein_normalize,\
                                        model.protein_sequence:protein_sequence, model.protein_sequence_normalize:protein_sequence_normalize,\
                                        model.virus_sequence:virus_sequence, model.virus_sequence_normalize:virus_sequence_normalize,\
                                        model.virus_protein:virus_protein, model.virus_protein_normalize:virus_protein_normalize,\
                                        model.protein_virus:protein_virus, model.protein_virus_normalize:protein_virus_normalize,\



                                        model.drug_protein:drug_protein, model.drug_protein_normalize:drug_protein_normalize,\
                                        model.protein_drug:protein_drug, model.protein_drug_normalize:protein_drug_normalize,\
                                        model.virus_drug:virus_drug, model.virus_drug_normalize:virus_drug_normalize,\


                                        model.drug_protein_mask:DTI_mask,\
                                        learning_rate: lr,\
                                        model.protein_protein_pos_idx : protein_protein_pos_idx,\
                                        model.protein_protein_neg_idx : protein_protein_neg_idx,\
                                        model.virus_protein_pos_idx : virus_protein_pos_idx,\
                                        model.virus_protein_neg_idx : virus_protein_neg_idx,\


                                        model.drug_protein_pos_idx : drug_protein_pos_idx,\
                                        model.drug_protein_neg_idx : drug_protein_neg_idx
                                        })

            #every opts.p steps of gradient descent, evaluate the performance, other choices of this number are possible
            if (i+1) % int(opts.p) == 0 and verbose == True:
                print ('step',(i+1),'total and dtiloss',tloss, dtiloss)
                pred_list = []
                ground_truth = []
                for ele in DTItrain:
                    pred_list.append(results_dti[ele[0],ele[1]])
                    ground_truth.append(ele[2])
                DTItrain_auc = roc_auc_score(ground_truth, pred_list)
                DTItrain_aupr = average_precision_score(ground_truth, pred_list)

                pred_list = []
                ground_truth = []
                for ele in DTIvalid:
                    pred_list.append(results_dti[ele[0],ele[1]])
                    ground_truth.append(ele[2])
                DTIvalid_auc = roc_auc_score(ground_truth, pred_list)
                DTIvalid_aupr = average_precision_score(ground_truth, pred_list)

                if DTIvalid_aupr >= best_valid_aupr:
                    best_valid_aupr = DTIvalid_aupr
                    best_pred = results_dti
                    #save_path = saver.save(sess, './model_'+str(opts.i)+'_'+str(r+1))


                    v_pred_list = []
                    v_ground_truth = []
                    h_pred_list = []
                    h_ground_truth = []
                    for ele in DTItest:
                        if ele[1] < num_virus:
                            v_pred_list.append(results_dti[ele[0],ele[1]])
                            v_ground_truth.append(ele[2])
                        else:
                            h_pred_list.append(results_dti[ele[0],ele[1]])
                            h_ground_truth.append(ele[2])

                    v_test_auc = roc_auc_score(v_ground_truth, v_pred_list)
                    v_test_aupr = average_precision_score(v_ground_truth, v_pred_list)
                    h_test_auc = roc_auc_score(h_ground_truth, h_pred_list)
                    h_test_aupr = average_precision_score(h_ground_truth, h_pred_list)



                print ('DTI train auc aupr,', DTItrain_auc, DTItrain_aupr, 'DTI valid auc aupr,', DTIvalid_auc, DTIvalid_aupr)
                print ('Test virus dti auc, aupr', v_test_auc, v_test_aupr, 'Test human dti auc, aupr', h_test_auc, h_test_aupr)
    return best_pred, DTI_test_mask


for r in range(int(opts.r)):
    print ('repetition round',r+1)

    virus_dti = np.load('../data/VDTI_net.npy')
    human_dti = np.load('../data/HDTI_net.npy')
    print (np.shape(virus_dti), np.shape(human_dti))
    network_dti = np.hstack([virus_dti, human_dti])

    whole_positive_index_dti = []
    whole_negative_index_dti = []
    for i in range(np.shape(network_dti)[0]):
        for j in range(np.shape(network_dti)[1]):
            if int(network_dti[i][j]) == 1:
                whole_positive_index_dti.append([i,j])
            elif int(network_dti[i][j]) == 0:
                whole_negative_index_dti.append([i,j])

    negative_sample_index_dti = np.random.choice(np.arange(len(whole_negative_index_dti)),size=10*len(whole_positive_index_dti),replace=False)

    data_set_dti = np.zeros((len(negative_sample_index_dti)+len(whole_positive_index_dti),3),dtype=int)
    count = 0
    for i in whole_positive_index_dti:
        data_set_dti[count][0] = i[0]
        data_set_dti[count][1] = i[1]
        data_set_dti[count][2] = 1
        count += 1
    for i in negative_sample_index_dti:
        data_set_dti[count][0] = whole_negative_index_dti[i][0]
        data_set_dti[count][1] = whole_negative_index_dti[i][1]
        data_set_dti[count][2] = 0
        count += 1

    rs = np.random.randint(0,1000,1)[0]
    kf = StratifiedKFold(data_set_dti[:,2], n_folds=10, shuffle=True, random_state=rs)

    counter = 0
    for train_index, test_index in kf:
        DTItrain, DTItest = data_set_dti[train_index], data_set_dti[test_index]
        DTItrain, DTIvalid =  train_test_split(DTItrain, test_size=0.05, random_state=rs)


        best_pred, test_mask = train_and_evaluate(DTItrain, DTIvalid, DTItest, graph=graph, num_steps=int(opts.e))
        
        counter += 1
        np.save('../output/NeoDTI_pred_rep_'+str(r+1)+'_fold_'+str(counter), best_pred)
        np.save('../output/NeoDTI_mask_rep_'+str(r+1)+'_fold_'+str(counter), test_mask)
print ('all finished')
