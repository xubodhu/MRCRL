import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.layers as layers
from tqdm import tqdm
import time
import os
import pickle
from get_data import load_word2vec,mt, mt_seperate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122
def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag
def initial_sentence(relation2id,word2id,path_name,types):

    fixlen = 70
    f = open(path_name, 'r', encoding='utf-8')
    train_sen = []
    train_ans = []
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        en1 = content[0]
        en2 = content[1]
        relation = relation2id[content[2]]
        label = [0 for i in range(len(relation2id))]
        y_id = relation
        label[y_id] = 1
        train_ans.append(label)

        sentence = content[3:-1]
        en1pos = 0
        en2pos = 0
        for i in range(len(sentence)):
            if sentence[i] == en1:
                en1pos = i
            if sentence[i] == en2:
                en2pos = i
        output = []
        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])
        for i in range(min(fixlen, len(sentence))):
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]
            output[i][0] = word
        train_sen.append(output)
    train_word = []
    train_pos1 = []
    train_pos2 = []
    for tmp in train_sen:
        tmp_train_word = []
        tmp_train_pos1 = []
        tmp_train_pos2 = []
        for i in tmp:
            tmp_train_word.append(i[0])
            tmp_train_pos1.append(i[1])
            tmp_train_pos2.append(i[2])
        train_word.append(tmp_train_word)
        train_pos1.append(tmp_train_pos1)
        train_pos2.append(tmp_train_pos2)
    np.save("data/train_x.npy", train_sen)
    np.save("cnndata/cnn_"+types+"_word.npy",train_word)
    np.save("cnndata/cnn_"+types+"_pos1.npy",train_pos1)
    np.save("cnndata/cnn_"+types+"_pos2.npy",train_pos2)
    np.save("cnndata/cnn_"+types+"_y.npy", train_ans)


class Settings(object):
    def __init__(self):
        self.vocab_size = 114042
        self.len_sentence = 70
        self.num_epochs = 50
        self.num_classes = 53
        self.cnn_size = 230
        self.num_layers = 1
        self.pos_size = 5
        self.pos_num = 123
        self.word_embedding = 50
        self.keep_prob = 0.5
        self.batch_size = 160
        self.num_steps = 10000
        self.lr= 0.01

class CNN():
    def __init__(self, word_embeddings, setting):

        self.vocab_size = setting.vocab_size
        self.len_sentence = len_sentence = setting.len_sentence
        self.num_epochs = setting.num_epochs
        self.num_classes = num_classes = setting.num_classes
        self.cnn_size = setting.cnn_size
        self.num_layers = setting.num_layers
        self.pos_size = setting.pos_size
        self.pos_num = setting.pos_num
        self.word_embedding = setting.word_embedding
        self.lr = setting.lr

        word_embedding = tf.get_variable(initializer=word_embeddings, name='word_embedding')
        pos1_embedding = tf.get_variable('pos1_embedding', [self.pos_num, self.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding', [self.pos_num, self.pos_size])
        # relation_embedding = tf.get_variable('relation_embedding', [self.num_classes, self.cnn_size])

        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_word')
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_pos2')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32)

        self.input_word_ebd = tf.nn.embedding_lookup(word_embedding, self.input_word)
        self.input_pos1_ebd = tf.nn.embedding_lookup(pos1_embedding, self.input_pos1)
        self.input_pos2_ebd = tf.nn.embedding_lookup(pos2_embedding, self.input_pos2)

        self.inputs = tf.concat(axis=2, values=[self.input_word_ebd, self.input_pos1_ebd, self.input_pos2_ebd])
        self.inputs = tf.reshape(self.inputs, [-1, self.len_sentence, self.word_embedding + self.pos_size * 2, 1])

        conv = layers.conv2d(inputs=self.inputs, num_outputs=self.cnn_size, kernel_size=[3, 60], stride=[1, 60],
                             padding='SAME')

        max_pool = layers.max_pool2d(conv, kernel_size=[70, 1], stride=[1, 1])
        self.sentence = tf.reshape(max_pool, [-1, self.cnn_size])

        tanh = tf.nn.tanh(self.sentence)
        drop = layers.dropout(tanh, keep_prob=self.keep_prob)

        self.outputs = layers.fully_connected(inputs=drop, num_outputs=self.num_classes, activation_fn=tf.nn.softmax)

        '''
        self.y_index =  tf.argmax(self.input_y,1,output_type=tf.int32)
        self.indexes = tf.range(0, tf.shape(self.outputs)[0]) * tf.shape(self.outputs)[1] + self.y_index
        self.responsible_outputs = - tf.reduce_mean(tf.log(tf.gather(tf.reshape(self.outputs, [-1]),self.indexes)))
        '''
        # loss
        # self.cross_loss = -tf.reduce_mean( tf.log(tf.reduce_sum( self.input_y  * self.outputs ,axis=1)))
        self.cross_loss = -tf.reduce_mean(tf.reduce_sum(self.input_y * tf.log(self.outputs), axis=1))
        self.reward = tf.log(tf.reduce_sum(self.input_y * self.outputs, axis=1))

        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())

        self.final_loss = self.cross_loss + self.l2_loss

        # accuracy
        self.pred = tf.argmax(self.outputs, axis=1)
        self.pred_prob = tf.reduce_max(self.outputs, axis=1)

        self.y_label = tf.argmax(self.input_y, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.y_label), 'float'))

        # minimize loss
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.final_loss)

        self.tvars = tf.trainable_variables()

        # manual update parameters
        self.tvars_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.tvars_holders.append(placeholder)

        self.update_tvar_holder = []
        for idx, var in enumerate(self.tvars):
            update_tvar = tf.assign(var, self.tvars_holders[idx])
            self.update_tvar_holder.append(update_tvar)


def train_union(train_data,valid_data,save_path):
    random.shuffle(train_data)
    vec, word2id, relation2id = load_word2vec()
    train_sen, train_ans = mt(word2id, relation2id, train_data)
    train_word, train_pos1, train_pos2 = mt_seperate(train_sen)
    wordembedding = vec

    word = train_word
    pos1 = train_pos1
    pos2 = train_pos2
    cnn_train_y = train_ans
    valid_sen, valid_ans = mt(word2id, relation2id, valid_data)
    valid_word, valid_pos1, valid_pos2 = mt_seperate(valid_sen)
    valid_word_01 = valid_word
    valid_pos1_01 = valid_pos1
    valid_pos2_01 = valid_pos2
    settings = Settings()
    settings.num_epochs = 15
    settings.batch_size = 2
    settings.lr = 0.1
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(cnn_train_y[0])
    settings.num_steps = len(train_word) // settings.batch_size

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    mx_epoch=0
    f_max = 0
    with tf.Graph().as_default():
        sess = tf.Session(config=tf_config)
        with sess.as_default():

            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = CNN(word_embeddings=wordembedding, setting=settings)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            for epoch in range(1,settings.num_epochs+1):

                bar = range(settings.num_steps)

                for _ in bar:

                    sample_list = random.sample(range(len(cnn_train_y)),settings.batch_size)
                    batch_word = [word[x] for x in sample_list]
                    batch_pos1 = [pos1[x] for x in sample_list]
                    batch_pos2 = [pos2[x] for x in sample_list]
                    batch_train_y = [cnn_train_y[x] for x in sample_list]

                    feed_dict = {}
                    feed_dict[model.input_word] = batch_word
                    feed_dict[model.input_pos1] = batch_pos1
                    feed_dict[model.input_pos2] = batch_pos2
                    feed_dict[model.input_y] = batch_train_y
                    feed_dict[model.keep_prob] = settings.keep_prob

                    _,loss,accuracy=sess.run([model.train_op, model.final_loss, model.accuracy],feed_dict=feed_dict)
                    #break
                #saver.save(sess, save_path=save_path)
                with open('data/valid_0.1_label_entitypair.pkl', 'rb') as input:
                    label_entitypair = pickle.load(input)
                pred_entitypair = {}
                batch_size = 100
                steps = len(valid_word_01) // batch_size + 1
                for step in range(steps):
                    batch_valid_word = valid_word_01[batch_size * step: batch_size * (step + 1)]
                    batch_valid_pos1 = valid_pos1_01[batch_size * step: batch_size * (step + 1)]
                    batch_valid_pos2 = valid_pos2_01[batch_size * step: batch_size * (step + 1)]
                    batch_valid_date = valid_data[batch_size * step:batch_size * (step + 1)]
                    batch_entitypair = []
                    for line in batch_valid_date:
                        items = line.split('\t')
                        e1 = items[0]
                        e2 = items[1]
                        batch_entitypair.append(e1 + '$' + e2)
                    feed_dict = {}
                    feed_dict[model.input_word] = batch_valid_word
                    feed_dict[model.input_pos1] = batch_valid_pos1
                    feed_dict[model.input_pos2] = batch_valid_pos2
                    feed_dict[model.keep_prob] = 1
                    batch_relation, batch_prob = sess.run([model.pred,model.pred_prob], feed_dict=feed_dict)

                    assert (len(batch_relation) == len(batch_prob) and len(batch_relation) == len(batch_entitypair))
                    for i in range(len(batch_relation)):
                        if batch_relation[i] != 0:
                            tmp_key = batch_entitypair[i]
                            tmp_value = (batch_prob[i], batch_relation[i])
                            if tmp_key not in pred_entitypair.keys():
                                pred_entitypair[tmp_key] = []
                                pred_entitypair[tmp_key] = tmp_value
                            elif tmp_value[0] > pred_entitypair[tmp_key][0]:
                                pred_entitypair[tmp_key] = tmp_value
                list_pred = []
                for key in pred_entitypair.keys():
                    tmp_prob = pred_entitypair[key][0]
                    tmp_relation = pred_entitypair[key][1]
                    tmp_entitypair = key
                    list_pred.append((tmp_prob, tmp_entitypair, tmp_relation))
                list_pred = sorted(list_pred, key=lambda x: x[0], reverse=True)
                true_positive = 0
                for i, item in enumerate(list_pred):
                    tmp_entitypair = item[1]
                    tmp_relation = item[2]
                    label_relations = label_entitypair[tmp_entitypair]
                    if tmp_relation in label_relations:
                        true_positive += 1
                i += 1
                file = open("data/num_entitypair_true.txt", "r")
                num_entitypair_true = eval(file.read())
                if(i == 0):
                    p = 0
                else:
                    p = float(true_positive/i)
                r = float(true_positive/num_entitypair_true)
                if((p+r)==0):
                    f = 0
                else:
                    f = 2*(p*r)/(p+r)
                print('epoch:{}     f:{}     p:{}     r:{}'.format(epoch,f,p,r))
                if f>f_max:
                    f_max = f
                    mx_epoch = epoch

    return f_max,mx_epoch


def train(train_data,save_path):
    random.shuffle(train_data)
    vec, word2id, relation2id = load_word2vec()
    train_sen, train_ans = mt(word2id, relation2id, train_data)
    train_word, train_pos1, train_pos2 = mt_seperate(train_sen)
    wordembedding = vec

    print('reading training data')

    cnn_train_word = train_word
    cnn_train_pos1 = train_pos1
    cnn_train_pos2 = train_pos2
    cnn_train_y    = train_ans
    settings = Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(cnn_train_y[0])
    settings.num_steps = len(cnn_train_word) // settings.batch_size

    initial_sentence(relation2id, word2id, "tp_tenp.txt", "tp_test")
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        sess = tf.Session(config=tf_config)
        with sess.as_default():

            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = CNN(word_embeddings=wordembedding, setting=settings)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=300)

            t_max = 0
            t_max_p_r =[]
            for epoch in range(1,settings.num_epochs+1):

                bar = tqdm(range(settings.num_steps), desc='epoch {}, loss=0.000000, accuracy=0.000000'.format(epoch))

                for _ in bar:

                    sample_list = random.sample(range(len(cnn_train_y)),settings.batch_size)
                    batch_train_word = [cnn_train_word[x] for x in sample_list]
                    batch_train_pos1 = [cnn_train_pos1[x] for x in sample_list]
                    batch_train_pos2 = [cnn_train_pos2[x] for x in sample_list]
                    batch_train_y = [cnn_train_y[x] for x in sample_list]

                    feed_dict = {}
                    feed_dict[model.input_word] = batch_train_word
                    feed_dict[model.input_pos1] = batch_train_pos1
                    feed_dict[model.input_pos2] = batch_train_pos2
                    feed_dict[model.input_y] = batch_train_y
                    feed_dict[model.keep_prob] = settings.keep_prob

                    _,loss,accuracy=sess.run([model.train_op, model.final_loss, model.accuracy],feed_dict=feed_dict)
                    bar.set_description('epoch {} loss={:.6f} accuracy={:.6f}'.format(epoch, loss, accuracy))
                    #break
                saver.save(sess, save_path="./model/"+str(epoch)+save_path)

                test_word = np.load("cnndata/cnn_tp_test_word.npy")
                test_pos1 = np.load("cnndata/cnn_tp_test_pos1.npy")
                test_pos2 = np.load("cnndata/cnn_tp_test_pos2.npy")
                test_y = np.load("cnndata/cnn_tp_test_y.npy")
                with open('tp_tenp.txt', 'r', encoding='utf-8') as input:
                    test_data = input.readlines()
                no_na = {}
                i = 0
                for line in test_data:
                    line = line.strip()
                    items = line.split('\t')
                    relation = items[2]
                    if relation != "NA":
                        if relation not in no_na.keys():
                            no_na[relation] = []
                            no_na[relation].append(i)
                        else:
                            no_na[relation].append(i)
                    i += 1
                test_word_l = []
                test_pos1_l = []
                test_pos2_l = []
                test_y_l = []
                print(no_na.keys())
                for key in no_na.keys():
                    test_word_l.append([test_word[x] for x in no_na[key]])
                    test_pos1_l.append([test_pos1[x] for x in no_na[key]])
                    test_pos2_l.append([test_pos2[x] for x in no_na[key]])
                    test_y_l.append([test_y[x] for x in no_na[key]])
                relation = []
                relation_pred = []
                for batch in range(len(test_y_l)):
                    batch_word = test_word_l[batch]
                    batch_pos1 = test_pos1_l[batch]
                    batch_pos2 = test_pos2_l[batch]
                    batch_y = test_y_l[batch]
                    batch_relation = [np.argmax(i) for i in batch_y]

                    feed_dict = {}
                    feed_dict[model.input_word] = batch_word
                    feed_dict[model.input_pos1] = batch_pos1
                    feed_dict[model.input_pos2] = batch_pos2
                    feed_dict[model.keep_prob] = 1
                    batch_relation_pred, batch_prob = sess.run([model.pred, model.pred_prob], feed_dict=feed_dict)

                    relation.append(batch_relation)
                    relation_pred.append(batch_relation_pred)
                p = []
                r = []
                t_n = 0
                for i in range(len(relation)):
                    TP = 0
                    FP = 0
                    for j in range(len(relation[i])):
                        if relation_pred[i][j] != 0:
                            if relation[i][j] == relation_pred[i][j]:
                                TP += 1
                                t_n += 1
                            else:
                                FP += 1
                    if (TP + FP) != 0:
                        p.append(TP / (TP + FP))
                    else:
                        p.append(0)
                    r.append(TP / (len(relation[i])))
                macro_p = 0
                macro_r = 0
                num = 0
                for i in range(len(relation)):
                    num += len(relation[i])
                    macro_p += p[i]
                    macro_r += r[i]
                macro_p /= len(p)
                macro_r /= len(r)
                if (macro_p + macro_r) != 0:
                    macro_f1 = 2 * (macro_p * macro_r) / (macro_p + macro_r)
                else:
                    macro_f1 = 0
                acc = t_n / num
                print(p)
                print(r)
                print(acc)
                if acc > t_max:
                    t_max =acc
                    t_max_p_r = [p ,r]
    return t_max,  t_max_p_r
