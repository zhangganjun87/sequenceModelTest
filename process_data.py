# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:12:01 2019

@author: Administrator
"""
import time
from os import path
import numpy as np
import tensorflow as tf
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
import random

def process_data(file):
    fi = open(file, 'r')
#    fo = open('processed_data', 'w')
    nextday_state = -1  #次日是升为1，是跌为0
    data_map = {}
    for line in fi:
        item_list = []
        items = line.strip().split('\t')   #list中元素的格式都是str,接下来把每个数据转换成float类型，如果是带%的，直接去掉%,不除以100
        date_day = items[0]        #日期
        kaipan = float(items[1].replace(',',''))        #开盘
        zuigao = float(items[2].replace(',',''))        #最高
        zuidi = float(items[3].replace(',',''))        #最低
        shoupan = float(items[4].replace(',',''))        #收盘
        shengdie = float(items[5].replace('%',''))        #升跌  %
        shoupanrijun = float(items[6].replace('%',''))  #has %        #收盘/日均
        shoupanrijun_10 = float(items[7].replace('%',''))  #has %        #收盘/10日均
        jiaoyiliang = float(items[8].replace(',',''))        #交易量（股）
        jiaoyijine = float(items[9].replace(',',''))        #交易金额（元）
        qianliu = float(items[10].replace('%','').replace(',','').replace('- --','0'))        #钱流
        rijun_1 = float(items[11].replace(',',''))        #日均价
        junjiashengdie_1 = float(items[12].replace('%',''))  #has %        #升跌%
        rijun_3 = float(items[13].replace(',',''))          #3天均
        junjiashengdie_3 = float(items[14].replace('%',''))  #has %        #升跌%
        rijun_5 = float(items[15].replace(',',''))        #5天均
        junjiashengdie_5 = float(items[16].replace('%',''))  #has %        #升跌%
        rijun_10 = float(items[17].replace(',',''))        #10天均
        junjiashengdie_10 = float(items[18].replace('%',''))  #has %        #升跌%
        shengdiejun10ri = float(items[19].replace('%',''))  #has %        #升跌均10天
        
        #把日期转换成时间戳
        date_day = date_day + ' 00:00:00'
        timeArray = time.strptime(date_day, '%Y/%m/%d %H:%M:%S')
        timeStamp = int(time.mktime(timeArray))
        
        #把这些数据写到新的list中
        item_list = [kaipan,zuigao,zuidi,shoupan,shengdie,shoupanrijun,shoupanrijun_10,jiaoyiliang,
                     jiaoyijine,qianliu,rijun_1,junjiashengdie_1,rijun_3,junjiashengdie_3,rijun_5,
                     junjiashengdie_5,rijun_10,junjiashengdie_10,shengdiejun10ri,nextday_state]
        
        #把list和时间添加到map中
        data_map[timeStamp] = item_list
        
        #判断前一天的目标分类
        if shengdie >= 0:
            nextday_state = 1
        else:
            nextday_state = 0
        #判断前一天的升跌值
#        nextday_state = shengdie
            
    fi.close()
    
    #对map按照时间戳由小到大进行排序(时间由早到晚)
    sorted_data_map = sorted(data_map.items(), key=lambda x:x[0])
    sorted_data_map.pop()   #删除最近一天的值，因为这天的标签值还不知道，是默认值
#    print(sorted_data_map)
    return sorted_data_map


def batch_generator(datas_list, n_seqs, n_steps):
    data_list = []
    batch_size = n_seqs * n_steps
    for datas in datas_list:
        data_list.append(datas[1])
    n_batches = int(len(data_list) // batch_size)
#    print(len(data_list))
#    print(n_batches)
    sh = random.randint(0, (n_batches-1)*batch_size)
    datas = data_list[sh : sh + batch_size]
    datas_np = np.array(datas)
    datas_new = datas_np.reshape((n_seqs,n_steps,20))
    x_ = datas_new[:,:,:-1]
    y_ = datas_new[:,:,-1]
    y_ex = np.expand_dims(y_, axis=2)
    return x_, y_ex

def get_a_cell(num_units):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    return lstm
            
#参数
num_units = 128        #隐层的维度
num_layers = 9
batch_size = 32
n_steps = 16
num_classes = 1
grad_clip = 5
learning_rate = 0.001
max_steps = 100000


with tf.Graph().as_default():
    datas_list = process_data('raw_data.txt')
#    with tf.name_scope('inputs'):
    lstm_inputs = tf.placeholder(tf.float64, shape=(batch_size, n_steps, 19), name='inputs')
    targets = tf.placeholder(tf.float64, shape=(batch_size, n_steps, 1), name='inputs')
        
#    with tf.name_scope('lstm'):
    cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(num_units) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float64)
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, lstm_inputs, initial_state=initial_state)
    seq_output = tf.concat(lstm_outputs, 1)
    x = tf.reshape(seq_output, [-1, num_units])
    
#    with tf.variable_scope('softmax'):
    softmax_w = tf.Variable(tf.truncated_normal([num_units, num_classes],dtype=tf.float64, stddev=0.1),dtype=tf.float64)
    softmax_b = tf.Variable(tf.zeros(num_classes,dtype=tf.float64),dtype=tf.float64)
                    
    logits = tf.matmul(x, softmax_w) + softmax_b
    proba_prediction = tf.nn.sigmoid(logits, name='predictions')
    
#    with tf.name_scope('loss'):
    y_reshaped = tf.reshape(targets, logits.get_shape())
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=proba_prediction, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
        
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    with tf.Session() as sess:
        step = 0
        sess.run(tf.global_variables_initializer())
#        new_state = sess.run(initial_state)
        while True:
            step += 1
            input_x,y = batch_generator(datas_list, batch_size, n_steps)
            start = time.time()
            feed = {lstm_inputs: input_x,targets: y} #,initial_state: new_state
            
            batch_loss, new_state, _ = sess.run([loss,final_state,optimizer],feed_dict=feed)
#            print('%%%%%%%%%%')
#            print(batch_loss)
#            print('%%%%%%%%%%')
            end = time.time()
            if step % 200 == 0:
                print('step: {}/{}... '.format(step, max_steps),
                      'loss: {:.6f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end - start)))
            if (step % 10000 == 0):
                tf.train.Saver().save(sess, path.join('modeSave/', 'model'), global_step=step)
            if step >= max_steps:
                break



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    