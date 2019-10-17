# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:33:32 2019

@author: wanghl
"""

import tensorflow as tf
import tensorflow.layers as layers


def conv2d(
    img,
    n_filters,
    rate=1,
    kernel_size=[3, 3],
    reuse=False,
    bias=True,
    activation='relu',
    ):
    return layers.conv2d(
        img,
        filters=n_filters,
        kernel_size=kernel_size,
        dilation_rate=rate,
        activation=activation,
        padding='same',
        reuse=reuse,
        use_bias=bias,
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
        )


def encoding(
    img,
    n_filters,
    rate=1,
    kernel_size=[3, 3],
    reuse=False,
    ):
    with tf.name_scope('shared_encoding'):
        net = conv2d(img, n_filters, rate, kernel_size, reuse=reuse)
        
        return net

def decoding(
    img,
    n_filters,
    rate=1,
    kernel_size=[3, 3],
    reuse=False,
    ):
    with tf.name_scope('shared_decoding'):
        net = conv2d(img, n_filters, rate, kernel_size, reuse=reuse)

        return net
    
def decode(img):
    with tf.name_scope('shared_decode'):
        f_0 = conv2d(img, n_filters=64, kernel_size=[3, 3])#1-mask
        f_1 = drdb(f_0, rate=2)
        f_2 = drdb(f_1, rate=2)
        f_3 = drdb(f_2, rate=2)
        f_4 = tf.concat([f_1, f_2, f_3], axis=3)
        f_5 = conv2d(f_4, n_filters=64, kernel_size=[3, 3])
        return f_5
    



def mask_network(i_1, i_2, i_3):
    with tf.name_scope('mask_network'):
        z_1 = encoding(i_1, n_filters=64, kernel_size=[3, 3])
        z_r = encoding(i_2, n_filters=64, kernel_size=[3, 3],
                       reuse=True)
        z_3 = encoding(i_3, n_filters=64, kernel_size=[3, 3],
                       reuse=True)
        m_1 = tf.sigmoid(tf.reduce_mean(tf.multiply(z_1,z_r)))
        m_3 = tf.sigmoid(tf.reduce_mean(tf.multiply(z_3,z_r)))

        m_11=1-m_1
        m_33=1-m_3
        
        z_11 = tf.multiply(z_1, m_1)
        z_12 = tf.multiply(z_r, m_1)
        z_33 = tf.multiply(z_3, m_3)
        z_32 = tf.multiply(z_r, m_3)
        z_s1  = tf.concat([z_11, z_12], axis=3)
        z_s3  = tf.concat([z_33, z_32], axis=3)


        z_111 = tf.multiply(z_1, m_11)
        z_122 = tf.multiply(z_r, m_11)
        z_333 = tf.multiply(z_3, m_33)
        z_322 = tf.multiply(z_r, m_33)
        z_s11  = tf.concat([z_111, z_122], axis=3)
        z_s33  = tf.concat([z_333, z_322], axis=3)

        return (z_s1, z_s3, z_s11, z_s33, z_r)


def dconv2d(x,n_filters=32,rate=2,kernel_size=[3, 3]):
    with tf.name_scope('dconv2d'):
        output = conv2d(x, n_filters=n_filters,
                        kernel_size=kernel_size, rate=rate)
        return output


def drdb(
    x,
    kernel_size=[3, 3],
    rate=2,
    growth_rate=32,
    ):
    with tf.name_scope('drdb'):
        x_1 = dconv2d(x, n_filters=growth_rate, rate=rate)
        x_2 = dconv2d(tf.concat([x, x_1], axis=3),
                      n_filters=growth_rate, rate=rate)
        x_3 = dconv2d(tf.concat([x, x_1, x_2], axis=3),
                      n_filters=growth_rate, rate=rate)
        x_4 = dconv2d(tf.concat([x, x_1, x_2, x_3], axis=3),
                       n_filters=growth_rate, rate=rate)
        x_5 = dconv2d(tf.concat([x, x_1, x_2, x_3, x_4], axis=3),
                       n_filters=growth_rate, rate=rate)

        output = tf.concat([
            x,
            x_1,
            x_2,
            x_3,
            x_4,
            x_5,
            ], axis=3)
        output = conv2d(output, n_filters=64, kernel_size=[1, 1])
        return x + output
    
    
def merging_network(z_s1, z_s3,z_s11,z_s33,z_r):
    with tf.name_scope('merging_network'):
        f_1 = decoding(z_s1, n_filters=64, kernel_size=[3, 3])#mask
        f_2 = decoding(z_s3, n_filters=64, kernel_size=[3, 3])
        f_3 = decode(z_s11)
        f_4 = decode(z_s33)
        f_5 = tf.concat([f_1,f_2,f_3,f_4,z_r], axis=3)
        f_6 = conv2d(f_5, n_filters=64, kernel_size=[3, 3])
        f_7 = conv2d(f_6, n_filters=3, kernel_size=[3, 3])
        f_7 =  tf.sigmoid(f_7)
        return f_7
        


def fusion_model(le, me, he):
    with tf.name_scope('fusion_model'):
        (z_s1, z_s3,z_s11,z_s33,z_r) = mask_network(le, me, he)
        out = merging_network(z_s1, z_s3,z_s11,z_s33, z_r)
        return out


