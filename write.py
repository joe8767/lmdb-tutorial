# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:44:12 2015

@author: joe

Reference:
http://deepdish.io/2015/04/28/creating-lmdb-in-python/
"""

import numpy as np
import lmdb
import caffe

N = 100

X = np.zeros((N, 3, 24, 32), dtype=np.uint8)
Y = np.zeros(N, dtype=np.int64)

for i in range(N):
    Y[i] = i

map_size = X.nbytes * 10

env = lmdb.open('mylmdb', map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        # convert data to bytes to storage
        datum.data = X[i].tobytes()
        datum.label = int(Y[i])
        # generate a id as key
        str_id = '{:08}'.format(i)
        # put(key, value)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())