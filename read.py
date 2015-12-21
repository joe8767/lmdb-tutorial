# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:53:30 2015

@author: joe

Reference:
http://deepdish.io/2015/04/28/creating-lmdb-in-python/
"""

import numpy as np
import lmdb
import caffe

env = lmdb.open('mylmdb', readonly=True)

with env.begin() as txn:
    # using b'00000000' as key, value returned as raw_datum
    raw_datum = txn.get(b'00000000')
    
# parse value returned from lmdb
datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

# parse string to uint8 (write process: datum.data = X[i].tobytes())
flat_x = np.fromstring(datum.data, dtype=np.uint8)

# reshape using other propoties (write process: datum.channels = X.shape[1]...)
x = flat_x.reshape(datum.channels, datum.height, datum.width)
# (write process: datum.label = int(Y[i]))
y = datum.label

# Iteration
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        print datum.channels
        print datum.height
        print datum.width
        print datum.label
        
        raw_data = datum.data # hex values
        # data is uint8 in write process, turn it back to uint8 from hex
        int_data = np.fromstring(raw_data, dtype=np.uint8) 
        # reshape to get the data
        data = int_data.reshape(datum.channels, datum.height, datum.width)










