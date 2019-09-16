import numpy as np
import os

X = np.fromfile("/home/sms/magnusro/frap_ann/generate_clean_data/x_1.bin", dtype = np.float32)
y = np.fromfile("/home/sms/magnusro/frap_ann/generate_clean_data/y_1.bin", dtype = np.float32)

shape      = (1, 110, 256, 256)
n_params   = 3
        
X = np.reshape(X, (-1, *shape))
y = np.reshape(y, (-1, n_params))
os.mkdir("data/spatiotemporal")

for i in range(X.shape[0]):
    np.save(os.path.join("data/spatiotemporal", "x_"+str(i+1)), X[i, :, :, :, :])
    np.save(os.path.join("data/spatiotemporal", "y_"+str(i+1)), y[i, :])