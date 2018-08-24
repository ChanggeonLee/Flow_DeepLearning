import matplotlib.pyplot as plt
import h5py
import numpy as np

shape = [128, 256]
# open the file
with h5py.File("fluid_flow_0002.h5") as dataH5:
  # get the keys
  key_list = dataH5.keys()
  print(key_list)

  flow = np.array(dataH5['Velocity_0'][:])
  flow = flow.reshape([shape[0], shape[1]+128, 3])[0:shape[0],0:shape[1],0:2]

  boundary = np.array(dataH5['Gamma'][:])
  boundary = boundary.reshape([shape[0], shape[1]+128, 1])[0:shape[0],0:shape[1],:]

  sflow_plot = np.concatenate([flow], axis=1)
  boundary_concat = np.concatenate(3*[boundary], axis=2)
  sflow_plot = np.sqrt(np.square(sflow_plot[:,:,0])) - .05 *boundary_concat[:,:,0]

  plt.imshow(sflow_plot)
  plt.colorbar()
  plt.show()
