import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
rc('text', usetex=True)

# Results in csv format with ni,nj,processes,time,iter,idim,jdim,isize,jsize
results = pd.read_csv('results.csv')

means = results.groupby(['ni', 'nj', 'processes', 'isize', 'jsize']).mean()
stds = results.groupby(['ni', 'nj', 'processes', 'isize', 'jsize'])['time'].std()

# Return the data to its original data frame structure
means = means.reset_index()
stds = stds.reset_index()

def plot1():
  fig, ax = plt.subplots()
  colors = ['r', 'y', 'g', 'b']
  nis = [192, 256, 512, 768]
  images = ["edge192x128.pgm", "edge256x192.pgm", "edge512x384.pgm", "edge768x768.pgm"]
  plt.xscale('log')
  plt.yscale('log')

  # Extract each of the images from the ni parameter
  for i, ni in enumerate(nis):
    data = means[(means.ni == ni)]
    data_err = stds[(stds.ni == ni)]
    data.plot(ax=ax, kind='scatter', x='processes', y='time', yerr=data_err, c=colors[i], marker='_', s=100, label=images[i])

  plt.xlabel(r"\(number of processes\)")
  plt.ylabel(r"\(time/ms\)")
  plt.title(r"Average time of one iteration against the number of MPI processes")
  plt.legend(loc='best')
  plt.savefig("time_processes.png")

def plot2():
  
  # Calculate the size of the buf in each image
  means['buf_size'] = means['isize'] * means['jsize']

  # Let's plot the time against the buf size
  fig, ax = plt.subplots()
  colors = ['r', 'y', 'g', 'b']
  nis = [192, 256, 512, 768]
  images = ["edge192x128.pgm", "edge256x192.pgm", "edge512x384.pgm", "edge768x768.pgm"]
  plt.xscale('log')
  plt.yscale('log')

  for i, ni in enumerate(nis):
    data = means[(means.ni == ni)]
    data.plot(ax=ax, kind='scatter', x='buf_size', y='time', c=colors[i], label=images[i])

  plt.xlabel(r"\(buffer size\)")
  plt.ylabel(r"\(time/ms\)")
  plt.title(r"Average time of one iteration against the buffer size on each process")
  plt.legend(loc='best')
  plt.savefig("time_buf_size_log.png")


# Let's make one final graph showing the speed up
# First calculate the time taken when number of processes is 1
def plot3():
  fig, ax = plt.subplots()
  colors = ['r', 'y', 'g', 'b']
  nis = [192, 256, 512, 768]
  images = ["edge192x128.pgm", "edge256x192.pgm", "edge512x384.pgm", "edge768x768.pgm"]
  plt.xscale('log')
  plt.yscale('log')
  single = []

  for i, ni in enumerate(nis):
    im_means = means[(means.ni == ni)]
    single =  im_means[(means.processes == 1)].time.values[0]
    im_means['speed_up'] = im_means['time'].rdiv(single)

    im_means.plot(ax=ax, kind='scatter', x='processes', y='speed_up', c=colors[i], label=images[i])

  x_s = [1, 384]
  ax.plot(np.array([0.8, 25]), np.array([0.8, 25]), color='k', linestyle='dashed', linewidth=1)
  ax.axvline(x=24)

  plt.xlabel(r"\(number of processes\)")
  plt.ylabel(r"\(Speed Up\)")
  plt.title(r"Speed up against the number of MPI processes")
  plt.legend(loc='best')
  plt.savefig("strong_scaling.png")

# Find the largest speed up
single = means[(means.ni == 768) & (means.processes == 1)].time.values[0]
maximum = max(means[(means.ni == 768)]['time'].rdiv(single))

print("Max speed up :", maximum)

# Plot a graph of the weak scaling relationship
def plot4():
  buf_sizes = [3072, 6144, 12288, 24576]
  fig, ax = plt.subplots()
  colors = ['r', 'y', 'g', 'b']
  plt.xscale('log')

  for i, buf_size in enumerate(buf_sizes):
    data = means[(means.buf_size == buf_size)]
    data_err = stds[(means.buf_size == buf_size)]
    data.plot(ax=ax, kind='scatter', x='processes', y='time', yerr=data_err, c=colors[i], marker = '_', s=100, label=f"Domain size = {buf_size}")

  plt.xlabel(r"\(number of processes\)")
  plt.ylabel(r"\(Time/ms\)")
  plt.title(r"Demonstation of weak scaling relation")
  plt.legend(loc='best')
  plt.savefig("weak_scaling.png")


plot1()
plot2()
plot3()
plot4()
