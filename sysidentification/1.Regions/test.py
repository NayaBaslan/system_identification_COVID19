import random
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(3,3, figsize=(15, 8), sharex=True, sharey=True)

for i, ax in enumerate(axs.flat):
    ax.scatter(*np.random.normal(size=(2,200)))
    ax.set_title(f'Title {i}')

# set labels
plt.setp(axs[-1, :], xlabel='x axis label')
plt.setp(axs[:, 0], ylabel='y axis label')


fig, axs = plt.subplots(4,1, figsize=(15, 8), sharex=True, sharey=True)
plt.setp(axs[0], ylabel='susceptible')
plt.setp(axs[1], ylabel='infected')
plt.setp(axs[2], ylabel='recovered')
plt.setp(axs[3], ylabel='dead')

ax[3].plot(x, y)

plt.show()