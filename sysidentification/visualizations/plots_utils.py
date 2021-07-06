from matplotlib import pyplot as plt
import numpy as np
import pdb
import os
from matplotlib import pyplot as plt
import numpy as np
import pdb
import os
my_path = os.path.abspath(__file__)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


months = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November','December']
months_2 = ['August', 'September', 'October', 'November','December']
first_second_wave = ['April', 'May', 'June', 'July', 'August', 'September', 'October','November','December', 'January', 'February', 'March','April']


def plot_new_infections(weeks, infections, dataset):

    ind = np.arange(weeks)  # the x locations for the groups
    width = 0.9  # the width of the bars
    fig = plt.figure()
    plt.title(r"Weekly Infections in " + str(dataset), fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.bar(ind,infections, width, color='chocolate')
    plt.ylabel(r"New Weekly Infections", fontsize=14)
    plt.xticks(np.arange(1, weeks +1, step=5), months, rotation=30, fontsize=12)
    plt.savefig(os.getcwd() + "/sysidentification/figures/datasource/weekly_infections.pdf", bbox_inches='tight')


def plot_new_deaths(weeks, deaths, dataset):

    ind = np.arange(weeks)  # the x locations for the groups
    width = 0.9  # the width of the bars
    fig = plt.figure()
    plt.title(r"Weekly Deaths in " + str(dataset), fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.bar(ind,deaths, width, color='chocolate')
    plt.ylabel(r"New Weekly Deaths", fontsize=14)
    plt.xticks(np.arange(1, weeks +1, step=5), months, rotation=30, fontsize=12)
    plt.savefig(os.getcwd() + "/sysidentification/figures/datasource/weekly_deaths.pdf", bbox_inches='tight')


def plot_active(t, active, age_groups, N):
    plt.title(r"Total Active Infections Germany ", fontsize = 14)
    colors = [r"lightsteelblue", "slategrey", "cornflowerblue", "royalblue", "mediumblue"]
    scaley = 1e3
    for i in range(age_groups):
        # plt.plot(t, active[i, :], label='Age Group %s ' % i)
        plt.plot(t, active[i, :] / scaley, color=colors[i])
    plt.legend(["A 00-19", "A 20-39", "A 40-59", "A 60-79", "A  80+"], fontsize=14)
    if (N==20):
        plt.xticks(np.arange(1, N +1, step=4), months, rotation=30, fontsize=14)
    elif(N==17):
        plt.xticks(np.arange(1, N + 1, step=4), months_2, rotation=30, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Active Infections x 1e3", fontsize=14)
    plt.xlabel(r"Months", fontsize=14)

    plt.savefig(os.getcwd() + "/figures/coupling/new_data/active_infections.pdf", bbox_inches='tight')