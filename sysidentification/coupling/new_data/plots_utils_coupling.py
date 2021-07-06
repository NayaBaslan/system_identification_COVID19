from matplotlib import pyplot as plt
import numpy as np
import pdb
import os

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


my_path = os.path.abspath(__file__)
months = [r'March','April', r'May', r'June', r'July', r'August', r'September', r'October', r'November',r'December']
months_2 = ['August', 'September', 'October', 'November','December']
first_second_wave = ['April', 'May', 'June', 'July', 'August', 'September', 'October','November','December', 'January', 'February', 'March','April']


def plot_active(t, active, age_groups, N):
    plt.title(r"Total Active Infections Germany ", fontsize = 14)
    colors = ["navy", "forestgreen", "darkgoldenrod", "brown", "purple"]
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
    plt.ylabel(r'Active Infections $\cdot 10^3$', fontsize=14)

    plt.xlabel(r"Months", fontsize=14)
    plt.xticks(np.arange(1, N +1, step=4), months, rotation=30, fontsize=14)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')

    plt.savefig(os.getcwd() + "/figures/coupling/new_data/active_infections.pdf", bbox_inches='tight')

    # plt.semilogy()
    # plt.yscale('log')

def plot_active_sim(t, active, age_groups, N, bounds):
    plt.title(r"Total Simulated Active Infections Germany ", fontsize = 14)
    colors = [r"navy", "forestgreen", "darkgoldenrod", "brown", "purple"]
    colors_fill = ["mediumslateblue", "springgreen", "peachpuff", "lightpink", "lavender"]

    scaley = 1e3
    for i in range(age_groups):
        # plt.plot(t, active[i, :], label='Age Group %s ' % i)
        plt.plot(t, active[i, :] / scaley, color=colors[i])

        plt.fill_between(t, active[i, :]/ scaley-5*i/10, active[i, :]/ scaley+5*i/10, facecolor=colors_fill[i])
    plt.legend(["A 00-19", "A 20-39", "A 40-59", "A 60-79", "A  80+"], fontsize=14)
    if (N==20):
        plt.xticks(np.arange(1, N +1, step=4), months, rotation=30, fontsize=14)
    elif(N==17):
        plt.xticks(np.arange(1, N + 1, step=4), months_2, rotation=30, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r'Active Infections $\cdot 10^3$', fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')

    plt.savefig(os.getcwd() + "/figures/coupling/new_data/active_sim_infections2.pdf", bbox_inches='tight')

def plot_active_sim_real(t, active_model, active_real, age_groups, N, bounds):
    plt.title(r"Real vs. Model Active Infections for A00-19 ", fontsize = 14)
    colors = ["navy", "forestgreen", "darkgoldenrod", "brown", "purple"]
    colors_fill = ["mediumslateblue", "springgreen", "peachpuff", "lightpink", "lavender"]

    scaley = 1e3
    # plt.plot(t, active_real[0, :], label='Age Group %s ' % 0)
    # plt.plot(t, active_model[0, :], label='Age Group %s ' % 0)
    # plt.fill_between(t, active_model[0, :]/ scaley-5*0/10, active_model[0, :]/ scaley+5*0/10, facecolor=colors_fill[0])

    fig, axs = plt.subplots(5,  sharex=True, sharey=True)
    fig.suptitle(r"Real vs. Model Active Infections per Age Group", fontsize = 14)
    fig.text(0.04, 0.5, r'Active Infections $\cdot 10^3$', fontsize=14, va='center', rotation='vertical')
    pdb.set_trace()
    axs[0].plot(t, active_real[0, :]/ scaley, color='darkslategray', label='Age Group %s ' % 0)

    axs[0].plot(t, active_model[0, :]/ scaley, color='gold', label='Age Group %s ' % 0)
    axs[0].fill_between(t, active_model[0, :] / scaley - 5 * 0 / 10, active_model[0, :] / scaley + 5 * 0 / 10, facecolor=colors_fill[0])
    axs[0].set_title(r'Age Group A00-19')

    axs[1].plot(t, active_real[1, :]/ scaley, color='darkslategray', label='Age Group %s ' % 0, )
    axs[1].plot(t, active_model[1, :] / scaley,color='gold', label='Age Group %s ' % 0)
    axs[1].fill_between(t, active_model[1, :] / scaley - 5 * 1 / 10, active_model[1, :] / scaley + 5 * 1 / 10, facecolor=colors_fill[0])
    axs[1].set_title(r'Age Group A20-39')
    axs[2].plot(t, active_real[2, :]/ scaley, color='darkslategray', label='Age Group %s ' % 0)
    axs[2].plot(t, active_model[2, :] / scaley,color='gold', label='Age Group %s ' % 0)
    axs[2].fill_between(t, active_model[2, :] / scaley - 5 * 2 / 10, active_model[2, :] / scaley + 5 * 2 / 10,  facecolor=colors_fill[0])
    axs[2].set_title(r'Age Group A40-59')
    axs[3].plot(t, active_real[3, :]/ scaley, color='darkslategray', label='Age Group %s ' % 0)
    axs[3].plot(t, active_model[3, :] / scaley, color='gold',label='Age Group %s ' % 0)
    axs[3].set_title(r'Age Group A60-79')
    axs[3].fill_between(t, active_model[3, :] / scaley - 5 * 3 / 10, active_model[3, :] / scaley + 5 * 3 / 10,  facecolor=colors_fill[0])
    axs[4].plot(t, active_real[4, :] / scaley, color='darkslategray', label='Age Group %s ' % 0)
    axs[4].plot(t, active_model[4, :] / scaley, color='gold',label='Age Group %s ' % 0)
    axs[4].fill_between(t, active_model[4, :] / scaley - 5 * 4 / 10, active_model[4, :] / scaley + 5 * 4 / 10,  facecolor=colors_fill[0])
    axs[4].set_title(r'Age Group 80+')

    plt.sca(axs[0])
    plt.yticks(fontsize=14)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')
    plt.sca(axs[1])
    plt.yticks(fontsize=14)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')
    plt.sca(axs[2])
    plt.yticks(fontsize=14)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')
    plt.sca(axs[3])
    plt.yticks(fontsize=14)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')
    plt.sca(axs[4])
    plt.yticks(fontsize=14)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')

    # for ax in axs.flat:
    #     ax.set(xlabel='Months', ylabel=r'Active Infections $\cdot 10^3$')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    #
    # for i in range(age_groups):
    #     plt.figure()
    #     plt.plot(t, active_real[i, :], label='Age Group %s ' % 0)
    #     plt.plot(t, active_model[i, :], label='Age Group %s ' % 0)
    #     # plt.plot(t, active[i, :], label='Age Group %s ' % i)
    #     # plt.plot(t, active[i, :] / scaley, color=colors[i])

    plt.legend([r"Simualated Active Infections", r"Real Active Infections"], fontsize=14)
    if (N==20):
        plt.xticks(np.arange(1, N +1, step=4), months, rotation=30, fontsize=14)
    elif(N==17):
        plt.xticks(np.arange(1, N + 1, step=4), months_2, rotation=30, fontsize=14)
    plt.yticks(fontsize=14)
    # plt.ylabel(r'Active Infections $\cdot 10^3$', fontsize=14)
    plt.xlabel("Months", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')

    plt.savefig(os.getcwd() + "/figures/coupling/new_data/active_sim_infections.pdf", bbox_inches='tight')

def plot_deaths_sim(t, dead, age_groups, N, bounds):
    plt.title(r"Total Simulated Deaths Germany ", fontsize = 14)
    colors = ["navy", "forestgreen", "darkgoldenrod", "brown", "purple"]
    colors_fill = ["mediumslateblue", "springgreen", "peachpuff", "lightpink", "lavender"]

    scaley = 1e3
    for i in range(age_groups):
        # plt.plot(t, active[i, :], label='Age Group %s ' % i)
        plt.plot(t, dead[i, :] / scaley, color=colors[i])

        plt.fill_between(t, dead[i, :]/ scaley-5*i/100, dead[i, :]/ scaley+5*i/100, facecolor=colors_fill[i])
    plt.legend(["A 00-19", "A 20-39", "A 40-59", "A 60-79", "A  80+"], fontsize=14)
    if (N==20):
        plt.xticks(np.arange(1, N +1, step=4), months, rotation=30, fontsize=14)
    elif(N==17):
        plt.xticks(np.arange(1, N + 1, step=4), months_2, rotation=30, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Deaths x 1e3", fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')

    plt.savefig(os.getcwd() + "/figures/coupling/new_data/sim_deaths.pdf", bbox_inches='tight')

def plot_deaths_real(t, dead, age_groups, N):
    plt.title(r"Total Deaths Germany (Real Data)", fontsize = 14)
    colors = ["navy", "forestgreen", "darkgoldenrod", "brown", "purple"]
    colors_fill = ["mediumslateblue", "springgreen", "peachpuff", "lightpink", "lavender"]

    scaley = 1e3
    for i in range(age_groups):
        # plt.plot(t, active[i, :], label='Age Group %s ' % i)
        plt.plot(t, dead[i, :] / scaley, color=colors[i])
    plt.legend(["A 00-19", "A 20-39", "A 40-59", "A 60-79", "A  80+"], fontsize=14)
    if (N==20):
        plt.xticks(np.arange(1, N +1, step=4), months, rotation=30, fontsize=14)
    elif(N==17):
        plt.xticks(np.arange(1, N + 1, step=4), months_2, rotation=30, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Deaths x 1e3", fontsize=14)
    plt.xlabel("Months", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')

    plt.savefig(os.getcwd() + "/figures/coupling/new_data/real_deaths.pdf", bbox_inches='tight')

def plot_cumulative(male, female, diverse):

    N = 5
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27  # the width of the bars

    fig = plt.figure()
    plt.title(r"Cumulative Infections in Germany on 06/12/2020", fontsize = 14)
    plt.yticks(fontsize = 14)
    ax = fig.add_subplot(111)

    scaley = 1e3
    yvals = male/scaley
    rects1 = ax.bar(ind, yvals, width, color='chocolate')
    zvals = female/scaley
    rects2 = ax.bar(ind + width, zvals, width, color='wheat')
    kvals = diverse/scaley
    rects3 = ax.bar(ind + width * 2, kvals, width, color='darkcyan')

    ax.set_ylabel(r'Cumulative Infections $\cdot 10^3$', fontsize = 14)
    ax.set_xlabel(r'Age Groups', fontsize = 14)

    ax.set_xticks(ind + width)
    ax.set_xticklabels(('A 00-19', 'A 20-39', 'A 40-59', 'A 60-79','A  80 +'), fontsize = 14)
    ax.legend((rects1[0], rects2[0], rects3[0]), ('Male', 'Female', 'Diverse'), fontsize = 14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/Cumulative_infections.pdf", bbox_inches='tight')

def plot_betas_diag(betas, N, size_one, age_groups, start):
    current_beta = betas[27 * size_one:27 * size_one + size_one]
    beta_mat = np.zeros((age_groups, age_groups))
    indices = np.tril_indices(age_groups)
    beta_mat[indices] = np.reshape(current_beta, 15)

    beta_t = beta_mat + beta_mat.T - np.diag(np.diag(beta_mat))

    max_abs = np.max(np.abs(beta_t))
    for i in range(N-1):
      current_beta = betas[i*size_one :i*size_one + size_one]
      beta_mat = np.zeros((age_groups, age_groups))
      indices = np.tril_indices(age_groups)
      beta_mat[indices] = np.reshape(current_beta,15)

      beta_t = beta_mat + beta_mat.T - np.diag(np.diag(beta_mat))

      # max_abs = np.max(np.abs(beta_t))
      plt.figure()
      plt.title(r"Transmission Rate Matrix  Calendar Week " + str(start + i), fontsize = 14)
      plt.imshow(beta_t, cmap='RdBu', vmin=-max_abs, vmax=max_abs)
      # plt.colorbar()
      cbar = plt.colorbar()
      for t in cbar.ax.get_yticklabels():
          t.set_fontsize(14)
      plt.xlabel(r"Age Groups", fontsize = 14)
      plt.ylabel(r"Age Groups", fontsize = 14)

      plt.xticks(ticks= np.arange(age_groups), labels=['A 00-19', 'A 20-39', 'A 40-59', 'A 60-79', 'A  80+'], rotation=30, fontsize = 14)
      plt.yticks(ticks=np.arange(age_groups), labels=['A 00-19', 'A 20-39', 'A 40-59', 'A 60-79', 'A  80+'],rotation=30, fontsize = 14)

      # plt.set_xticklabels(('A 00-19', 'A 20-39', 'A 40-59', 'A 60-79', 'A  80 +'), fontsize=14)

      plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas/betas" + str(i) + ".pdf", bbox_inches='tight')


# def plot_betas_tv(tspan, beta_00, beta_11, beta_22, beta_33, beta_44):
#     plt.plot(tspan, beta_00)
#     plt.plot(tspan, beta_11)
#     plt.plot(tspan, beta_22)
#     plt.plot(tspan, beta_33)
#     plt.plot(tspan, beta_44)
#     plt.legend(["beta_00", "beta_11", "beta_22", "beta_33", "beta_44"], loc="upper right")
#     plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv.pdf", bbox_inches='tight')

def plot_betas_tv(t, betas, N, size_one, age_groups):
    beta_00 = []; beta_11 = []; beta_22 = [] ; beta_33 = [] ; beta_44 = []; beta_55 = []
    beta_01 = []; beta_02 = []; beta_03 = [];  beta_04 = [];
    beta_12 = []; beta_13 = []; beta_14 = [];
    beta_23 = []; beta_24 = [];
    beta_34 = [];

    avg_betas = 0
    for i in range(N-1):
      current_beta = betas[i*size_one :i*size_one + size_one]
      beta_mat = np.zeros((age_groups, age_groups))
      indices = np.tril_indices(age_groups)
      beta_mat[indices] = np.reshape(current_beta,15)
      avg_betas = avg_betas + beta_mat


      beta_t = beta_mat + beta_mat.T - np.diag(np.diag(beta_mat))
      beta_00.append(np.diag(beta_mat)[0])
      beta_11.append(np.diag(beta_mat)[1])
      beta_22.append(np.diag(beta_mat)[2])
      beta_33.append(np.diag(beta_mat)[3])
      beta_44.append(np.diag(beta_mat)[4])

      beta_01.append(beta_mat[0,1])
      beta_02.append(beta_mat[0, 2])
      beta_03.append(beta_mat[0, 3])
      beta_04.append(beta_mat[0, 4])

      beta_12.append(beta_mat[1, 1])
      beta_13.append(beta_mat[1, 2])
      beta_14.append(beta_mat[1, 3])

      beta_23.append(beta_mat[2, 3])
      beta_24.append(beta_mat[2, 4])

      beta_34.append(beta_mat[3, 4])
      #
      max_abs = np.max(np.abs(beta_t))
    plt.figure()
    plt.plot(t, beta_00)
    plt.title(r'Tranmission Rate: $\beta_{00}$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Value of Transmission rate", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv/beta_00.pdf", bbox_inches='tight')

    plt.figure()
    plt.plot(t, beta_11)
    plt.title(r'Tranmission Rate: $\beta_{11}$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Value of Transmission rate", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv/beta_11.pdf", bbox_inches='tight')

    plt.figure()
    plt.plot(t, beta_22)
    plt.title(r'Tranmission Rate: $\beta_{22}$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Value of Transmission rate", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv/beta_22.pdf", bbox_inches='tight')

    plt.figure()
    plt.plot(t, beta_33)
    plt.title(r'Tranmission Rate: $\beta_{33}$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Value of Transmission rate", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv/beta_33.pdf", bbox_inches='tight')

    plt.figure()
    plt.plot(t, beta_44)
    plt.title(r'Tranmission Rate: $\beta_{44}$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Value of Transmission rate", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv/beta_44.pdf", bbox_inches='tight')

    plt.figure()
    plt.plot(t, beta_01)
    plt.title(r'Tranmission Rate: $\beta_{01}$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Value of Transmission rate", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv/beta_01.pdf", bbox_inches='tight')

    plt.figure()
    plt.plot(t, beta_02)
    plt.title(r'Tranmission Rate: $\beta_{02}$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Value of Transmission rate", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv/beta_02.pdf", bbox_inches='tight')

    plt.figure()
    plt.plot(t, beta_03)
    plt.title(r'Tranmission Rate: $\beta_{03}$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Value of Transmission rate", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv/beta_03.pdf", bbox_inches='tight')

    plt.figure()
    plt.plot(t, beta_04)
    plt.title(r'Tranmission Rate: $\beta_{04}$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Value of Transmission rate", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv/beta_04.pdf", bbox_inches='tight')

    plt.figure()
    plt.plot(t, beta_12)
    plt.title(r'Tranmission Rate: $\beta_{12}$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Value of Transmission rate", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv/beta_12.pdf", bbox_inches='tight')

    plt.figure()
    plt.plot(t, beta_13)
    plt.title(r'Tranmission Rate: $\beta_{13}$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Value of Transmission rate", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv/beta_13.pdf", bbox_inches='tight')

    plt.figure()
    plt.plot(t, beta_14)
    plt.title(r'Tranmission Rate: $\beta_{14}$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Value of Transmission rate", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv/beta_14.pdf", bbox_inches='tight')

    plt.figure()
    plt.plot(t, beta_23)
    plt.title(r'Tranmission Rate: $\beta_{23}$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Value of Transmission rate", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv/beta_23.pdf", bbox_inches='tight')

    plt.figure()
    plt.plot(t, beta_24)
    plt.title(r'Tranmission Rate: $\beta_{24}$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Value of Transmission rate", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv/beta_24.pdf", bbox_inches='tight')

    plt.figure()
    plt.plot(t, beta_34)
    plt.title(r'Tranmission Rate: $\beta_{34}$', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Value of Transmission rate", fontsize=14)
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.savefig(os.getcwd() + "/figures/coupling/new_data/betas_tv/beta_34.pdf", bbox_inches='tight')


def plot_betas(t, betas, run_id, ub_betas, lb_betas,N, dataset):

    fig = plt.step(t, betas, color='dimgray')
    plt.title(r"Transmission rate " + run_id)

    if (N == 150):
        plt.xticks(np.arange(0, N+1, step=30), first_wave)
        plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(run_id) + "_" + str(dataset) + "first_wave.pdf")
    if (N == 100):
        plt.xticks(np.arange(0, N + 1, step=30), second_wave)
        plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(run_id) + "_" + str(dataset) + "second_wave.pdf")
    if (N == 330):
        plt.xticks(np.arange(0, N + 1, step=30), first_second_wave)
        plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(run_id) + "_" + str(dataset) + "first_second_wave.pdf")

    ub_betas = np.reshape(ub_betas, (N-1))
    lb_betas = np.reshape(lb_betas, (N-1))
    plt.fill_between(t, lb_betas, ub_betas, facecolor='lightgrey')

def plot_state_trajectories(t, S_model, I_model, R_model, D_model, S_real, I_real, R_real, D_real, N1, all_S_ub, all_S_lb,
                            ub_S, lb_S, ub_I, lb_I, ub_R, lb_R, ub_D, lb_D, dataset):
    fig, axs = plt.subplots(4, 1, figsize=(15, 8), sharex=True, sharey=False)
    plt.suptitle(r"State Trajectory Estimations")
    plt.setp(axs[0], ylabel='susceptible')
    plt.setp(axs[1], ylabel='infected')
    plt.setp(axs[2], ylabel='recovered')
    plt.setp(axs[3], ylabel='dead')

    axs[0].plot(t, S_model, color='dimgray')
    all_S_lb = np.reshape(all_S_lb, (N1))
    all_S_ub = np.reshape(all_S_ub, (N1))

    ub_S = np.reshape(ub_S, (N1))
    lb_S = np.reshape(lb_S, (N1))

    ub_I = np.reshape(ub_I, (N1))
    lb_I = np.reshape(lb_I, (N1))

    ub_R = np.reshape(ub_R, (N1))
    lb_R = np.reshape(lb_R, (N1))

    ub_D = np.reshape(ub_D, (N1))
    lb_D = np.reshape(lb_D, (N1))

    axs[0].fill_between(t, lb_S, ub_S, facecolor='lightgrey')
    axs[0].plot(t, S_real, color='k')
    axs[0].legend(["Model Data Susceptible", "Real Data Susceptible"], loc="upper right")
    axs[0].axvline(x=43-13, color='r')
    axs[0].axvline(x=61-13, color='r')
    axs[0].axvline(x=105-13, color='r')
    axs[0].axvline(x=217-13, color='r')
    axs[0].axvline(x=256-13, color='r')
    axs[0].axvline(x=267-13, color='r')
    axs[0].axvline(x=273-13, color='r')

    axs[1].plot(t, I_model, color='lightskyblue')
    axs[1].plot(t, I_real, color='navy')
    axs[1].legend(["Model Data Infected (Active Cases)", "Real Data Infected (Active Cases)"], loc="upper right")
    axs[1].fill_between(t, lb_I, ub_I, facecolor='lightsteelblue')
    axs[1].axvline(x=43-13, color='r')
    axs[1].axvline(x=61-13, color='r')
    axs[1].axvline(x=105-13, color='r')
    axs[1].axvline(x=217-13, color='r')
    axs[1].axvline(x=256-13, color='r')
    axs[1].axvline(x=267-13, color='r')
    axs[1].axvline(x=273-13, color='r')

    axs[2].plot(t, R_model, color='mediumseagreen')
    axs[2].plot(t, R_real, color='darkgreen')
    axs[2].legend(["Model Data Recovered", "Real Data Recovered"], loc="upper right")
    axs[2].fill_between(t, lb_R, ub_R, facecolor='lightgreen')
    axs[2].axvline(x=43-13, color='r')
    axs[2].axvline(x=61-13, color='r')
    axs[2].axvline(x=105-13, color='r')
    axs[2].axvline(x=217-13, color='r')
    axs[2].axvline(x=256-13, color='r')
    axs[2].axvline(x=267-13, color='r')
    axs[2].axvline(x=273-13, color='r')

    axs[3].plot(t, D_model, color='salmon')
    axs[3].plot(t, D_real, color='darkred')
    axs[3].fill_between(t, lb_D, ub_D, facecolor='mistyrose')
    axs[3].legend(["Model Data Dead", "Real Data Dead"], loc="upper right")
    axs[3].set_xticks(np.arange(0, N1 + 2, step=30))
    axs[3].axvline(x=43-13, color='r')
    axs[3].axvline(x=61-13, color='r')
    axs[3].axvline(x=105-13, color='r')
    axs[3].axvline(x=217-13, color='r')
    axs[3].axvline(x=256-13, color='r')
    axs[3].axvline(x=267-13, color='r')
    axs[3].axvline(x=273-13, color='r')

    if (N1 == 150):
        axs[3].set_xticklabels(first_wave)
        plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(dataset) + "_first_wave_state_estimations.pdf")
    if (N1 == 100):
        axs[3].set_xticklabels(second_wave)
        plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(dataset) + "_second_wave_state_estimations.pdf")
    if (N1 == 330):
        axs[3].set_xticklabels(first_second_wave)
        plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(dataset) + "_first_second_wave_state_estimations.pdf")

def plot_residuals(t, res_S, res_I, res_R, res_D, N1,  ub_res_S, lb_res_S, ub_res_I, lb_res_I,ub_res_R, lb_res_R,ub_res_D, lb_res_D, dataset):
    fig, axs = plt.subplots(4, 1, figsize=(15, 8), sharex=True, sharey=False)
    # plt.title("Residual Values")
    plt.suptitle("Residual Values")
    plt.setp(axs[0], ylabel='residuals')
    plt.setp(axs[1], ylabel='residuals')
    plt.setp(axs[2], ylabel='residuals')
    plt.setp(axs[3], ylabel='residuals')
    plt.setp(axs[3], xlabel='time')
    axs[3].set_xticks(np.arange(0, N1 + 2, step=30))

    ub_res_S = np.reshape(ub_res_S, (N1))
    lb_res_S = np.reshape(lb_res_S, (N1))
    axs[0].fill_between(t, lb_res_S, ub_res_S, facecolor='lightgrey')
    axs[0].step(t, res_S, color='k')
    axs[0].legend(["Residuals Susceptible"], loc="upper right")

    ub_res_I = np.reshape(ub_res_I, (N1))
    lb_res_I = np.reshape(lb_res_I, (N1))
    axs[1].fill_between(t, lb_res_I, ub_res_I, facecolor='lightsteelblue')
    axs[1].step(t, res_I, color='navy')
    axs[1].legend(["Residuals Infected"], loc="upper right")

    ub_res_R = np.reshape(ub_res_R, (N1))
    lb_res_R = np.reshape(lb_res_R, (N1))
    axs[2].fill_between(t, lb_res_R, ub_res_R, facecolor='lightgreen')
    axs[2].step(t, res_R, color='darkgreen')
    axs[2].legend(["Residuals Recovered"], loc="upper right")

    ub_res_D = np.reshape(ub_res_D, (N1))
    lb_res_D = np.reshape(lb_res_D, (N1))
    axs[3].fill_between(t, lb_res_D, ub_res_D, facecolor='mistyrose')
    axs[3].step(t, res_D, color='darkred')
    axs[3].legend(["Residuals Dead"], loc="upper right")

    if (N1 == 150):
        axs[3].set_xticklabels(first_wave)
        plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(dataset) + "first_wave_residuals.pdf")
    if (N1 == 100):
        axs[3].set_xticklabels(second_wave)
        plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(dataset) + "second_wave_residuals.pdf")
    if (N1 == 330):
        axs[3].set_xticklabels(first_second_wave)
        plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(dataset) + "first_second_residuals.pdf")

def show_cov(sigma_theta, max_abs, dataset):
    plt.imshow(sigma_theta, cmap='RdBu', vmin=-max_abs, vmax=max_abs)
    plt.colorbar()
    plt.title("rCovariance Matrix of the Parameter Estimates")
    plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(dataset)+ "_covariance_theta.pdf")

def plot_errors(t, error_S, error_I, error_R, error_D, pop, N1, dataset):
    plt.plot(t, error_S / pop * 100)
    plt.plot(t, error_I / pop * 100)
    plt.plot(t, error_R / pop * 100)
    plt.plot(t, error_D / pop * 100)
    plt.title("rAbsolute Error (% of Total Population of Germany)")
    plt.legend(["Error Susceptible ", "Error Infected", "Error Recovered", "Error Dead"], loc="upper right")

    if (N1 == 150):
        plt.xticks(np.arange(0, N1 + 1, step=30),first_wave)
        plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(dataset) + "first_wave_errors.pdf")
    if (N1 == 100):
        plt.xticks(np.arange(0, N1 + 1, step=30),second_wave)
        plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(dataset) + "second_wwave_errors.pdf")
    if (N1 == 330):
        plt.xticks(np.arange(0, N1 + 1, step=30),first_second_wave)
        plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(dataset) + "first_second_wave_errors.pdf")

def plot_active_sim_bound(t, active, age_groups, N,  ub, lb):
    plt.title(r"Total Simulated Active Infections Germany ", fontsize = 14)
    colors = ["navy", "forestgreen", "darkgoldenrod", "brown", "purple"]
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
    plt.xticks(np.arange(1, N + 1, step=4), months, rotation=30, fontsize=14)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')

    plt.savefig(os.getcwd() + "/figures/coupling/new_data/active_sim_infections_ub_lb.pdf", bbox_inches='tight')