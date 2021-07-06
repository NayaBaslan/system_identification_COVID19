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


second_wave = ['November','December', 'January', 'February', 'March','April']
first_second_wave = ['April', 'May', 'June', 'July', 'August', 'September', 'October','November','December', 'January', 'February', 'March','April']

months = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November','December']
first_wave = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November','December']

months = ['April', 'May', 'June', 'July', 'August', 'September', 'October','November','December', 'January', 'February', 'March','April']
months_weekly = ['March', 'April', 'May', 'June', 'July', 'August', 'September', 'October','November','December', 'January', 'February', 'March','April']


def plot_betas(t, betas, ub_betas, lb_betas,N, dataset, alpha_reg):
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
    fig = plt.step(t, betas, color='dimgray')
    # plt.title(r'$ \alpha $', fontsize='small')

    if ((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2))) == 7):

        plt.title(r'Values of $\beta$ Using $\bar{\alpha} = 10^7 $' , fontsize=18)

    elif ((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2))) == 6):
        plt.title(r'Values of $\beta$ using $\bar{\alpha} = 10^6$' , fontsize=18)

    elif ((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2))) == 5):
        plt.title(r'Values of $\beta$ using $\bar{\alpha} = 10^5$', fontsize=18)

    plt.grid(b=True, which='major', color='lightgray', linestyle='-')

    ub_betas = np.reshape(ub_betas, (N-1))
    lb_betas = np.reshape(lb_betas, (N-1))
    plt.fill_between(t, lb_betas, ub_betas, facecolor='lightgrey')
    plt.xticks(np.arange(0, N + 1, step=30), months_weekly, rotation=30, fontsize=14)
    plt.ylim((0, 1.7))
    plt.xlabel("Months", fontsize=14)
    plt.ylabel("Time Varying Transmission Rate" , fontsize=14)
    plt.yticks(fontsize=14)

    # if (N == 150):
    #     plt.xticks(np.arange(0, N+1, step=30), first_wave, rotation=30,fontsize=14)
    #     plt.savefig(os.getcwd() + "/sysidentification/figures/freiburg/" + "_" + str((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2)))) + "first_wave.pdf", bbox_inches='tight')
    # if (N == 100):
    #     plt.xticks(np.arange(0, N + 1, step=30), second_wave, rotation=30,fontsize=14)
    #     plt.savefig(os.getcwd() + "/sysidentification/figures/freiburg/"  + "_" + str((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2)))) + "second_wave.pdf", bbox_inches='tight')
    # if (N == 330):
    #     plt.xticks(np.arange(0, N + 1, step=30), months_weekly,rotation=30, fontsize=14)
    plt.savefig(os.getcwd() + "/sysidentification/figures/freiburg/betas/" +  "betas_save" + str((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2)))) + ".pdf", bbox_inches='tight')


def plot_r0(t, r0, r0_lb, r0_ub,N, dataset, alpha_reg):
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
    fig = plt.plot(t, r0, color='dimgray')
    # plt.title(r'$ \alpha $', fontsize='small')
    if ((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2))) == 7):
        plt.title(r' Values of $R_0 $ using $\bar{\alpha} = 10^7 $' , fontsize=18)
    elif ((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2))) == 6):
        plt.title(r' Values of $R_0 $ using $\bar{\alpha} = 10^6 $', fontsize=18)
    elif ((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2))) == 5):
        plt.title(r' Values of $R_0 $ using $\bar{\alpha} = 10^5 $', fontsize=18)

    plt.grid(b=True, which='major', color='lightgray', linestyle='-')

    plt.ylim((0, 2.5))

    ro_ub = np.reshape(r0_ub, (N-1))
    ro_lb = np.reshape(r0_lb, (N-1))
    plt.fill_between(t, r0_lb, r0_ub, facecolor='lightgrey')
    plt.xticks(np.arange(0, N + 1, step=30), months_weekly, rotation=30, fontsize=14)
    plt.xlabel("Months", fontsize=14)
    plt.ylabel("Time Varying Reproduction Number" , fontsize=14)
    plt.yticks(fontsize=14)


    plt.savefig(os.getcwd() + "/sysidentification/figures/freiburg/r0/" +  "r0_" + str((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2)))) + ".pdf", bbox_inches='tight')

def plot_r0_4(t, r0, r0_lb, r0_ub,N, dataset, alpha_reg):
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

    r0_4 = r0[0:N-1:4]
    t_4 = t[0:N - 1:4]

    # fig = plt.plot(t, r0, color='dimgray')
    fig = plt.plot(t_4, r0_4, color='dimgray')

    # plt.title(r'$ \alpha $', fontsize='small')
    if ((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2))) == 7):
        plt.title(r' 4 Time Varying Basic Reproduction Number $R_0 $ with $\alpha = 10^7 $' , fontsize=18)
    elif ((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2))) == 6):
        plt.title(r' 4 Time Varying Basic Reproduction Number $R_0 $ with $\alpha = 10^6 $', fontsize=18)
    elif ((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2))) == 5):
        plt.title(r' 4 Time Varying Basic Reproduction Number $R_0 $ with $\alpha = 10^5 $', fontsize=18)

    plt.grid(b=True, which='major', color='lightgray', linestyle='-')


    ro_ub = np.reshape(r0_ub, (N-1))
    ro_lb = np.reshape(r0_lb, (N-1))
    plt.fill_between(t, r0_lb, r0_ub, facecolor='lightgrey')
    plt.xticks(np.arange(0, N + 1, step=30), months_weekly, rotation=30, fontsize=14)
    plt.xlabel("Months", fontsize=14)
    plt.ylabel("Time Varying Reproduction Number" , fontsize=14)
    plt.yticks(fontsize=14)


    plt.savefig(os.getcwd() + "/sysidentification/figures/freiburg/r0/" +  "r0_4_" + str((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2)))) + ".pdf", bbox_inches='tight')



def plot_state_trajectories(t, S_model, I_model, R_model, D_model, S_real, I_real, R_real, D_real, N1, all_S_ub, all_S_lb,
                            ub_S, lb_S, ub_I, lb_I, ub_R, lb_R, ub_D, lb_D, dataset):
    fig, axs = plt.subplots(4, 1, figsize=(15, 8), sharex=True, sharey=False)
    plt.suptitle("State Trajectory Estimations", fontsize=18)
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

def plot_state_trajectories_reg(t, I_model,  D_model, I_model2, D_model2, I_real,  D_real, alpha1, alpha2, N1,
                             ub_I, lb_I, ub_D, lb_D,   ub_I2, lb_I2,  ub_D2, lb_D2,dataset,alpha_vec1, alpha_vec2):

    fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True, sharey=False)
    plt.suptitle("State Trajectory Estimations for Active Infections and Deaths in Freiburg", fontsize=18)

    axs[0].set_ylabel("Active Infections", fontsize=14)
    axs[1].set_ylabel("Deaths", fontsize=14)

    axs[0].grid(b=True, which='major', color='lightgray', linestyle='-')
    axs[1].grid(b=True, which='major', color='lightgray', linestyle='-')

    scaley = np.max(I_real)/alpha_vec1[0]
    scaley1 = np.max(I_real) / alpha_vec1[0]

    scaley2 = np.max(D_real)/alpha_vec2[0]

    tspan = np.arange(N1-2)

    plt.yticks(fontsize=14)

    ub_I = np.reshape(ub_I, (N1))
    lb_I = np.reshape(lb_I, (N1))

    ub_I2 = np.reshape(ub_I2, (N1))
    lb_I2 = np.reshape(lb_I2, (N1))

    ub_D = np.reshape(ub_D, (N1))
    lb_D = np.reshape(lb_D, (N1))

    ub_D2 = np.reshape(ub_D2, (N1))
    lb_D2 = np.reshape(lb_D2, (N1))

    axs[0].plot(t, I_model, color='lightskyblue')
    axs[0].plot(t, I_model2, color='olive')
    axs[0].plot(t, I_real, color='navy')
    # axs[0].plot(tspan, alpha_vec1 * scaley1, color='darkred', linestyle='dashed')

    axs[0].axvspan(43 - 13-3, 43 - 13+3, alpha=0.2, color='k')
    axs[0].axvspan(61 - 13 -3,61 - 13 +3, alpha=0.2, color='k')
    axs[0].axvspan(105 - 13 - 3, 105 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(217 - 13 - 3, 217 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(256 - 13 - 3, 256 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(267 - 13 - 3, 267 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(278 - 13 - 3, 278 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(288 - 13 - 3, 288 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(293 - 13 +5 - 3, 293 - 13  + 5+ 3, alpha=0.2, color='k')

    # alpha_reg_vec2[105 - 13:115 - 4 - 13] = 1e3
    # alpha_reg_vec2[217 - 13:227 - 4 - 13] = 1e3
    # alpha_reg_vec2[256 - 13:266 - 4 - 13] = 1e3
    # alpha_reg_vec2[267 - 13:277 - 4 - 13] = 1e3
    # alpha_reg_vec2[278 - 13:288 - 4 - 13] = 1e3
    # alpha_reg_vec2[288 - 13:298 - 4 - 13] = 1e3
    # alpha_reg_vec2[293 + 5 - 13:303 - 4 + 5 - 13] = 1e3

    str((int)(np.log(alpha1) / np.log(100) * 2))
    # axs[0].legend([r' Simulated Active Infections with $ \alpha = 10^6 $' + str((int)(np.rint((np.log(alpha1) / np.log(100) * 2)))), r' Simulated Active Infections with $ \alpha_{reg} = 1 e $' + str((int)(np.rint((np.log(alpha2) / np.log(100) * 2)))),
    #                "Real Active Infections", r' Regression Vector'], loc="upper center", fontsize=14)
    if ((int)(np.rint((np.log(alpha1) / np.log(100) * 2))) == 7  and (int)(np.rint((np.log(alpha2) / np.log(100) * 2))) == 5):

        axs[0].legend([r' Simulated Active Infections with $ \bar{\alpha} = 10^7 $',
                   r' Simulated Active Infections with $ \bar{\alpha} = 10^5$',
                   "Real Active Infections", r' Days for new Lockdown Measures'], loc="upper center", fontsize=14)
    elif ((int)(np.rint((np.log(alpha1) / np.log(100) * 2))) == 6  and (int)(np.rint((np.log(alpha2) / np.log(100) * 2))) == 5):
        axs[0].legend([r' Simulated Active Infections with $ \bar{\alpha} = 10^6 $',
                       r' Simulated Active Infections with $ \bar{\alpha} = 10^5$',
                       "Real Active Infections", r' Days for new Lockdown Measures'], loc="upper center", fontsize=14)

    # axs[0].fill_between(t, lb_I2, ub_I2, facecolor='lemonchiffon')
    axs[0].fill_between(t, lb_I2, ub_I2, facecolor='lightyellow')
    axs[0].fill_between(t, lb_I, ub_I, facecolor='lightsteelblue')

    axs[0].annotate('[1]', xy=(1, 100), xytext=(43-16, np.max(I_model) + np.max(I_model)/4  ), fontsize=14)
    axs[0].annotate('[2]', xy=(1, 100), xytext=(61 - 16, np.max(I_model)+ np.max(I_model)/4), fontsize=14)
    axs[0].annotate('[3]', xy=(1, 100), xytext=(105 - 16, np.max(I_model)+ np.max(I_model)/4), fontsize=14)
    axs[0].annotate('[4]', xy=(1, 100), xytext=(217 - 16, np.max(I_model)+ np.max(I_model)/4), fontsize=14)
    axs[0].annotate('[5]', xy=(1, 100), xytext=(256 - 16, np.max(I_model)+ np.max(I_model)/4), fontsize=14)
    axs[0].annotate('[6]', xy=(1, 100), xytext=(267 - 16, np.max(I_model)+ np.max(I_model)/4), fontsize=14)
    axs[0].annotate('[7]', xy=(1, 100), xytext=(278 - 16, np.max(I_model)+ np.max(I_model)/4), fontsize=14)
    axs[0].annotate('[8]', xy=(1, 100), xytext=(288 - 16, np.max(I_model) + np.max(I_model) / 4), fontsize=14)
    axs[0].annotate('[9]', xy=(1, 100), xytext=(293 +5 - 16, np.max(I_model) + np.max(I_model) / 4), fontsize=14)

    axs[1].plot(t, D_model, color='lightskyblue')
    axs[1].plot(t, D_model, color='olive')
    axs[1].plot(t, D_real, color='navy')
    axs[1].fill_between(t, lb_D, ub_D, facecolor='lightsteelblue')
    # axs[1].fill_between(t, lb_D2, ub_D2, facecolor='lemonchiffon')
    # axs[1].plot(tspan, alpha_vec2 * scaley2, color='darkred', linestyle='dashed')
    axs[1].axvspan(43 - 13 - 3, 43 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(61 - 13 - 3, 61 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(105 - 13 - 3, 105 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(217 - 13 - 3, 217 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(256 - 13 - 3, 256 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(267 - 13 - 3, 267 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(278 - 13 - 3, 278 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(288 - 13 - 3, 288 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(293 - 13 + 5- 3, 293 - 13 +5  + 3, alpha=0.2, color='k')
    # axs[1].legend([r' Simulated Deaths $ \alpha = 1 e $' + str((int)(np.rint((np.log(alpha1) / np.log(100) * 2)))),r' Simulated Deaths with $ \alpha = 1 e $' + str((int)(np.rint((np.log(alpha2) / np.log(100) * 2)))), "Real Deaths", r' Regression Vector '], loc="upper center", fontsize=14)
    # axs[1].legend([r' Simulated Deaths $ \alpha_{reg} = 1 e $' + str((int)(np.rint((np.log(alpha1) / np.log(100) * 2)))),r' Simulated Deaths with $ \alpha_{reg} = 1 e $' + str((int)(np.rint((np.log(alpha2) / np.log(100) * 2)))), "Real Deaths"], loc="upper center", fontsize=14)

    if ((int)(np.rint((np.log(alpha1) / np.log(100) * 2))) == 7  and (int)(np.rint((np.log(alpha2) / np.log(100) * 2))) == 5):
        axs[1].legend([r' Simulated Deaths $ \bar{\alpha} = 10^7 $' ,
                   r' Simulated Deaths with $ \bar{\alpha} = 10^5$'
                      , "Real Deaths", r' Days for new Lockdown Measures '], loc="upper center", fontsize=14)
    elif ((int)(np.rint((np.log(alpha1) / np.log(100) * 2))) == 6  and (int)(np.rint((np.log(alpha2) / np.log(100) * 2))) == 5):
        axs[1].legend([r' Simulated Deaths $ \bar{\alpha} = 10^6 $',
                       r' Simulated Deaths with $ \bar{\alpha} = 10^5$'
                          , "Real Deaths", r' Days for new Lockdown Measures '], loc="upper center", fontsize=14)

    else:
        print("False")


    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[1].tick_params(axis='both', which='minor', labelsize=14)

    axs[1].annotate('[1]', xy=(1, 100), xytext=(43-16,np.max(D_real)+ np.max(D_real)/7 +5), fontsize=14)
    axs[1].annotate('[2]', xy=(1, 100), xytext=(61 - 16, np.max(D_real)+ np.max(D_real)/7 +5 ), fontsize=14)
    axs[1].annotate('[3]', xy=(1, 100), xytext=(105 - 16, np.max(D_real)+ np.max(D_real)/7+5), fontsize=14)
    axs[1].annotate('[4]', xy=(1, 100), xytext=(217 - 16, np.max(D_real)+ np.max(D_real)/7+5), fontsize=14)
    axs[1].annotate('[5]', xy=(1, 100), xytext=(256 - 16, np.max(D_real)+ np.max(D_real)/7+5), fontsize=14)
    axs[1].annotate('[6]', xy=(1, 100), xytext=(267 - 16, np.max(D_real)+ np.max(D_real)/7+5), fontsize=14)
    axs[1].annotate('[7]', xy=(1, 100), xytext=(278 - 16, np.max(D_real)+ np.max(D_real)/7+5), fontsize=14)
    axs[1].annotate('[8]', xy=(1, 100), xytext=(288 - 16, np.max(D_real)+ np.max(D_real)/7+5), fontsize=14)
    axs[1].annotate('[9]', xy=(1, 100), xytext=(293 +5 - 16, np.max(D_real)+ np.max(D_real)/7+5), fontsize=14)

    axs[1].set_xticks(np.arange(0, N1 + 1, step=30))

    axs[1].set_xticklabels(months_weekly, rotation=30, fontsize=14)


    plt.savefig(os.getcwd() + "/sysidentification/figures/freiburg/active_deaths_alpha" + str((int)(np.rint((np.log(alpha1) / np.log(100) * 2))))+".pdf",  bbox_inches='tight')

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


def show_cov(alpha, sigma_theta, max_abs, dataset, N):
    plt.imshow(sigma_theta, cmap='RdBu', vmin=-max_abs, vmax=max_abs)
    plt.colorbar()
    if ((int)(np.rint((np.log(alpha) / np.log(100) * 2))) == 7):
       plt.title(r' Covariance Matrix with $ \bar{\alpha} = 10^7 $', fontsize=14)
    if ((int)(np.rint((np.log(alpha) / np.log(100) * 2))) == 6):
        plt.title(r' Covariance Matrix with $ \bar{\alpha} = 10^6 $',
                  fontsize=14)
    if ((int)(np.rint((np.log(alpha) / np.log(100) * 2))) == 5):
        plt.title(r' Covariance Matrix  with $ \bar{\alpha} = 10^5 $',
                  fontsize=14)

    if (N==330):
        plt.xticks(np.arange(0, N + 1, step=30), months_weekly, rotation=30, fontsize=14)
        plt.yticks(np.arange(0, N + 1, step=30), months_weekly, rotation=30, fontsize=14)
    elif (N==47):
        plt.xticks(np.arange(0, N + 1, step=4), months_weekly, rotation=30, fontsize=14)
        plt.yticks(np.arange(0, N + 1, step=4), months_weekly, rotation=30, fontsize=14)

    plt.xlabel(r' Time Varying Transmission Rate $\beta$',fontsize=14)
    plt.xticks(np.arange(0, N + 1, step=30), months_weekly, rotation=30, fontsize=14)
    plt.yticks(np.arange(0, N + 1, step=30), months_weekly, rotation=30, fontsize=14)

    plt.ylabel(r' Time Varying Transmission Rate $\beta$',fontsize=14)

    plt.savefig(os.getcwd() + "/sysidentification/figures/freiburg/covariances/" + str((int)(np.rint((np.log(alpha) / np.log(100) * 2))))+  "_covariance_theta.pdf", bbox_inches='tight')



def plot_errors(t, error_S, error_I, error_R, error_D, pop, N1, dataset):
    plt.plot(t, error_S / pop * 100)
    plt.plot(t, error_I / pop * 100)
    plt.plot(t, error_R / pop * 100)
    plt.plot(t, error_D / pop * 100)
    plt.title("Absolute Error (% of Total Population of Germany)")
    plt.legend(["Error Susceptible ", "Error Infected", "Error Recovered", "Error Dead"], loc="upper right" )

    if (N1 == 150):
        plt.xticks(np.arange(0, N1 + 1, step=30),first_wave)
        plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(dataset) + "first_wave_errors.pdf")
    if (N1 == 100):
        plt.xticks(np.arange(0, N1 + 1, step=30),second_wave)
        plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(dataset) + "second_wwave_errors.pdf")
    if (N1 == 330):
        plt.xticks(np.arange(0, N1 + 1, step=30),first_second_wave)
        plt.savefig(os.getcwd() + "/sysidentification/figures/" + str(dataset) + "first_second_wave_errors.pdf")


def plot_active_stuttgart(t,stuttgart, boblingen, esslingen, göppingen , ludwigsburg,remsmurrkreis , heilbronn ,hohenlohekreis,maintauberkreis,  schwaebisch,N):
    plt.title("Total Daily Active Administrative District of Stuttgart ", fontsize = 14)
    colors = ["lightsteelblue", "slategrey", "cornflowerblue", "royalblue", "mediumblue", "blue", "slateblue", "darkslateblue", "mediumslateblue", "mediumpurple"]
    scaley = 1e3
    # plt.plot(t, active[i, :], label='Age Group %s ' % i)
    plt.plot(t, stuttgart / scaley, color=colors[0])
    plt.plot(t, boblingen / scaley, color=colors[1])
    plt.plot(t, esslingen / scaley, color=colors[2])
    plt.plot(t, göppingen / scaley, color=colors[3])
    plt.plot(t, ludwigsburg / scaley, color=colors[4])
    plt.plot(t, remsmurrkreis / scaley, color=colors[5])
    plt.plot(t, heilbronn/ scaley, color=colors[6])
    plt.plot(t, hohenlohekreis / scaley, color=colors[7])
    plt.plot(t, maintauberkreis / scaley, color=colors[8])
    plt.plot(t, schwaebisch / scaley, color=colors[9])

    plt.legend(["Stuttgart", "Böblingen", "Esslingen", "Göppingen", "Ludwigsburg", "Rems-Murr-Kreis", "Heilbronn", "Hohenlohekreis", "Main-Tauber-Kreis", "Schwäbisch Hall"], fontsize=9)

    plt.xticks(np.arange(1, N + 1, step=30), months, rotation=30, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Active Infections x 1e3", fontsize=14)
    plt.xlabel("Months", fontsize=14)
    plt.xticks(np.arange(0, N + 1, step=30),months)
    plt.savefig(os.getcwd() + "/sysidentification/figures/datasource/active_stuttgart.pdf", bbox_inches='tight')


datasets  = ["Freiburg", "Emmendingen", "Ortenaukreis", "Rottweil", "Tuttlingen", "Konstanz", "Waldshut", "Lörrach", "Breisgauhochscwarzwald", "Schwarzwaldbaarkreis"]


def plot_active_freiburg(t,freiburg, emmendingen, ortenaukreis, rottweil , tuttlingen,konstanz , waldshut ,lorrach,breisgauhochschwarzwald,  schwarzwaldbaarkreis,N):
    plt.title(r"Daily Active Infections in the Administrative District of Freiburg ", fontsize = 14)
    colors = [r"darkorange", "darkkhaki", "brown", "gold", "orange", "darkgoldenrod", "salmon", "chocolate", "sandybrown", "peru"]
    scaley = 1e3
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')
    # plt.plot(t, active[i, :], label='Age Group %s ' % i)

    plt.plot(t, freiburg / scaley, color=colors[0])
    plt.plot(t, emmendingen / scaley, color=colors[1])
    plt.plot(t, ortenaukreis / scaley, color=colors[2])
    plt.plot(t, rottweil / scaley, color=colors[3])
    plt.plot(t, tuttlingen / scaley, color=colors[4])
    plt.plot(t, konstanz / scaley, color=colors[5])
    plt.plot(t, waldshut/ scaley, color=colors[6])
    plt.plot(t, lorrach / scaley, color=colors[7])
    plt.plot(t, breisgauhochschwarzwald/ scaley, color=colors[8])
    plt.plot(t, schwarzwaldbaarkreis / scaley, color=colors[9])

    plt.legend(["Freiburg", "Emmendingen", "Ortenaukreis", "Rottweil", "Tuttlingen", "Konstanz", "Waldshut", "Lörrach", "Breisgau-Hochschwarzwald", "Schwarzwald-Baar-Kreis"], fontsize=14)

    plt.xticks(np.arange(1, N + 1, step=30), months, rotation=30, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r'Active Infections $\cdot 10^3$', fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.xticks(np.arange(0, N + 1, step=30),months)
    plt.savefig(os.getcwd() + "/sysidentification/figures/datasource/active_freiburg.pdf", bbox_inches='tight')

def plot_weekly_active_stuttgart(t,stuttgart, boblingen, esslingen, göppingen , ludwigsburg,remsmurrkreis , heilbronn ,hohenlohekreis,maintauberkreis,  schwaebisch,N):
    plt.title(r"Total Weekly Active Infections in the Administrative District of Stuttgart ", fontsize = 14)
    colors = [r"lightsteelblue", "slategrey", "cornflowerblue", "royalblue", "mediumblue", "blue", "slateblue", "darkslateblue", "mediumslateblue", "mediumpurple"]
    scaley = 1e3
    # plt.plot(t, active[i, :], label='Age Group %s ' % i)

    plt.plot(t, freiburg / scaley, color=colors[0])
    plt.plot(t, emmendingen / scaley, color=colors[1])
    plt.plot(t, ortenaukreis / scaley, color=colors[2])
    plt.plot(t, rottweil / scaley, color=colors[3])
    plt.plot(t, tuttlingen / scaley, color=colors[4])
    plt.plot(t, konstanz / scaley, color=colors[5])
    plt.plot(t, waldshut / scaley, color=colors[6])
    plt.plot(t, lorrach / scaley, color=colors[7])
    plt.plot(t, breisgauhochschwarzwald / scaley, color=colors[8])
    plt.plot(t, schwarzwaldbaarkreis / scaley, color=colors[9])

    plt.legend(["Stuttgart", "Böblingen", "Esslingen", "Göppingen", "Ludwigsburg", "Rems-Murr-Kreis", "Heilbronn", "Hohenlohekreis", "Main-Tauber-Kreis", "Schwäbisch Hall"], fontsize=14)

    plt.xticks(np.arange(1, N + 1, step=30), months, rotation=30, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r"Active Weekly Infections x 1e3", fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.xticks(np.arange(0, N + 1, step=4),months)
    plt.savefig(os.getcwd() + "/sysidentification/figures/datasource/active_weekly_stuttgart.pdf", bbox_inches='tight')

def plot_weekly_active_freiburg(t,stuttgart, boblingen, esslingen, göppingen , ludwigsburg,remsmurrkreis , heilbronn ,hohenlohekreis,maintauberkreis,  schwaebisch,N):
    plt.title(r"Weekly Active Infections in the Administrative District of Freiburg ", fontsize = 14)
    colors = [r"darkorange", "darkkhaki", "brown", "gold", "orange", "darkgoldenrod", "salmon", "chocolate", "sandybrown", "peru"]
    scaley = 1e3
    # plt.plot(t, active[i, :], label='Age Group %s ' % i)

    plt.plot(t, stuttgart / scaley, color=colors[0])
    plt.plot(t, boblingen / scaley, color=colors[1])
    plt.plot(t, esslingen / scaley, color=colors[2])
    plt.plot(t, göppingen / scaley, color=colors[3])
    plt.plot(t, ludwigsburg / scaley, color=colors[4])
    plt.plot(t, remsmurrkreis / scaley, color=colors[5])
    plt.plot(t, heilbronn / scaley, color=colors[6])
    plt.plot(t, hohenlohekreis / scaley, color=colors[7])
    plt.plot(t, maintauberkreis / scaley, color=colors[8])
    plt.plot(t, schwaebisch / scaley, color=colors[9])

    plt.legend(["Freiburg", "Emmendingen", "Ortenaukreis", "Rottweil", "Tuttlingen", "Konstanz", "Waldshut", "Lörrach", "Breisgau-Hochschwarzwald", "Schwarzwald-Baar-Kreis"], fontsize=14)

    plt.xticks(np.arange(1, N + 1, step=30), months, rotation=30, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r'Active Infections  [$ 10^3$]', fontsize=14)
    plt.xlabel(r"Months", fontsize=14)
    plt.xticks(np.arange(0, N + 1, step=4),months)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')
    plt.savefig(os.getcwd() + "/sysidentification/figures/datasource/active_weekly_freiburg.pdf", bbox_inches='tight')

def plot_rwert(t, N, rwert, rwert_lb, rwert_ub):

    plt.plot(t, rwert, color='crimson')
    plt.title("Daily Time Varying Reproduction Number of COVID-19", fontsize=14)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')
    plt.xticks(np.arange(0, N + 1, step=30), months_weekly, rotation=30, fontsize=14)

    rwert_ub = np.reshape(rwert_ub, (N))
    rwert_lb = np.reshape(rwert_lb, (N))
    # plt.plot(t, rwert_ub, color='crimson')

    rwert_lb = np.array(rwert_lb, dtype=float)
    rwert_ub = np.array(rwert_ub, dtype=float)

    plt.fill_between(t, rwert_lb, rwert_ub, facecolor='pink')
    plt.xlabel("Months", fontsize=14)
    plt.ylabel(r'Reproduction Number $R_0$', fontsize=14)
    plt.yticks(fontsize=14)

    plt.savefig(os.getcwd() + "/figures/rwert/rwert_daily.pdf", bbox_inches='tight')




def plot_rwert_weekly(t_weeks, weeks, rwert_weekly, rwert_lb_weekly, rwert_ub_weekly):
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

    plt.figure()
    plt.plot(t_weeks, rwert_weekly, color='crimson')
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')

    plt.title("Weekly Time Varying Reproduction Number of COVID-19", fontsize=14)

    plt.xticks(np.arange(0, weeks + 1, step=4), months_weekly, rotation=30, fontsize=14)

    rwert_ub_weekly = np.reshape(rwert_ub_weekly, (weeks))
    rwert_lb_weekly = np.reshape(rwert_lb_weekly, (weeks))
    # plt.plot(t, rwert_ub, color='crimson')

    rwert_lb_weekly = np.array(rwert_lb_weekly, dtype=float)
    rwert_ub_weekly = np.array(rwert_ub_weekly, dtype=float)

    plt.fill_between(t_weeks, rwert_lb_weekly, rwert_ub_weekly, facecolor='pink')
    plt.xlabel("Months", fontsize=14)
    plt.ylabel(r'Reproduction Number $R_0$', fontsize=14)
    plt.yticks(fontsize=14)

    plt.savefig(os.getcwd() + "/figures/rwert/rwert_weekly.pdf", bbox_inches='tight')

def plot_cfr(t, cfr, N):

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
    fig = plt.plot(t,cfr, color='darkred')
    plt.title(r'Estimating the Case Fatality Rate', fontsize=18)

    # plt.title(r' Time Varying Transmission Rate $\beta $ with $\alpha_{reg} = 1 e $' + str((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2)))), fontsize=18)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')


    plt.xlabel("Months", fontsize=16)
    plt.ylabel(r"Case Fatality Rate " , fontsize=14)
    plt.yticks(fontsize=14)

    plt.xticks(np.arange(0, N + 1, step=30), months_weekly,rotation=30, fontsize=14)
    plt.savefig(os.getcwd() + "/sysidentification/figures/freiburg/" +  "cfr.pdf", bbox_inches='tight')


def plot_alpha(t, alpha_reg_vector,alpha, N):

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
    scaley = alpha/10
    fig = plt.plot(t, alpha_reg_vector/scaley, color='darkred')
    # plt.title(r'$ \alpha $', fontsize='small')

    # plt.title(r' Time Varying Transmission Rate $\beta $ with $\alpha_{reg} = 1 e $' + str((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2)))), fontsize=18)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')


    plt.xlabel("Months", fontsize=16)
    plt.ylabel(r"$\alpha$ values x 1e"+str((int)(np.rint((np.log(alpha) / np.log(100) * 2)))-1) , fontsize=14)
    plt.yticks(fontsize=14)

    plt.xticks(np.arange(0, N + 1, step=30), months_weekly,rotation=30, fontsize=14)
    plt.savefig(os.getcwd() + "/sysidentification/figures/freiburg/" +  "alpha_values" + str((int)(np.rint((np.log(alpha) / np.log(100) * 2)))) + ".pdf", bbox_inches='tight')


def plot_predictions(t, I_model,  D_model,  I_real,  D_real, alpha1,  N1,
                             ub_I, lb_I, ub_D, lb_D ,dataset, alpha_vec1, predicted, N_pred, active_pred, dead_pred ):


    fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True, sharey=False)
    plt.suptitle(r"State Trajectory Estimations and Predictions for Active Infections and Deaths in Freiburg", fontsize=18)

    axs[0].set_ylabel("Active Infections", fontsize=14)
    axs[1].set_ylabel("Deaths", fontsize=14)

    axs[0].grid(b=True, which='major', color='lightgray', linestyle='-')
    axs[1].grid(b=True, which='major', color='lightgray', linestyle='-')

    t_pred = np.arange(N1 +1, N_pred+1+N1, step=1)


    scaley = np.max(I_real)/alpha_vec1[0]
    scaley1 = np.max(I_real) / alpha_vec1[0]
    scaley2 =  np.max(D_real)/alpha_vec1[0]

    tspan = np.arange(N1-2)

    plt.yticks(fontsize=14)

    ub_I = np.reshape(ub_I, (N1))
    lb_I = np.reshape(lb_I, (N1))

    bound_extra = np.ones(N_pred) * (I_model[N1-1]-lb_I[N1-1])



    ub_D = np.reshape(ub_D, (N1))
    lb_D = np.reshape(lb_D, (N1))


    # axs[0].plot(t, I_model, color='lightskyblue')
    I_pred = predicted[:,1]

    D_pred = predicted[:, 3]

    I_all = np.concatenate((I_model, I_pred), axis=0)
    N_all = N1 + N_pred
    tspan2 = np.arange(N_all)

    N_ava = np.shape(active_pred)[0]
    tspan3 = np.arange(N_ava)


    axs[0].plot(t, I_model, color='lightskyblue')
    axs[0].plot(tspan3, active_pred, color='navy')
    axs[0].plot(t_pred, I_pred, color='darkgoldenrod')

    # axs[0].plot(tspan, alpha_vec1 * scaley1, color='darkred', linestyle='dashed')

    axs[0].axvspan(43 - 13 - 3, 43 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(61 - 13 - 3, 61 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(105 - 13 - 3, 105 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(217 - 13 - 3, 217 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(256 - 13 - 3, 256 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(267 - 13 - 3, 267 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(278 - 13 - 3, 278 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(288 - 13 - 3, 288 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(293 - 13 +5- 3, 293 - 13 +5 + 3, alpha=0.2, color='k')
    N_ava = np.shape(active_pred)[0]





    str((int)(np.log(alpha1) / np.log(100) * 2))
    # axs[0].legend([r' Simulated Active Infections with $ \bar{\alpha} = 1 e $' + str((int)(np.rint((np.log(alpha1) / np.log(100) * 2)))),r"Real Active Infections", r'Predicted Infections',r' Days for new Lockdown Measures'], loc="upper center", fontsize=14)

    axs[0].legend([r' Simulated Active Infections with $ \bar{\alpha} = 10^8 $' ,r"Real Active Infections", r'Predicted Infections',r' Days for new Lockdown Measures'], loc="upper center", fontsize=14)

    axs[0].fill_between(t, lb_I, ub_I, facecolor='lightsteelblue')
    axs[0].fill_between(t_pred, I_pred - bound_extra,I_pred + bound_extra, facecolor='papayawhip')




    axs[0].annotate('[1]', xy=(1, 100), xytext=(43-16, np.max(I_model) + np.max(I_model)/3   -150  +50), fontsize=14)
    axs[0].annotate('[2]', xy=(1, 100), xytext=(61 - 16, np.max(I_model)+ np.max(I_model)/3-150+50), fontsize=14)
    axs[0].annotate('[3]', xy=(1, 100), xytext=(105 - 16, np.max(I_model)+ np.max(I_model)/3-150+50), fontsize=14)
    axs[0].annotate('[4]', xy=(1, 100), xytext=(217 - 16, np.max(I_model)+ np.max(I_model)/3-150+50), fontsize=14)
    axs[0].annotate('[5]', xy=(1, 100), xytext=(256 - 16, np.max(I_model)+ np.max(I_model)/3-150+50), fontsize=14)
    axs[0].annotate('[6]', xy=(1, 100), xytext=(267 - 16, np.max(I_model)+ np.max(I_model)/3-150+50), fontsize=14)
    axs[0].annotate('[7]', xy=(1, 100), xytext=(278 - 16, np.max(I_model)+ np.max(I_model)/3-150+50), fontsize=14)
    axs[0].annotate('[8]', xy=(1, 100), xytext=(288 - 16, np.max(I_model) + np.max(I_model) / 3-150+50), fontsize=14)
    axs[0].annotate('[9]', xy=(1, 100), xytext=(293+5 - 16, np.max(I_model) + np.max(I_model) / 3-150+50), fontsize=14)

    bound_extra_dead = np.ones(N_pred) * (D_model[N1 - 1] - lb_D[N1 - 1])


    axs[1].plot(t, D_model, color='lightskyblue')
    axs[1].plot(tspan3, dead_pred, color='navy')
    axs[1].plot(t_pred, D_pred, color='darkgoldenrod')
    axs[1].fill_between(t, lb_D, ub_D, facecolor='lightsteelblue')
    axs[1].fill_between(t_pred, D_pred - bound_extra_dead, D_pred + bound_extra_dead, facecolor='papayawhip')
    # axs[1].plot(tspan, alpha_vec1 * scaley2, color='darkred', linestyle='dashed')

    axs[1].axvspan(43 - 13 - 3, 43 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(61 - 13 - 3, 61 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(105 - 13 - 3, 105 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(217 - 13 - 3, 217 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(256 - 13 - 3, 256 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(267 - 13 - 3, 267 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(278 - 13 - 3, 278 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(288 - 13 - 3, 288 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(293 - 13 +5- 3, 293 - 13 +5 + 3, alpha=0.2, color='k')
    axs[1].legend([r' Simulated Deaths with $ \bar{\alpha} = 10^8 $' , r"Real Deaths",r' Predicted Deaths', r' Days for new Lockdown Measures '], loc="upper center", fontsize=14)
    # axs[1].legend([r' Simulated Deaths $ \alpha_{reg} = 1 e $' + str((int)(np.rint((np.log(alpha1) / np.log(100) * 2)))),r' Simulated Deaths with $ \alpha_{reg} = 1 e $' + str((int)(np.rint((np.log(alpha2) / np.log(100) * 2)))), "Real Deaths"], loc="upper center", fontsize=14)



    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[1].tick_params(axis='both', which='minor', labelsize=14)

    axs[1].annotate('[1]', xy=(1, 100), xytext=(43-16,np.max(D_real)+ np.max(D_real)/7+20-5 +35), fontsize=14)
    axs[1].annotate('[2]', xy=(1, 100), xytext=(61 - 16, np.max(D_real)+ np.max(D_real)/7+20+35-5), fontsize=14)
    axs[1].annotate('[3]', xy=(1, 100), xytext=(105 - 16, np.max(D_real)+ np.max(D_real)/7+20+35-5), fontsize=14)
    axs[1].annotate('[4]', xy=(1, 100), xytext=(217 - 16, np.max(D_real)+ np.max(D_real)/7+20+35-5), fontsize=14)
    axs[1].annotate('[5]', xy=(1, 100), xytext=(256 - 16, np.max(D_real)+ np.max(D_real)/7+20+35-5), fontsize=14)
    axs[1].annotate('[6]', xy=(1, 100), xytext=(267 - 16, np.max(D_real)+ np.max(D_real)/7+20+35-5), fontsize=14)
    axs[1].annotate('[7]', xy=(1, 100), xytext=(278 - 16, np.max(D_real)+ np.max(D_real)/7+20+35-5), fontsize=14)
    axs[1].annotate('[8]', xy=(1, 100), xytext=(288 - 16, np.max(D_real)+ np.max(D_real)/7+20+35-5), fontsize=14)
    axs[1].annotate('[9]', xy=(1, 100), xytext=(293+5 - 16, np.max(D_real)+ np.max(D_real)/7+20+35-5), fontsize=14)


    axs[1].set_xticks(np.arange(0, N_all + 1, step=30))

    axs[1].set_xticklabels(months_weekly, rotation=30, fontsize=14)



    plt.savefig(os.getcwd() + "/sysidentification/figures/freiburg/pred_active_deaths_alpha" + str((int)(np.rint((np.log(alpha1) / np.log(100) * 2))))+".pdf",  bbox_inches='tight')


def plot_state_betas(t, betas1, betas2 , I_model,  D_model, I_model2, D_model2, I_real,  D_real, alpha1, alpha2, N1,
                             ub_I, lb_I, ub_D, lb_D,   ub_I2, lb_I2,  ub_D2, lb_D2,dataset,alpha_vec1, alpha_vec2):
    t2 = np.arange(0, N1 - 1, 1)

    fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True, sharey=False)
    plt.suptitle(r"Estimating the Number of Infections, Deaths and the Transmission Rate $\beta$", fontsize=18)

    axs[0].set_ylabel("Active Infections", fontsize=14)
    axs[1].set_ylabel(r'Tranmission Rate $\beta$', fontsize=14)
    # axs[2].set_ylabel(r'Tranmission Rate $\beta$', fontsize=14)

    axs[0].grid(b=True, which='major', color='lightgray', linestyle='-')
    axs[1].grid(b=True, which='major', color='lightgray', linestyle='-')
    # axs[2].grid(b=True, which='major', color='lightgray', linestyle='-')

    scaley = np.max(I_real)/alpha_vec1[0]
    scaley1 = np.max(I_real) / alpha_vec1[0]

    scaley2 = np.max(D_real)/alpha_vec2[0]

    tspan = np.arange(N1-2)

    plt.yticks(fontsize=14)

    ub_I = np.reshape(ub_I, (N1))
    lb_I = np.reshape(lb_I, (N1))

    ub_I2 = np.reshape(ub_I2, (N1))
    lb_I2 = np.reshape(lb_I2, (N1))

    ub_D = np.reshape(ub_D, (N1))
    lb_D = np.reshape(lb_D, (N1))

    ub_D2 = np.reshape(ub_D2, (N1))
    lb_D2 = np.reshape(lb_D2, (N1))

    axs[0].plot(t, I_model, color='lightskyblue')
    axs[0].plot(t, I_model2, color='olive')
    axs[0].plot(t, I_real, color='navy')
    # axs[0].plot(tspan, alpha_vec1 * scaley1, color='darkred', linestyle='dashed')
    axs[0].axvspan(43 - 13 - 3, 43 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(61 - 13 - 3, 61 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(105 - 13 - 3, 105 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(217 - 13 - 3, 217 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(256 - 13 - 3, 256 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(267 - 13 - 3, 267 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(278 - 13 - 3, 278 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(288 - 13 - 3, 288 - 13 + 3, alpha=0.2, color='k')
    axs[0].axvspan(293 - 13 +5 - 3, 293 - 13  +5 + 3, alpha=0.2, color='k')

    str((int)(np.log(alpha1) / np.log(100) * 2))
    # axs[0].legend([r' Simulated Active Infections with $ \alpha = 10^6 $' + str((int)(np.rint((np.log(alpha1) / np.log(100) * 2)))), r' Simulated Active Infections with $ \alpha_{reg} = 1 e $' + str((int)(np.rint((np.log(alpha2) / np.log(100) * 2)))),
    #                "Real Active Infections", r' Regression Vector'], loc="upper center", fontsize=14)
    if ((int)(np.rint((np.log(alpha1) / np.log(100) * 2))) == 7  and (int)(np.rint((np.log(alpha2) / np.log(100) * 2))) == 5):

        axs[0].legend([r' Simulated Active Infections with $ \alpha = 10^7 $',
                   r' Simulated Active Infections with $ \alpha = 10^5$',
                   "Real Active Infections", r' Days for new Lockdown Measures'], loc="upper center", fontsize=14)
    elif ((int)(np.rint((np.log(alpha1) / np.log(100) * 2))) == 6  and (int)(np.rint((np.log(alpha2) / np.log(100) * 2))) == 5):
        axs[0].legend([r' Simulated Active Infections with $ \bar{\alpha} = 10^6 $',
                       r' Simulated Active Infections with $ \bar{\alpha} = 10^5$',
                       "Real Active Infections", r' Days for new Lockdown Measures'], loc="upper center", fontsize=14)

    # axs[0].fill_between(t, lb_I2, ub_I2, facecolor='lemonchiffon')
    axs[0].fill_between(t, lb_I2, ub_I2, facecolor='lightyellow')
    axs[0].fill_between(t, lb_I, ub_I, facecolor='lightsteelblue')

    axs[0].annotate('[1]', xy=(1, 100), xytext=(43-16, np.max(I_model) + np.max(I_model)/4 ), fontsize=14)
    axs[0].annotate('[2]', xy=(1, 100), xytext=(61 - 16, np.max(I_model)+ np.max(I_model)/4), fontsize=14)
    axs[0].annotate('[3]', xy=(1, 100), xytext=(105 - 16, np.max(I_model)+ np.max(I_model)/4), fontsize=14)
    axs[0].annotate('[4]', xy=(1, 100), xytext=(217 - 16, np.max(I_model)+ np.max(I_model)/4), fontsize=14)
    axs[0].annotate('[5]', xy=(1, 100), xytext=(256 - 16, np.max(I_model)+ np.max(I_model)/4), fontsize=14)
    axs[0].annotate('[6]', xy=(1, 100), xytext=(267 - 16, np.max(I_model)+ np.max(I_model)/4), fontsize=14)
    axs[0].annotate('[7]', xy=(1, 100), xytext=(278 - 16, np.max(I_model)+ np.max(I_model)/4), fontsize=14)
    axs[0].annotate('[8]', xy=(1, 100), xytext=(288 - 16, np.max(I_model) + np.max(I_model) / 4), fontsize=14)
    axs[0].annotate('[9]', xy=(1, 100), xytext=(293 +5 - 16, np.max(I_model) + np.max(I_model) / 4), fontsize=14)

    # axs[1].plot(t, D_model, color='lightskyblue')
    # axs[1].plot(t, D_model2, color='olive')
    axs[1].plot(t2, betas1, color='lightskyblue')
    axs[1].plot(t2, betas2, color='olive')



    axs[1].axvspan(43 - 13 - 3, 43 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(61 - 13 - 3, 61 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(105 - 13 - 3, 105 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(217 - 13 - 3, 217 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(256 - 13 - 3, 256 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(267 - 13 - 3, 267 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(278 - 13 - 3, 278 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(288 - 13 - 3, 288 - 13 + 3, alpha=0.2, color='k')
    axs[1].axvspan(293 - 13 +5- 3, 293 - 13 +5 + 3, alpha=0.2, color='k')


    axs[1].set_xticks(np.arange(0, N1 + 1, step=30))

    axs[1].set_xticklabels(months_weekly, rotation=30, fontsize=14)

    # axs[1].fill_between(t, lb_D, ub_D, facecolor='lightsteelblue')
    # axs[1].fill_between(t, lb_D2, ub_D2, facecolor='lemonchiffon')

    # axs[1].plot(tspan, alpha_vec2 * scaley2, color='darkred', linestyle='dashed')
    #
    # axs[2].plot(t2, betas2, color='darkslategray')
    # axs[2].axvspan(43 - 13 - 3, 43 - 13 + 3, alpha=0.2, color='k')
    # axs[2].axvspan(61 - 13 - 3, 61 - 13 + 3, alpha=0.2, color='k')
    # axs[2].axvspan(105 - 13 - 3, 105 - 13 + 3, alpha=0.2, color='k')
    # axs[2].axvspan(217 - 13 - 3, 217 - 13 + 3, alpha=0.2, color='k')
    # axs[2].axvspan(256 - 13 - 3, 256 - 13 + 3, alpha=0.2, color='k')
    # axs[2].axvspan(267 - 13 - 3, 267 - 13 + 3, alpha=0.2, color='k')
    # axs[2].axvspan(278 - 13 - 3, 278 - 13 + 3, alpha=0.2, color='k')
    # axs[2].axvspan(288 - 13 - 3, 288 - 13 + 3, alpha=0.2, color='k')
    # axs[2].axvspan(293 - 13 +5 - 3, 293 - 13  +5+ 3, alpha=0.2, color='k')
    # # axs[1].legend([r' Simulated Deaths $ \alpha = 1 e $' + str((int)(np.rint((np.log(alpha1) / np.log(100) * 2)))),r' Simulated Deaths with $ \alpha = 1 e $' + str((int)(np.rint((np.log(alpha2) / np.log(100) * 2)))), "Real Deaths", r' Regression Vector '], loc="upper center", fontsize=14)
    # axs[1].legend([r' Simulated Deaths $ \alpha_{reg} = 1 e $' + str((int)(np.rint((np.log(alpha1) / np.log(100) * 2)))),r' Simulated Deaths with $ \alpha_{reg} = 1 e $' + str((int)(np.rint((np.log(alpha2) / np.log(100) * 2)))), "Real Deaths"], loc="upper center", fontsize=14)

    if ((int)(np.rint((np.log(alpha1) / np.log(100) * 2))) == 7  and (int)(np.rint((np.log(alpha2) / np.log(100) * 2))) == 5):
        axs[1].legend([r' Simulated Deaths $ \bar{\alpha} = 10^7 $' ,
                   r' Simulated Deaths with $ \bar{\alpha} = 10^5$'
                      , "Real Deaths", r' Days for new Lockdown Measures '], loc="upper center", fontsize=14)
    elif ((int)(np.rint((np.log(alpha1) / np.log(100) * 2))) == 6  and (int)(np.rint((np.log(alpha2) / np.log(100) * 2))) == 5):
        axs[1].legend([r' Using Regularization with $ \bar{\alpha} = 10^6 $',
                       r' Using Regularization with $ \bar{\alpha} = 10^5$',
                           r' Days for new Lockdown Measures'], loc="upper center", fontsize=14)

    else:
        print("False")
    # if ((int)(np.rint((np.log(alpha1) / np.log(100) * 2))) == 7  and (int)(np.rint((np.log(alpha2) / np.log(100) * 2))) == 5):
    #     axs[2].legend([r' Simulated Deaths $ \alpha = 10^7 $' ,
    #                r' Simulated Deaths with $ \alpha = 10^5$'
    #                   , "Real Deaths", r' Days for new Lockdown Measures'], loc="upper center", fontsize=14)
    # elif ((int)(np.rint((np.log(alpha1) / np.log(100) * 2))) == 6  and (int)(np.rint((np.log(alpha2) / np.log(100) * 2))) == 5):
    #     axs[2].legend([r' Transmission Rate with $ \alpha = 10^5$'
    #                       , "Real Deaths", r' Days for new Lockdown Measures'], loc="upper center", fontsize=14)
    #
    # else:
    #     print("False")


    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[1].tick_params(axis='both', which='minor', labelsize=20)
    # axs[2].tick_params(axis='both', which='minor', labelsize=14)

    axs[1].annotate('[1]', xy=(1, 100), xytext=(43-16,np.max(D_real)+ np.max(D_real)/7 +5), fontsize=14)
    axs[1].annotate('[2]', xy=(1, 100), xytext=(61 - 16, np.max(D_real)+ np.max(D_real)/7 +5 ), fontsize=14)
    axs[1].annotate('[3]', xy=(1, 100), xytext=(105 - 16, np.max(D_real)+ np.max(D_real)/7+5), fontsize=14)
    axs[1].annotate('[4]', xy=(1, 100), xytext=(217 - 16, np.max(D_real)+ np.max(D_real)/7+5), fontsize=14)
    axs[1].annotate('[5]', xy=(1, 100), xytext=(256 - 16, np.max(D_real)+ np.max(D_real)/7+5), fontsize=14)
    axs[1].annotate('[6]', xy=(1, 100), xytext=(267 - 16, np.max(D_real)+ np.max(D_real)/7+5), fontsize=14)
    axs[1].annotate('[7]', xy=(1, 100), xytext=(278 - 16, np.max(D_real)+ np.max(D_real)/7+5), fontsize=14)
    axs[1].annotate('[8]', xy=(1, 100), xytext=(288 - 16, np.max(D_real)+ np.max(D_real)/7+5), fontsize=14)
    axs[1].annotate('[9]', xy=(1, 100), xytext=(293 +5 - 16, np.max(D_real)+ np.max(D_real)/7+5), fontsize=14)

    # axs[2].set_xticks(np.arange(0, N1 + 1, step=30))

    # axs[2].set_xticklabels(months_weekly, rotation=30, fontsize=14)


    plt.savefig(os.getcwd() + "/sysidentification/figures/freiburg/active_deaths_alpha2" + str((int)(np.rint((np.log(alpha1) / np.log(100) * 2))))+".pdf",  bbox_inches='tight')