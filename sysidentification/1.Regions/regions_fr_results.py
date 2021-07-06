from casadi import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sird_model import sird_model_params
import os
import pdb
from plots_utils import *

import xlsxwriter

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

datasets  = ["Freiburg", "Emmendingen", "Ortenaukreis", "Rottweil", "Tuttlingen", "Konstanz", "Waldshut", "Lörrach", "Breisgauhochscwarzwald", "Schwarzwaldbaarkreis"]
ids       = ["8311", "8316", "8317", "8325", "8327", "8335", "8337", "8336", "8315", "8326"]

populations = [231195, 166408, 430953, 139878, 140766, 286305, 171003, 228736, 263601, 212506 ]

dataset = "Freiburg"

for i in range(len(datasets)):
    if (datasets[i] == dataset):
        print("yes from10 at i", i )
        pop = populations[i]
        id = ids[i]

import xlsxwriter
workbook = xlsxwriter.Workbook('solution_casadi.xlsx')
worksheet = workbook.add_worksheet()

print(os.getcwd())
os.chdir('../')
os.chdir('../')

print(os.getcwd())

# Source: https://github.com/jgehrcke/covid-19-germany-gae

df = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha7_N.xlsx', 'Sheet1')
df1 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha7_N.xlsx', 'Sheet2')
df7 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha7_N.xlsx', 'Sheet3')
df8 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha7_N.xlsx', 'Sheet4')
df10 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha7_N.xlsx', 'Sheet5')


df2 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha5_N.xlsx', 'Sheet1')
df3 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha5_N.xlsx', 'Sheet2')
df6 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha5_N.xlsx', 'Sheet3')
df6 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha5_N.xlsx', 'Sheet3')
df9 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha5_N.xlsx', 'Sheet4')
df11 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha5_N.xlsx', 'Sheet5')
df12 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha5_N.xlsx', 'Sheet6')


df4 = pd.read_csv(os.getcwd() + '/covid-19-germany-gae/cases-rki-by-ags.csv')
df5 = pd.read_csv(os.getcwd() + '/covid-19-germany-gae/deaths-rki-by-ags.csv')

active1 = df.iloc[1,:]
active1 = active1.to_numpy()
dead1 = df.iloc[2,:]
dead1 = dead1.to_numpy()


lb_I1 = df1.iloc[1,:]
lb_I1 = lb_I1.to_numpy()
ub_I1 = df1.iloc[2,:]
ub_I1 = ub_I1.to_numpy()
lb_D1 = df1.iloc[3,:]
lb_D1 = lb_D1.to_numpy()
ub_D1 = df1.iloc[4,:]
ub_D1 = ub_D1.to_numpy()

# betas1 = df

betas1 = df7.iloc[1,:]
betas1 = betas1.to_numpy()
betas1_lb = df7.iloc[2,:]
betas1_lb = betas1_lb.to_numpy()
betas1_ub = df7.iloc[3,:]
betas1_ub = betas1_ub.to_numpy()

gamma1 = df8.iloc[1,0]
mu1 = df8.iloc[1,1]

betas2 = df6.iloc[1,:]
betas2 = betas2.to_numpy()

betas2_lb = df6.iloc[2,:]

betas2_lb = betas2_lb.to_numpy()
betas2_ub = df6.iloc[3,:]
betas2_ub = betas2_ub.to_numpy()
gamma2 = df9.iloc[1,0]
mu2 = df9.iloc[1,1]

R0_1 = betas1/gamma1
R0_1_lb = betas1_lb/gamma1
R0_1_ub = betas1_ub/gamma1

R0_2 = betas2/gamma2
R0_2_lb = betas2_lb/gamma2
R0_2_ub = betas2_ub/gamma2


active2 = df2.iloc[1,:]
active2 = active2.to_numpy()
dead2 = df2.iloc[2,:]
dead2 = dead2.to_numpy()

lb_I2 = df3.iloc[1,:]
lb_I2 = lb_I2.to_numpy()
ub_I2 = df3.iloc[2,:]
ub_I2 = ub_I2.to_numpy()
lb_D2 = df3.iloc[3,:]
lb_D2 = lb_D2.to_numpy()
ub_D2 = df3.iloc[4,:]
ub_D2 = ub_D2.to_numpy()

N = len(ub_D1)

all_columns = df4.columns
all_columns = all_columns.to_numpy()

for i in range(np.shape(all_columns)[0]):
    if (all_columns[i] == id):
        print("yes at i", i )
        cumulative_infected = df4.iloc[:, i]
        deaths = df5.iloc[:, i]
        break

N_all = len(cumulative_infected)

cumulative_infected = cumulative_infected.to_numpy()
deaths = deaths.to_numpy()
deaths = deaths[15:len(deaths)]   # shape 379

new_infections = np.zeros(N_all)
new_deaths = np.zeros(len(deaths))

# x_data = pd.date_range('2018-04-01', periods=5, freq='MS')
#

for i in range(N_all-1):
    new_infections[i] = cumulative_infected[i+1] - cumulative_infected[i]

for i in range(len(deaths)-1):
    new_deaths[i] = deaths[i+1] - deaths[i]

active_infections = np.zeros(N_all - 15)
real_active = np.zeros(N_all - 15)
for i in range(len(active_infections)):
    active_infections[i] = np.sum(new_infections[i:i+15])
    real_active[i] = active_infections[i] - new_deaths[i] # perfect

N_max = len(active_infections)   # 379 days

recovered = np.zeros(len(active_infections))
susceptible = np.zeros(len(active_infections))


for i in range(len(active_infections)):
    recovered[i] = cumulative_infected[i+15] - real_active[i] - deaths[i]
    susceptible[i] = pop - real_active[i] - deaths[i] - recovered[i]


N_pred = 47
N1 = 330 + 80 - 40 -30

N_pred = 47
N1 = 330

start = 16 # = 18.03.2020 with N=330 10.02.2021
active = active_infections[start:start+N1]
active_pred = active_infections[start:start+N1+N_pred]

# active_predicted =
dead = deaths[start:start+N1]
dead_pred = deaths[start:start+N1+N_pred]

recovered = recovered[start:start+N1]
susceptible = susceptible[start:start+N1]

cumulated = cumulative_infected[start:start+N1]



predicted = df12.iloc[1:N_pred+1,0:4]
predicted = predicted.to_numpy()


tspan = np.arange(0, N1, 1)

alpha1 = 1e6
alpha2 = 1e5

alpha_reg = 1e7
alpha_reg_vec = np.ones((N1-2)) * alpha_reg
alpha_reg_vec[43 - 13:53-4 - 13] = 1e5
alpha_reg_vec[61 - 13:71-4 - 13] = 1e5
alpha_reg_vec[105 - 13:115-4 - 13] = 1e5
alpha_reg_vec[217 - 13:227-4 - 13] = 1e5
alpha_reg_vec[256 - 13:266-4 - 13] = 1e5
alpha_reg_vec[267 - 13:277-4 - 13] = 1e5
alpha_reg_vec[278 - 13:288-4 - 13] = 1e5
alpha_reg_vec[288 - 13:298-4 - 13] = 1e5
alpha_reg_vec[293+5 - 13:303-4+5 - 13] = 1e5


alpha_reg2 = 1e5
alpha_reg_vec2 = np.ones((N1-2)) * alpha_reg2
alpha_reg_vec2[43 - 13:53-4 - 13] = 1e3
alpha_reg_vec2[61 - 13:71-4 - 13] = 1e3
alpha_reg_vec2[105 - 13:115-4 - 13] = 1e3
alpha_reg_vec2[217 - 13:227-4 - 13] = 1e3
alpha_reg_vec2[256 - 13:266-4 - 13] = 1e3
alpha_reg_vec2[267 - 13:277-4 - 13] = 1e3
alpha_reg_vec2[278 - 13:288-4 - 13] = 1e3
alpha_reg_vec2[288 - 13:298-4 - 13] = 1e3
alpha_reg_vec2[293+5 - 13:303-4+5 - 13] = 1e3


plot_state_trajectories_reg(tspan, active1,  dead1, active2, dead2, active,  dead, alpha1, alpha2, N1,  ub_I1, lb_I1, ub_D1, lb_D1,   ub_I2, lb_I2,  ub_D2, lb_D2,dataset, alpha_reg_vec, alpha_reg_vec2)
# plt.figure()

# plot_predictions(tspan, active2,  dead2, active,  dead, alpha2, N1,  ub_I2, lb_I2, ub_D2, lb_D2,dataset, alpha_reg_vec2, predicted, N_pred, active_pred, dead_pred)
t = np.arange(0, N1-1, 1)
plt.figure()
plot_betas(t, betas1, betas1_ub, betas1_lb, N1, dataset, alpha1)

plt.figure()
plot_betas(t, betas2, betas2_ub, betas2_lb, N1, dataset, alpha2)

plt.figure()

R0_1 = betas1/gamma1
R0_1_lb = betas1_lb/gamma1
R0_1_ub = betas1_ub/gamma1

R0_2 = betas2/gamma2
R0_2_lb = betas2_lb/gamma2
R0_2_ub = betas2_ub/gamma2

plt.figure()
plot_r0(t, R0_1, R0_1_lb, R0_1_ub,N, dataset, alpha1)
plt.figure()
plot_r0(t, R0_2, R0_2_lb, R0_2_ub,N, dataset, alpha2)
plt.figure()
plot_r0_4(t, R0_2, R0_2_lb, R0_2_ub,N, dataset, alpha2)


cfr = np.ones(N)
for i in range(N):
    cfr[i] = deaths[i]/np.sum(active[0:i+1])*100


plt.figure()
plot_cfr(tspan, cfr, N1)
sigma_theta = df10.iloc[1:332,0:331].to_numpy()
sigma_theta2 = df11.iloc[1:332,0:331].to_numpy()

max_abs = np.max(np.abs(sigma_theta))
max_abs2 = np.max(np.abs(sigma_theta2))

plt.figure()
show_cov(alpha1,sigma_theta, max_abs, dataset, N1)
plt.figure()
show_cov(alpha2,sigma_theta2, max_abs2, dataset, N1)


t_reg = np.arange(0,N1-2)
plt.figure()
plot_alpha(t_reg, alpha_reg_vec, alpha_reg, N1)
plt.figure()
plot_alpha(t_reg, alpha_reg_vec2, alpha_reg2, N1)

plt.figure()
plot_state_betas(tspan, betas1, betas2,  active1,  dead1, active2, dead2, active,  dead, alpha1, alpha2, N1,  ub_I1, lb_I1, ub_D1, lb_D1,   ub_I2, lb_I2,  ub_D2, lb_D2,dataset, alpha_reg_vec, alpha_reg_vec2)


plt.show()
pdb.set_trace()
