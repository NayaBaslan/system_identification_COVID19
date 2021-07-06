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

df = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha6_N.xlsx', 'Sheet1')
df1 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha6_N.xlsx', 'Sheet2')
df7 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha6_N.xlsx', 'Sheet3')
df8 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha6_N.xlsx', 'Sheet4')
df10 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha6_N.xlsx', 'Sheet5')


df2 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha8_N.xlsx', 'Sheet1')
df3 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha8_N.xlsx', 'Sheet2')
df6 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha8_N.xlsx', 'Sheet3')
df6 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha8_N.xlsx', 'Sheet3')
df9 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha8_N.xlsx', 'Sheet4')
df11 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha8_N.xlsx', 'Sheet5')
df12 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha8_N.xlsx', 'Sheet6')

# df2 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha6_N.xlsx', 'Sheet1')
# df3 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha6_N.xlsx', 'Sheet2')
# df6 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha6_N.xlsx', 'Sheet3')
# df6 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha6_N.xlsx', 'Sheet3')
# df9 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha6_N.xlsx', 'Sheet4')
# df11 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha6_N.xlsx', 'Sheet5')
# df12 = pd.read_excel(os.getcwd() + '/sysidentification/simulation_results/freiburg/regions_freiburg_alpha6_N.xlsx', 'Sheet6')



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
N1 = 280
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

alpha1 = 1e11
alpha2 = 1e5

alpha_reg = 1e6
alpha_reg_vec = np.ones((N1-2)) * alpha_reg
alpha_reg_vec[43 - 13:53-4 - 13] = 1e7
alpha_reg_vec[61 - 13:71-4 - 13] = 1e7
alpha_reg_vec[105 - 13:115-4 - 13] = 1e7
alpha_reg_vec[217 - 13:227-4 - 13] = 1e7
alpha_reg_vec[256 - 13:266-4 - 13] = 1e7
alpha_reg_vec[267 - 13:277-4 - 13] = 1e7
alpha_reg_vec[278 - 13:288-4 - 13] = 1e7
alpha_reg_vec[288 - 13:298-4 - 13] = 1e7
alpha_reg_vec[293+5 - 13:303-4+5 - 13] = 1e7


alpha_reg2 = 1e11
alpha_reg_vec2 = np.ones((N1-2)) * alpha_reg2
alpha_reg_vec2[43 - 13:53-4 - 13] = 1e10
alpha_reg_vec2[61 - 13:71-4 - 13] = 1e10
alpha_reg_vec2[105 - 13:115-4 - 13] = 1e10
alpha_reg_vec2[217 - 13:227-4 - 13] = 1e10
alpha_reg_vec2[256 - 13:266-4 - 13] = 1e10
alpha_reg_vec2[267 - 13:277-4 - 13] = 1e10
alpha_reg_vec2[278 - 13:288-4 - 13] = 1e10
alpha_reg_vec2[288 - 13:298-4 - 13] = 1e10
alpha_reg_vec2[293+5 - 13:303-4+5 - 13] = 1e10


# plot_state_trajectories_reg(tspan, active1,  dead1, active2, dead2, active,  dead, alpha1, alpha2, N1,  ub_I1, lb_I1, ub_D1, lb_D1,   ub_I2, lb_I2,  ub_D2, lb_D2,dataset, alpha_reg_vec, alpha_reg_vec2)
# plt.figure()

plot_predictions(tspan, active2,  dead2, active,  dead, alpha2, N1,  ub_I2, lb_I2, ub_D2, lb_D2,dataset, alpha_reg_vec2, predicted, N_pred, active_pred, dead_pred)

plt.show()

contact_matrix_europe = c(5.13567073170732, 1.17274819632136, 0.982359525171638, 2.21715890088845, 1.29666356906914, 0.828866413937242, 0.528700773224482, 0.232116187961884, 0.0975205061876398, 1.01399087153423, 10.420788530466, 1.5084165224448, 1.46323525034693, 2.30050630727188, 1.0455742822567, 0.396916593664865, 0.276112578159939, 0.0867321859134207, 0.787940961549209, 1.39931415327149, 4.91448118586089, 2.39551550152373, 2.08291844616138, 1.67353143324194, 0.652483430981848, 0.263165822550241, 0.107498717856296, 1.53454251726848, 1.17129688889679, 2.06708280469829, 3.91165644171779, 2.74588910732349, 1.66499320847473, 1.02145416818956, 0.371633336270256, 0.112670158106901, 0.857264438638371, 1.7590640625625, 1.71686658407219, 2.62294018855816, 3.45916114790287, 1.87635185962704, 0.862205884832066, 0.523958801433231, 0.205791955532149, 0.646645383952458, 0.943424739130445, 1.62776721065554, 1.87677409215498, 2.21415705015835, 2.5920177383592, 1.10525460534109, 0.472961105423521, 0.282448363507455, 0.504954014454259, 0.438441714821823, 0.77694120330432, 1.40954408148402, 1.24556204828388, 1.35307720400585, 1.70385674931129, 0.812686154912104, 0.270111273681845, 0.305701280434649, 0.420580126969344, 0.432113761275257, 0.707170907986224, 1.04376196943771, 0.798427737704416, 1.12065725135372, 1.33035714285714, 0.322575366839763, 0.237578345845701, 0.24437789962337, 0.326505855457376, 0.396586297530862, 0.758318763302674, 0.881999483055259, 0.688988121391528, 0.596692087603768, 0.292682926829268)