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

import xlsxwriter
workbook = xlsxwriter.Workbook('solution_casadi.xlsx')
worksheet = workbook.add_worksheet()

print(os.getcwd())
os.chdir('../')
os.chdir('../')

print(os.getcwd())

# Source: https://github.com/jgehrcke/covid-19-germany-gae
datasets  = ["Freiburg", "Emmendingen", "Ortenaukreis", "Rottweil", "Tuttlingen", "Konstanz", "Waldshut", "Lörrach", "Breisgauhochscwarzwald", "Schwarzwaldbaarkreis"]
ids       = ["8311", "8316", "8317", "8325", "8327", "8335", "8337", "8336", "8315", "8326"]

datasets  = ["Stuttgart", "Böblingen", "Esslingen", "Göppingen", "Ludwigsburg", "Rems-Murr-Kreis", "Heilbronn", "Hohenlohekreis", "Main-Tauber-Kreis", "Schwäbisch Hall"]
ids       = ["8111", "8115", "8116", "8117", "8118", "8119", "8121", "8126", "8128", "8127"]

datasets  = ["Freiburg", "Emmendingen", "Ortenaukreis", "Rottweil", "Tuttlingen", "Konstanz", "Waldshut", "Lörrach", "Breisgauhochscwarzwald", "Schwarzwaldbaarkreis"]
ids       = ["8311", "8316", "8317", "8325", "8327", "8335", "8337", "8336", "8315", "8326"]

populations = [231195, 166408, 430953, 139878, 140766, 286305, 171003, 228736, 263601, 212506 ]
# populations = [635911, 392807, 535024, 258145, 545423, 427248 , 126592, 112655, 132399, 196761]

dataset = "Freiburg"

for i in range(len(datasets)):
    if (datasets[i] == dataset):
        print("yes from10 at i", i )
        pop = populations[i]
        id = ids[i]

df = pd.read_csv(os.getcwd() + '/covid-19-germany-gae/cases-rki-by-ags.csv')
df1 = pd.read_csv(os.getcwd() + '/covid-19-germany-gae/deaths-rki-by-ags.csv')

all_columns = df.columns
all_columns = all_columns.to_numpy()

for i in range(np.shape(all_columns)[0]):
    if (all_columns[i] == id):
        print("yes at i", i )
        cumulative_infected = df.iloc[:, i]
        deaths = df1.iloc[:, i]
        break

# cumulative_infected = df.iloc[:, 191]

# Preprocessing
########################################################################################

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


# for i in range(len(active_infections)):
#     recovered[i] = cumulative_infected[i+15] - real_active[i] - deaths[i]
#     susceptible[i] = pop - real_active[i] - deaths[i] - recovered[i]

for i in range(len(active_infections)):
    recovered[i] = cumulative_infected[i + 15] - active_infections[i] - deaths[i]
    susceptible[i] = pop - active_infections[i] - deaths[i] - recovered[i]

########################################################################################

# Phase 1
########################################################################################
# Phase 1 : 01.04.2020 - 30.06.2020 (First Wave)
# first day = 01.04.2020 cumulative_infected[30]

# N1 = 91
N_pred = 47
N1 = 330 + 80 - 40 -30 +20

N_pred = 47
N1 = 330


start = 16 # = 18.03.2020 with N=330 10.02.2021   N=290  01.01        N=280 22.12.
# 43 wearing masks 61 shops reopen 105 kindergartens and schools reopen 217 wearing masks in city center 256 wearing masks in workplace
# 267 lockdown + curfew 273 closing all non-essential shops

####### 16 = April 02 2021 first wave   (33)
####### 230 = November 02 2021 second wave (247)

##### For first wave:  start = 15 N1 = 150
##### For second wave: start = 230 N1 = 100
##### For first and second wave: start = 16 N1 = 330

active = active_infections[start:start+N1]
dead = deaths[start:start+N1]
recovered = recovered[start:start+N1]
susceptible = susceptible[start:start+N1]

cumulated = cumulative_infected[start+15:start+N1+15]

data1 = np.stack((susceptible , active, recovered, dead), axis=-1)
data = data1.astype(int)   # shape (193,2)

# Define the model: states, controls, and parameters
########################################################################################
nx = 4
nu = 1

symbol = SX.sym

beta = symbol('beta')
gamma = symbol('gamma')
mu = symbol('mu')

controls = beta
params = vertcat(gamma,mu)

S = symbol('S')
I = symbol('I')
R = symbol('R')
D = symbol('D')

states = vertcat(S,I,R,D)

u_guess = DM([0.05])

dSdt, dIdt, dRdt, dDdt = sird_model_params(states, controls, params)

rhs = vertcat(dSdt, dIdt, dRdt, dDdt)

ode = Function('ode',[states,controls,params],[rhs])

# Define the NLP and Solve it
#############################################################################

N_steps_per_sample = 1
dt = 0.1

# Build an integrator for this system: Runge Kutta 4 integrator
k1_age = ode(states,controls,params)
k2_age = ode(states+dt/2.0*k1_age,controls,params)
k3_age = ode(states+dt/2.0*k2_age,controls,params)
k4_age = ode(states+dt*k3_age,controls,params)

states_final = states+dt/6.0*(k1_age+2*k2_age+2*k3_age+k4_age)


one_step_age = Function('one_step_age',[states, controls,params],[states_final])

X_age = states

for i in range(N_steps_per_sample):
  X_age = one_step_age(X_age, controls, params)

one_sample = Function('one_sample_age',[states, controls,params], [X_age])


print(one_sample)
x_traj = symbol("x_traj_age", nx, N1)
beta_time_var = symbol("beta_time_var", nu * (N1-1), 1)

infected_init = active[0]
dead_init = dead[0]
recovered_init = recovered[0]

x0 = DM([pop - infected_init - dead_init - recovered_init, infected_init,recovered_init, dead_init])

xcurrent = x0
x_traj[:, 0] = x0


for i in range(N1-1):
  xcurrent = one_sample(xcurrent, beta_time_var[nu*i: (i+1)*nu],params)
  x_traj[:, i+1] = xcurrent

y_sym = x_traj

for i in range(np.shape(data)[0]):
  for j in range(np.shape(data)[1]):
    if data[i][j] < 0 :
      print("ERROR: Found less than zero")

model_mismatch = data.T - y_sym


g = vertcat(beta_time_var,params)

lbg = np.zeros(np.shape(g))
ubg = np.ones(np.shape(g))*1000000

alpha_reg = 1e7

alpha_reg = 1e8
alpha_reg_vec = np.ones((N1-2)) * alpha_reg
alpha_reg_vec[43 - 13:53 - 13] = 1e7
alpha_reg_vec[61 - 13:71 - 13] = 1e7
alpha_reg_vec[105 - 13:115 - 13] = 1e7
alpha_reg_vec[217 - 13:227 - 13] = 1e7
alpha_reg_vec[256 - 13:266 - 13] = 1e7
alpha_reg_vec[267 - 13:277 - 13] = 1e7
alpha_reg_vec[278 - 13:288 - 13] = 1e7
alpha_reg_vec[288 - 13:290 - 13] = 1e7
alpha_reg_vec[293 - 13:298 - 13] = 1e7

alpha_reg = 1e11
alpha_reg_vec = np.ones((N1-2)) * alpha_reg
alpha_reg_vec[43 - 13:53 - 13] = 1e10
alpha_reg_vec[61 - 13:71 - 13] = 1e10
alpha_reg_vec[105 - 13:115 - 13] = 1e10
alpha_reg_vec[217 - 13:227 - 13] = 1e10
alpha_reg_vec[256 - 13:266 - 13] = 1e10
alpha_reg_vec[267 - 13:277 - 13] = 1e10
alpha_reg_vec[278 - 13:288 - 13] = 1e10
alpha_reg_vec[288 - 13:290 - 13] = 1e10
alpha_reg_vec[293 - 13:298 - 13] = 1e10


# alpha_reg = 1e7
# alpha_reg_vec = np.ones((N1-2)) * alpha_reg
# alpha_reg_vec[43 - 13:53 - 13] = 1e6
# alpha_reg_vec[61 - 13:71 - 13] = 1e6
# alpha_reg_vec[105 - 13:115 - 13] = 1e6
# alpha_reg_vec[217 - 13:227 - 13] = 1e6
# alpha_reg_vec[256 - 13:266 - 13] = 1e6
# alpha_reg_vec[267 - 13:277 - 13] = 1e6
# alpha_reg_vec[278 - 13:288 - 13] = 1e6
# alpha_reg_vec[288 - 13:290 - 13] = 1e6
# alpha_reg_vec[293 - 13:298 - 13] = 1e6

alpha_reg = 1e6
alpha_reg_vec = np.ones((N1-2)) * alpha_reg
alpha_reg_vec[43 - 13:53 - 13] = 1e4
alpha_reg_vec[61 - 13:71 - 13] = 1e4
alpha_reg_vec[105 - 13:115 - 13] = 1e4
alpha_reg_vec[217 - 13:227 - 13] = 1e4
alpha_reg_vec[256 - 13:266 - 13] = 1e4
alpha_reg_vec[267 - 13:277 - 13] = 1e4
alpha_reg_vec[278 - 13:288 - 13] = 1e4

alpha_reg = 1e6
alpha_reg_vec = np.ones((N1-2)) * alpha_reg
alpha_reg_vec[43 - 13 -3:43 - 13 +3+3] = 1e5
alpha_reg_vec[61 - 13-3:71 - 13+3] = 1e5
alpha_reg_vec[105 - 13-3:115 - 13+3] = 1e5
alpha_reg_vec[217 - 13-3:227 - 13+3] = 1e5
alpha_reg_vec[256 - 13-3:266 - 13+3] = 1e5
alpha_reg_vec[267 - 13-3:277 - 13+3] = 1e5
alpha_reg_vec[278 - 13-3:288 - 13+3] = 1e5
alpha_reg_vec[288 - 13-3:290 - 13+3] = 1e5
alpha_reg_vec[293 - 13+5-3:298 - 13+3+5] = 1e5

alpha_reg = 1e8
alpha_reg_vec = np.ones((N1-2)) * alpha_reg
alpha_reg_vec[43 - 13 -3:43 - 13 +3+3] = 1e7
alpha_reg_vec[61 - 13-3:61 - 13+3] = 1e7
alpha_reg_vec[105 - 13-3:105 - 13+3] = 1e7
alpha_reg_vec[217 - 13-3:217 - 13+3] = 1e7
alpha_reg_vec[256 - 13-3:256 - 13+3] = 1e7
alpha_reg_vec[267 - 13-3:267 - 13+3] = 1e7
alpha_reg_vec[278 - 13-3:278 - 13+3] = 1e7
alpha_reg_vec[288 - 13-3:288 - 13+3] = 1e7
alpha_reg_vec[293 - 13+5-3:293 - 13+3+5] = 1e7

alpha_reg = 1e5
alpha_reg_vec = np.ones((N1-2)) * alpha_reg
alpha_reg_vec[43 - 13 -3:43 - 13 +3+3] = 1e4
alpha_reg_vec[61 - 13-3:61 - 13+3] = 1e4
alpha_reg_vec[105 - 13-3:105 - 13+3] = 1e4
alpha_reg_vec[217 - 13-3:217 - 13+3] = 1e4
alpha_reg_vec[256 - 13-3:256 - 13+3] = 1e4
alpha_reg_vec[267 - 13-3:267 - 13+3] = 1e4
alpha_reg_vec[278 - 13-3:278 - 13+3] = 1e4
alpha_reg_vec[288 - 13-3:280 - 13+3] = 1e4
alpha_reg_vec[293 - 13+5-3:293 - 13+3+5] = 1e4

# alpha_reg = 1e7
# alpha_reg_vec = np.ones((N1-2)) * alpha_reg
# alpha_reg_vec[43 - 13 -3:43 - 13 +3+3] = 1e5
# alpha_reg_vec[61 - 13-3:61 - 13+3] = 1e5
# alpha_reg_vec[105 - 13-3:105 - 13+3] = 1e5
# alpha_reg_vec[217 - 13-3:217 - 13+3] = 1e5
# alpha_reg_vec[256 - 13-3:256 - 13+3] = 1e5
# alpha_reg_vec[267 - 13-3:267 - 13+3] = 1e5
# alpha_reg_vec[278 - 13-3:278 - 13+3] = 1e5
# alpha_reg_vec[288 - 13-3:280 - 13+3] = 1e5
# alpha_reg_vec[293 - 13+5-3:293 - 13+3+5] = 1e5

# alpha_reg = np.ones(N1-2) * 1e7

# alpha_reg = 1e6

workbook = xlsxwriter.Workbook('sysidentification/simulation_results/freiburg/regions_freiburg_alpha'+ str((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2)))) + '_N.xlsx')
worksheet = workbook.add_worksheet()

reg = symbol('reg', N1-2)
for i in range(np.shape(beta_time_var)[0]-1):
    reg[i] = beta_time_var[i+1] - beta_time_var[i]

u_test = vertcat(beta_time_var,params)

# Regression
##################################################################################
# give different values when non-pharmaceutical interventions are applied

##############################################################################################

# Weighting
##############################################################################################

W = np.ones((4,N1))
for i in range(4):
    for j in range(60):
        W[i,j] = 0.001
    for j in range(60,N1):
        W[i, j] = 1

##############################################################################################

# nlp = {'x':u_test, 'f':dot(model_mismatch*np.sqrt(W), np.sqrt(W)*model_mismatch) + alpha_reg*dot(reg,reg) ,'g':g}

nlp = {'x':u_test, 'f':dot(model_mismatch*np.sqrt(W), np.sqrt(W)*model_mismatch) + dot(reg*np.sqrt(alpha_reg_vec),np.sqrt(alpha_reg_vec)*reg) ,'g':g}

#
# nlp = {'x':u_test, 'f':dot(model_mismatch, model_mismatch) + alpha_reg*dot(reg,reg) ,'g':g}
# nlp = {'x':u_test, 'f':dot(model_mismatch, model_mismatch) + dot(reg*alpha_reg,reg) ,'g':g}

solver = nlpsol("solver", "ipopt", nlp)   # DM(1.95599)

u_guess = np.ones(np.shape(u_test))
sol = solver(x0=u_guess,lbg=lbg, ubg=ubg)

###### Covariance Estimation
hessLag = solver.get_function('nlp_hess_l')
sol['lam_g']
hessLag_sol = hessLag(sol['x'], [] , 1,sol['lam_g'])

hessLag_sol = hessLag_sol + hessLag_sol.T - np.diag(np.diag(hessLag_sol))
hess_sparsity = hessLag.sparsity_out(0)

hess_sparsity.is_symmetric()
cov = inv(hessLag_sol)

sparsity_matrix = DM(hess_sparsity).full()

hessLag_inv = inv(hessLag_sol)
sparsity_matrix = sparsity_matrix + sparsity_matrix.T - np.diag(np.diag(sparsity_matrix))

sigma_theta = sol['f'] * hessLag_inv / (4*N1 - (N1+1)) * 10
sigma_theta = sol['f'] * hessLag_inv / (4*N1 - (N1+1)) * 10

# sigma_theta = hessLag_inv * 10000

sigma_betas = np.diag(sigma_theta)
sigma_betas = np.sqrt(sigma_betas)/5

f1 = y_sym[0,:]
f2 = y_sym[1,:]
f3 = y_sym[2,:]
f4 = y_sym[3,:]

J1 = Function('j1', [u_test], [jacobian(f1, u_test)])
J2 = Function('j2', [u_test], [jacobian(f2, u_test)])
J3 = Function('j3', [u_test], [jacobian(f3, u_test)])
J4 = Function('j4', [u_test], [jacobian(f4, u_test)])

J1_sol = J1(sol['x'])
J2_sol = J2(sol['x'])
J3_sol = J3(sol['x'])
J4_sol = J4(sol['x'])

J_all = jacobian(y_sym, u_test)

sigma_y = []
sigma_S = []
sigma_I = []
sigma_R = []
sigma_D = []

for i in range(N1):
    sigma_y.append(J1_sol[i,:] @ sigma_theta @ J1_sol[i,:].T)
    sigma_S.append(J1_sol[i,:] @ sigma_theta @ J1_sol[i,:].T)
    sigma_I.append(J2_sol[i, :] @ sigma_theta @ J2_sol[i, :].T)
    sigma_R.append(J3_sol[i, :] @ sigma_theta @ J3_sol[i, :].T)
    sigma_D.append(J4_sol[i, :] @ sigma_theta @ J4_sol[i, :].T)

sigma_y = np.sqrt(sigma_y)
sigma_S = np.sqrt(sigma_S)*2
sigma_I = np.sqrt(sigma_I)
sigma_R = np.sqrt(sigma_R)
sigma_D = np.sqrt(sigma_D)/5

all_params = sol['x']
betas = all_params[0:N1-1]
sigma = all_params[N1-1]
mu = all_params[N1]


# sigma_betas = np.diag(sigma_theta)
# sigma_betas = np.sqrt(sigma_betas)
ub_betas = betas + sigma_betas[0:N1-1]
lb_betas = betas - sigma_betas[0:N1-1]

ub_betas = ub_betas.full()
lb_betas = lb_betas.full()

betas_all = horzcat(betas,lb_betas,ub_betas)
betas_all = betas_all.full()

ub_sigma = sigma + sigma_betas[N1-1]
lb_sigma = sigma - sigma_betas[N1-1]
ub_mu = mu + sigma_betas[N1]
lb_mu = mu - sigma_betas[N1]

ub_sigma = ub_sigma.full()
lb_sigma = lb_sigma.full()
ub_mu = ub_mu.full()
lb_mu = lb_mu.full()

# Get the state estimates using NLP Solution
#############################################################################
x_solution = DM(nx,N1)
xcurrent = x0
days = N1
x_solution[:, 0] = x0

x_predicted = DM(nx,N_pred)

for i in range(N1-1):
  xcurrent = one_sample(xcurrent, betas[nu*i: (i+1)*nu], vertcat(sigma,mu))
  x_solution[:, i+1] = xcurrent


for i in range(N_pred):
    xcurrent = one_sample(xcurrent, betas[nu*(N1-2 -5): (N1-2+1 -5)*nu], vertcat(sigma,mu))
    x_predicted[:,i] = xcurrent

all_S = np.reshape(x_solution[0,:],(-1,1))
all_I = np.reshape(x_solution[1,:],(-1,1))
all_R = np.reshape(x_solution[2,:],(-1,1))
all_D = np.reshape(x_solution[3,:],(-1,1))

os.chdir(os.getcwd() + '/sysidentification/simulation_results/freiburg')

array = horzcat(all_I,all_D)
array = array.full()

row = 2

for col, data in enumerate(array):
    worksheet.write_column(row, col, data)

worksheet = workbook.add_worksheet()


max_abs = np.max(np.abs(sigma_theta.full()))

run_id = "alpha" + str(alpha_reg)
os.chdir('../')
os.chdir('../')
os.chdir('../')

################################
# With Covariance Estimates

sigma_y = np.reshape(sigma_y,(-1,1))
sigma_S = np.reshape(sigma_S,(-1,1))
sigma_I = np.reshape(sigma_I,(-1,1))
sigma_R = np.reshape(sigma_R,(-1,1))
sigma_D = np.reshape(sigma_D,(-1,1))

all_S_ub = all_S + sigma_y
all_S_lb = all_S - sigma_y

ub_S = all_S + sigma_S
lb_S = all_S - sigma_S

ub_I = all_I + sigma_I
lb_I = all_I - sigma_I

ub_R = all_R + sigma_R*5
lb_R = all_R - sigma_R*5

ub_D = all_D + sigma_D
lb_D = all_D - sigma_D

lb_ub = horzcat(lb_I, ub_I, lb_D, ub_D)
lb_ub = lb_ub.full()

for col, data in enumerate(lb_ub):
    worksheet.write_column(row, col, data)

worksheet = workbook.add_worksheet()
betas_mat = betas.full()

for col, data in enumerate(betas_all):
    worksheet.write_column(row, col, data)

worksheet = workbook.add_worksheet()
worksheet.write_column(row,0,sigma.full())
worksheet.write_column(row,1,mu.full())

worksheet = workbook.add_worksheet()
for col, data in enumerate(sigma_theta.full()):
    worksheet.write_column(row, col, data)

worksheet = workbook.add_worksheet()
for col, data in enumerate(x_predicted.full()):
    worksheet.write_column(row, col, data)

workbook.close()

plt.figure(1)
# show_cov(alpha_reg,sigma_
# theta, max_abs, dataset, run_id, N1)

################################
tspan = np.arange(0, N1, 1)
tspan_controls = np.arange(0, N1-1, 1)

plt.figure(2)
# plot_betas(tspan_controls,betas,run_id,  ub_betas, lb_betas, N1, dataset, alpha_reg)

plt.figure(3)
plot_state_trajectories(tspan, all_S, all_I, all_R, all_D, susceptible, active, recovered, dead, N1, all_S_ub, all_S_lb,
                        ub_S, lb_S, ub_I, lb_I, ub_R, lb_R, ub_D, lb_D, dataset)

error_S = np.zeros(N1)
error_I = np.zeros(N1)
error_R = np.zeros(N1)
error_D = np.zeros(N1)

res_S = np.zeros(N1)
res_I = np.zeros(N1)
res_R = np.zeros(N1)
res_D = np.zeros(N1)

for i in range(N1):
    error_S[i] = np.absolute(all_S[i] - susceptible[i])
    error_I[i] = np.absolute(all_I[i] - active[i])
    error_R[i] = np.absolute(all_R[i] - recovered[i])
    error_D[i] = np.absolute(all_D[i] - dead[i])

for i in range(N1):
    res_S[i] = all_S[i] - susceptible[i]
    res_I[i] = all_I[i] - active[i]
    res_R[i] = all_R[i] - recovered[i]
    res_D[i] = all_D[i] - dead[i]

ub_res_S = res_S + np.reshape(sigma_S, (N1))
lb_res_S = res_S - np.reshape(sigma_S, (N1))

ub_res_I = res_I + np.reshape(sigma_I, (N1))
lb_res_I = res_I - np.reshape(sigma_I, (N1))

ub_res_R = res_R + np.reshape(sigma_R, (N1))
lb_res_R = res_R - np.reshape(sigma_R, (N1))

ub_res_D = res_D + np.reshape(sigma_D, (N1))
lb_res_D = res_D - np.reshape(sigma_D, (N1))

plt.figure(7)
plot_errors(tspan, error_S, error_I, error_R, error_D, pop, N1, dataset)

plt.figure(9)
plot_residuals(tspan, res_S, res_I, res_R, res_D,N1, ub_res_S, lb_res_S,ub_res_I, lb_res_I,ub_res_R, lb_res_R,ub_res_D, lb_res_D , dataset)


plt.show()