from casadi import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import os
import pdb
import xlsxwriter

from sird_coupling_symmetric import *
from rk4 import *
start = 23 # weeek number
# 13 Aprtil, 18 May, 23 June , 28 July,  31 August, 24 September
month = "August"
# start = 13
# start = 30alpha_re
N = 6

weeks_num = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
weeks_name = ["March", "April", "April", "April", "April", "May", "May", "May", "May", "May", "June","June","June","June",
              "July", "July", "July", "July", "August", "August", "August", "August","August", "September" , "September"]

workbook = xlsxwriter.Workbook('solution_casadi'+str(start)+'week' +str(N)+'_N.xlsx')
worksheet = workbook.add_worksheet()

nx = 4
nu = 3

pop_germany_total = 83905306

symbol = SX.sym

# Controls and Parameters
# beta = symbol('beta',(9,9))
beta_low = symbol('beta',Sparsity.lower(9))
# beta = beta_low + beta_low.T - diag(diag(beta_low))

gamma = symbol('gamma',(1,9))
mu = symbol('mu',(1,9))


# u = vertcat(beta)
params = vertcat(gamma, mu)
os.chdir('../')
os.chdir('../')


# df2 = pd.read_excel(os.getcwd() + '/code/data/data_germany/weekly_active_infections_age_gender_ger.xlsx')
# df3 = pd.read_excel(os.getcwd() + '/code/data/data_germany/weekly_cumulatice_infected_male.xlsx')
# df4 = pd.read_excel(os.getcwd() + '/code/data/data_germany/dead.xlsx')
# df5 = pd.read_excel(os.getcwd() + '/code/data/data_germany/weekly_cumulatice_infected_female.xlsx')
# df6 = pd.read_excel(os.getcwd() + '/code/data/data_germany/weekly_cumulatice_infected_diverse.xlsx')

df7 = pd.read_excel(os.getcwd() + '/sysidentification/data/male_IRD.xlsx')
df8 = pd.read_excel(os.getcwd() + '/sysidentification/data/age_groups_pop_germany.xlsx')


#############################################################################
#                 MALE
#############################################################################

os.chdir(os.getcwd() + '/sysidentification/coupling/results')

#####
# Age Group 0-9

#   Age Group      Value of d
#     0-9              0
#    10-19             1
#    20-29             2
#    30-39             3
#    40-49             4
#    50-59             5
#    60-69             6
#    70-79             7
#     80+              8

# N = 15                         # weeks WORKIGN


# problem with 4, 7, 8

male_active2 = df7.iloc[82:82+9, start:start + N]
male_active2 = male_active2.to_numpy()
male_recovered2 = df7.iloc[110:110+9, start:start + N]
male_recovered2 = male_recovered2.to_numpy()
male_dead2 = df7.iloc[95:95+9,  start:start + N]
male_dead2 = male_dead2.to_numpy()

male_pop_ger_age = df8.iloc[0:9, 7]
male_pop_ger_age = male_pop_ger_age.to_numpy()


active_age = male_active2
dead_age = male_dead2
recovered_age = male_recovered2
susceptible_age = np.ones(np.shape(dead_age))

pop = male_pop_ger_age



for i in range(np.shape(dead_age)[0]):
    for j in range(np.shape(dead_age)[1]):
        susceptible_age[i][j] = pop[i] - (active_age[i][j] + dead_age[i][j] + recovered_age[i][j])


final_data = np.ones((36,N))

for i in range(0,9):
  for j in range(N):
    final_data[i][j] = susceptible_age[i][j]
    final_data[i+9][j] = active_age[i][j]
    final_data[i+18][j] = recovered_age[i][j]
    final_data[i+27][j] = dead_age[i][j]



for i in range(np.shape(dead_age)[0]):
    for j in range(np.shape(dead_age)[1]):
        if (dead_age[i][j] < 0):
            print("found less than zero")

        if (recovered_age[i][j] < 0):
            print("found less than zero")

        if (susceptible_age[i][j] < 0):
            print("found less than zero")

        if (active_age[i][j] < 0):
            print("found less than zero")


############ MODELING #####################
S = symbol('S',(9,1))
I = symbol('I',(9,1))
R = symbol('R',(9,1))
D = symbol('D',(9,1))

states = vertcat(S,I,R,D)

# controls = u


dSdt, dIdt, dRdt, dDdt = sird_model_sym(states, beta_low ,params)

rhs = vertcat(dSdt, dIdt, dRdt, dDdt)

# Form an ode function

# ode_age = Function('ode_age',[states,controls, params],[rhs])
ode_age = Function('ode_age',[states, beta_low, params],[rhs])

############ Creating a simulator ##########
N_steps_per_sample = 1
dt = 0.1

# states_final_age = rk4_age(ode_age, states, controls, params, dt)
states_final_age = rk4_age(ode_age, states, beta_low, params, dt)


# # TODO: make function + separate file!
# # Build an integrator for this system: Runge Kutta 4 integrator
# k1_age = ode_age(states_ages,controls_age)
# k2_age = ode_age(states_ages+dt/2.0*k1_age,controls_age)
# k3_age = ode_age(states_ages+dt/2.0*k2_age,controls_age)
# k4_age = ode_age(states_ages+dt*k3_age,controls_age)
#
# states_final_age = states_ages+dt/6.0*(k1_age+2*k2_age+2*k3_age+k4_age)
#
#
# one_step_age = Function('one_step_age',[states, controls, params],[states_final_age])
one_step_age = Function('one_step_age',[states, beta_low, params],[states_final_age])

X_age = states

for i in range(N_steps_per_sample):
  X_age = one_step_age(X_age, beta_low, params)

# Create a function that simulates all step propagation on a sample
# one_sample_age = Function('one_sample_age',[states, controls, params], [X_age])
one_sample_age = Function('one_sample_age',[states, beta_low, params], [X_age])


print(one_sample_age)
############ Simulating the system ##########
# TODO: double check if map accum can be used
# all_samples = one_sample.mapaccum("all_samples", N)
# print(all_samples)
# x_traj = all_samples(x0, u_time_var)

x_traj = symbol("x_traj", nx*9, N)
u_time_var = symbol("u_time_var", 9*9, (N-1))

for i in range(N-1):
    u_time_var2 = symbol('x', Sparsity.lower(9))
    u_time_var2 = reshape(u_time_var2,-1,1)
    u_time_var[:,i] = u_time_var2


# for i in range(np.shape(u_time_var)[1]-1):
#     u_time = u_time_var[:,i]
#     u_time = reshape(u_time, 9,9)
#     for i in range(np.shape(u_time)[0]):
#         for j in range(np.shape(u_time)[1]):
#             u_time[i,j] = u_time[j,i]

infected_init_age = np.ones((9,1))
for i in range(len(infected_init_age)):
  infected_init_age[i] = active_age[i][0]

dead_init_age = np.ones((9,1))
for i in range(len(dead_init_age)):
  dead_init_age[i] = dead_age[i][0]

recovered_init_age = np.ones((9,1))
for i in range(len(recovered_init_age)):
  recovered_init_age[i] = recovered_age[i][0]

x0 = np.ones((9*4,1))

for i in range(len(dead_init_age)):
  x0[i] = pop[i] - infected_init_age[i] - dead_init_age[i] - recovered_init_age[i]
  x0[i+9] = infected_init_age[i]
  x0[i+18] = recovered_init_age[i]
  x0[i+27] = dead_init_age[i]

xcurrent = x0
x_traj[:, 0] = x0

for i in range(N-1):
  current_control = u_time_var[:,i]
  current_control_t = reshape(current_control,9,9)
  xcurrent= one_sample_age(xcurrent, current_control_t,params)
  x_traj[:, i+1] = xcurrent

size = (int)((9*9-9)/2 + 9)*(N-1)
dec_variables = []
dec_var = symbol('dev_var', size)
for i in range(N-1):
  current_control = u_time_var[:,i]
  current_control_t = reshape(current_control,9,9)
  #
  for k in range(9):
      for j in range(k + 1):
          dec_variables.append(current_control_t[k,j])

for i in range(size):
    dec_var[i] = dec_variables[i]



y_sym = x_traj

model_mismatch = final_data - y_sym

model_ird = y_sym[9:36,:]
real_ird = final_data[9:36,:]

model_mismatch_ird = model_ird - real_ird

u_test = reshape(u_time_var,-1,1)

u_test = vertcat(u_test, gamma.T, mu.T)
u_test = vertcat(dec_var, gamma.T, mu.T)

lbg_age = np.zeros(np.shape(u_test))
ubg_age = np.ones(np.shape(u_test))*1000000

lbg_sigma = np.zeros((N-2)*9)
lbg_mu = np.zeros((N-2)*9)

ubg_sigma = np.zeros((N-2)*9)
ubg_mu = np.zeros((N-2)*9)

lbg_all = vertcat(lbg_age, )
ubg_all = vertcat(ubg_age, ubg_sigma , ubg_mu)

g_age = u_test

g_all = vertcat(g_age)
all_betas = u_test[0:size]
size_one = (int)( size / (N-1) )
reg = symbol('reg', size_one)
# alpha_reg = 1e17
# alpha_reg = 1e90
alpha_reg = 1e120
alpha_reg = 1e7
# alpha_reg = 1e120


# 1e7 for N=6 works

for i in range(N-2):
  betas = all_betas[i*45:i*45+45]
  betas_next = all_betas[(i+1)*45:(i+1)*45+45]

  for j in range(np.shape(betas)[0]):
    reg[j] = betas[j] - betas_next[j]




nlp_age = {'x':u_test, 'f':dot(model_mismatch, model_mismatch) + alpha_reg*dot(reg,reg),'g':g_age}
# nlp_age = {'x':u_test, 'f':dot(model_mismatch, model_mismatch) ,'g':g_age}

solver_age = nlpsol("solver", "ipopt", nlp_age)   # DM(1.95599)

u_guess_age = np.ones(np.shape(u_test))
sol_age = solver_age(x0=u_guess_age,lbg=lbg_age, ubg=ubg_age)

pdb.set_trace()
##################################################################################

print("working till here")
all_params_age = sol_age["x"]

x_solution = DM(nx*9,N)
xcurrent = x0

days = N
x_solution[:, 0] = x0
solution_params = sol_age["x"]
betas =  solution_params[0:9*9*(N-1)]

gamma = solution_params[9*9*(N-1):9*9*(N-1)+9]
mu = solution_params[9*9*(N-1)+9:9*9*(N-1)+18]
avg_betas = 0
for i in range(N-1):
  u_sol = solution_params[i*81:i*81+81]
  avg_betas += reshape(u_sol,(9,9))

  xcurrent = one_sample_age(xcurrent, reshape(u_sol,(9,9)), vertcat(gamma.T,mu.T))
  x_solution[:, i+1] = xcurrent

avg_betas = avg_betas / (N-1)
max_abs = np.max(np.abs(avg_betas.full()))
plt.imshow(avg_betas, cmap='RdBu', vmin=-max_abs, vmax=max_abs)
plt.colorbar()
plt.title("Average Transmission Rate August")
plt.xlabel("Age Groups")
plt.ylabel("Age Groups")
os.chdir('../')
os.chdir('../')

plt.savefig(os.getcwd() + "/figures/coupling/" + str(month)+ "_avg_betas.pdf")

all_S_ages = x_solution[0:9,:]
all_I_ages = x_solution[9:18,:]
all_R_ages = x_solution[18:27,:]
all_D_ages = x_solution[27:36,:]

sol_array = x_solution.toarray()
param_array = sol_age['x'].toarray()
param_array = np.transpose(param_array)

row = 2

array = np.transpose(sol_array)
for col, data in enumerate(array):
    worksheet.write_column(row, col, data)

worksheet = workbook.add_worksheet()

for col, data in enumerate(param_array):
    worksheet.write_column(row, col, data)

pdb.set_trace()
workbook.close()
plt.show()
pdb.set_trace()

