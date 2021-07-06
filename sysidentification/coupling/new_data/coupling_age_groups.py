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
from plots_utils_coupling import *

start = 13 + 20
N = 17 # max = 37 alpha_reg 1e9
start = 13
N = 37

start = 13
N = 17 # max = 37 alpha_reg 1e9
N = 37

alpha_reg = 1e120

alpha_reg = 1e60
alpha_reg = 1e100

alpha_reg = 1e9
alpha_reg = 1e12
alpha_reg = 1e15
alpha_reg = 1e17 # doesnt work
alpha_reg = 1e7
alpha_reg = 1e9






age_groups = 5
weeks_num = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]

if (start==13):
    print("yes here")
    weeks_name = ["March", "April", "April", "April", "April", "May", "May", "May", "May", "May", "June","June","June","June",
              "July", "July", "July", "July", "August", "August", "August", "August","August", "September" , "September"]
elif (start==13+20):
    print("yes here 2")
    weeks_name = [ "August", "August", "August", "August", "August", "September", "September"]

else:
    print("nothing")
workbook = xlsxwriter.Workbook('solution_casadi'+str(N)+ str((int)(np.rint((np.log(alpha_reg) / np.log(100) * 2))))  + 'N'+ str(N)+ '.xlsx')
worksheet = workbook.add_worksheet()

nx = 4
nu = 3

pop_germany_total = 83905306

symbol = SX.sym

beta_low = symbol('beta',Sparsity.lower(age_groups))

gamma = symbol('gamma',(1,age_groups))
mu = symbol('mu',(1,age_groups))

# u = vertcat(beta)
params = vertcat(gamma, mu)
os.chdir('../')
os.chdir('../')

df1 = pd.read_excel(os.getcwd() + '/data/coupling/five_age_groups_active.xlsx')
df2 = pd.read_excel(os.getcwd() + '/data/coupling/five_age_groups_cumulative.xlsx')
df3 = pd.read_excel(os.getcwd() + '/data/coupling/five_age_groups_dead.xlsx')
df4 = pd.read_excel(os.getcwd() + '/data/coupling/age_groups_pop_germany_red.xlsx')
#############################################################################
#                 MALE
#############################################################################

active = df1.iloc[45:45+5, 25+start + 2*(start-13) :25+start + 2*(start-13) + 3*N:3]
active = active.to_numpy()

cumulative = df2.iloc[76:76+5, start -1 : start-1 + N]
cumulative = cumulative.to_numpy()

cumulative_female = df2.iloc[91:91+5, start -1 : start-1 + N]
cumulative_female = cumulative_female.to_numpy()

cumulative_male = df2.iloc[103:103+5, start -1 : start-1 + N]
cumulative_male = cumulative_male.to_numpy()

cumulative_diverse = df2.iloc[116:116+5, start -1 : start-1 + N]
cumulative_diverse = cumulative_diverse.to_numpy()

dead = df3.iloc[58:58+5, start-10 + 2*(start-13):start-10 + 2*(start-13) + 3*N:3]
pdb.set_trace()
dead = dead.to_numpy()

pop = df4.iloc[10:15, 9]
pop = pop.to_numpy()

susceptible = np.ones(np.shape(dead))
recovered = np.ones(np.shape(dead))


for i in range(np.shape(dead)[0]):
    for j in range(np.shape(dead)[1]):
        recovered[i][j] = cumulative[i][j] - active[i][j] - dead[i][j]
        if (recovered[i][j] < 0):
            print("Found less than zero recovered",i,j, recovered[i][j])
            recovered[i][j] = 0

        susceptible[i][j] = pop[i] - active[i][j] - dead[i][j] - recovered[i][j]


final_data = np.ones((4*age_groups,N))

pdb.set_trace()
for i in range(0,age_groups):
  for j in range(N):
    final_data[i][j] = susceptible[i][j]
    final_data[i+5][j] = active[i][j]
    final_data[i+10][j] = recovered[i][j]
    final_data[i+15][j] = dead[i][j]


for i in range(np.shape(dead)[0]):
    for j in range(np.shape(dead)[1]):
        if (dead[i][j] < 0):
            print("found less than zero, dead" ,i,j)

        if (recovered[i][j] < 0):
            print("found less than zero recovered",i,j)

        if (susceptible[i][j] < 0):
            print("found less than zero suscpetible",i,j)

        if (active[i][j] < 0):
            print("found less than zero active",i,j)

t = np.arange(0, N, 1)

############ MODELING #####################
S = symbol('S',(age_groups,1))
I = symbol('I',(age_groups,1))
R = symbol('R',(age_groups,1))
D = symbol('D',(age_groups,1))

states = vertcat(S,I,R,D)

dSdt, dIdt, dRdt, dDdt = sird_model_sym_red(states, beta_low ,params)

rhs = vertcat(dSdt, dIdt, dRdt, dDdt)

# Form an ode function

ode_age = Function('ode_age',[states, beta_low, params],[rhs])

############ Creating a simulator ##########
N_steps_per_sample = 1
dt = 0.1

states_final_age = rk4_age(ode_age, states, beta_low, params, dt)

one_step_age = Function('one_step_age',[states, beta_low, params],[states_final_age])

X_age = states

for i in range(N_steps_per_sample):
  X_age = one_step_age(X_age, beta_low, params)

one_sample_age = Function('one_sample_age',[states, beta_low, params], [X_age])

print(one_sample_age)

x_traj = symbol("x_traj", nx*age_groups, N)
u_time_var = symbol("u_time_var", age_groups*age_groups, (N-1))

for i in range(N-1):
    u_time_var2 = symbol('x', Sparsity.lower(age_groups))
    u_time_var2 = reshape(u_time_var2,-1,1)
    u_time_var[:,i] = u_time_var2

infected_init = np.ones((age_groups,1))
for i in range(len(infected_init)):
  infected_init[i] = active[i][0]

dead_init = np.ones((age_groups,1))
for i in range(len(dead_init)):
  dead_init[i] = dead[i][0]

recovered_init = np.ones((age_groups,1))
for i in range(len(recovered_init)):
  recovered_init[i] = recovered[i][0]

x0 = np.ones((age_groups*4,1))

for i in range(len(dead_init)):
  x0[i] = pop[i] - infected_init[i] - dead_init[i] - recovered_init[i]
  x0[i+age_groups] = infected_init[i]
  x0[i+2*age_groups] = recovered_init[i]
  x0[i+3*age_groups] = dead_init[i]

xcurrent = x0
x_traj[:, 0] = x0

for i in range(N-1):
  current_control = u_time_var[:,i]
  current_control_t = reshape(current_control,age_groups,age_groups)
  xcurrent= one_sample_age(xcurrent, current_control_t,params)
  x_traj[:, i+1] = xcurrent

size = (int)((age_groups*age_groups-age_groups)/2 + age_groups)*(N-1)

dec_variables = []
dec_var = symbol('dev_var', size)
for i in range(N-1):
  current_control = u_time_var[:,i]
  current_control_t = reshape(current_control,age_groups,age_groups)
  #
  for k in range(age_groups):
      for j in range(k + 1):
          dec_variables.append(current_control_t[k,j])

for i in range(size):
    dec_var[i] = dec_variables[i]

y_sym = x_traj
model_mismatch = final_data - y_sym


model_ird = y_sym[5:20,:]
real_ird = final_data[5:20,:]

model_mismatch_ird = model_ird - real_ird

u_test = reshape(u_time_var,-1,1)

u_test = vertcat(u_test, gamma.T, mu.T)
u_test = vertcat(dec_var, gamma.T, mu.T)

lbg_age = np.zeros(np.shape(u_test))
ubg_age = np.ones(np.shape(u_test))*1000000

lbg_sigma = np.zeros((N-2)*age_groups)
lbg_mu = np.zeros((N-2)*age_groups)

ubg_sigma = np.zeros((N-2)*age_groups)
ubg_mu = np.zeros((N-2)*age_groups)

lbg_all = vertcat(lbg_age, )
ubg_all = vertcat(ubg_age, ubg_sigma , ubg_mu)

g_age = u_test

g_all = vertcat(g_age)
all_betas = u_test[0:size]
size_one = (int)( size / (N-1) )
reg = symbol('reg', size_one)

for i in range(N-2):
  betas = all_betas[i*size_one:i*size_one+size_one]
  betas_next = all_betas[(i+1)*size_one:(i+1)*size_one+size_one]

  for j in range(np.shape(betas)[0]):
    reg[j] = betas[j] - betas_next[j]

pdb.set_trace()
nlp_age = {'x':u_test, 'f':dot(model_mismatch, model_mismatch) + alpha_reg*dot(reg,reg),'g':g_age}

solver_age = nlpsol("solver", "ipopt", nlp_age)   # DM(1.95599)

u_guess_age = np.ones(np.shape(u_test))
sol_age = solver_age(x0=u_guess_age,lbg=lbg_age, ubg=ubg_age)

##################################################################################
# Covariance Estimation

hessLag = solver_age.get_function('nlp_hess_l')

hessLag_sol = hessLag(sol_age['x'], [] , 1,sol_age['lam_g'])

hessLag_sol = hessLag_sol + hessLag_sol.T - np.diag(np.diag(hessLag_sol))

hess_sparsity = hessLag.sparsity_out(0)

hess_sparsity.is_symmetric()
cov = inv(hessLag_sol)

sparsity_matrix = DM(hess_sparsity).full()

hessLag_inv = inv(hessLag_sol)
sparsity_matrix = sparsity_matrix + sparsity_matrix.T - np.diag(np.diag(sparsity_matrix))

sigma_theta = sol_age['f'] * hessLag_inv / (4*5*N - (N+1))

pdb.set_trace()
sigma_theta = np.abs(sigma_theta)
sigma_theta = np.sqrt(sigma_theta)
sigma_theta = np.sqrt(sigma_theta)
sigma_theta = np.sqrt(sigma_theta)
sigma_theta = np.sqrt(sigma_theta)

sigma_betas = np.diag(sigma_theta)
sigma_betas = np.sqrt(sigma_betas)




f1 = y_sym[0:5,:]
f2 = y_sym[5:10,:]
f3 = y_sym[10:15,:]
f4 = y_sym[15:20,:]

J1 = Function('j1', [u_test], [jacobian(f1, u_test)])
J2 = Function('j2', [u_test], [jacobian(f2, u_test)])
J3 = Function('j3', [u_test], [jacobian(f3, u_test)])
J4 = Function('j4', [u_test], [jacobian(f4, u_test)])

J1_sol = J1(sol_age['x'])
J2_sol = J2(sol_age['x'])
J3_sol = J3(sol_age['x'])
J4_sol = J4(sol_age['x'])

J_all = jacobian(y_sym, u_test)

sigma_y = []
sigma_S = []
sigma_I = []
sigma_R = []
sigma_D = []

for i in range(N*5):
    sigma_y.append(J1_sol[i,:] @ sigma_theta @ J1_sol[i,:].T)
    sigma_S.append(J1_sol[i,:] @ sigma_theta @ J1_sol[i,:].T)
    sigma_I.append(J2_sol[i, :] @ sigma_theta @ J2_sol[i, :].T)
    sigma_R.append(J3_sol[i, :] @ sigma_theta @ J3_sol[i, :].T)
    sigma_D.append(J4_sol[i, :] @ sigma_theta @ J4_sol[i, :].T)

sigma_y = np.sqrt(sigma_y)
sigma_S = np.sqrt(sigma_S)
sigma_I = np.sqrt(sigma_I)
sigma_R = np.sqrt(sigma_R)
sigma_D = np.sqrt(sigma_D)

sigma_I = np.reshape(sigma_I,(5,N))
sigma_D = np.reshape(sigma_D,(5,N))



##################################################################################

print("working till here")
all_params_age = sol_age["x"]

x_solution = DM(nx*age_groups,N)
xcurrent = x0

days = N
x_solution[:, 0] = x0
solution_params = sol_age["x"]
betas =  solution_params[0:size]

gamma = solution_params[size:size+age_groups]
mu = solution_params[size + age_groups:size+2*age_groups]
avg_betas = 0

beta_0 = np.zeros(N-1)

#####################################

beta_00 = []; beta_11 = []; beta_22 = [] ; beta_33 = [] ; beta_44 = []; beta_55 = []



for i in range(N-1):
  current_beta = betas[i*size_one :i*size_one + size_one]
  beta_mat = np.zeros((age_groups, age_groups))
  indices = np.tril_indices(age_groups)
  beta_mat[indices] = np.reshape(current_beta.full(),15)
  avg_betas = avg_betas + beta_mat

  xcurrent = one_sample_age(xcurrent, beta_mat, vertcat(gamma.T, mu.T))
  x_solution[:, i + 1] = xcurrent

  beta_00.append(np.diag(beta_mat)[0])
  beta_11.append(np.diag(beta_mat)[1])
  beta_22.append(np.diag(beta_mat)[2])
  beta_33.append(np.diag(beta_mat)[3])
  beta_44.append(np.diag(beta_mat)[4])

plt.figure(N)
tspan = np.arange(0, N-1, 1)

avg_betas = avg_betas / (N-1)
avg_betas = avg_betas + avg_betas.T - np.diag(np.diag(avg_betas))
max_abs = np.max(np.abs(avg_betas))


all_S = x_solution[0:age_groups,:]
all_I = x_solution[age_groups:2*age_groups,:]
all_R = x_solution[2*age_groups:3*age_groups,:]
all_D = x_solution[3*age_groups:4*age_groups,:]


all_I_ub = all_I.full() + np.sqrt(sigma_I)
all_I_lb = all_I.full() - np.sqrt(sigma_I)

all_D_ub = all_D.full() + np.sqrt(sigma_D)
all_D_lb = all_D.full() - np.sqrt(sigma_D)

bounds = vertcat(all_I_ub, all_I_lb, all_D_ub, all_D_lb)
bounds = bounds.full()
bounds = np.transpose(bounds)
pdb.set_trace()
sol_array = x_solution.toarray()
param_array = sol_age['x'].toarray()
param_array = np.transpose(param_array)

row = 2

os.chdir(os.getcwd() + '/simulation_results/coupling')

array = np.transpose(sol_array)
for col, data in enumerate(array):
    worksheet.write_column(row, col, data)

worksheet = workbook.add_worksheet()

# header_format = workbook.add_format({
#     'bold': True,
#     'text_wrap': True,
#     'valign': 'top',
#     'fg_color': '#D7E4BC',
#     'border': 1})

for col, data in enumerate(param_array):
    worksheet.write_column(row, col, data)


worksheet = workbook.add_worksheet()

for col, data in enumerate(bounds):
    worksheet.write_column(row, col, data)


workbook.close()


pdb.set_trace()

