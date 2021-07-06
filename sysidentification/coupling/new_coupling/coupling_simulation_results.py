from casadi import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import os
from plots_utils_coupling import *

month = "August"
start = 13 + 20
start = 13

N = 17 # max = 37 alpha_reg 1e9
N = 37
# N = 5
age_groups = 9

weeks_num = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]

if (start==13):
    print("yes here")
    weeks_name = ["March", "April", "April", "April", "April", "May", "May", "May", "May", "May", "June","June","June","June",
              "July", "July", "July", "July", "August", "August", "August", "August","August", "September" , "September"]
elif (start==13+20):
    print("yes here 2")
    weeks_name = [ "August", "August", "August", "August", "August", "September", "September"]

else:
    print("nothing")

nx = 4
nu = 3

pop_germany_total = 83905306

symbol = SX.sym


beta_low = symbol('beta',Sparsity.lower(age_groups))

gamma = symbol('gamma',(1,age_groups))
mu = symbol('mu',(1,age_groups))


params = vertcat(gamma, mu)
os.chdir('../')
os.chdir('../')

df1 = pd.read_excel(os.getcwd() + '/data/coupling/five_age_groups_active.xlsx')
df2 = pd.read_excel(os.getcwd() + '/data/coupling/five_age_groups_cumulative.xlsx')
df3 = pd.read_excel(os.getcwd() + '/data/coupling/five_age_groups_dead.xlsx')
df4 = pd.read_excel(os.getcwd() + '/data/coupling/age_groups_pop_germany_red.xlsx')

active = df1.iloc[27:27+age_groups, 25+start + 2*(start-13) :25+start + 2*(start-13) + 3*N:3]
active = active.to_numpy()

cumulative = df2.iloc[56:56+age_groups, start -1 : start-1 + N]
cumulative = cumulative.to_numpy()
cumulative_female = df2.iloc[91:91+5, start -1 : start-1 + N]
cumulative_female = cumulative_female.to_numpy()

cumulative_male = df2.iloc[103:103+5, start -1 : start-1 + N]
cumulative_male = cumulative_male.to_numpy()

cumulative_diverse = df2.iloc[116:116+5, start -1 : start-1 + N]
cumulative_diverse = cumulative_diverse.to_numpy()

dead = df3.iloc[8:8+age_groups, start-10 + 2*(start-13):start-10 + 2*(start-13) + 3*N:3]
dead = dead.to_numpy()
dead = dead.astype(int)

for i in range(np.shape(dead)[0]):
    for j in range(np.shape(dead)[1]-1):
        if (dead[i,j] > dead[i,j+1]):
            dead[i,j+1] = dead[i,j]

pop = df4.iloc[0:age_groups, 9]

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



nx = 4
nu = 3
symbol = SX.sym
beta_age = symbol('beta_age',(9,9))
gamma_age = symbol('gamma_age',(1,9))
mu_age = symbol('mu_age',(1,9))

u_ages = vertcat(beta_age,gamma_age,mu_age)
#


# begin week 13


# df1 = pd.read_excel(os.getcwd() + '/simulation_results/solution_casadi20_N.xlsx', 'Sheet1')
# df2 = pd.read_excel(os.getcwd() + '/simulation_results/solution_casadi20_N.xlsx', 'Sheet2', index_col=None)

# df1 = pd.read_excel(os.getcwd() + '/simulation_results/solution_casadi17_N.xlsx', 'Sheet1')
# df2 = pd.read_excel(os.getcwd() + '/simulation_results/solution_casadi17_N.xlsx', 'Sheet2', index_col=None)

df1 = pd.read_excel(os.getcwd() + '/simulation_results/coupling/solution_casadi.xlsx', 'Sheet1')
df2 = pd.read_excel(os.getcwd() + '/simulation_results/coupling/solution_casadi.xlsx', 'Sheet2', index_col=None)
df3 = pd.read_excel(os.getcwd() + '/simulation_results/coupling/solution_casadi.xlsx', 'Sheet3', index_col=None)

bounds = df3.iloc[0:4*age_groups,0:37]
bounds = bounds.to_numpy()

col_str_counts = np.sum(df1.applymap(lambda x: 1 if isinstance(x, str) else 0), axis=0)
N = np.shape(col_str_counts)[0]
x_solution = df1.iloc[1:37, 0:N]
x_solution = x_solution.to_numpy()

# params = df1.iloc[,0]
col_str_counts2 = np.sum(df2.applymap(lambda x: 1 if isinstance(x, str) else 0), axis=1)
size_param = np.shape(col_str_counts2)[0]

susceptible_sim = x_solution[0:age_groups,:]
active_sim = x_solution[age_groups:2*age_groups,:]
recovered_sim = x_solution[2*age_groups:3*age_groups,:]
dead_sim = x_solution[3*age_groups:4*age_groups,:]

for i in range(np.shape(x_solution)[0]):
    for j in range(np.shape(x_solution)[1]):
        if (x_solution[i][j] < 0):
            print("Found less than zero in solution",i,j, x_solution[i][j])

t = np.arange(0, N, 1)
plt.figure()
plot_active(t, active, age_groups, N)
plt.figure()
plot_active_sim(t, active_sim, age_groups, N, bounds)
plt.figure()
plot_active_sim_real(t, active_sim, active, age_groups, N, bounds)
# pdb.set_trace()
plt.figure()
plot_deaths_real(t, dead, age_groups, N)
plt.figure()
plot_deaths_sim(t, dead_sim, age_groups, N, bounds)
params = df2.iloc[0:size_param, 0]
# betas =  params[0:size]

contact_matrix = params[0:age_groups*age_groups]
contact_matrix = contact_matrix.to_numpy()
contact_matrix = np.reshape(contact_matrix,(age_groups,age_groups))+0.1

betas = params[age_groups*age_groups:age_groups*age_groups+N-1]
betas = betas.to_numpy()

plot_betas_matix(betas, contact_matrix, N, age_groups, start)


# plt.figure(3)
# plot_cumulative(cumulative_male[:,N-1], cumulative_female[:,N-1], cumulative_diverse[:,N-1])


# beta_00 = []; beta_11 = []; beta_22 = [] ; beta_33 = [] ; beta_44 = []; beta_55 = []
# plt.figure()
# plot_betas_diag(betas, N, size_one, age_groups, start)
# tspan = np.arange(0, N-1, 1)
# plt.figure()
# plot_betas_tv(tspan, betas, N, size_one, age_groups)

# for i in range(N-1):
#   current_beta = betas[i*size_one :i*size_one + size_one]
#   beta_mat = np.zeros((age_groups, age_groups))
#   indices = np.tril_indices(age_groups)
#   beta_mat[indices] = np.reshape(current_beta,15)
#   avg_betas = avg_betas + beta_mat
#
#
#   beta_t = beta_mat + beta_mat.T - np.diag(np.diag(beta_mat))
#   beta_00.append(np.diag(beta_mat)[0])
#   beta_11.append(np.diag(beta_mat)[1])
#   beta_22.append(np.diag(beta_mat)[2])
#   beta_33.append(np.diag(beta_mat)[3])
#   beta_44.append(np.diag(beta_mat)[4])
#   #
#   max_abs = np.max(np.abs(beta_t))
#   plt.figure(i)
#   plt.imshow(beta_t, cmap='RdBu', vmin=-max_abs, vmax=max_abs)
#   plt.colorbar()
alpha_reg =1e8
t2 = np.arange(0, N-1, 1)
plt.figure()
plot_betas(t2, betas/2, N)

plt.show()
pdb.set_trace()