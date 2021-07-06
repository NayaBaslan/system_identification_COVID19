from casadi import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sird_model import sird_model_params
import os
import pdb

import xlsxwriter
workbook = xlsxwriter.Workbook('solution_casadi.xlsx')
worksheet = workbook.add_worksheet()

print(os.getcwd())
os.chdir('../')
print(os.getcwd())

# Source: https://github.com/jgehrcke/covid-19-germany-gae
dataset = "freiburg"
df = pd.read_excel(os.getcwd() + '/data/cases-rki-by-ags.xlsx')
df1 = pd.read_excel(os.getcwd() + '/data/deaths-rki-by-ags.xlsx')
# cumulative_infected = df.iloc[:, 191]

# Preprocessing
########################################################################################

if (dataset == "freiburg"):
    cumulative_infected = df.iloc[:, 190]
    deaths = df1.iloc[:, 190]
    pop = 263601

N_all = len(cumulative_infected)
cumulative_infected = cumulative_infected.to_numpy()
deaths = deaths.to_numpy()
deaths = deaths[15:len(deaths)]   # shape 379

new_infections = np.zeros(N_all)
new_deaths = np.zeros(len(deaths))

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



########################################################################################

# Phase 1
########################################################################################
# Phase 1 : 01.04.2020 - 30.06.2020 (First Wave)
# first day = 01.04.2020 cumulative_infected[30]

# N1 = 91
N1 = 150
active = active_infections[16:16+N1]
dead = deaths[16:16+N1]
recovered = recovered[16:16+N1]
susceptible = susceptible[16:16+N1]


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



alpha_reg = 1e6
reg = symbol('reg', N1-2)
for i in range(np.shape(beta_time_var)[0]-1):
    reg[i] = beta_time_var[i+1] - beta_time_var[i]

u_test = vertcat(beta_time_var,params)
nlp = {'x':u_test, 'f':dot(model_mismatch, model_mismatch) + alpha_reg*dot(reg,reg) ,'g':g}


solver = nlpsol("solver", "ipopt", nlp)   # DM(1.95599)
# solver2 = nlpsol("solver", "ipopt", nlp2)   # DM(1.95599)


u_guess = np.ones(np.shape(u_test))
sol = solver(x0=u_guess,lbg=lbg, ubg=ubg)
# sol2 = solver2(x0=u_guess,lbg=lbg, ubg=ubg)

###### Covariance Estimation
hessLag = solver.get_function('nlp_hess_l')
sol['lam_g']
hessLag_sol = hessLag(sol['x'], [] , 1,sol['lam_g'])

pdb.set_trace()

hessLag_sol = hessLag_sol + hessLag_sol.T - np.diag(np.diag(hessLag_sol))
hess_sparsity = hessLag.sparsity_out(0)

hess_sparsity.is_symmetric()
cov = inv(hessLag_sol)

sparsity_matrix = DM(hess_sparsity).full()

hessLag_inv = inv(hessLag_sol)
sparsity_matrix = sparsity_matrix + sparsity_matrix.T - np.diag(np.diag(sparsity_matrix))

plt.figure(80)
max_abs = np.max(np.abs(sparsity_matrix))
plt.imshow(sparsity_matrix,cmap='RdBu',vmin = -max_abs, vmax = max_abs)
plt.colorbar()



sigma_theta = sol['f'] * hessLag_inv / (4*N1 - (N1+1))
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

sigmas_dummy_values = np.ones(N1)*10
sigma_y = []

for i in range(N1):
    # sigma_y = [sigma_y, J1_sol[i,:] @ hessLag_sol @ J1_sol[i,:].T]
    sigma_y.append(J1_sol[i,:] @ sigma_theta @ J1_sol[i,:].T)

sigma_y = np.sqrt(sigma_y)

all_params = sol['x']
betas = all_params[0:N1-1]
sigma = all_params[N1-1]
mu = all_params[N1]



# Get the state estimates using NLP Solution
#############################################################################
x_solution = DM(nx,N1)
xcurrent = x0
days = N1
x_solution[:, 0] = x0


for i in range(N1-1):
  xcurrent = one_sample(xcurrent, betas[nu*i: (i+1)*nu], vertcat(sigma,mu))
  x_solution[:, i+1] = xcurrent


all_S = np.reshape(x_solution[0,:],(-1,1))
all_I = np.reshape(x_solution[1,:],(-1,1))
all_R = np.reshape(x_solution[2,:],(-1,1))
all_D = np.reshape(x_solution[3,:],(-1,1))


plt.figure(70)
# max_abs = np.max(np.abs(np.log(sigma_theta.full())))
# plt.imshow(np.log(sigma_theta),cmap='RdBu',vmin = -max_abs, vmax = max_abs)

max_abs = np.max(np.abs(sigma_theta.full()))
plt.imshow(sigma_theta,cmap='RdBu',vmin = -max_abs, vmax = max_abs)

plt.colorbar()

################################
# With Covariance Estimates
sigmas_dummy_values = np.reshape(sigmas_dummy_values,(-1,1))
sigma_y = np.reshape(sigma_y,(-1,1))
sigmas_dummy_values = np.reshape(sigmas_dummy_values,(-1,1))
all_S_ub = all_S + sigma_y
all_S_lb = all_S - sigma_y

all_S_ub_dummy = all_S + sigmas_dummy_values
all_S_lb_dummy = all_S - sigmas_dummy_values

################################
tspan = np.arange(0, N1, 1)
tspan_controls = np.arange(0, N1-1, 1)

plt.figure(1)
plt.step(tspan_controls,betas,'-.')
plt.title("Transmission Rate - Time Varying Parameter")

plt.figure(2)
plt.title("Susceptible")
plt.plot(tspan, all_S, 'k-')

all_S_lb = np.reshape(all_S_lb,(N1))
all_S_ub = np.reshape(all_S_ub,(N1))
plt.fill_between(tspan, all_S_lb, all_S_ub)
plt.plot(tspan, susceptible)
plt.legend(["Model Data Susceptible","Real Data Susceptible"])

plt.figure(28)
plt.title("Susceptible")
plt.plot(tspan, all_S, 'k-')

all_S_lb_dummy = np.reshape(all_S_lb_dummy,(N1))
all_S_ub_dummy = np.reshape(all_S_ub_dummy,(N1))
plt.fill_between(tspan, all_S_lb_dummy, all_S_ub_dummy)
plt.plot(tspan, susceptible)
plt.legend(["Model Data Susceptible","Real Data Susceptible"])

plt.figure(30)
plt.title("Susceptible")
plt.plot(tspan, all_S, 'k-')
plt.plot(tspan, susceptible)
plt.legend(["Model Data Susceptible","Real Data Susceptible"])


plt.figure(3)
plt.title("Infected")
plt.plot(tspan, all_I)
plt.plot(tspan, active)
plt.legend(["Model Data Infected (Active Cases)","Real Data Infected (Active Cases)"])

plt.figure(4)
plt.title("Recovered")
plt.plot(tspan, all_R)
plt.plot(tspan, recovered)
plt.legend(["Model Data Recovered","Real Data Recovered"])

plt.figure(5)
plt.title("Dead")
plt.plot(tspan, all_D)
plt.plot(tspan, dead)
plt.legend(["Model Data Dead","Real Data Dead"])

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

plt.figure(7)
plt.plot(tspan,error_S/pop*100)
plt.plot(tspan,error_I/pop*100)
plt.plot(tspan,error_R/pop*100)
plt.plot(tspan,error_D/pop*100)
plt.title("Absolute Error (% of Total Population of Germany)")
plt.legend(["Error Susceptible ","Error Infected","Error Recovered","Error Dead"], loc ="upper right")

plt.figure(8)
plt.title("Residuals Susceptible")
plt.step(tspan,res_S,'-.')
plt.figure(9)
plt.title("Residuals Infected")
plt.step(tspan,res_I,'-.')
plt.figure(10)
plt.title("Residuals Recovered")
plt.step(tspan,res_R,'-.')
plt.figure(11)
plt.title("Residuals Dead")
plt.step(tspan,res_D,'-.')
# plt.legend(["Residual Susceptible ","Residual Infected","Residual Recovered","Residual Dead"], loc ="upper right")

pdb.set_trace()
plt.show()
