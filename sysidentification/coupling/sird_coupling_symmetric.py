from casadi import *
import pdb

def sird_model_sym_test(x, u, p):

  dSdt = [] ;  dIdt = [];  dRdt = []; dDdt = []
  u = u + u.T - diag(diag(u))
  for i in range(9):
    S = x[0+i]; I = x[i+9]; R = x[i+18]; D = x[i+27];
    beta = u[0:9, i]; gamma = p[0, i]; mu = p[1, i]
    Is = x[9:18, 0]
    N_pop0 = S + I + R + D
    prod = S*beta*Is
    S_sum = sum(vertsplit(prod,1))
    dSdt_i = - S_sum / N_pop0  # -  gamma_0*Is[0] - mu_0*Is[0]
    dIdt_i = S_sum / N_pop0 - gamma * Is[i] - mu * Is[i]
    dRdt_i = gamma * Is[i]
    dDdt_i = mu * Is[i]

    dSdt = vertcat(dSdt, dSdt_i)
    dIdt = vertcat(dIdt, dIdt_i)
    dRdt = vertcat(dRdt, dRdt_i)
    dDdt = vertcat(dDdt, dDdt_i)

  return [dSdt, dIdt, dRdt, dDdt]

def sird_model_sym(x, u, p):

  dSdt = [] ;  dIdt = [];  dRdt = []; dDdt = []
  u = u + u.T - diag(diag(u))
  for i in range(9):
    S = x[0+i]; I = x[i+9]; R = x[i+18]; D = x[i+27];
    beta = u[0:9, i]; gamma = p[0, i]; mu = p[1, i]
    Is = x[9:18, 0]
    N_pop0 = S + I + R + D
    prod = S*beta*Is
    S_sum = sum(vertsplit(prod,1))
    dSdt_i = - S_sum / N_pop0  # -  gamma_0*Is[0] - mu_0*Is[0]
    dIdt_i = S_sum / N_pop0 - gamma * Is[i] - mu * Is[i]
    dRdt_i = gamma * Is[i]
    dDdt_i = mu * Is[i]

    dSdt = vertcat(dSdt, dSdt_i)
    dIdt = vertcat(dIdt, dIdt_i)
    dRdt = vertcat(dRdt, dRdt_i)
    dDdt = vertcat(dDdt, dDdt_i)

  return [dSdt, dIdt, dRdt, dDdt]


def sird_model_sym_red(x, u, p):

  dSdt = [] ;  dIdt = [];  dRdt = []; dDdt = []
  u = u + u.T - diag(diag(u))
  for i in range(5):
    S = x[0+i]; I = x[i+5]; R = x[i+10]; D = x[i+15];
    beta = u[0:5, i]; gamma = p[0, i]; mu = p[1, i]

    Is = x[5:10, 0]
    N_pop0 = S + I + R + D
    prod = S*beta*Is
    S_sum = sum(vertsplit(prod,1))
    dSdt_i = - S_sum / N_pop0  # -  gamma_0*Is[0] - mu_0*Is[0]
    dIdt_i = S_sum / N_pop0 - gamma * Is[i] - mu * Is[i]
    dRdt_i = gamma * Is[i]
    dDdt_i = mu * Is[i]

    dSdt = vertcat(dSdt, dSdt_i)
    dIdt = vertcat(dIdt, dIdt_i)
    dRdt = vertcat(dRdt, dRdt_i)
    dDdt = vertcat(dDdt, dDdt_i)

  return [dSdt, dIdt, dRdt, dDdt]