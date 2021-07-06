from casadi import *
import pdb

def sird_model_beta(x, u, sigma, mu):
  # TODO: use beta, gamma, instead of u[i]
  S = x[0] ; I = x[1] ; R = x[2] ; D = x[3]
  beta = u
  N_pop = S + I + R + D
  print("The population is: ", N_pop)
  dSdt = -beta * S * I / N_pop
  dIdt = beta * S * I / N_pop - sigma * I - mu * I
  dRdt = sigma * I
  dDdt = mu * I
  return vertcat(dSdt, dIdt, dRdt, dDdt)


def sird_model(x, u):
  # TODO: use beta, gamma, instead of u[i]
  S = x[0] ; I = x[1] ; R = x[2] ; D = x[3]
  beta = u[0] ; gamma = u[1] ; mu = u[2]
  N_pop = S + I + R + D
  print("The population is: ", N_pop)
  dSdt = -beta * S * I / N_pop
  dIdt = beta * S * I / N_pop - gamma * I - mu * I
  dRdt = gamma * I
  dDdt = mu * I
  return vertcat(dSdt, dIdt, dRdt, dDdt)

def sird_model_params(x, u, p):
  S = x[0] ; I = x[1] ; R = x[2] ; D = x[3]
  beta = u ; gamma = p[0] ; mu = p[1]
  N_pop = S + I + R + D
  dSdt = -beta * S * I / N_pop
  dIdt = beta * S * I / N_pop - gamma * I - mu * I
  dRdt = gamma * I
  dDdt = mu * I
  return [dSdt, dIdt, dRdt, dDdt]