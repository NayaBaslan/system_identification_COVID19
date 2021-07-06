from casadi import *

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


def sird_model_age(x, u):
  # TODO: use beta, gamma, instead of u[i]
  # S = x[0] ; I = x[1] ; R = x[2] ; D = x[3]

  S_0 = x[0][0]; I_0 = x[9][0]; R_0 = x[18][0]; D_0 = x[27][0];
  beta_0 = u[0:9,0];  gamma_0 = u[9,0];  mu_0 = u[10,0]
  Is = x[9:18, 0]
  N_pop0 = S_0 + I_0 + R_0 + D_0
  prod = S_0*beta_0*Is
  S_sum2 = vertsplit(prod,1)
  S_sum = cumsum(S_0*beta_0*Is)
  ## try sum()
  # TODO try for loop, try sum , use vertsplit

  dS0dt = - S_sum[8]/N_pop0  #-  gamma_0*Is[0] - mu_0*Is[0]
  dI0dt =  S_sum[8]/N_pop0 -  gamma_0*Is[0] - mu_0*Is[0]
  dR0dt = gamma_0*Is[0]
  dD0dt = mu_0*Is[0]

  S_1 = x[1][0]; I_1 = x[10][0]; R_1 = x[19][0]; D_1 = x[28][0];
  beta_1 = u[0:9,1];  gamma_1 = u[9,1];  mu_1 = u[10,1]
  Is = x[9:18, 0]
  N_pop1 = S_1 + I_1 + R_1 + D_1
  S_sum = cumsum(S_1 * beta_1 * Is)
  dS1dt = - S_sum[8] / N_pop1  # -  gamma_0*Is[0] - mu_0*Is[0]
  dI1dt = S_sum[8]/N_pop1 - gamma_1 * Is[1] - mu_1 * Is[1]
  dR1dt = gamma_1*Is[1]
  dD1dt = mu_1*Is[1]

  S_2 = x[2][0];  I_2 = x[11][0];  R_2 = x[20][0];  D_2 = x[29][0];
  beta_2 = u[0:9, 2];
  gamma_2 = u[9, 2];
  mu_2 = u[10, 2]
  Is = x[9:18, 0]
  N_pop2 = S_2 + I_2 + R_2 + D_2
  S_sum = cumsum(S_2 * beta_2 * Is)
  dS2dt = - S_sum[8] / N_pop2  # -  gamma_0*Is[0] - mu_0*Is[0]
  dI2dt = S_sum[8]/N_pop2 - gamma_2 * Is[2] - mu_2 * Is[2]
  dR2dt = gamma_2 * Is[2]
  dD2dt = mu_2 * Is[2]

  S_3 = x[3][0];  I_3 = x[12][0];  R_3 = x[21][0];  D_3 = x[30][0];
  beta_3 = u[0:9, 3];
  gamma_3 = u[9, 3];
  mu_3 = u[10, 3]
  Is = x[9:18, 0]
  N_pop3 = S_3 + I_3 + R_3 + D_3
  S_sum = cumsum(S_3 * beta_3 * Is)
  dS3dt = - S_sum[8] / N_pop3  # -  gamma_0*Is[0] - mu_0*Is[0]
  dI3dt = S_sum[8]/N_pop3 - gamma_3 * Is[3] - mu_3 * Is[3]
  dR3dt = gamma_3 * Is[3]
  dD3dt = mu_3 * Is[3]

  S_4 = x[4][0];  I_4 = x[13][0];  R_4 = x[22][0];  D_4 = x[31][0];
  beta_4 = u[0:9, 4];
  gamma_4 = u[9, 4];
  mu_4 = u[10, 4]
  Is = x[9:18, 0]
  N_pop4 = S_4 + I_4 + R_4 + D_4
  S_sum = cumsum(S_4 * beta_4 * Is)
  dS4dt = - S_sum[8] / N_pop4  # -  gamma_0*Is[0] - mu_0*Is[0]
  dI4dt = S_sum[8]/N_pop4 - gamma_4 * Is[4] - mu_4 * Is[4]
  dR4dt = gamma_4 * Is[4]
  dD4dt = mu_4 * Is[4]

  S_5 = x[5][0];  I_5 = x[14][0];  R_5 = x[23][0];  D_5 = x[32][0];
  beta_5 = u[0:9, 5];
  gamma_5 = u[9, 5];
  mu_5 = u[10, 5]
  Is = x[9:18, 0]
  N_pop5 = S_5 + I_5 + R_5 + D_5

  S_sum = cumsum(S_5 * beta_5 * Is)
  dS5dt = - S_sum[8] / N_pop5  # -  gamma_0*Is[0] - mu_0*Is[0]
  dI5dt = S_sum[8]/N_pop5 - gamma_5 * Is[5] - mu_5 * Is[5]
  dR5dt = gamma_5 * Is[5]
  dD5dt = mu_5 * Is[5]

  S_6 = x[6][0];  I_6 = x[15][0];  R_6 = x[24][0];  D_6 = x[33][0];
  beta_6 = u[0:9, 6];
  gamma_6 = u[9, 6];
  mu_6 = u[10, 6]
  Is = x[9:18, 0]
  N_pop6 = S_6 + I_6 + R_6 + D_6
  S_sum = cumsum(S_6 * beta_6 * Is)
  dS6dt = - S_sum[8] / N_pop6  # -  gamma_0*Is[0] - mu_0*Is[0]
  dI6dt = S_sum[8]/N_pop6 - gamma_6 * Is[6] - mu_6 * Is[6]
  dR6dt = gamma_6 * Is[6]
  dD6dt = mu_6 * Is[6]

  S_7 = x[7][0];  I_7 = x[16][0];  R_7 = x[25][0];  D_7 = x[34][0];
  beta_7 = u[0:9, 7];
  gamma_7 = u[9, 7];
  mu_7 = u[10, 7]
  Is = x[9:18, 0]
  N_pop7 = S_7 + I_7 + R_7 + D_7
  S_sum = cumsum(S_7 * beta_7 * Is)
  dS7dt = - S_sum[8] / N_pop7  # -  gamma_0*Is[0] - mu_0*Is[0]
  dI7dt = S_sum[8]/N_pop7 - gamma_7 * Is[7] - mu_7 * Is[7]
  dR7dt = gamma_7 * Is[7]
  dD7dt = mu_7 * Is[7]

  S_8 = x[8][0];  I_8 = x[17][0];  R_8 = x[26][0];  D_8 = x[35][0];
  beta_8 = u[0:9, 8];
  gamma_8 = u[9, 8];
  mu_8 = u[10, 8]
  Is = x[9:18, 0]
  N_pop8 = S_8 + I_8 + R_8 + D_8
  S_sum = cumsum(S_8 * beta_8 * Is)
  dS8dt = - S_sum[8] / N_pop8  # -  gamma_0*Is[0] - mu_0*Is[0]
  dI8dt = S_sum[8] /N_pop8 - gamma_8 * Is[8] - mu_8 * Is[8]
  dR8dt = gamma_8 * Is[8]
  dD8dt = mu_8 * Is[8]

  dSdt = vertcat(dS0dt, dS1dt, dS2dt, dS3dt, dS4dt, dS5dt, dS6dt, dS7dt, dS8dt)
  dIdt = vertcat(dI0dt, dI1dt, dI2dt, dI3dt, dI4dt, dI5dt, dI6dt, dI7dt, dI8dt)
  dRdt = vertcat(dR0dt, dR1dt, dR2dt, dR3dt, dR4dt, dR5dt, dR6dt, dR7dt, dR8dt)
  dDdt = vertcat(dD0dt, dD1dt, dD2dt, dD3dt, dD4dt, dD5dt, dD6dt, dD7dt, dD8dt)

  # pdb.set_trace()
  return [dSdt, dIdt, dRdt, dDdt]