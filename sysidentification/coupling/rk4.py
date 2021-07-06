from casadi import *
import pdb

def rk4_age(ode_age, states_ages, controls_age, params,  dt):
    k1_age = ode_age(states_ages, controls_age,params)
    k2_age = ode_age(states_ages + dt / 2.0 * k1_age, controls_age, params)
    k3_age = ode_age(states_ages + dt / 2.0 * k2_age, controls_age, params)
    k4_age = ode_age(states_ages + dt * k3_age, controls_age, params)

    states_final_age = states_ages + dt / 6.0 * (k1_age + 2 * k2_age + 2 * k3_age + k4_age)

    return states_final_age