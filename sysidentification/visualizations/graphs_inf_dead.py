from casadi import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pdb
from plots_utils import *

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
populations = [231195, 166408, 430953, 139878, 140766, 286305, 171003, 228736, 263601, 212506 ]

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

new_inf = new_infections[15:len(deaths)]
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
N1 = 330
start = 13
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

new_infections = new_infections[start:start+N1]
new_deaths = new_deaths[start:start+N1]
weeks = 47
new_inf_week = []
new_death_week = []

for i in range(weeks):
    new_inf_week.append(np.sum(new_infections[i*7:i*7+7]))
    new_death_week.append(np.sum(new_deaths[i*7:i*7+7]))

weekly_inf = np.zeros(len(new_inf_week))
weekly_dead = np.zeros(len(new_death_week))
for i in range(weeks):
    weekly_inf[i] = new_inf_week[i]
    weekly_dead[i] = new_death_week[i]
data1 = np.stack((susceptible , active, recovered, dead), axis=-1)
data = data1.astype(int)   # shape (193,2)

data = data.T
plot_new_infections(weeks, weekly_inf, dataset)
plot_new_deaths(weeks, weekly_dead, dataset)

pdb.set_trace()
plt.show()
pdb.set_trace()