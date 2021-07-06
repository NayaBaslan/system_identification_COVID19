from casadi import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sird_model import sird_model_params
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

datasets  = ["Stuttgart", "Böblingen", "Esslingen", "Göppingen", "Ludwigsburg", "Rems-Murr-Kreis", "Heilbronn", "Hohenlohekreis", "Main-Tauber-Kreis", "Schwäbisch Hall"]
ids       = ["8111", "8115", "8116", "8117", "8118", "8119", "8121", "8126", "8128", "8127"]

datasets  = ["Freiburg", "Emmendingen", "Ortenaukreis", "Rottweil", "Tuttlingen", "Konstanz", "Waldshut", "Lörrach", "Breisgauhochscwarzwald", "Schwarzwaldbaarkreis"]
ids       = ["8311", "8316", "8317", "8325", "8327", "8335", "8337", "8336", "8315", "8326"]

populations = [635911, 392807, 535024, 258145, 545423, 427248 , 126592, 112655, 132399, 196761]

populations = [231195, 166408, 430953, 139878, 140766, 286305, 171003, 228736, 263601, 212506 ]
dataset = "Hohenlohekreis"

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
for i in range(np.shape(all_columns)[0]):
    if (all_columns[i] == ids[0]):
        print("found Stuttgart")
        cumulative_infected_stuttgart = df.iloc[:, i]
        deaths_stuttgart = df1.iloc[:, i]
    if (all_columns[i] == ids[1]):
        print("found Böblingen")
        cumulative_infected_boblingen = df.iloc[:, i]
        deaths_boblingen = df1.iloc[:, i]
    if (all_columns[i] == ids[2]):
        print("found Esslingen")
        cumulative_infected_esslingen = df.iloc[:, i]
        deaths_esslingen = df1.iloc[:, i]
    if (all_columns[i] == ids[3]):
        print("found Göppingen")
        cumulative_infected_goppingen = df.iloc[:, i]
        deaths_goppingen = df1.iloc[:, i]
    if (all_columns[i] == ids[4]):
        print("found Ludwigsburg")
        cumulative_infected_ludwigsburg = df.iloc[:, i]
        deaths_ludwigsburg = df1.iloc[:, i]
    if (all_columns[i] == ids[5]):
        print("found Rems-Murr-Kreis")
        cumulative_infected_remsmurrkreis = df.iloc[:, i]
        deaths_remsmurrkreis = df1.iloc[:, i]
    if (all_columns[i] == ids[6]):
        print("found Heilbronn")
        cumulative_infected_heilbronn = df.iloc[:, i]
        deaths_heilbronn = df1.iloc[:, i]
    if (all_columns[i] == ids[7]):
        print("found Hohenlohekreis")
        cumulative_infected_hohenlohekreis = df.iloc[:, i]
        deaths_hohenlohekreis = df1.iloc[:, i]
    if (all_columns[i] == ids[8]):
        print("found Main-Tauber-Kreis")
        cumulative_infected_maintauberkreis = df.iloc[:, i]
        deaths_maintauberkreis = df1.iloc[:, i]
    if (all_columns[i] == ids[9]):
        print("found Schwäbisch Hall")
        cumulative_infected_schwaebisch = df.iloc[:, i]
        deaths_schwaebisch = df1.iloc[:, i]


N_all = len(cumulative_infected)
cumulative_infected = cumulative_infected.to_numpy()
deaths = deaths.to_numpy()
deaths = deaths[15:len(deaths)]   # shape 379

new_infections = np.zeros(N_all)
new_infections_stuttgart = np.zeros(N_all)
new_infections_boblingen = np.zeros(N_all)
new_infections_esslingen = np.zeros(N_all)
new_infections_goppingen = np.zeros(N_all)
new_infections_ludwigsburg = np.zeros(N_all)
new_infections_remmsmurkreis = np.zeros(N_all)
new_infections_heilbronn = np.zeros(N_all)
new_infections_hohenlohekreis = np.zeros(N_all)
new_infections_maintauberkreis = np.zeros(N_all)
new_infections_schwaebisch = np.zeros(N_all)

new_deaths = np.zeros(len(deaths))
new_deaths_stuttgart = np.zeros(len(deaths))
new_deaths_boblingen = np.zeros(len(deaths))
new_deaths_esslingen = np.zeros(len(deaths))
new_deaths_goppingen = np.zeros(len(deaths))
new_deaths_ludwigsburg = np.zeros(len(deaths))
new_deaths_remmsmurkreis = np.zeros(len(deaths))
new_deaths_heilbronn = np.zeros(len(deaths))
new_deaths_hohenlohekreis = np.zeros(len(deaths))
new_deaths_maintauberkreis = np.zeros(len(deaths))
new_deaths_schwaebisch = np.zeros(len(deaths))

# x_data = pd.date_range('2018-04-01', periods=5, freq='MS')
#
for i in range(N_all-1):
    new_infections[i] = cumulative_infected[i+1] - cumulative_infected[i]
    new_infections_stuttgart[i] = cumulative_infected_stuttgart[i + 1] - cumulative_infected_stuttgart[i]
    new_infections_boblingen[i] = cumulative_infected_boblingen[i + 1] - cumulative_infected_boblingen[i]
    new_infections_esslingen[i] = cumulative_infected_esslingen[i + 1] - cumulative_infected_esslingen[i]
    new_infections_goppingen[i] = cumulative_infected_goppingen[i + 1] - cumulative_infected_goppingen[i]
    new_infections_ludwigsburg[i] = cumulative_infected_ludwigsburg[i + 1] - cumulative_infected_ludwigsburg[i]
    new_infections_remmsmurkreis[i] = cumulative_infected_remsmurrkreis[i + 1] - cumulative_infected_remsmurrkreis[i]
    new_infections_heilbronn[i] = cumulative_infected_heilbronn[i+ 1] - cumulative_infected_heilbronn[i]
    new_infections_hohenlohekreis[i] = cumulative_infected_hohenlohekreis[i+ 1] - cumulative_infected_hohenlohekreis[i]
    new_infections_maintauberkreis[i] = cumulative_infected_maintauberkreis[i+ 1] - cumulative_infected_maintauberkreis[i]
    new_infections_schwaebisch[i] = cumulative_infected_schwaebisch[i+ 1] - cumulative_infected_schwaebisch[i]

for i in range(len(deaths)-1):
    new_deaths[i] = deaths[i+1] - deaths[i]
    new_deaths_stuttgart[i] = deaths_stuttgart[i+1] - deaths_stuttgart[i]
    new_deaths_boblingen[i] = deaths_boblingen[i+1] - deaths_boblingen[i]
    new_deaths_esslingen[i] = deaths_esslingen[i+1] - deaths_esslingen[i]
    new_deaths_goppingen[i] = deaths_goppingen[i+1] - deaths_goppingen[i]
    new_deaths_ludwigsburg[i] = deaths_ludwigsburg[i+1] - deaths_ludwigsburg[i]
    new_deaths_remmsmurkreis[i] = deaths_remsmurrkreis[i+1] - deaths_remsmurrkreis[i]
    new_deaths_heilbronn[i] = deaths_heilbronn[i+1] - deaths_heilbronn[i]
    new_deaths_hohenlohekreis[i] = deaths_hohenlohekreis[i+1] - deaths_hohenlohekreis[i]
    new_deaths_maintauberkreis[i] = deaths_maintauberkreis[i+1] - deaths_maintauberkreis[i]
    new_deaths_schwaebisch[i] = deaths_schwaebisch[i+1] - deaths_schwaebisch[i]


active_infections = np.zeros(N_all - 15)
active_infections_stuttgart = np.zeros(N_all - 15)
active_infections_boblingen = np.zeros(N_all - 15)
active_infections_esslingen = np.zeros(N_all - 15)
active_infections_goppingen = np.zeros(N_all - 15)
active_infections_ludwigsburg = np.zeros(N_all - 15)
active_infections_remmsmurkreis = np.zeros(N_all - 15)
active_infections_heilbronn = np.zeros(N_all - 15)
active_infections_hohenlohekreis = np.zeros(N_all - 15)
active_infections_maintauberkreis = np.zeros(N_all - 15)
active_infections_schwaebisch = np.zeros(N_all - 15)



real_active = np.zeros(N_all - 15)
for i in range(len(active_infections)):
    active_infections[i] = np.sum(new_infections[i:i+15])
    active_infections_stuttgart[i] = np.sum(new_infections_stuttgart[i:i + 15])
    active_infections_boblingen[i] = np.sum(new_infections_boblingen[i:i + 15])
    active_infections_esslingen[i] = np.sum(new_infections_esslingen[i:i + 15])
    active_infections_goppingen[i] = np.sum(new_infections_goppingen[i:i + 15])
    active_infections_ludwigsburg[i] = np.sum(new_infections_ludwigsburg[i:i + 15])
    active_infections_remmsmurkreis[i] = np.sum(new_infections_remmsmurkreis[i:i + 15])
    active_infections_heilbronn[i] = np.sum(new_infections_heilbronn[i:i + 15])
    active_infections_hohenlohekreis[i] = np.sum(new_infections_hohenlohekreis[i:i + 15])
    active_infections_maintauberkreis[i] = np.sum(new_infections_maintauberkreis[i:i + 15])
    active_infections_schwaebisch[i] = np.sum(new_infections_schwaebisch[i:i + 15])

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
N1 = 330
start = 17 # = 18.03.2020 with N=330 10.02.2021
# 43 wearing masks 61 shops reopen 105 kindergartens and schools reopen 217 wearing masks in city center 256 wearing masks in workplace
# 267 lockdown + curfew 273 closing all non-essential shops

####### 16 = April 02 2021 first wave   (33)
####### 230 = November 02 2021 second wave (247)

##### For first wave:  start = 15 N1 = 150
##### For second wave: start = 230 N1 = 100
##### For first and second wave: start = 16 N1 = 330

active = active_infections[start:start+N1]
active_stuttgart = active_infections_stuttgart[start:start+N1]
active_boblingen = active_infections_boblingen[start:start+N1]
active_esslingen = active_infections_esslingen[start:start+N1]
active_goppingen = active_infections_goppingen[start:start+N1]
active_ludwigsburg = active_infections_ludwigsburg[start:start+N1]
active_remmsmurkreis = active_infections_remmsmurkreis[start:start+N1]
active_heilbronn = active_infections_heilbronn[start:start+N1]
active_hohenlohekreis = active_infections_hohenlohekreis[start:start+N1]
active_maintauberkreis = active_infections_maintauberkreis[start:start+N1]
active_schwaebisch = active_infections_schwaebisch[start:start+N1]


dead = deaths[start:start+N1]
recovered = recovered[start:start+N1]
susceptible = susceptible[start:start+N1]

cumulated = cumulative_infected[start:start+N1]

data1 = np.stack((susceptible , active, recovered, dead), axis=-1)
data = data1.astype(int)   # shape (193,2)



# Define the model: states, controls, and parameters
########################################################################################
nx = 4
nu = 1

tspan = np.arange(0, N1, 1)

# plot_active_stuttgart(tspan, active_stuttgart, active_boblingen, active_esslingen , active_goppingen ,active_ludwigsburg ,active_remmsmurkreis , active_heilbronn ,active_hohenlohekreis,active_maintauberkreis,  active_schwaebisch,N1)

plot_active_freiburg(tspan, active_stuttgart, active_boblingen, active_esslingen , active_goppingen ,active_ludwigsburg ,active_remmsmurkreis , active_heilbronn ,active_hohenlohekreis,active_maintauberkreis,  active_schwaebisch,N1)

weeks = 47
active_weeks_stuttgart = np.zeros(weeks)
active_weeks_boblingen = np.zeros(weeks)
active_weeks_esslingen = np.zeros(weeks)
active_weeks_goppingen = np.zeros(weeks)
active_weeks_ludwigsburg = np.zeros(weeks)
active_weeks_remsmurrkreis = np.zeros(weeks)
active_weeks_heilbronn = np.zeros(weeks)
active_weeks_hohenlogekreis = np.zeros(weeks)
active_weeks_mainrauberkreis = np.zeros(weeks)
active_weeks_schwaebish = np.zeros(weeks)


for i in range(weeks):
    active_weeks_stuttgart[i] = active_stuttgart[i*7]
    active_weeks_boblingen[i] = active_boblingen[i*7]
    active_weeks_esslingen[i] = active_esslingen[i*7]
    active_weeks_goppingen[i] = active_goppingen[i*7]
    active_weeks_ludwigsburg[i] = active_ludwigsburg[i*7]
    active_weeks_remsmurrkreis[i] = active_remmsmurkreis[i*7]
    active_weeks_heilbronn[i] = active_heilbronn[i*7]
    active_weeks_hohenlogekreis[i] = active_hohenlohekreis[i*7]
    active_weeks_mainrauberkreis[i] = active_maintauberkreis[i*7]
    active_weeks_schwaebish[i] = active_schwaebisch[i*7]

t_weeks = np.arange(0, weeks, 1)
# plot_weekly_active_stuttgart(t_weeks, active_weeks_stuttgart, active_weeks_boblingen, active_weeks_esslingen, active_weeks_goppingen, active_weeks_ludwigsburg, active_weeks_remsmurrkreis, active_weeks_heilbronn, active_weeks_hohenlogekreis, active_weeks_mainrauberkreis, active_weeks_schwaebish, weeks)

active_weeks_freiburg = np.zeros(weeks)
active_weeks_emmendingen = np.zeros(weeks)
active_weeks_ortenaukreis = np.zeros(weeks)
active_weeks_rottweil = np.zeros(weeks)
active_weeks_tuttlingen = np.zeros(weeks)
active_weeks_konstanz = np.zeros(weeks)
active_weeks_waldshut = np.zeros(weeks)
active_weeks_lorrach = np.zeros(weeks)
active_weeks_breisgau = np.zeros(weeks)
active_weeks_schwarz = np.zeros(weeks)


for i in range(weeks):
    active_weeks_freiburg[i] = active_stuttgart[i*7]
    active_weeks_emmendingen[i] = active_boblingen[i*7]
    active_weeks_ortenaukreis[i] = active_esslingen[i*7]
    active_weeks_rottweil[i] = active_goppingen[i*7]
    active_weeks_tuttlingen[i] = active_ludwigsburg[i*7]
    active_weeks_konstanz[i] = active_remmsmurkreis[i*7]
    active_weeks_waldshut[i] = active_heilbronn[i*7]
    active_weeks_lorrach[i] = active_hohenlohekreis[i*7]
    active_weeks_breisgau[i] = active_maintauberkreis[i*7]
    active_weeks_schwarz[i] = active_schwaebisch[i*7]
plt.figure()
plot_weekly_active_freiburg(t_weeks, active_weeks_freiburg, active_weeks_emmendingen, active_weeks_ortenaukreis, active_weeks_rottweil, active_weeks_tuttlingen, active_weeks_konstanz, active_weeks_waldshut, active_weeks_lorrach, active_weeks_breisgau, active_weeks_schwarz, weeks)
pdb.set_trace()
plt.show()