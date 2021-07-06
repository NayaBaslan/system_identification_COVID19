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
alpha_reg = 1e5
alpha_reg = 1e8

age_groups = 9
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

beta = symbol('beta',(9,9))
gamma = symbol('gamma',(1,age_groups))
mu = symbol('mu',(1,age_groups))
beta_tv = symbol('beta_tv',1)

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

pdb.set_trace()