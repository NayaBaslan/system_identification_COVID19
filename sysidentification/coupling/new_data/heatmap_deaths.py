import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.rcParams['text.usetex'] = True
import pandas as pd
import os
import pdb
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

a = os.getcwd()
print(os.getcwd())
os.chdir(os.getcwd())
a = os.getcwd()
# import pdb; pdb.set_trace()

#############


month = "August"
start = 13 + 20
start = 13

N = 17 # max = 37 alpha_reg 1e9
N =37
# N = 5
age_groups = 5

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



os.chdir('../')
os.chdir('../')

df1 = pd.read_excel(os.getcwd() + '/data/coupling/five_age_groups_active.xlsx')
df2 = pd.read_excel(os.getcwd() + '/data/coupling/five_age_groups_cumulative.xlsx')
df3 = pd.read_excel(os.getcwd() + '/data/coupling/five_age_groups_dead.xlsx')
df4 = pd.read_excel(os.getcwd() + '/data/coupling/age_groups_pop_germany_red.xlsx')

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
dead = dead.to_numpy()
for i in range(np.shape(dead)[0]):
    for j in range(np.shape(dead)[1]-1):
        if (dead[i,j] > dead[i,j+1]):
            dead[i,j+1] = dead[i,j]

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



age_groups = ["A0-19","A20-39","A40-59","A60-79","A80+"]

weeks = ["13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28" , "29",
         "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49"]
months = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November','December']

# np.shape(weeks) = 37
# np.shape(age_groups) = 9

# import pdb; pdb.set_trace()

fig, ax = plt.subplots(figsize=(15, 5))
dead = dead.astype(int)
# ax.imshow(np.random.rand(8, 90), interpolation='nearest', aspect='auto')

im = ax.imshow(dead)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)

# We want to show all ticks...
plt.xticks(np.arange(1, N +1, step=4), months, rotation=30, fontsize=14)
ax.set_yticks(np.arange(len(age_groups)))
# ... and label them with the respective list entries
ax.set_xticklabels(months, fontsize=14)
ax.set_yticklabels(age_groups, fontsize=14)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
# plt.colorbar(im)
# pdb.set_trace()
# Loop over data dimensions and create text annotations.
for i in range(len(age_groups)):
    for j in range(len(weeks)):
        text = ax.text(j, i, (int)(dead[i, j]/10),
                       ha="center", va="center", color="w")

# ax.set_title("Active Infections Germany ")

ax.set_title(r' Deaths Germany $\cdot 10$', fontsize=14)
# pdb.set_trace()
# fig.tight_layout()
# import pdb ; pdb.set_trace()

# plt.savefig(os.getcwd() + "/figures/coupling/new_data/heatmap_deaths.pdf", bbox_inches='tight')
plt.savefig(os.getcwd() + "/figures/coupling/new_data/heatmap_deaths.pdf")


plt.show()


