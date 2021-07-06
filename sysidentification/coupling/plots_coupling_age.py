from __future__ import division

# start time 12:50 end 13:11 to formulate nlp
from casadi import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import os


import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pdb
import numpy as np
nx = 4
nu = 3

symbol = SX.sym

beta_age = symbol('beta_age',(9,9))
gamma_age = symbol('gamma_age',(1,9))
mu_age = symbol('mu_age',(1,9))

u_ages = vertcat(beta_age,gamma_age,mu_age)
#
os.chdir('../')
os.chdir('../')

# begin week 13
df2 = pd.read_excel(os.getcwd() + '/sysidentification/coupling/results/solution_casadi20week5_N.xlsx')

col_str_counts = np.sum(df2.applymap(lambda x: 1 if isinstance(x, str) else 0), axis=0)
N = np.shape(col_str_counts)[0]
x_solution_age = df2.iloc[1:37, 0:N]
x_solution_age = x_solution_age.to_numpy()

col_str_counts = np.sum(df2.applymap(lambda x: 1 if isinstance(x, str) else 0), axis=0)

all_S_ages = x_solution_age[0:9,:]
all_I_ages = x_solution_age[9:18,:]
all_R_ages = x_solution_age[18:27,:]
all_D_ages = x_solution_age[27:36,:]

pdb.set_trace()
tspan2 = np.arange(0+13, N+13, 1)
plt.figure(1)
plt.title("Active Infections - Real vs Model for all Age Groups")
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Sort colors by hue, saturation, value and name.
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]

n = len(sorted_names)

a_color = ['black', 'k', 'dimgray', 'dimgrey', 'gray', 'grey', 'darkgray', 'darkgrey', 'silver', 'lightgray', 'lightgrey', 'gainsboro',
 'whitesmoke', 'w', 'white', 'snow', 'rosybrown', 'lightcoral', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'r', 'red',
 'mistyrose', 'salmon', 'tomato', 'darksalmon', 'coral', 'orangered', 'lightsalmon', 'sienna', 'seashell', 'chocolate', 'saddlebrown',
 'sandybrown', 'peachpuff', 'peru', 'linen', 'bisque', 'darkorange', 'burlywood', 'antiquewhite', 'tan', 'navajowhite', 'blanchedalmond',
 'papayawhip', 'moccasin', 'orange', 'wheat', 'oldlace', 'floralwhite', 'darkgoldenrod', 'goldenrod', 'cornsilk', 'gold', 'lemonchiffon',
 'khaki', 'palegoldenrod', 'darkkhaki', 'ivory', 'beige', 'lightyellow', 'lightgoldenrodyellow', 'olive', 'y', 'yellow', 'olivedrab', 'yellowgreen',
 'darkolivegreen', 'greenyellow', 'chartreuse', 'lawngreen', 'honeydew', 'darkseagreen', 'palegreen', 'lightgreen', 'forestgreen', 'limegreen', 'darkgreen',
 'g', 'green', 'lime', 'seagreen', 'mediumseagreen', 'springgreen', 'mintcream', 'mediumspringgreen', 'mediumaquamarine', 'aquamarine', 'turquoise', 'lightseagreen',
 'mediumturquoise', 'azure', 'lightcyan', 'paleturquoise', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'c', 'aqua', 'cyan', 'darkturquoise', 'cadetblue',
 'powderblue', 'lightblue', 'deepskyblue', 'skyblue', 'lightskyblue', 'steelblue', 'aliceblue', 'dodgerblue', 'lightslategray', 'lightslategrey', 'slategray',
 'slategrey', 'lightsteelblue', 'cornflowerblue', 'royalblue', 'ghostwhite', 'lavender', 'midnightblue', 'navy', 'darkblue', 'mediumblue', 'b', 'blue', 'slateblue'
    , 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'thistle',
 'plum', 'violet', 'purple', 'darkmagenta', 'm', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink', 'lavenderblush',
 'palevioletred', 'crimson', 'pink', 'lightpink']



for i in range(9):
  plt.plot(tspan2,np.transpose(x_solution_age[i+9,:]),color=a_color[i*2])


plt.legend(["I0 model","I0 real","I1 model","I1 real","I2 model","I2 real","I3 model","I3 real","I4 model","I4 real","I5 model","I5 real","I6 model","I6 real","I7 model","I7 real","I8 model","I8 real",])
plt.xlabel("Calendar Weeks")
plt.ylabel("Number of Cases")
plt.figure(2)
plt.title("Recovered Cases - Real vs Model for all Age Groups")
for i in range(9):
  plt.plot(tspan2,np.transpose(x_solution_age[i+18,:]),color=a_color[(i+9)*2])


plt.legend(["R0 model","R0 real","R1 model","R1 real","R2 model","R2 real","R3 model","R3 real","R4 model","R4 real","R5 model","R5 real","R6 model","R6 real","R7 model","R7 real","R8 model","R8 real",])

plt.figure(3)
plt.title("Dead Cases - Real vs Model for all Age Groups")
for i in range(9):
  plt.plot(tspan2,np.transpose(x_solution_age[i+27,:]),color=a_color[(i+18)*2])


plt.legend(["D0 model","D0 real","D1 model","D1 real","D2 model","D2 real","D3 model","D3 real","D4 model","D4 real","D5 model","D5 real","D6 model","D6 real","D7 model","D7 real","D8 model","D8 real",])

plt.figure(4)
plt.title("Infected, Recoverd, and Dead for all Age Groups")
lines = []
for i in range(9):

    lines += plt.plot(tspan2, np.transpose(x_solution_age[i+9, :]),label="{} infected".format(i),color=a_color[(i+27)*2])
    lines += plt.plot(tspan2, np.transpose(x_solution_age[i+18, :]),label="{} recovered".format(i),color=a_color[(i+28)*2])
    lines += plt.plot(tspan2, np.transpose(x_solution_age[i+27, :]),label="{} dead".format(i),color=a_color[(i+29)*2])


labels = [l.get_label() for l in lines]
plt.legend(lines, labels)


plt.show()
pdb.set_trace()

