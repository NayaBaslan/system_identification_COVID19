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
df = pd.read_excel(os.getcwd() + '/data/Rwert/Nowcasting_Zahlen.xlsx', 'Nowcast_R')
rwert = df.iloc[16:346,7]
rwert = rwert.to_numpy()

rwert_lb = df.iloc[16:346,8]
rwert_lb = rwert_lb.to_numpy()

rwert_ub = df.iloc[16:346,9]
rwert_ub = rwert_ub.to_numpy()

N = np.shape(rwert)[0]
t = np.arange(0, N, 1)

plot_rwert(t, N, rwert, rwert_lb, rwert_ub)

# weekly
weeks = 47
t_weeks = np.arange(0, weeks, 1)
rwert_weekly = np.zeros(weeks)
rwert_lb_weekly = np.zeros(weeks)
rwert_ub_weekly = np.zeros(weeks)

for i in range(weeks):
    rwert_weekly[i] = rwert[i*7]
    rwert_ub_weekly[i] = rwert_ub[i*7]
    rwert_lb_weekly[i] = rwert_lb[ i* 7]


plot_rwert_weekly(t_weeks, weeks, rwert_weekly, rwert_lb_weekly, rwert_ub_weekly)



plt.show()