import math
from IPython.core.pylabtools import figsize
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import pymc as pm

def calculate_n18_flag_difficulty(flag_n18__data):
    no_captures_flags_number = 0
    for i in flag_n18__data.keys():
        if flag_n18__data[i] == 0:
            no_captures_flags_number += 1

    return 1 - (no_captures_flags_number / len(flag_n18__data.keys()))


def calculate_flag_difficulty(participants_and_flags_data, flag_number):
    not_captures_flags_number = 0
    for key in participants_and_flags_data.keys():
        if str(flag_number) in participants_and_flags_data[key]:
            not_captures_flags_number += 1

    return 1 - (not_captures_flags_number / (len(participants_and_flags_data) - 1))

def split_flags(data):
    flags = {}
    for i in data:
        flags[i[0]] = i[1].split(',')
    return flags

def calculate_participant_skill(participants_and_flags_data, participant_number, flags_difficulty):
    participant_skill = 0
    if '' in participants_and_flags_data[participant_number]:
        return 1.0
    for i in participants_and_flags_data[participant_number]:
        participant_skill += flags_difficulty[i]

    return 1 - participant_skill / len(participants_and_flags_data[participant_number])

def calculate_model_Rasha(flag_difficulty, participant_skill):
    return (math.exp(participant_skill - flag_difficulty)) / (1 + math.exp(participant_skill - flag_difficulty))

# Формирование данных по КП и участникам.
# ==========================================
data = genfromtxt('s1.csv', delimiter=';', dtype=None, encoding='utf-8')

participants_and_flags_data = split_flags(data)


# Расчет сложности взятия для каждого КП.
flags_difficulty = {}

for i in range(20):
    if i == 17:
        continue
    flags_difficulty[str(i+1)] = calculate_flag_difficulty(participants_and_flags_data, i+1)


# Расчет навыков взятия КП участников.
participant_skills = {}

for i in participants_and_flags_data.keys():
    if i == 'N':
        continue
    participant_skills[i] = calculate_participant_skill(participants_and_flags_data, i, flags_difficulty)


# Расчет вероятностей взятия КП по модели Раша.
data_for_distribution = {}

for i in participant_skills.keys():
    for j in flags_difficulty.keys():
        data_for_distribution[str(i), str(j)] = calculate_model_Rasha(flags_difficulty[j], participant_skills[i])


# Расчет вероятностей взятия КП по модели Раша для 18-го КП.
flag_n18__data = {'1': 1,
                  '13': 1,
                  '26': 1,
                  '21': 1,
                  '24': 1,
                  '15': 1,
                  '19': 0,
                  '9': 0,
                  '2': 0}

flag_n18_difficulty = calculate_n18_flag_difficulty(flag_n18__data)

data_for_distribution_for_18n_flag = {}

for i in participant_skills.keys():
    if i in flag_n18__data.keys():
        continue
    data_for_distribution_for_18n_flag[i, '18'] = calculate_model_Rasha(flag_n18_difficulty, participant_skills[i])


# Расчет координат для всех КП кроме 18-го.
x_coords = []
y_coords = []

for i in data_for_distribution.keys():
    x_coords.append(int(i[0]))
    y_coords.append(data_for_distribution[i])


# Расчет координат для 18-го КП.
x_coords_for_n18_flag = []
y_coords_for_n18_flag = []

for i in data_for_distribution_for_18n_flag.keys():
    x_coords_for_n18_flag.append(int(i[0]))
    y_coords_for_n18_flag.append(data_for_distribution_for_18n_flag[i])


# Создание интерфейса с графиками.
figsize(10, 10)
fig = plt.figure()
plt.subplots_adjust(hspace=0.3)


# Отрисовка графика 1
x = np.array(x_coords)
y = np.array(y_coords)

ax_1 = fig.add_subplot(221)
ax_1.scatter(x, y)
ax_1.set_xlabel('Номер участника')
ax_1.set_ylabel('Вероятность взятия флага')
ax_1.set_title(r'''Рапределение вероятностей захвата
КП (всех кроме 18-го) участниками''', loc='center')


# Расчет апостериорного распределения вероятностей для всех КП, кроме 18-го.
d = [0.56, 0.41, 0.34]

tau_1 = pm.Uniform('tau_1', lower=0, upper=1)

lambda_1 = pm.Exponential('lambda_1', 0.05)
lambda_2 = pm.Exponential('lambda_2', 0.05)
lambda_3 = pm.Exponential('lambda_3', 0.05)

observed_1 = pm.Normal('observed_1', tau_1, lambda_1, observed=True, value=d[0])
observed_2 = pm.Normal('observed_2', tau_1, lambda_2, observed=True, value=d[1])
observed_3 = pm.Normal('observed_3', tau_1, lambda_3, observed=True, value=d[2])

model = pm.Model([tau_1, lambda_1, lambda_2, lambda_3, observed_1, observed_2, observed_3])

mcmc = pm.MCMC(model)
mcmc.sample(25000, 2500)

p2_trace = mcmc.trace('tau_1')[:]


# Отрисовка графика 2.
ax_2 = fig.add_subplot(222)
ax_2.hist(p2_trace, histtype='stepfilled', bins=140, alpha=0.85, color="#A60628", density=True)
ax_2.set_xlabel('Вероятность взятия флага')
ax_2.set_title(r'''Апостериорное распределение вероятностей
$d_{i}$ и $p_{j}$ (для всех кроме 18-го КП)''', loc='center')
plt.xlim(0, 1)


# Отрисовка графика 3.
x = np.array(x_coords_for_n18_flag)
y = np.array(y_coords_for_n18_flag)

ax_3 = fig.add_subplot(223)
ax_3.scatter(x, y)
ax_3.set_xlabel('Номер участника')
ax_3.set_ylabel('Вероятность взятия 18-го КП')
ax_3.set_title('Рапределение вероятностей\nзахвата 18-го КП участниками', loc='center')


# Расчет апостериорного распределения вероятностей для 18-го КП.
d = 0.45

tau_1 = pm.Uniform('tau_1', lower=0, upper=1)
lambda_1 = pm.Exponential('lambda_1', 0.005)
observed_1 = pm.Normal('observed_1', tau_1, lambda_1, observed=True, value=d)

model = pm.Model([lambda_1, tau_1])

mcmc = pm.MCMC(model)
mcmc.sample(25000, 2500)

p2_trace = mcmc.trace('tau_1')[:]


# Отрисовка графика 4
ax_4 = fig.add_subplot(224)
ax_4.hist(p2_trace, histtype='stepfilled', bins=140, alpha=0.85, color="#A60628", density=True)
ax_4.set_xlabel('Вероятность взятия 18-го КП')
ax_4.set_title(r'''Апостериорное распределение
вероятностей $d_{i}$ и $p_{j}$ для 18-го КП''', loc='center')
plt.xlim(0, 1)

plt.show()
