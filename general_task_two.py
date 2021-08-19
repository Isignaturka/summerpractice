import pymc as pm
from matplotlib import pyplot as plt

N = 100 # Количество студентов участвующих в опросе
D = [25, 35, 50, 75]

for i in D:
  # Процент списывающих студентов
  p_cheat = pm.Uniform("freq_cheating", 0, 1)

  # Частота получения ответа "ДА"
  p_yes = p_cheat / 2 + 0.25

  # Наблюдения
  observations = pm.Binomial("obs", N, p_yes,
                            observed=True, value=i)

  model = pm.Model([p_cheat, p_yes, observations])

  # Алгоритм вероятностного вывода
  mcmc = pm.MCMC(model)
  mcmc.sample(25000, 2500)

  p_trace = mcmc.trace("freq_cheating")[:]

  #Строим гистограмму 
  plt.figure(figsize=(14, 7))
  plt.hist(p_trace, histtype="stepfilled", density=False, alpha=0.85, bins=30,
          label=f'Апостериорное распределение D={i}', color="#348ABD")
  plt.xlim(0, 1)
  plt.legend();

plt.show()
