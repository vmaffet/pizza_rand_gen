#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def pizza_gen(n, mu = 0, sigma = 0.1):

  if isinstance(mu, (int, float)):
    mu = mu*np.ones(n)
  elif type(mu) is not np.array:
    print('Error mu type')
    raise

  if isinstance(sigma, (int, float)):
    sigma = sigma*np.ones(n)
  elif type(sigma) is not np.array:
    print('Error sigma type')
    raise

  gen_n = mu[0]
  speed = 0
  acc   = 0
  yield gen_n

  rands = np.random.normal(mu, sigma)

  streak = 3
  for i in range(1,n):
    if (rands[i] - gen_n)*acc > 0:
      streak += 1
    else:
      streak = 4

    if rands[i] > gen_n:
      acc = sigma[i]/streak
    else:
      acc = -sigma[i]/streak

    speed += acc
    gen_n += speed
    yield gen_n


if __name__ == "__main__":

  N = 750

  data = np.array(list(pizza_gen(N, sigma=0.025)))

  theta = np.linspace(0, 2*np.pi, N, False)
  R = 5

  X = (R+data)*np.sin(theta)
  Y = (R+data)*np.cos(theta)

  plt.plot(X,Y)
  plt.show()

'''
  data = [0] * 100

  plt.ion()

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_ylim(-1, 1)
  line1, = ax.plot(data)

  for x in pizza_gen(N):
    data = [x]+data[:-1]
    line1.set_ydata(data)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)
'''
