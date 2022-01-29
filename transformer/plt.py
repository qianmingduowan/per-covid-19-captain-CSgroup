import matplotlib.pyplot as plt

x = [i for i in range(10)]
y = x
plt.plot(x, y, color='green', label='aaa')
plt.legend()
plt.savefig('plt.png')