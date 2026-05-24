# This file is AI generated
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("milestone_02_velocity_field.csv")
X = df.pivot(index='y', columns='x', values='vx').values
Y = df.pivot(index='y', columns='x', values='vy').values

plt.streamplot(range(15), range(10), X, Y)
plt.title("Velocity field at t=0")
plt.savefig("velocity.png")
plt.show()