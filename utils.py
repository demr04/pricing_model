import numpy as np
import matplotlib.pyplot as plt

def step(t):
    return float((t[-1]-t[0])/len(t))

def RK4_step(var, t, grad, *params):
    dt = step(t)

    k1 = grad(var,t,*params)
    k2 = grad(var+0.5*dt*k1, t+0.5*dt, *params)
    k3 = grad(var+0.5*dt*k2, t+0.5*dt, *params)
    k4 = grad(var+dt*k3, t+dt, *params)

    return dt*(k1+k2+k3+k4)/6

def simulation(var, t, grad,*params):
    
    result = []

    for looper in range(len(t)):
        result.append(var)
        var = var + RK4_step(var, t, grad, *params)

        if np.min(var) <0:
            break

    return result

def fix_dim(data, t):
    if len(t) != len(data):
        t = np.linspace(t[0], t[-1], len(data))
    
    return t

def critic_point_index(x):

    for i in range(1, len(x)-1):
        if np.sign(x[i-1]-x[i]) != np.sign(x[i]-x[i+1]):
            yield i

def inverse(x, func, *params):
    y =  func(x, *params)

    branches = []
    idx_branch = 0
    state = False

    for item in critic_point_index(y):

        state = True

        if idx_branch == 0:
            branches.append([0, item])
            idx_branch += 1
            continue

        branches.append([branches[idx_branch-1][1], item])
        idx_branch += 1

    if state == True:
        branches.append([branches[-1][1], len(x)-1])

    else:
        branches.append([0,len(x-1)])
    
    return y, x, branches

x = np.linspace(-1, 3, num=50)
y = (x-2)*(x-1)*(x+3)*(x+1)*(x+2)
#y = -(x)**2 +3*x


branches = []
idx_branch = 0
state = False

for item in critic_point_index(y):

    state = True

    if idx_branch == 0:
        branches.append([0, item])
        idx_branch += 1
        continue

    if  branches[idx_branch-1][1] == item-1:
        continue

    branches.append([branches[idx_branch-1][1]-1, item])
    idx_branch += 1

if state == True:
    branches.append([branches[-1][1]-1, len(x)-1])

else:
    branches.append([0,len(x-1)])

print(branches)
for i in range(len(y)):
    print('{value}\t{i}'.format(value = round(y[i],2),i = i))

plt.plot(y,x)


for item in branches:
    start = item[0]
    stop = item[1]
    plt.plot(y[start:stop],x[start:stop])

plt.legend(['1','2','3'])

plt.show()


