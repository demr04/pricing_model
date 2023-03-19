import numpy as np
import matplotlib.pyplot as plt
import utils as u

#also inverse of fatigue function
def utility(x, k, x_max):

    n0 = np.divide(k, x_max-k)
    n1 = np.divide(x_max, x_max-k)

    return x_max*np.power(x, n0)-np.power(x, n1)

def burden(consume, price, budget):
    return np.divide(consume*np.power(price, 2), budget)

def gains(work, price):
    return price*(np.log(work)+np.log(price)-1.0)

def fatigue(k, x_max):

    x = np.linspace(0, x_max, num=1000)
    
    pairs = u.inverse(x, utility, [0], k, x_max)[0]
    return pairs

def find_func_x(x0, pairs):
    state = 0
    values = []
    for pair in pairs:
        print(values)
        if state == 1 and x0 < pair[0]:
            values.append(pair[1])
            break
        
        if x0 > pair[0]:
            values.append(pair[1])
            state = 1
            break
        
    
    return np.divide(values[0] + values[1], 2)

def grad_market(var, t, *params):

    weights = params[0]
    params_func = params[1]
    consume = var[0]
    work = var[1]
    price = var[2]

    a1 = weights[0][0]
    a2 = weights[0][1]
    b1 = weights[1][0]
    b2 = weights[1][0]
    c = weights[2]

    k_consume = params_func[0][0]
    consume_max = params_func[0][1]
    k_fatigue = params_func[1][0]
    work_max = params_func[1][1]
    budget = params_func[2]

    fatigue_template = fatigue(k_fatigue, work_max)

    dconsume = a1*utility(consume, k_consume, consume_max) - a2*burden(consume, price, budget)
    dwork = b1*gains(work, price)-b2*find_func_x(work, fatigue_template)
    dprice = c*(consume-work)
    
    
    dvar = np.array([dconsume, dwork, dprice])
    
    return dvar

t = np.linspace(0, 100, num=1000)
consume = 10
work = 12
price = 50

var = np.array([consume, work, price])
weights = [[2,3],[1,2],0.1]
params_func = [[100, 150],[200, 300], 1000000]

# result = u.simulation(var, t, grad_market, weights, params_func)

def pol(x):
    return x**2-x**3+x**4

result = u.inverse(t, pol)

"""
#result[:][0] = consume
#result[:][1] = work
#result[:][2] = price
plt.plot(t, result[:][0])
plt.plot(t, result[:][1])
plt.plot(t, result[:][2])

plt.legend(['Demand','Supply','Price'])
plt.show()

"""
