import numpy as np
import pylab as plt

N_Days = 252
N_Runs = 10000
strike = 100
Spot_Price = 100
volatility=0.15
np.random.seed(25)
rets = np.random.randn(N_Runs, N_Days)*volatility/np.sqrt(252)
print(rets.shape)
traces = np.cumprod(1+rets,1)*Spot_Price
put = np.mean((strike-traces[:,-1])*(((traces[:,-1]-strike)< 0)))



call = np.mean((traces[:, -1] - strike)*((traces[:, -1]-strike) > 0))

print('put prisen')
print(put)

def get_price_w_rf(right, T, S, X, v, rf, N=10000):
    D = np.exp(-rf*(T/252))
    prices = np.cumprod(1+(np.random.randn(T, N) * v / np.sqrt(252)),axis=0)*S
    if right == 'c':
        return np.sum((prices[-1,:]-X*D)[prices[-1,:]>X*D])/prices.shape[1]
    else:
        return -np.sum((prices[-1,:]-X*D)[prices[-1,:]>X*D])/prices.shape[1]

print(get_price_w_rf('c', 126, 144, 144,0.15, 0.02))