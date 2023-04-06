from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import random as rm
from tqdm import tqdm
from NCBC_helper import sample_sphere, get_K
import pickle
import os
from CBC_helper import functional_steiner, proj, least_squares



seed = 99
np.random.seed(seed)
rm.seed(seed)
T = 40

# time-varying system from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9303845
A0 = np.array([[1.5, 0],[1.,1.]])
B0 = 0.05* np.array([[1.],[2/3]])
nx = A0.shape[0]
nu = B0.shape[1]
n = (nx, nu)
x0 = 2*np.random.rand(nx,1) # initial state
W = 0.
w_list = np.zeros((T, nx))
# w_list[:T-8,:] = np.random.uniform(size=(T-10,nx))
# w_list[:T-8,:] = W * np.array(rm.choices([-0.5 ,1.], k= (T-8)*nx)).reshape((T-10,nx)) 

# bookkeeping
x_list = [x0, ]
u_list = [np.zeros((nu,1)), ]
theta_list = []
G_list = [np.vstack((x0, u_list[0])), ]
ub_list = []
lb_list = []
B_list = [None] * (T-1)
A_list = [None] * (T-1)

# main loop
for t in tqdm(range(1,T)):
    
    A = np.array([[1.5, 0.0025*t],[-0.1 * np.cos(0.3*t), 1.+ 0.05**(3/2) * np.sin(0.5*t)* np.sqrt(t)]])
    B = 0.05* np.array([[1.],[(2 + 0.1*t)/(3 + 0.1*t)]])
    A_list[t-1] = A
    B_list[t-1] = B
    
    x_next = A @ x_list[t-1] + B @ u_list[t-1] + w_list[t-1].reshape((nx,1))
    x_list.append(x_next)
    ub_list.append(x_next + np.ones_like(x_next))
    lb_list.append(x_next - np.ones_like(x_next) * (W+1e-5))

    # select the Steiner point as the consistent hypothesis model
    theta_hat = functional_steiner(G_list, h_lb_list=lb_list, h_ub_list=ub_list, n=n, t=t)
    theta_t = proj(theta_hat, t, n, G_list, lb_list, ub_list)
    A_t = theta_t[:, :nx]
    B_t = theta_t[:, nx:]
    theta_list.append(theta_t)
    
    # compute control action
    K_t = get_K(A=A_t, B=B_t, nx=nx, nu=nu)
    u_t = -K_t @ x_list[-1] # x_list[-1] = x_next
    explore_noise = np.random.uniform(-1., 1., u_list[-1].shape)
    u_list.append(u_t + explore_noise)
    print('u_t = {u}, x_t = {x}'.format(u=u_t, x=np.linalg.norm(x_list[-1])))
    G_list.append(np.vstack((x_list[-1], u_list[-1])))
    
    
# save data for MLJS
filename = os.path.join('out', 'ltv_unstable_seed99_T40_5t-square')
with open(f'{filename}.pkl', 'wb') as f:
    pickle.dump(file=f, obj=dict(
       x_list=x_list, u_list=u_list, theta_list=theta_list,  w_list=w_list))

    
# compute trajectories from open loop and fixed LQR controller 
x_open = [x0,]
u_open = [np.random.uniform(-1., 1., u_list[-1].shape),]
x_ls = [x0,]
u_ls = [0.*np.ones_like(u_list[-1]),]

for t in range(1,T):
    x_open.append(A_list[t-1] @ x_open[t-1] + B_list[t-1] @  u_open[t-1] + w_list[t-1].reshape((nx,1)))
    u_open.append(np.random.uniform(-1., 1., u_list[-1].shape))
    x_ls.append(A_list[t-1] @ x_ls[t-1] + B_list[t-1] @ u_ls[t-1] + w_list[t-1].reshape((nx,1)))
    
    A_LS, B_LS = least_squares(x_ls, u_ls, nx, nu)
    B_LS += 0.0001 * np.ones_like(B_LS)
    K_LS = get_K(A_LS, B_LS, nx, nu)
    u_ls.append(-K_LS @ x_ls[-1])
    
    

plt.semilogy(np.linalg.norm(np.array(x_open), axis=1), label='bounded random injection')
plt.semilogy(np.linalg.norm(np.array(x_list), axis=1), label='algorithm')
plt.semilogy(np.linalg.norm(np.array(x_ls), axis=1), label='Online Least Square')
plt.legend()
plt.savefig('LTV_unstable_seed99.png')

# save data
filename = os.path.join('out', 'ltv_seed99_unstable_T40_5t-square')
with open(f'{filename}.pkl', 'wb') as f:
    pickle.dump(file=f, obj=dict(
        x_open=x_open, u_open = u_open, x_list=x_list, u_list=u_list, theta_list=theta_list, x_ls=x_ls,
        u_ls=u_ls, w_list=w_list))
