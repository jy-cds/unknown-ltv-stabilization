from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import random as rm
from tqdm import tqdm
from NCBC_helper import sample_sphere, get_K
import pickle
import os
from CBC_helper import functional_steiner, proj, least_squares




seed = 520
np.random.seed(seed)
rm.seed(seed)
T = 40


# time-varying system from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9303845

nx = 2
nu = 1
n = (nx, nu)
# x0 = np.random.rand(nx,1) # initial state
x0 = np.array([[0.5],[-0.5]])
W = 1.5
w_list = np.zeros((T, nx))


x_list = [x0, ]
u_list = [3.*np.ones((nu,1)), ]
theta_list = []
G_list = [np.vstack((x0, u_list[0])), ]
ub_list = []
lb_list = []


for t in tqdm(range(1,T)):
    
    
    # x_next = A @ x_list[t-1] + B @ u_list[t-1] + w_list[t-1].reshape((nx,1))
    x_next = np.array([x_list[t-1][0] + 0.1*x_list[t-1][1],-0.049 * np.sin(x_list[t-1][0]) + 0.998*x_list[t-1][1] + 0.1*u_list[t-1].flatten() ])
    x_list.append(x_next)
    ub_list.append(x_next + np.ones_like(x_next) * (W))
    lb_list.append(x_next - np.ones_like(x_next) * (W))

    # select the Steiner point as the consistent hypothesis model
    theta_hat = functional_steiner(G_list, h_lb_list=lb_list, h_ub_list=ub_list, n=n, t=t, box_constraints=[0.01 , 2])
    theta_t = proj(theta_hat, t, n, G_list, lb_list, ub_list)
    A_t = theta_t[:, :nx]
    B_t = theta_t[:, nx:]
    theta_list.append(theta_t)
    
    # compute control action
    K_t = get_K(A=A_t, B=B_t, nx=nx, nu=nu)
    u_t = -K_t @ x_list[-1] # x_list[-1] = x_next
    u_list.append(u_t)
    print('u_t = {u}, x_t = {x}'.format(u=u_t, x=np.linalg.norm(x_list[-1])))
    G_list.append(np.vstack((x_list[-1], u_list[-1])))
    
# save data
filename = os.path.join('out', 'nonlinear_seed88_T40_5t-square')
with open(f'{filename}.pkl', 'wb') as f:
    pickle.dump(file=f, obj=dict(
       x_list=x_list, u_list=u_list, theta_list=theta_list,  w_list=w_list))

# compute trajectories from open loop and fixed LQR controller 
x_open = [x0,]
x_ls = [x0,]
u_ls = [0.*np.ones_like(u_list[-1]),]

for t in range(1,T):
    x_open.append(np.array([x_open[t-1][0] + 0.01*x_open[t-1][1] , -0.049 * np.sin(x_open[t-1][0]) + 0.998*x_open[t-1][1] ]))
    x_ls.append( np.array([x_ls[t-1][0] + 0.01*x_ls[t-1][1] , -0.049 * np.sin(x_ls[t-1][0]) + 0.998*x_ls[t-1][1] + 0.01*u_ls[t-1].flatten() ]))
    
    A_LS, B_LS = least_squares(x_ls, u_ls, nx, nu)
    B_LS += 1e-6 * np.ones_like(B_LS)
    K_LS = get_K(A_LS, B_LS, nx, nu)
    u_ls.append(-K_LS @ x_ls[-1])
    
    

plt.semilogy(np.linalg.norm(np.array(x_open), axis=1), label='open loop')
plt.semilogy(np.linalg.norm(np.array(x_list), axis=1), label='algorithm')
plt.semilogy(np.linalg.norm(np.array(x_ls), axis=1), label='Online Least Square')
plt.legend()
plt.savefig('nonlinear_seed88.png')

# save data
filename = os.path.join('out', 'nonlinear_seed88_T40_5t-square')
with open(f'{filename}.pkl', 'wb') as f:
    pickle.dump(file=f, obj=dict(
        x_open=x_open, x_list=x_list, u_list=u_list, theta_list=theta_list, x_ls=x_ls,
        u_ls=u_ls, w_list=w_list))