from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import random as rm
from tqdm import tqdm
from NCBC_helper import sample_sphere, get_K
import pickle
import os
from CBC_helper import functional_steiner, proj, least_squares

######################## fixed parameters ########################
seed_list = [88,9,10,11]
seed = 88
np.random.seed(seed)
rm.seed(seed)
T = 40


# Synthetic Switching System Setup 

# A = np.matrix([[0.99, 1.5], [0, 0.99]])
# B = np.matrix([[1, 0], [0, 1]])


# Dynamics taken from Xiong & Lam "Stabilization of discrete-time Markovian jump linear systems via time-delayed controllers" 
A1 = np.array([[1.5,1.],[0.,0.5]])
B1 = np.array([[0.],[1.]])
A2 = np.array([[0.6,0],[0.1,1.2]])
B2 = np.array([[1.],[1.]])
Q = np.array([[0.8, 0.2], [0.1, 0.9]]) # transition probability matrix
switched_dynamics = ({"A": A1, "B": B1},{"A": A2, "B": B2})
transition_index = [None]*T
transition_index[0] = int(np.random.choice(2,1,[0.5,0.5])) # initialize system with equal probability


################################################################### 
nx = A1.shape[0]
nu = B1.shape[1]
n = (nx, nu)
x0 = np.random.rand(nx,1) # initial state
W = 3.
w_list = np.zeros((T, nx))
w_list[:T-8,:] = np.random.uniform(size=(T-8,nx))
w_list[:T-8,:] = W * np.array(rm.choices([-1. ,1.], k= (T-8)*nx)).reshape((T-8,nx)) 




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
    
    A = switched_dynamics[transition_index[t-1]]['A']
    B = switched_dynamics[transition_index[t-1]]['B']
    A_list[t-1] = A
    B_list[t-1] = B
    
    # for the last 20 time steps, stop changing the system
    if t <= T - 15:
        transition_index[t] = int(np.random.choice(2, 1, p=Q[transition_index[t-1]])) # get the next dynamical matrix
    else:
        transition_index[t] = transition_index[t-1]
    

    x_next = A @ x_list[t-1] + B @ u_list[t-1] + w_list[t-1].reshape((nx,1))
    x_list.append(x_next)
    ub_list.append(x_next + np.ones_like(x_next) * W)
    lb_list.append(x_next - np.ones_like(x_next) * W)
    
    '''
    if t == 1:
        r = proj(Theta_hat=np.zeros((nx, nx+nu)), t=t, n=n, G_list=G_list, h_lb_list=lb_list, h_ub_list=ub_list, get_distance=True)
    else:
        eva = min_wf(t, n, r, G_list, lb_list, ub_list)
        if eva > 3*r/2 - r/100:
            print('t = {t}: r is updated from {r_prev} to {r}'.format(t=t, r_prev=r, r = eva))
            r = eva
    '''
    
    # select the Steiner point as the consistent hypothesis model
    # theta_hat = steiner(G_list, h_lb_list=lb_list, h_ub_list=ub_list, n=n, eps=r/(t**2), t=t, R=2*r)
    theta_hat = functional_steiner(G_list, h_lb_list=lb_list, h_ub_list=ub_list, n=n, t=t)
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


# compute trajectories from open loop and fixed LQR controller 
x_open = [x0,]
x_ls = [x0,]
u_ls = [0.*np.ones_like(u_list[-1]),]
# K_fixed = get_K(A=A, B=B_list[0], nx=nx, nu=nu)

# compute offline optimal trajectory
# x_opt = [x0,]
# P = [None] * (T-1)
# for k in range(T-2,-1,-1): # counting from T-2 back to 0
#     if k == T-2:
#         P[k] = Q
#     else:
#         P[k] = Q + A.transpose()@P[k+1]@A - A.transpose() @ P[k+1] @ B_list[k] @ linalg.inv(R + B.transpose() @ P[k+1] @ B) @ B.transpose() @ P[k+1] @ A

for t in range(1,T):
    A = switched_dynamics[transition_index[t-1]]['A']
    B = switched_dynamics[transition_index[t-1]]['B']
    transition_index[t] = int(np.random.choice(2, 1, p=Q[transition_index[t-1]])) # get the next dynamical matrix
    
    A_list[t-1] = A
    B_list[t-1] = B
    transition_index[t] = int(np.random.choice(2, 1, p=Q[transition_index[t-1]])) # get the next dynamical matrix
    
    
    
    x_open.append(A_list[t-1] @ x_open[t-1] + w_list[t-1].reshape((nx,1)))
    x_ls.append(A_list[t-1] @ x_ls[t-1] + B_list[t-1] @ u_ls[t-1] + w_list[t-1].reshape((nx,1)))
    
    A_LS, B_LS = least_squares(x_ls, u_ls, nx, nu)
    B_LS += 0.0001 * np.ones_like(B_LS)
    K_LS = get_K(A_LS, B_LS, nx, nu)
    u_ls.append(-K_LS @ x_ls[-1])
    
    
    # K_opt = linalg.inv(R + B.transpose() @ P[t+1])
    # x_opt.append((A - B_list[t-1] @ K_opt) @ x_opt[-1] + w_list[t-1].reshape((nx,1)))
    

plt.semilogy(np.linalg.norm(np.array(x_open), axis=1), label='open loop')
plt.semilogy(np.linalg.norm(np.array(x_list), axis=1), label='algorithm')
plt.semilogy(np.linalg.norm(np.array(x_ls), axis=1), label='Online Least Square')
# plt.semilogy(np.linalg.norm(np.array(x_lqr), axis=1), label='fix LQR')
# plt.semilogy(np.linalg.norm(np.array(x_opt), axis=1), label='optimal')

plt.legend()
plt.savefig('seed88.png')

# save data for MLJS
filename = os.path.join('out', 'mljs_seed88_T40_5t-square')
with open(f'{filename}.pkl', 'wb') as f:
    pickle.dump(file=f, obj=dict(
        x_open=x_open, x_list=x_list, u_list=u_list, theta_list=theta_list, x_ls=x_ls,
        u_ls=u_ls, switches=transition_index, w_list=w_list))

    

