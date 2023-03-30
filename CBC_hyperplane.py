from __future__ import annotations
import numpy as np
import mosek
import cvxpy as cp
import matplotlib.pyplot as plt
import control
import random as rm
from tqdm import tqdm
from NCBC_helper import sample_sphere, get_K
from dynamics import LTV_dynamics
import matplotlib.pyplot as plt
from scipy import linalg

# CBC functions
def min_wf(t: int, n: tuple[int, int], r: float, G_list: list[np.ndarray], h_lb_list: list[np.ndarray], h_ub_list:list[np.ndarray]) -> float:
    ''' Computes an estimate of the minimum value of the work function after observing x(t+1)
    
    Args
    - t: current time step
    - n: tuple, in the form of (nx, nu) where nx and nu are the dimension of the state and control respectively
    - r: radius of the sublevel set of the work function
    - G_list, h_lb_list, h_ub_list: time-concatenated half space constraints from trajectory data
    
    Returns: float
    - the estimated minimum value of work function after observing x(t+1)
    '''

    # setting up the optimization variables
    lam = cp.Variable((t,1))
    z = {} # z is a list of t nx by (nx+nu) matrices
    x = {}
    for i in range(t+1):
        x[i] = cp.Variable((n[0], np.sum(n)))
        z[i] = cp.Variable((n[0], np.sum(n)))
    
    # setting up the constraints for t=1
    
    for i in range(1,t+1):
        if i == 1:
            constraints = [z[i] == x[i]]
        else:
            constraints += [z[i] == x[i]-x[i-1]]
        
        constraints += [x[i] @ G_list[i-1] >= h_lb_list[i-1]]
        constraints += [x[i] @ G_list[i-1] <= h_ub_list[i-1]]
        constraints += [cp.norm(z[i], p='fro') <= lam[i-1]]

    prob = cp.Problem(cp.Minimize(cp.sum(lam)), constraints)
    prob.solve(solver='MOSEK', mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
    
    if prob.status in ["infeasible", "unbounded"]:
        print('MIN_WF failed to solve due to {reason}'.format(prob.status))
    
    return prob.value


def compute_support(Z: np.ndarray, eps: float, t: int, n: tuple[int, int], r: float, G_list: list[np.ndarray], h_lb_list: list[np.ndarray], h_ub_list:list[np.ndarray]) -> np.ndarray:
    ''' Given a set of random vectors concatenated into Z, compute the support function evaluated at each of the random vectors for the current convex body (sublevel set of work function after observing x(t+1) with level r)
    
    Args
    - Z: concatenated random vectors sampled uniformly from the unit sphere with N samples
    - eps: error tolerance for solving the second order cone program
    - t: current time step
    - n: tuple, in the form of (nx, nu) where nx and nu are the dimension of the state and control respectively
    - r: radius of the sublevel set of the work function
    - G_list, h_lb_list, h_ub_list: time-concatenated half space constraints from trajectory data
    
    Returns: Y: ndarray(N, nx, nx+nu), concatenated vector of the extreme points in sublevel set of the work function
    '''
    
    Y = np.zeros((Z.shape[0], n[0], np.sum(n)))
    for k, z_k in enumerate(Z):
        y = cp.Variable((n[0],np.sum(n))) # y is an element in the consistent parameter set with dimension nx by (nx+nu)
        b = cp.Variable()
        lam = cp.Variable((t,1))
        z = {} # z is a list of t nx by (nx+nu) matrices
        x = {}
        for i in range(1,t+1):
            x[i] = cp.Variable((n[0], np.sum(n)))
            z[i] = cp.Variable((n[0], np.sum(n)))
    
        # setting up the constraints for t=1    
        for i in range(1,t+1):
            if i == 1:
                constraints = [x[i] == z[i]] # x_1 = z_1 since x_0 = 0
            else:
                constraints += [z[i] == x[i]-x[i-1]]
                
            constraints += [x[i] @ G_list[i-1] >= h_lb_list[i-1]]
            constraints += [x[i] @ G_list[i-1] <= h_ub_list[i-1]]
            constraints += [cp.norm(z[i], p='fro') <= lam[i-1]]
        
        constraints += [cp.norm((y-x[t]), p='fro') <= b]
        constraints += [b + cp.sum(lam) <= 2*r]
        
        prob = cp.Problem(cp.Minimize(z_k @ cp.vec(y)), constraints)
        # prob.solve(solver='MOSEK', mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual, 'MSK_DPAR_INTPNT_CO_TOL_NEAR_REL': eps}) 
        prob.solve(solver='MOSEK', mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}) 
        
        if prob.status in ["infeasible", "unbounded"]:
            print('Support function failed to compute due to {reason}'.format(prob.status))
        else:
            Y[k] = y.value
            
    return Y



def steiner(G_list: list[np.ndarray], h_lb_list: list[np.ndarray], h_ub_list: list[np.ndarray], n: int, eps: float, t:int, R: float,  delta: float = 1.) -> tuple[np.ndarray, np.ndarray]:
    ''' Compute the Steiner point of the sublevel set of the current work function after observing x(t+1)
    
    Args
    - G_list, h_lb_list, h_ub_list: time-concatenated half space constraints from trajectory data
    - n: tuple, in the form of (nx, nu) where nx and nu are the dimension of the state and control respectively
    - eps: error tolerance for solving the second order cone program
    - t: current time step
    - R: radius of the sublevel set of the work function
    - delta: probability of the computed steiner point being more than eps away from the analytical steiner point
    
    Returns Theta = [A, B]
    '''
    d = n[0] * np.sum(n)
    N = int(np.min( [((d+1) ** 2 + R**2) / (eps**2 * delta) , 300]))
    Z = sample_sphere(d=d , N=N) # Z is a N by d matrix with N iid sampled R^d unit vector
    P = compute_support(Z=Z, eps=eps/d, t=t, n=n, r=R, G_list=G_list, h_lb_list=h_lb_list, h_ub_list=h_ub_list)
    Theta = np.mean(P, axis=0) # Theta[:, :nx], Theta[:, nx:] = A, B
    return Theta

def proj(Theta_hat, t, n, G_list, h_lb_list, h_ub_list, get_distance=False):
    # closed-form computation of the projection on to the intersection of the hyperplane
    Theta = cp.Variable((n[0], np.sum(n)))
    constraints = [Theta @ G_list[-1] >= h_lb_list[-1]]
    constraints += [Theta @ G_list[-1] <= h_ub_list[-1]]
    prob = cp.Problem(cp.Minimize(cp.norm(Theta - Theta_hat, p='fro')), constraints)
    prob.solve(solver = 'MOSEK', mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
    
    if prob.status in ["infeasible", "unbounded"]:
        print('Projection step failed to compute due to {reason}'.format(prob.status))
        
    if get_distance:
        return prob.value
    else:
        return Theta.value


def least_squares(x_list, u_list, nx, nu):
    xt1 = np.hstack(x_list[1:])
    xt = np.hstack(x_list[:-1])
    ut  = np.hstack(u_list)
    zt = np.vstack((xt,ut))
    # print("x",xt,'\n',"Next x",xt1,'\n',"u",ut)
    A = cp.Variable((nx,nx))
    B = cp.Variable((nx,nu))
    obj = cp.Minimize(cp.norm(xt1 - cp.hstack((A,B)) @  zt, 'fro' ))
    prob = cp.Problem(obj) #,constraints
    prob.solve(solver=cp.MOSEK)
    return A.value,B.value


######################## fixed parameters ########################
np.random.seed(520)
T = 30


# Synthetic Switching System Setup 

'''
# LTV Dynamics inspired by https://arxiv.org/pdf/2206.02507.pdf. Open-loop unstable
A = np.array([[1.,0.5],[1.2,0.]]) #
B = np.array([[0.],[0.]])  #np.array([[0.],[t/20]])
''' 
A = np.matrix([[0.99, 1.5], [0, 0.99]])
B = np.matrix([[1, 0], [0, 1]])


# Dynamics taken from https://arxiv.org/pdf/2206.02507.pdf
A1 = np.array([[1.,0.5],[0.,1.]])
B1 = np.array([[0.],[1.2]])
A2 = np.array([[1.,1.5],[0.,1.]])
B2 = np.array([[1.],[0.9]])
Q = np.array([[0.1, 0.9], [0.2, 0.8]]) # transition probability matrix
switched_dynamics = ({"A": A1, "B": B1},{"A": A2, "B": B2})
transition_index = [None]*T
transition_index[0] = int(np.random.choice(2,1,[0.5,0.5])) # initialize system with equal probability


################################################################### 
nx = A.shape[0]
nu = B1.shape[1]
n = (nx, nu)
x0 = np.random.rand(nx,1) # initial state
W = 2.
w_list = np.zeros((T, nx))
w_list[:T-5,:] = np.random.uniform(size=(T-5,nx))
w_list[:T-5,:] = W * np.array(rm.choices([-2. ,1.], k= (T-5)*nx)).reshape((T-5,nx)) 
Q = np.eye(nx)
R = np.eye(nu)



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
    # if t <=25:
    #     B = np.array([[0.],[(np.e)**(t/20)]])
    # B_list[t-1] = B
    
    # A[0,1] = np.abs(np.sin(np.pi*(t-1)/2))*np.exp((t-1)/60)
    # A[1,0] = np.abs(np.cos(np.pi*(t-1)/2))*np.exp((t-1)/60)
    # A_list[t-1] = A
    
    A = switched_dynamics[transition_index[t-1]]['A']
    B = switched_dynamics[transition_index[t-1]]['B']
    transition_index[t] = int(np.random.choice(2, 1, p=Q[transition_index[t-1]])) # get the next dynamical matrix
    
    A_list[t-1] = A
    B_list[t-1] = B
    x_next = A @ x_list[t-1] + B @ u_list[t-1] + w_list[t-1].reshape((nx,1))
    x_list.append(x_next)
    ub_list.append(x_next + np.ones_like(x_next) * W)
    lb_list.append(x_next - np.ones_like(x_next) * W)
    if t == 1:
        r = proj(Theta_hat=np.zeros((nx, nx+nu)), t=t, n=n, G_list=G_list, h_lb_list=lb_list, h_ub_list=ub_list, get_distance=True)
    else:
        eva = min_wf(t, n, r, G_list, lb_list, ub_list)
        if eva > 3*r/2 - r/100:
            print('t = {t}: r is updated from {r_prev} to {r}'.format(t=t, r_prev=r, r = eva))
            r = eva
    
    # select the Steiner point as the consistent hypothesis model
    theta_hat = steiner(G_list, h_lb_list=lb_list, h_ub_list=ub_list, n=n, eps=r/(t**2), t=t, R=2*r)
    theta_t = proj(theta_hat, t, n, G_list, lb_list, ub_list)
    A_t = theta_t[:, :nx]
    B_t = theta_t[:, nx:]
    theta_list.append(theta_t)
    
    # compute control action
    K_t = get_K(A=A_t, B=B_t, nx=nx, nu=nu)
    u_t = -K_t @ x_list[-1] # x_list[-1] = x_next
    u_list.append(u_t)
    print('u_t = {u}'.format(u=u_t))
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
plt.show()

    
    