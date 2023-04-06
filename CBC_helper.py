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
import pickle
import os



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
    # prob.solve(solver='MOSEK', mosek_params={'MSK_IPAR_INTPNT_SOLVE_FORM': mosek.solveform.dual})
    
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

def socp(Z: np.ndarray, t: int, n: tuple[int, int], G_list: list[np.ndarray], h_lb_list: list[np.ndarray], h_ub_list:list[np.ndarray], box_constraints: list[float] = [-2. , 3.]) -> np.ndarray:
    ''' Given a set of random vectors concatenated into Z, compute the support function evaluated at each of the random vectors for the current convex body (sublevel set of work function after observing x(t+1) with level r)
    
    Args
    - Z: concatenated random vectors sampled uniformly from the unit sphere with N samples
    - eps: error tolerance for solving the second order cone program
    - t: current time step
    - n: tuple, in the form of (nx, nu) where nx and nu are the dimension of the state and control respectively
    - G_list, h_lb_list, h_ub_list: time-concatenated half space constraints from trajectory data
    - box_constriants: lower and upper bound on the value of each entry of A and B
    
    Returns: 
    - Y: ndarray(N, nx, nx+nu), concatenated vector of the extreme points in sublevel set of the work function
    - val: ndarray(N,), a list of the SOCP optimal value
    '''
    
    Y = np.zeros((Z.shape[0], n[0], np.sum(n)))
    val = np.zeros((Z.shape[0],))
    for k, z_k in enumerate(Z):
        y = cp.Variable((n[0],np.sum(n))) # y is an element in the consistent parameter set with dimension nx by (nx+nu)
        b = cp.Variable()
        lam = cp.Variable((t,1))
        z = {} # z is a list of t nx by (nx+nu) matrices, casted as dictionary here.
        x = {}
        for i in range(1,t+1):
            x[i] = cp.Variable((n[0], np.sum(n)))
            z[i] = cp.Variable((n[0], np.sum(n)))
    
        # setting up the constraints: z[i] = x[i]-x[i-1] is the dummy  
        for i in range(1,t+1):
            if i == 1:
                constraints = [x[i] == z[i]] # x_1 = z_1 since we assume x_0 = 0
            else:
                constraints += [z[i] == x[i]-x[i-1]]
                
            constraints += [x[i] @ G_list[i-1] >= h_lb_list[i-1]]
            constraints += [x[i] @ G_list[i-1] <= h_ub_list[i-1]]
            constraints += [x[i] <= box_constraints[1]]
            constraints += [x[i] >= box_constraints[0]]
            constraints += [cp.norm(z[i], p='fro') <= lam[i-1]]
        constraints += [cp.norm((y-x[t]), p='fro') <= b]
        
        prob = cp.Problem(cp.Minimize( cp.sum(lam) + b - z_k @ cp.vec(y, order='C')), constraints) # cp.vec here flattens in row-major ('C') 
        # prob.solve(solver='MOSEK', mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}) 
        prob.solve(solver = 'MOSEK', mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':   'MSK_SOLVE_DUAL'})
        
        
        if prob.status in ["infeasible", "unbounded"]:
            print('Support function failed to compute due to {reason}'.format(prob.status))
        else:
            Y[k] = y.value
            val[k] = prob.value
            
    return Y, val

def functional_steiner(G_list: list[np.ndarray], h_lb_list: list[np.ndarray], h_ub_list: list[np.ndarray], n: int, t:int, box_constraints: list[float] = [-2. , 3.]) -> tuple[np.ndarray, np.ndarray]:
    ''' Compute the Steiner point of the sublevel set of the current work function after observing x(t+1)
    
    Args
    - G_list, h_lb_list, h_ub_list: time-concatenated half space constraints from trajectory data
    - n: tuple, in the form of (nx, nu) where nx and nu are the dimension of the state and control respectively
    - t: current time step
    - box_constriants: lower and upper bound on the value of each entry of A and B
    
    Returns Theta = [A, B]
    '''
    d = n[0] * np.sum(n)
    N = np.max([5*t**2, 500]) # number of samples
    Z = sample_sphere(d=d , N=N) # Z is a N by d matrix with N iid sampled R^d unit vector
    _, val = socp(Z=Z, t=t, n=n, G_list=G_list, h_lb_list=h_lb_list, h_ub_list=h_ub_list, box_constraints=box_constraints) 
    P =  Z * val[:,None]
    Theta = -d * np.mean(P, axis=0) # This is a vectorized form 

    return Theta.reshape((n[0], np.sum(n)), order='C') # Theta[:, :nx], Theta[:, nx:] = A, B



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
    N = int(np.min( [((d+1) ** 2 + R**2) / (eps**2 * delta) , 500])) # number of samples
    Z = sample_sphere(d=d , N=N) # Z is a N by d matrix with N iid sampled R^d unit vector
    P = compute_support(Z=Z, eps=eps/d, t=t, n=n, r=R, G_list=G_list, h_lb_list=h_lb_list, h_ub_list=h_ub_list)
    Theta = np.mean(P, axis=0) # Theta[:, :nx], Theta[:, nx:] = A, B
    return Theta

def proj(Theta_hat, t, n, G_list, h_lb_list, h_ub_list, get_distance=False, box_constraints: list[float] = [-2. , 3.]):
    # closed-form computation of the projection on to the intersection of the hyperplane
    Theta = cp.Variable((n[0], np.sum(n)))
    constraints = [Theta @ G_list[-1] >= h_lb_list[-1]]
    constraints += [Theta @ G_list[-1] <= h_ub_list[-1]]
    constraints += [Theta <= box_constraints[1]]
    constraints += [Theta >= box_constraints[0]]
    prob = cp.Problem(cp.Minimize(cp.norm(Theta - Theta_hat, p='fro')), constraints)
    prob.solve(solver = 'MOSEK', mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM':   'MSK_SOLVE_DUAL'})
    
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

