import numpy as np
import mosek
import cvxpy as cp
import control
seed = 88
np.random.seed(seed)

def support(Z, d, G, h_lb, h_ub , W, lb, ub):
    # Given a set of random vectors concatenated into Z, compute the support 
    # function evaluated at each of the random vectors
    # X = np.vstack([np.kron(np.eye(2), xt) for xt in x[:-1]])
    # U = np.vstack([np.kron(np.eye(2),ut ) for ut in u])
    # G = np.hstack((X,U))
    # h_ub = np.vstack([x_.reshape((2,1))  for x_ in x[1:]]) + W
    # h_lb = np.vstack([x_.reshape((2,1))  for x_ in x[1:]]) - W
    
    infeasible = False
    Y = np.zeros((Z.shape[0], d))
    y = cp.Variable((d,1))
    constraints = [y <= ub*np.ones((d,1))]
    constraints += [y >= lb*np.ones((d,1))]
    constraints += [G@y <= h_ub]
    constraints += [G@y >= h_lb]

    for i, zi in enumerate(Z):
        prob = cp.Problem(cp.Minimize(zi.transpose()@y), constraints)
        prob.solve()
        if prob.status in ["infeasible", "unbounded"]:
          infeasible = True
          break
        Y[i] = y.value.flatten()
        # print((h_ub - W) - G@y.value)
    return Y, infeasible

def sample_sphere(d,N=20):
    # Sample N random vectors on d dimensional sphere
    # dimension d and makes 10 samples
    x = np.random.normal(size=(N, d))
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
    return x


def get_parameter(G, h_lb, h_ub, nx, nu, W, lb, ub ):
    # theta = [A11,A12,A21,A22,B1,B2]
    # https://arxiv.org/pdf/1905.11877.pdf Algorithm 3.
    # d: dimension of the parameter space
    # r: Radius of a ball that contains the parameter polytope
    # epsilon: Steiner point approximation accuracy
    # x_list,u_list: state and control trajectory
    # W,lb,ub: disturbance bound, lower and upper bound on the parameter space.
    d = nx*(nx+nu)
    Z = sample_sphere(d) # Z is a N by d matrix with N iid sampled R^d gaussian vector
    P, infeasible = support(Z,d,G, h_lb, h_ub ,W,lb,ub) # P is a N by nx(nx+nu) matrix with N sampled points in the polytope
    if infeasible:
      A, B = None, None
    else:
      theta =  np.mean(P,0) # summing over all rows and divide by number of cols to get a row vector of size nx(nx+nu).
      A = theta[:nx**2].reshape((nx,nx))
      B = theta[nx**2:].reshape((nx,nu))

    return A, B, infeasible


def get_K(A,B,nx,nu):
    # Compute the optimal LQR controller for a given A,B matrix with identity cost matrices
    _,_,K = control.dare(A, B, np.eye(nx), np.eye(nu) )
    return K

def consistent(A, B, x, u, nx, nu, W, lb, ub ):
    # function to check whether a given A,B model is consistent with observations x and u trajectory
    X = np.vstack([np.kron(np.eye(nx), xt) for xt in x[:-1]])
    U = np.vstack([np.kron(np.eye(nx),ut ) for ut in u])
    G = np.hstack((X,U))
    h_ub = np.vstack([x_.reshape((nx,1))  for x_ in x[1:]]) + W
    h_lb = np.vstack([x_.reshape((nx,1))  for x_ in x[1:]]) - W
    y = np.hstack((A.flatten(), B.flatten())).reshape((nx*(nx+nu),1))
    consist = (G@y <= h_ub).all() and (G@y >= h_lb).all()
    return consist, G, h_lb, h_ub 