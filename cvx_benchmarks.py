import cvxpy
import scipy
import time
import numpy as np
import matplotlib
matplotlib.use('Agg') #for plotting w/out GUI on server
from matplotlib import pyplot as plt

filetype = "png" # "png" or "pdf"
figdpi = 200 #int: DPI of the PDF output file
output_filename = "benchmark"

def savePlot():
    plt.savefig(output_filename + "." + filetype, dpi=figdpi, format=filetype)
    print("Plot saved to " + output_filename + "." + filetype)

def compile_time(prob, solver='ECOS'):
    # Return the amount of time needed to call prob.solve(max_iters=1).
    #
    # Because we aren't giving the solver enough iterations to actually solve the problem,
    # cvxpy will throw an error saying that the solver "failed." We don't actually
    # care that the solver only had one iteration to work with, we only care that the 
    # solver read the problem data.
    start = time.time()
    try:
        prob.solve(solver=solver, max_iters=1)
        end = time.time()
    except cvxpy.SolverError:
        end = time.time()
    return end-start


def gen_A_b_c(m, n):
    # Generate data for a linear program which is guaranteed to be feasible.
    A = scipy.sparse.rand(m, n, density=0.1).toarray()
    x0 = np.random.rand(n, 1)
    b = np.dot(A, x0)
    c = np.random.randn(n, 1)
    return A, b, c


def scalarized_problem(A, b, c):
    # Generate a cvxpy Problem object of the planned form, where 
    #   (1) upper and lower bounds are specified on each element of x individually, and
    #   (2) equality constraints A x = b are specified one row at a time.
    rank = 30
    m,n = A.shape

    x = cvxpy.Variable([n,n])
    p = c

    kappa = 0.2
    beta = 2/rank
    epsilon = kappa * (1-beta)/((rank-1)*(1-beta)+1)


    objective = cvxpy.Minimize(p.T @ cvxpy.reshape(cvxpy.diag(x),(n,1)))
    constraints = [x >= 0]
    for i in range(0, n):
        constraints += [
        cvxpy.norm((np.reshape(A[:,i], (m,1)) - A @ cvxpy.reshape(x[:,i], (n,1))), p=1) <= 2*epsilon,
        x[i,i] <= 1 ]
        for j in range(0,n):
            constraints += [x[i,j] <= x[i,i]]

    constraints += [cvxpy.trace(x) == rank]
    prob = cvxpy.Problem(objective, constraints)
    return prob

ms1 = [250, 500, 1000, 1500, 2000]
n = 250
ts1 = list()
tv1 = list()
for m in ms1:
    A, b, c = gen_A_b_c(m, n)
    print(m)
    # p = vectorized_problem(A, b, c)
    # tv1.append(compile_time(p))
    p = scalarized_problem(A, b, c)
    t = compile_time(p)
    print(t)
    ts1.append(t)
    print()

plt.plot(ms1, ts1, 'r', label='scalarized')
# plt.plot(ms1, tv1, 'b', label='vectorized')
plt.legend(loc='upper left')
savePlot()
plt.show()

# def vectorized_problem(A, b, c):
#     # Generate a cvxpy Problem object of the planned form, where 
#     #   (1) upper and lower bounds are specified on the entire vector "x" at once, and
#     #   (2) equality constraints A x = b are specified by a single cvxpy Constraint object.
#     x = cvxpy.Variable(shape=(A.shape[1], 1))
#     constraints = [0 <= x, x <= 1, A * x == b]
#     objective = cvxpy.Minimize(c.T * x)
#     prob = cvxpy.Problem(objective, constraints)
#     return prob