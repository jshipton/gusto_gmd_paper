from firedrake import (PeriodicIntervalMesh, SpatialCoordinate, Constant,
                       FunctionSpace, Function, TestFunction, TrialFunction,
                       LinearVariationalProblem, LinearVariationalSolver,
                       exp, dx)
from firedrake.fml import all_terms, drop, keep, replace_subject, subject
from firedrake.pyplot import plot
from gusto.core import time_derivative, transport, diffusion
import matplotlib.pyplot as plt

L = 1
nx = 20
mesh = PeriodicIntervalMesh(nx, L)
x = SpatialCoordinate(mesh)[0]

V = FunctionSpace(mesh, "Lagrange", 2)
d = 0.05
qn = Function(V).interpolate(exp(-((x-0.5*L)/d)**2))
plot(qn)
plt.show()
c = Constant(0.5)
kappa = Constant(0.01)

# Equation
q = Function(V)
v = TestFunction(V)
residual = (
    time_derivative(q * v * dx) +
    transport(c * q.dx(0) * v * dx) +
    diffusion(kappa * q.dx(0) * v.dx(0) * dx)
    )
residual = subject(residual, q)

# Set up solvers
q_trial = TrialFunction(V)
dt = 0.01

# Set up solver for explicit transport step
lhs_explicit = residual.label_map(lambda t: t.has_label(time_derivative),
                                  map_if_true=keep,
                                  map_if_false=drop)
lhs_explicit = lhs_explicit.label_map(all_terms,
                                      map_if_true=replace_subject(q_trial))
rhs_explicit = residual.label_map(lambda t: t.has_label(diffusion),
                                  map_if_true=drop)
rhs_explicit = rhs_explicit.label_map(lambda t: t.has_label(time_derivative),
                                      map_if_false=lambda t: -dt*t)
rhs_explicit = rhs_explicit.label_map(all_terms,
                                      map_if_true=replace_subject(qn))
qstar = Function(V)
prob_explicit = LinearVariationalProblem(lhs_explicit.form,
                                         rhs_explicit.form,
                                         qstar)
solver_explicit = LinearVariationalSolver(prob_explicit)

# Set up solver for implicit transport step
lhs_implicit = residual.label_map(lambda t: t.has_label(transport),
                                  map_if_true=drop)
lhs_implicit = lhs_implicit.label_map(lambda t: t.has_label(time_derivative),
                                      map_if_false=lambda t: dt*t)
lhs_implicit = lhs_implicit.label_map(all_terms,
                                      map_if_true=replace_subject(q_trial))
rhs_implicit = residual.label_map(lambda t: t.has_label(time_derivative),
                                      map_if_false=drop)
rhs_implicit = rhs_implicit.label_map(all_terms,
                                      map_if_true=replace_subject(qstar))
qnp1 = Function(V)
prob_implicit = LinearVariationalProblem(lhs_implicit.form,
                                         rhs_implicit.form,
                                         qnp1)
solver_implicit = LinearVariationalSolver(prob_implicit)

# Timestepping loop
tmax = 0.5
t = 0
while t < tmax:
    t += dt
    solver_explicit.solve()
    solver_implicit.solve()
    qn.assign(qnp1)
    plot(qn)
    plt.show()
