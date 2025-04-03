"""
The Galewsky jet shallow water test, solved with the SIQN stepper and using
both an icosahedral mesh and a cubed-sphere mesh and with both vector invariant
u transport and vector advective u transport.
"""

from gusto import *
from firedrake import IcosahedralSphereMesh, Constant, ge, le, exp, cos, \
    conditional, interpolate, SpatialCoordinate, VectorFunctionSpace, \
    Function, assemble, dx, pi, CubedSphereMesh

import numpy as np

icos = True
vector_invariant = True

# --------------------------------------------------------------------------- #
# Test case parameters
# --------------------------------------------------------------------------- #

day = 24*60*60
dt = 112.5
tmax = 6*day
ndumps = 6
ref_level = 6
ncells = 90

# Shallow water parameters
R = 6371220.
H = 10000.

parameters = ShallowWaterParameters(H=H)

# ------------------------------------------------------------------------ #
# Set up model objects
# ------------------------------------------------------------------------ #

# Domain
if icos:
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=2)
    domain = Domain(mesh, dt, 'BDM', 1)

else:
    mesh = GeneralCubedSphereMesh(radius=R, num_cells_per_edge_of_panel=ncells,
                                  degree=2)
    domain = Domain(mesh, dt, 'RTCF', 1)

x = SpatialCoordinate(mesh)

# Equation
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R

if vector_invariant:
    u_transport_option = 'vector_invariant_form'
else:
    u_transport_option = 'vector_advection_form'

eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr,
                             u_transport_option=u_transport_option)

# estimate core count for Pileus
print(f'Estimated number of cores = {eqns.X.function_space().dim() / 50000} ')

# I/O
perturb = True
if perturb:
    dirname = "galewsky_jet_perturbed_icos%s_vector_invariant%s" % (icos, vector_invariant)
else:
    dirname = "moist_galewsky_jet_unperturbed"
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          dumplist_latlon=['D',
                                           'PotentialVorticity',
                                           'RelativeVorticity'],
                          dump_nc=True,
                          dump_vtus=True,
                          chkptfreq=1)

diagnostic_fields = [PotentialVorticity(), RelativeVorticity(),
                     CourantNumber()]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "D")]
transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  transport_methods,
                                  num_outer=2, num_inner=2)

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #

u0 = stepper.fields('u')
D0 = stepper.fields('D')

lamda, phi, _ = lonlatr_from_xyz(x[0], x[1], x[2])

# expressions for meridional and zonal velocity
u_max = 80.0
phi0 = pi/7.
phi1 = pi/2. - phi0
en = np.exp(-4./((phi1-phi0)**2))
u_zonal_expr = (u_max/en)*exp(1/((phi - phi0)*(phi - phi1)))
u_zonal = conditional(ge(phi, phi0),
                      conditional(le(phi, phi1), u_zonal_expr, 0.), 0.)
u_merid = 0.0

# get cartesian components of velocity
uexpr = xyz_vector_from_lonlatr(u_zonal, 0, 0, x)

Rc = Constant(R)
g = Constant(parameters.g)


def D_integrand(th):
    # Initial D field is calculated by integrating D_integrand w.r.t. phi
    # Assumes the input is between phi0 and phi1.
    # Note that this function operates on vectorized input.
    from numpy import exp, sin, tan
    f = 2.0*parameters.Omega*sin(th)
    u_zon = (80.0/en)*exp(1.0/((th - phi0)*(th - phi1)))
    return u_zon*(f + tan(th)*u_zon/R)


def Dval(X):
    # Function to return value of D at X
    from scipy import integrate

    # Preallocate output array
    val = np.zeros(len(X))

    angles = np.zeros(len(X))

    # Minimize work by only calculating integrals for points with
    # phi between phi_0 and phi_1.
    # For phi <= phi_0, the integral is 0
    # For phi >= phi_1, the integral is constant.

    # Precalculate this constant:
    poledepth, _ = integrate.fixed_quad(D_integrand, phi0, phi1, n=64)
    poledepth *= -R/parameters.g

    angles[:] = np.arcsin(X[:, 2]/R)

    for ii in range(len(X)):
        if angles[ii] <= phi0:
            val[ii] = 0.0
        elif angles[ii] >= phi1:
            val[ii] = poledepth
        else:
            # Fixed quadrature with 64 points gives absolute errors below 1e-13
            # for a quantity of order 1e-3.
            v, _ = integrate.fixed_quad(D_integrand, phi0, angles[ii], n=64)
            val[ii] = -(R/parameters.g)*v

    return val


# Get coordinates to pass to Dval function
W = VectorFunctionSpace(mesh, D0.ufl_element())
X = interpolate(mesh.coordinates, W)
D0.dat.data[:] = Dval(X.dat.data_ro)

# Adjust mean value of initial D
C = Function(D0.function_space()).assign(Constant(1.0))
area = assemble(C*dx)
Dmean = assemble(D0*dx)/area
D0 -= Dmean
D0 += Constant(parameters.H)

# optional perturbation
if perturb:
    alpha = Constant(1/3.)
    beta = Constant(1/15.)
    Dhat = Constant(120.)
    phi2 = Constant(pi/4.)
    g = Constant(parameters.g)
    D_pert = Function(D0.function_space()).interpolate(Dhat*cos(phi)*exp(-(lamda/alpha)**2)*exp(-((phi2 - phi)/beta)**2))
    D0 += D_pert


def initialise_fn():
    u0 = stepper.fields("u")
    D0 = stepper.fields("D")

    u0.project(uexpr, form_compiler_parameters={'quadrature_degree': 12})

    X = interpolate(mesh.coordinates, W)
    D0.dat.data[:] = Dval(X.dat.data_ro)
    area = assemble(C*dx)
    Dmean = assemble(D0*dx)/area
    D0 -= Dmean
    D0 += parameters.H
    if perturb:
        D_pert.interpolate(Dhat*cos(phi)*exp(-(lamda/alpha)**2)*exp(-((phi2 - phi)/beta)**2))
        D0 += D_pert


initialise_fn()

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)
