"""
The Williamson 5 shallow water test, solved with the SIQN stepper at refinement
level 6.
"""

from gusto import *
from firedrake import (IcosahedralSphereMesh, CubedSphereMesh,
                       SpatialCoordinate, as_vector, pi, sqrt, min_value, cos,
                       atan)

icos = False
vector_invariant = True

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

day = 24*60*60
dt = 112.5
tmax = 50*day
ndumps = 50
ref_level = 6
ncells = 90

# shallow water parameters
R = 6371220.
H = 5960.

parameters = ShallowWaterParameters(H=H)

# ------------------------------------------------------------------------ #
# Set up model objects
# ------------------------------------------------------------------------ #

# Domain
if icos:
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=2)
    domain = Domain(mesh, dt, 'BDM', 1)
    dx = 2*pi*R*cos(atan(1/2))/(5*2**ref_level)
    print(dx)

else:
    mesh = GeneralCubedSphereMesh(radius=R, num_cells_per_edge_of_panel=ncells,
                                  degree=2)
    domain = Domain(mesh, dt, 'RTCF', 1)
    dx = 10000/ncells
    print(dx)

x = SpatialCoordinate(mesh)

# Equation
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R
lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])
R0 = pi/9.
R0sq = R0**2
lamda_c = -pi/2.
lsq = (lamda - lamda_c)**2
theta_c = pi/6.
thsq = (theta - theta_c)**2
rsq = min_value(R0sq, lsq+thsq)
r = sqrt(rsq)
bexpr = 2000 * (1 - r/R0)

if vector_invariant:
    u_transport_option = 'vector_invariant_form'
else:
    u_transport_option = 'vector_advection_form'

eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=bexpr,
                             u_transport_option=u_transport_option)

# estimate core count for Pileus
print(f'Estimated number of cores = {eqns.X.function_space().dim() / 50000} ')

# I/O
dirname = "williamson_5_icos%s_vector_invariant%s" % (icos, vector_invariant)
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(
    dirname=dirname,
    dumplist_latlon=['D'],
    dump_nc=True,
    dump_vtus=True,
    dumpfreq=dumpfreq
)
diagnostic_fields = [Sum('D', 'topography'), RelativeVorticity(),
                     PotentialVorticity()]
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
u_max = 20.
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
g = parameters.g
Rsq = R**2
Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

u0.project(uexpr)
D0.interpolate(Dexpr)

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)
