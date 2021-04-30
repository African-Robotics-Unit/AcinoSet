import pyomo.environ as pyo


def create_new_model():
    model = pyo.ConcreteModel()
    model.x = pyo.Var([1,2], domain=pyo.NonNegativeReals)
    model.OBJ = pyo.Objective(expr = 2*model.x[1] + 3*model.x[2])
    model.Constraint1 = pyo.Constraint(expr = 3*model.x[1] + 4*model.x[2] >= 1)
    return model

# let's first solve it using plain ipopt, to check that it's working:
from pyomo.opt import SolverFactory
opt = SolverFactory('ipopt', executable='/tmp/build/bin/ipopt')
opt.solve(create_new_model())

# and then solve it again with HSL MA86:
opt.options['linear_solver'] = 'ma86'
opt.options['OF_ma86_scaling'] = 'none'  # a random parameter
opt.solve(create_new_model(), tee=True)