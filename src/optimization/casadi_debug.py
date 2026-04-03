import casadi
try:
    # This specifically tries to load the OSQP engine
    solver = casadi.conic('solver', 'osqp', {'x': casadi.MX.sym('x')})
    print("SUCCESS: OSQP is working!")
except Exception as e:
    print(f"STILL FAILING: {e}")