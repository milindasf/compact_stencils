"""
compute the finite difference coefficients for a given grid spacing
using symbolic python
1. Taylor expansions 
2. Lagrange polynomial approach 
    for compact finite differences, I think this will results a under determined system
"""
import sympy

"""
computes the Lagrange polynomials on specified grid points. 
"""
def lagrange_poly(x,order,i,xi=None):
    if xi==None:
        xi=sympy.symbols('x:%d'%(order+1))
    index = list(range(order+1))
    index.pop(i)
    return sympy.simplify(sympy.prod([(x-xi[j])/(xi[i]-xi[j]) for j in index]))


"""
forms the polynomial in x with interpolating specific values. 
"""
def interpolating_polynomial(x, order, xi=None, fxi=None):
    if xi==None:
        xi  = sympy.symbols('x:%d'%(order+1))
        fxi = sympy.symbols('f:%d'%(order+1))
    
    Lp = [lagrange_poly(x,order,i, xi=xi) for i in range(order+1)]
    S  = 0

    for i in range(order+1):
        S += Lp[i] * fxi[i]

    return S

"""
computes the fd stencil coefficients for uniform gird spacing 
using interpolating polynomials. 
"""
def fd_stencil(width, deriv, index=None):
    if index is None:
        index = [i for i in range(-width,width+1)]

    h     = sympy.Symbol('h')
    x     = sympy.Symbol('x')

    xi    = tuple([ii * h for ii in index])
    fxi   = tuple([sympy.Symbol('f[%d]'%ii) for ii in index]) 
    a     = tuple([sympy.Symbol('a[%d]'%ii) for ii in index]) 

    fx    = interpolating_polynomial(x, len(index)-1, xi, fxi)

    W     =  sympy.diff(fx,x,deriv).subs(x,0)
    
    for i, ii in enumerate(index):
        W = W - (a[i] * fxi[i])/(h** deriv)
    
    W = sympy.simplify(W)
    
    cc = list()
    for i, ii in enumerate(index):
        s = [(fxi[i],1)]
        for j, jj in enumerate(index):
            if j !=i:
                s.append((fxi[j],0))

        cc.append(W.subs(s,simultaneous=True))

    print(sympy.solve(cc, a, dict=True))

"""
Taylor expansion with a symbolic function for specified number of terms. 
""" 
def taylor_exp(f,x, h, n):
    return sum(h**i/sympy.factorial(i) * f(x).diff(x, i) for i in range(n))


"""
Computes both compact finite differences (CFD) and finite difference stencils (FD)
using Taylor expansion coefficient match. 

stencil assumed to be in the following form. 
A f'_{i-1} + f'_{i} + A f'_{i+1} = a(f_{i+1} - f_{i-1})/2h
when A=0 we get regular FD. 

idx_rhs : gird spacing in the approximations in the RHS of the equation
idx_lhs : gird spacing in the approximations in the LHS of the equation
c_rhs   : coefficient array for the RHS
c_lhs   : coefficient array for the LHS
deriv   : approximation derivative (i.e., 1st , 2nd derivatives etc)
order   : derivative order
"""

def fd_stencil_with_taylor(idx_rhs, idx_lhs, c_rhs, c_lhs, deriv, order):
    h     = sympy.Symbol('h')
    x     = sympy.Symbol('x')

    f     = sympy.Function('f')
    g     = lambda x: f(x).diff(x,deriv) #if isinstance(t, sympy.Symbol) else f(x).diff(x,deriv).subs(x, t) #f(x).diff(x,deriv)
    
    
    xi    = idx_rhs 
    yi    = idx_lhs 

    #print(a1)
    #print(a2)

    rhs = 0
    for i, hh in enumerate(xi):
        rhs += c_rhs[i] * taylor_exp(f, x, hh, order+1) 
    
    lhs = 0
    for i, hh in enumerate(yi):
        lhs +=c_lhs[i] * taylor_exp(g, x, hh, order) 

    # print("lhs")
    # sympy.pprint(sympy.simplify(lhs))
    # print("rhs")
    # sympy.pprint(sympy.simplify(rhs))
    #sympy.pprint(taylor_exp(g, x, h, order))
    
    W = sympy.simplify(lhs-rhs) 
    #W = sympy.simplify(W)
    W = sympy.expand(W)
    #sympy.pprint(W)
    
    cc = list()
    W1 = W.subs(h,1)
    
    #print("W1")
    #sympy.pprint(W1)
    for i in range(order+1):
        expr = W1.collect(f(x).diff(x,i))
        expr = expr.coeff(f(x).diff(x,i))
        if expr!=0:
            cc.append(expr)
    #print(cc)

    return sympy.solve(cc,dict=True), W

x         = sympy.Symbol('x')
h         = sympy.Symbol('h')
a1,a2,a3  = sympy.symbols(('a1','a2','a3'))
A, B      = sympy.symbols(('A','B'))

# regular finite-differences

idx_rhs = [-3*h, -2*h, -h, 0, h, 2*h, 3*h]
idx_lhs = [0]
c_lhs   = [-a3/(h), -a2/(h), -a1/(h), 0, a1/(h), a2/(h), a3/(h)]
c_rhs   = [1]

cc, W = fd_stencil_with_taylor(idx_rhs, idx_lhs, c_lhs, c_rhs, 1, 6)
print("6th order central FD")
print(cc)
print("")

idx_rhs = [0, h, 2*h, 3*h, 4*h, 5*h, 6*h]
idx_lhs = [0]
aa      =list(sympy.symbols('a0:7'))
c_lhs   = [a/h for a in aa]
c_rhs   = [1]

cc, W = fd_stencil_with_taylor(idx_rhs, idx_lhs, c_lhs, c_rhs, 1, 6)
print("6th order upwind FD")
print(cc)
print("")

# compact finite difference examples. 
idx_rhs    = [-3*h, -2*h, -h, 0, h, 2*h, 3*h]
idx_lhs    = [-2*h, -h, 0, h, 2*h]
c_rhs      = [-a3/(6*h), -a2/(4*h), -a1/(2*h), 0, a1/(2*h), a2/(4*h), a3/(6*h)]
c_lhs      = [B, A, 1, A, B]

cc, W   = fd_stencil_with_taylor(idx_rhs, idx_lhs, c_rhs, c_lhs, 1, 10)
print("10th order central CFD")
print(cc)
print("")

aa       = list(sympy.symbols('a0:8'))
c_rhs    = [a/h for a in aa]
c_lhs    = [1,A]
idx_rhs     = [i * h for i in range(len(aa))]
idx_lhs     = [0, h]

cc, W   = fd_stencil_with_taylor(idx_rhs, idx_lhs, c_rhs, c_lhs, 1, 8)
print("8th order central CFD, left boundary, i=0")
print(cc)
print("")


aa       = list(sympy.symbols('a0:4'))
c_rhs    = [a/h for a in aa]
c_lhs    = [A,1,A]
idx_rhs  = [i * h for i in range(len(aa))]
idx_lhs  = [-h, 0, h]

cc, W   = fd_stencil_with_taylor(idx_rhs, idx_lhs, c_rhs, c_lhs, 1, 4)
print("8th order central CFD, left boundary, i=1")
print(cc)
print("")


