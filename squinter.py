# squinter.py
# snap-together squeezed light interferometer circuits
# generalizes mziexample() and boostedexample()

# design rules that make everything one mechanism:
#   1. every input is a coherent state (α=0 vacuum), so ev() = normal-order + substitute stays exact
#   2. all squeezing is a circuit element, never an input state: "squeezed input" = squeezer on a fresh
#      vacuum line, exact because ⟨ξ|O|ξ⟩ = ⟨0|S†(ξ)OS(ξ)|0⟩
#   3. basis is the interleaved vector v = (a1,a1†,a2,a2†,...); every element is a matrix, v → Ev;
#      elements are specified by their action on annihilation ops, conjugate rows auto-filled
#   4. modes may be appended mid-circuit: M ← M⊕I₂ first (a late mode just idled until now)
#   5. loss = beamsplitter to a fresh vacuum ancilla that is never detected (discarding is free in
#      normal-ordered vacuum expectation — the partial trace happens automatically)

from sympy import sqrt,exp,sin,cos,sinh,cosh,tanh,I,pi,Rational
from sympy import sympify,symbols,diff,conjugate,simplify,pprint,cancel,expand,expand_trig
from sympy import Matrix,eye,zeros
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.operatorordering import normal_ordered_form

def nof(expr):
    return normal_ordered_form(expr.expand(),independent=True) # wrong result if not expanded! e.g. (a+b)*(at+bt) is unchanged

def mziexample(α=None,β=None,ϕ0=None): # α,β = input amplitudes, ϕ0 = phase shift
    α = α if α is not None else symbols('α',real=True)
    β = β if β is not None else symbols('β',real=True)
    ϕ = symbols('ϕ',real=True)
    αc,βc = α.conjugate(),β.conjugate()
    a,b = BosonOp('a'),BosonOp('b') # a,b are two input channels, a=top path, b=bottom path
    at,bt = a.adjoint(),b.adjoint()
    BS = sqrt(Rational(1,2))*Matrix([[1,-1],[1,1]]) # 50/50 beam splitter
    PH = Matrix([[exp(I*ϕ),0],[0,1]]) # phase shifters
    v = Matrix([a,b]) # input state
    A,B = BS*PH*BS*v # output state
    def ev(expr,subs={a:α,at:αc,b:β,bt:βc}): # expectation value
        return nof(expr).subs(subs)
    # evaluate phase noise at output port A
    N = ev(Dagger(A)*A) # print('N',N)
    NN = ev(Dagger(A)*A*Dagger(A)*A) # print('NN',NN)
    ΔNΔN = NN - N**2 # print('ΔN²',ΔNΔN)
    dNdϕ = diff(N,ϕ) # print('dN/dϕ',dNdϕ)
    ΔϕΔϕ = (ΔNΔN/dNdϕ**2).rewrite(exp,cos,sin).expand().simplify() # print('Δϕ²',ΔϕΔϕ)
    ΔϕΔϕ = ΔϕΔϕ if ϕ0 is None else ΔϕΔϕ.subs(ϕ,ϕ0)
    print('Δϕ² = ΔN²/(dN/dϕ)² =')
    pprint(ΔϕΔϕ)
    return ΔϕΔϕ

def boostedexample(α=None,β=None,r=None,ϕ0=None): # coherent light boosted SU(1,1) interferometer
    α = α if α is not None else symbols('α',real=True)
    β = β if β is not None else symbols('β',real=True)
    r = r if r is not None else symbols('r',positive=True)
    ϕ = symbols('ϕ',real=True)
    αc,βc = α.conjugate(),β.conjugate()
    a,b = BosonOp('a'),BosonOp('b') # a,b are two input channels, a=signal input, b=idler input
    at,bt = a.adjoint(),b.adjoint()
    µ,𝜈 = cosh(r),sinh(r)
    v = Matrix([a,at,b,bt]) # input state
    OPA1 = Matrix([[µ,0,0,+𝜈],[0,µ,+𝜈,0],[0,+𝜈,µ,0],[+𝜈,0,0,µ]])
    PH = Matrix([[exp(I*ϕ),0,0,0],[0,exp(-I*ϕ),0,0],[0,0,1,0],[0,0,0,1]])
    OPA2 = Matrix([[µ,0,0,-𝜈],[0,µ,-𝜈,0],[0,-𝜈,µ,0],[-𝜈,0,0,µ]])
    A,At,B,Bt = OPA2*PH*OPA1*v
    def ev(expr,subs={a:α,at:αc,b:β,bt:βc}): # expectation value
        return nof(expr).subs(subs)
    N = ev(At*A+Bt*B) # print('N',N.subs(ϕ,ϕ0))
    dNdϕ = diff(N,ϕ) # print('dN/dϕ',dNdϕ.subs(ϕ,ϕ0))
    NN = ev((At*A+Bt*B)**2) # print('NN',NN.subs(ϕ,ϕ0))
    ΔNΔN = NN - N**2 # print('ΔN²',ΔNΔN.subs(ϕ,ϕ0))
    ΔϕΔϕ = (ΔNΔN/dNdϕ**2).rewrite(exp,cos,sin).expand()#.simplify() # 
    ΔϕΔϕ = ΔϕΔϕ if ϕ0 is None else ΔϕΔϕ.subs(ϕ,ϕ0)
    print('Δϕ² = <ΔN²>/(d<N>/dϕ)² =',ΔϕΔϕ.subs(α,0).subs(β,0).limit(ϕ,0)) # equals 1/sinh²(2r) ref:Plick15,pg5
    return ΔϕΔϕ
    # SU(1,1) boosted interferometer notes
    #  - phase of the SU(1,1) interferometer depends on sum of the phases φs + φi in each path
    #    the signal and idler can share the same modulation path, halves Vπ
    #  - can also phase modulate the pump alone. or all three with pump dominating but probably not efficiently. net phase is ϕs+ϕi-ϕp
    #  - R = 1-η + η*exp(±2√(P ηSHG))
    #    ηSHG from R fit in many papers was usually much less than measured ηSHG (Kashiwazaki20 excepted)
    #    (due to multimode factor of 2?)
    #  - Internal loss has same effect as in squeezing, but external loss ~no little effect
    #  - but boosted has exponentially better phase sensitivity than squeezing alone

class Squinter:
    def __init__(self):
        self.labels,self.amps,self.M = [],{},zeros(0,0)
    def index(self,label):
        return self.labels.index(label)
    def input(self,label,α=0): # append a mode in coherent state α (default vacuum)
        assert label not in self.labels
        self.labels += [label]
        self.amps[label] = sympify(α)
        n = len(self.labels)
        M = eye(2*n)
        if n>1: M[:2*n-2,:2*n-2] = self.M
        self.M = M
        return self
    def element(self,modes,A,B=None,real=False):
        # builds a matrix E that when multiplied by an input vector returns the output vector.
        # while E operates on all modes, only those in the list 'modes' will be changed.
        # aᵢ' = Σⱼ Aᵢⱼaⱼ + Bᵢⱼaⱼ†. Conjugate rows are auto-filled. You only ever specify A and B — the action on the annihilation operators. The creation-operator rows are filled in automatically by conjugation.
        # B=None ⇒ passive element. Defaulting B to a zero matrix gives aᵢ' = Σⱼ Aᵢⱼ aⱼ with no a/a† mixing. Number-conserving. That's what phase and bs use.
        # Supplying a nonzero B is what makes an element active (squeezing, OPA), mixing annihilation and creation operators.
        m = len(modes)
        A = Matrix(A)
        B = Matrix(B) if B is not None else zeros(m)
        cj = (lambda x:x) if real else conjugate # real=True if sympy can't prove entries real (e.g. sqrt(1-η))
        ii = [self.index(l) for l in modes]
        E = eye(2*len(self.labels))
        for p in range(m):
            for q in range(m):
                E[2*ii[p]  ,2*ii[q]  ] = A[p,q]
                E[2*ii[p]  ,2*ii[q]+1] = B[p,q]
                E[2*ii[p]+1,2*ii[q]  ] = cj(B[p,q])
                E[2*ii[p]+1,2*ii[q]+1] = cj(A[p,q])
        self.M = E*self.M
        return self
    def phase(self,mode,ϕ):
        return self.element([mode],[[exp(I*ϕ)]])
    def bs(self,m1,m2,θ=pi/4): # θ=π/4: a'=(a−b)/√2, b'=(a+b)/√2, same convention as mziexample()
        return self.element([m1,m2],[[cos(θ),-sin(θ)],[sin(θ),cos(θ)]])
    def beamsplitter(self,m1,m2,θ=pi/4):
        return self.bs(m1,m2,θ=θ)
    def squeeze(self,mode,r,θ=0): # S†(ξ)aS(ξ) = a·cosh r − e^{iθ}a†·sinh r, ξ = r·e^{iθ}
        return self.element([mode],[[cosh(r)]],[[-exp(I*θ)*sinh(r)]])
    def opa(self,m1,m2,r,θ=0): # a' = a·cosh r + e^{iθ}b†·sinh r and a↔b; θ=0,+r matches OPA1,
        µ,ν = cosh(r),exp(I*θ)*sinh(r) # inverse is r→−r (or θ→θ+π); θ tracks pump phase
        return self.element([m1,m2],[[µ,0],[0,µ]],[[0,ν],[ν,0]])
    def loss(self, mode, η, label=None): # power transmission η
        if label is None: # auto-label 'loss','loss2','loss3',… if label not specified
            n = sum([1 for l in self.labels if l.startswith('loss')])
            label = f'loss{n+1:d}' if n else 'loss'
        self.input(label, 0)
        return self.element([mode, label], [[sqrt(η), sqrt(1-η)], [-sqrt(1-η), sqrt(η)]], real=True)
    def mixsqueezed(self,mode,label,r,θ=0,θbs=pi/4): # tap squeezed vacuum into an existing beam on a BS
        return self.input(label,0).squeeze(label,r,θ).bs(mode,label,θbs)
    # ---- outputs, expectation values, sensitivity ----
    def inputvector(self): # build the input vector [a₁  a₁†  a₂  a₂†  …]
        ops = []
        for l in self.labels:
            a = BosonOp(l)
            ops += [a,Dagger(a)]
        return Matrix(ops)
    def outputmode(self,mode):
        # returns the annihilation operator of a particular ouput mode, in terms of input operators
        # we never need the output creation operator row explicitly since it is the same thing daggered
        # (but we do need them for each new element multiplication)
        return (self.M[2*self.index(mode),:] * self.inputvector())[0]
    def N(self,*modes): # total photon number over the listed output ports
        return sum(Dagger(self.outputmode(l))*self.outputmode(l) for l in modes)
    def quad(self,mode,θlo=0): # homodyne quadrature X_θ at an output port
        A = self.outputmode(mode)
        return (exp(-I*θlo)*A+exp(I*θlo)*Dagger(A))/sqrt(2)
    def Nstats(self,weights):
        # ⟨O⟩ and ⟨ΔO²⟩ for O = Σₖ wₖ aₖ†aₖ directly from gaussian pair moments — no nof()
        # aₖ' = uₖ+fₖ with n=⟨fₖ†f_l⟩, ñ=⟨fₖf_l†⟩, m=⟨fₖf_l⟩, m̄=⟨fₖ†f_l†⟩; wick gives
        # Cov(Nₖ,N_l) = ūₖū_l m + uₖū_l n + ūₖu_l ñ + uₖu_l m̄ + m·m̄ + n·ñ
        # conjugated coefficients are read from the dagger rows 2k+1 of M (honors element(real=True))
        # exact for any linear chain and coherent inputs; |α|⁴ never forms, so float-safe by construction
        if isinstance(weights,str): weights = [weights]
        if not isinstance(weights,dict): weights = {l:1 for l in weights}
        R = range(len(self.labels))
        A  = {l:[self.M[2*self.index(l)  ,2*i  ] for i in R] for l in weights}
        B  = {l:[self.M[2*self.index(l)  ,2*i+1] for i in R] for l in weights}
        Ac = {l:[self.M[2*self.index(l)+1,2*i+1] for i in R] for l in weights}
        Bc = {l:[self.M[2*self.index(l)+1,2*i  ] for i in R] for l in weights}
        αs,αcs = [self.amps[l] for l in self.labels],[conjugate(self.amps[l]) for l in self.labels]
        u  = {l:sum(A[l][i]*αs[i]+B[l][i]*αcs[i] for i in R) for l in weights}
        uc = {l:sum(Ac[l][i]*αcs[i]+Bc[l][i]*αs[i] for i in R) for l in weights}
        n  = lambda k,l: sum(Bc[k][i]*B[l][i] for i in R)  # ⟨fₖ†f_l⟩
        ñ  = lambda k,l: sum(A[k][i]*Ac[l][i] for i in R)  # ⟨fₖf_l†⟩
        m  = lambda k,l: sum(A[k][i]*B[l][i] for i in R)   # ⟨fₖf_l⟩
        mc = lambda k,l: sum(Bc[k][i]*Ac[l][i] for i in R) # ⟨fₖ†f_l†⟩
        mean = sum(w*(uc[l]*u[l]+n(l,l)) for l,w in weights.items())
        var = sum(wk*wl*(uc[k]*uc[l]*m(k,l) + u[k]*uc[l]*n(k,l) + uc[k]*u[l]*ñ(k,l) + u[k]*u[l]*mc(k,l)
                         + m(k,l)*mc(k,l) + n(k,l)*ñ(k,l))
                  for k,wk in weights.items() for l,wl in weights.items())
        return mean.expand(),var.expand()
    def ev(self,expr): # expectation value
        subs = {}
        for l in self.labels:
            a = BosonOp(l)
            subs[a],subs[Dagger(a)] = self.amps[l],conjugate(self.amps[l])
        return nof(expr).subs(subs)
    # def dphisqr(self,obs,ϕ,ϕ0=None,simplifyit=False): # Δϕ² = ⟨ΔO²⟩/(d⟨O⟩/dϕ)²
    #     N = self.ev(obs)
    #     NN = self.ev(obs*obs)
    #     ΔNΔN = (NN - N**2).expand()
    #     dNdϕ = diff(N,ϕ)
    #     ΔϕΔϕ = (ΔNΔN/dNdϕ**2).rewrite(exp,cos,sin).expand() # same post-processing as mziexample()/boostedexample()
    #     ΔϕΔϕ = ΔϕΔϕ if not simplifyit else ΔϕΔϕ.simplify()
    #     return ΔϕΔϕ if ϕ0 is None else ΔϕΔϕ.subs(ϕ,ϕ0)
    def dphisqr(self,obs,ϕ,ϕ0=None,simplifyit=False): # Δϕ² = ⟨ΔO²⟩/(d⟨O⟩/dϕ)²
        # obs may be an operator expression (legacy nof path), or mode weights for O = Σ wₖNₖ:
        # a label, list of labels (wₖ=1), or dict {label:w} e.g. {'a':1,'b':-1} — gaussian path, no nof()
        if isinstance(obs,(str,dict,list,tuple,set)):
            N,ΔNΔN = self.Nstats(obs)
        else:
            N = self.ev(obs)
            NN = self.ev(obs*obs)
            ΔNΔN = (NN - N**2).expand() # expand cancels the |α|⁴ block exactly — float evaluation stays sane
        dNdϕ = diff(N,ϕ)
        ΔϕΔϕ = (ΔNΔN/dNdϕ**2).rewrite(exp,cos,sin) # .expand() # same post-processing as mziexample()/boostedexample()
        if simplifyit: ΔϕΔϕ = ΔϕΔϕ.simplify()
        return ΔϕΔϕ if ϕ0 is None else ΔϕΔϕ.subs(ϕ,ϕ0)
    def check(self): # commutation preserved ⇔ MΩMᵀ = Ω; returns the zero matrix iff consistent
        # Preserving [bi,bj†]=δij[b_i,b_j^\dagger]=\delta_{ij}
        # [bi,bj†]=δij, [bi,bj]=0[b_i,b_j]=0, [bi,bj]=0 requires AA†−BB†=I,ABT=BAT,AA^\dagger - BB^\dagger = I, \qquad AB^T = BA^T,AA†−BB†=I,ABT=BAT,
        # which in the interleaved basis is MΩM^T=Ω with Ω=⨁ᵢ(01−10) (\Omega = \bigoplusindex \begin{pmatrix}0&1\\-1&0\end{pmatrix}) — so a check() method verifies any chain you build.
        n = len(self.labels)
        Ω = zeros(2*n,2*n)
        for i in range(n):
            Ω[2*i,2*i+1],Ω[2*i+1,2*i] = 1,-1
        return simplify(self.M*Ω*self.M.T-Ω)
def tanhsimplify(e, r):
    from sympy import symbols, sin, cos, sinh, cosh, tanh, cancel, expand, expand_trig
    t = symbols('tanhr', positive=True)
    return expand_trig(e).subs(sinh(r), t*cosh(r)).subs([(sin(x)**2, 1-cos(x)**2) for x in {w.args[0] for w in e.atoms(cos)}]).expand().cancel().subs(t, tanh(r))
def sympy2float(expr, subs={}, dps=30, maxdps=480, rtol=1e-6, imtol=1e-4):
    # subs: {symbol: value}; exact sympy values preferred, ints/floats accepted
    # returns float(re) if converged and |im| ≤ imtol·|re|, else nan
    # for parameter scans lift the lambdify out and reuse it — compile dominates per-call cost
    import mpmath as mp
    from numpy import nan
    from sympy import sympify, lambdify, Expr
    expr = sympify(expr).subs({k:v for k,v in subs.items() if isinstance(v,Expr)})
    args = {k:v for k,v in subs.items() if not isinstance(v,Expr)}
    syms = sorted(expr.free_symbols, key=str)
    missing = [s for s in syms if s not in args]
    assert not missing, f'unresolved: {missing}'
    f = lambdify(syms, expr, 'mpmath')
    def run(prec): # conversion and evaluation both at prec so escalation is end to end
        with mp.workdps(prec):
            return mp.mpc(f(*[mp.mpmathify(args[s]) for s in syms]))
    try:
        zprev = run(dps)
        while True:
            dps *= 2
            z = run(dps)
            if abs(z-zprev) <= rtol*(abs(z)+abs(zprev)): break # also catches exact 0
            if dps >= maxdps: return nan # successive precisions disagree: report noise as nan
            zprev = z
    except ZeroDivisionError:
        return nan # singular working point, e.g. dN/dφ = 0
    return float(z.real) if abs(z.imag) <= imtol*(abs(z.real)+1e-99) else nan

def mzi(α=None,β=None,ϕ0=None): # reproduces mziexample()
    α = α if α is not None else symbols('α',real=True)
    β = β if β is not None else symbols('β',real=True)
    ϕ = symbols('ϕ',real=True)
    sq = Squinter().input('a',α).input('b',β).bs('a','b').phase('a',ϕ).bs('a','b')
    return sq.dphisqr(sq.N('a'),ϕ,ϕ0)

def boosted(α=None,β=None,ϕ0=None): # reproduces boostedexample()
    α = α if α is not None else symbols('α',real=True)
    β = β if β is not None else symbols('β',real=True)
    r = symbols('r',positive=True)
    ϕ = symbols('ϕ',real=True)
    sq = Squinter().input('a',α).input('b',β).opa('a','b',r).phase('a',ϕ).opa('a','b',-r)
    return sq.dphisqr(sq.N('a','b'),ϕ,ϕ0,simplifyit=0)

def degenboosted(α=None,θ=0,ϕ0=None):
    α = α if α is not None else symbols('α',real=True)
    r = symbols('r',positive=True)
    ϕ = symbols('ϕ',real=True)
    sq = Squinter().input('a',α).squeeze('a',r,θ).phase('a',ϕ).squeeze('a',-r,θ)
    return sq.dphisqr(sq.N('a'),ϕ,ϕ0,simplifyit=0)

def mzisqueezed(ϕ0=None): # Caves config: coherent + squeezed vacuum, difference detection
    α = symbols('α',real=True)
    r = symbols('r',positive=True)
    θ,ϕ = symbols('θ ϕ',real=True)
    sq = (Squinter().input('a',α)
                  .input('b',0).squeeze('b',r,θ)
                  .bs('a','b').phase('a',ϕ).bs('a','b'))
    return sq.dphisqr(sq.N('a')-sq.N('b'),ϕ,ϕ0,simplifyit=0)

def mzilossy(ϕ0=None): # coherent only, equal internal loss η both arms ⇒ exactly (lossless result)/η
    α = symbols('α',real=True)
    η = symbols('η',positive=True)
    ϕ = symbols('ϕ',real=True)
    sq = (Squinter().input('a',α).input('b',0)
                  .bs('a','b').phase('a',ϕ)
                  .loss('a',η).loss('b',η)
                  .bs('a','b'))
    return sq.dphisqr(sq.N('a'),ϕ,ϕ0)

if __name__ == '__main__':
    mziexample()
    mziexample(α=symbols('α'),β=0,ϕ0=pi/2)
    boostedexample()
    ϕ,θ = symbols('ϕ θ',real=True)
    print('MZI, coherent inputs (regression vs original mzi):')
    pprint(mzi())
    print('SU(1,1), vacuum seeded, ϕ→0 (expect 1/sinh²(2r), Plick15):')
    pprint(boosted(α=0,β=0).limit(ϕ,0))
    print('MZI, coherent + squeezed vacuum, N_a−N_b at ϕ0=π/2, θ=0 and θ=π:')
    dd = mzisqueezed(ϕ0=pi/2)
    pprint(dd.subs(θ,0).simplify())
    pprint(dd.subs(θ,pi).simplify())
    # checks: dd.subs(r,0) must reduce to the coherent case; at the better θ the large-α expansion
    # should give e^{−2r}/α² + O(sinh²r/α⁴) (Caves 1981)
    # symplectic sanity for any chain: assert c.check().is_zero_matrix
    # mzilossy(ϕ0=pi/2) should equal the lossless coherent result divided by η

