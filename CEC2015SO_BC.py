import numpy as np
import numba


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Bent_Cigar(x):
    dim = x.shape[0]
    return x[0]*x[0] + 1e+6 * np.sum(np.square(x[1:dim]))


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_BentCigar(x, M, o):
    M = np.ascontiguousarray(M)
    return Bent_Cigar(np.dot(M, x - o[0:x.shape[0]])) + 100


@numba.cfunc("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def f1(x, M, o):
    return Shifted_Rotated_BentCigar(x, M, o)


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Discus(x):
    dim = x.shape[0]
    return 1e+6 * (x[0] ** 2) + np.sum(np.square(x[1:dim]))


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Discus(x, M, o):
    M = np.ascontiguousarray(M)
    return Discus(np.dot(M, x - o[0:x.shape[0]])) + 200


@numba.cfunc("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def f2(x, M, o):
    return Shifted_Rotated_Discus(x, M, o)


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Weierstrass(x):
    dim = x.shape[0]
    D = numba.prange(dim)
    kmax = numba.prange(21)
    res = 0
    t_2 = 0
    for i in D:
        t_1 = 0
        t_2 = 0
        for k in kmax:
            a_k = pow(0.5, k)
            b_k = pow(3, k)
            t_1 += a_k * np.cos(2 * np.pi * b_k * (x[i] + 0.5))
            t_2 += a_k * np.cos(np.pi * b_k)
        res += t_1
    res -= dim * t_2
    return res


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Weierstrass(x, M, o):
    M = np.ascontiguousarray(M)
    return Weierstrass(np.dot(M, 0.005 * (x - o[0:x.shape[0]]))) + 300


@numba.cfunc("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def f3(x, M, o):
    return Shifted_Rotated_Weierstrass(x, M, o)


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Schwefel(x):
    res = 0
    dim = x.shape[0]
    D = numba.prange(dim)
    x += 4.209687462275036e+002
    for i in D:
        if x[i] > 500:
            res -= (500.0 - np.fmod(x[i], 500)) * np.sin(pow(500.0 - np.fmod(x[i], 500), 0.5))
            tmp = (x[i] - 500.0) / 100
            res += tmp * tmp / dim
        elif x[i] < -500:
            res -= (-500.0 + np.fmod(np.fabs(x[i]), 500)) * np.sin(pow(500.0 - np.fmod(np.fabs(x[i]), 500), 0.5))
            tmp = (x[i] + 500.0) / 100
            res += tmp * tmp / dim
        else:
            res -= x[i] * np.sin(pow(np.fabs(x[i]), 0.5))
    res += 4.189828872724338e+002 * dim
    return res


@numba.njit("f8(f8[:],f8[:])", nogil=True, fastmath=True)
def Shifted_Schwefel(x, o):
    return Schwefel(10 * (x - o[0:x.shape[0]])) + 400


@numba.cfunc("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def f4(x, M, o):
    return Shifted_Schwefel(x, o)


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Katsuura(x):

    dim = x.shape[0]
    D = numba.prange(dim)
    rag = numba.prange(1, 33, 1)
    res = 1.0
    tmp3 = pow(dim, 1.2)

    for i in D:
        temp = 0.0
        for j in rag:
            tmp1 = pow(2.0, j)
            tmp2 = tmp1 * x[i]
            temp += np.fabs(tmp2 - np.floor(tmp2 + 0.5)) / tmp1
        res *= pow(1.0 + (i + 1) * temp, 10.0 / tmp3)
    tmp1 = 10.0 / dim / dim
    res = res * tmp1 - tmp1
    return res


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Katsuura(x, M, o):
    M = np.ascontiguousarray(M)
    return Katsuura(np.dot(M, 0.05 * (x - o[0:x.shape[0]]))) + 500


@numba.cfunc("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def f5(x, M, o):
    return Shifted_Rotated_Katsuura(x, M, o)


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def HappyCat(x):
    x = np.ascontiguousarray(x)
    x -= 1
    dim = x.shape[0]
    t1 = np.sum(x)
    t2 = np.dot(x, x)
    return pow(np.abs(t2 - dim), 0.25) + (0.5 * t2 + t1) / dim + 0.5


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_HappyCat(x, M, o):
    M = np.ascontiguousarray(M)
    return HappyCat(np.dot(M, 0.05 * (x - o[0:x.shape[0]]))) + 600


@numba.cfunc("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def f6(x, M, o):
    return Shifted_Rotated_HappyCat(x, M, o)


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def HGBat(x):
    x = np.ascontiguousarray(x)
    x -= 1
    dim = x.shape[0]
    t1 = np.sum(x)
    t2 = np.dot(x, x)
    return np.sqrt(np.abs((t2**2) - (t1**2))) + (0.5 * t2 + t1) / dim + 0.5


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_HGBat(x, M, o):
    M = np.ascontiguousarray(M)
    return HGBat(np.dot(M, 0.05 * (x - o[0:x.shape[0]]))) + 700


@numba.cfunc("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def f7(x, M, o):
    return Shifted_Rotated_HGBat(x, M, o)


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Expended_Griewank_plus_Rosenbrock(x):
    dim = x.shape[0]
    D_1 = numba.prange(dim - 1)
    res = 0
    x += 1.0
    for i in D_1:
        tmp1 = x[i] * x[i] - x[i + 1]
        tmp2 = x[i] - 1.0
        temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
        res += (temp * temp) / 4000.0 - np.cos(temp) + 1.0

    tmp1 = x[dim - 1] * x[dim - 1] - x[0]
    tmp2 = x[dim - 1] - 1.0
    temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
    res += (temp * temp) / 4000.0 - np.cos(temp) + 1.0
    return res


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Expended_Griewank_plus_Rosenbrock(x, M, o):
    M = np.ascontiguousarray(M)
    return Expended_Griewank_plus_Rosenbrock(np.dot(M, 0.05 * (x - o[0:x.shape[0]]))) + 800


@numba.cfunc("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def f8(x, M, o):
    return Shifted_Rotated_Expended_Griewank_plus_Rosenbrock(x, M, o)


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Expanded_Scaffer_F6(x):

    def Scaffer_F6(x1, x2):
        t = x1*x1 + x2*x2
        return 0.5 + (pow(np.sin(np.sqrt(t)), 2)-0.5) / ((1 + 0.001 * t) ** 2)

    dim = x.shape[0]
    D_1 = numba.prange(dim-1)
    res = Scaffer_F6(x[dim-1], x[0])
    for i in D_1:
        res += Scaffer_F6(x[i], x[i+1])
    return res


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Shifted_Rotated_Expanded_Scaffer_F6(x, M, o):
    M = np.ascontiguousarray(M)
    return Expanded_Scaffer_F6(np.dot(M, x - o[0:x.shape[0]])) + 900


@numba.cfunc("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def f9(x, M, o):
    return Shifted_Rotated_Expanded_Scaffer_F6(x, M, o)


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Rastrigin(x):
    return np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x) + 10)


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def High_Conditioned_Elliptic(x):
    dim = x.shape[0]
    n = numba.prange(0, dim, 1)
    res = 0
    for i in n:
        res += pow(10, 6 * i / (dim - 1)) * (x[i] ** 2)
    return res


@numba.njit("f8(f8[:],f8[:,:],f8[:],int32[:])", nogil=True, fastmath=True)
def Hybrid_1(x, M, o, s):
    dim = x.shape[0]
    x = np.dot(np.ascontiguousarray(M), x - o[0:dim])
    x = x[s]
    fit = Schwefel(10 * x[0:0.3*dim])
    fit += Rastrigin(0.0512 * x[0.3*dim:0.6*dim])
    fit += High_Conditioned_Elliptic(x[0.6*dim:dim])
    return fit + 1000


@numba.cfunc("f8(f8[:],f8[:,:],f8[:],int32[:])", nogil=True, fastmath=True)
def f10(x, M, o, s):
    return Hybrid_1(x, M, o, s)


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Griewank(x):
    x = np.ascontiguousarray(x)
    y = np.arange(1, x.shape[0]+1, 1)
    return 0.00025 * np.dot(x, x) - np.prod(np.cos(x / np.sqrt(y))) + 1


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Rosenbrock(x):
    dim = x.shape[0]
    n = numba.prange(dim - 1)
    res = 0
    x += 1
    for i in n:
        res += (100 * (((x[i]**2) - x[i+1]) ** 2) + (x[i]-1)**2)
    return res


@numba.njit("f8(f8[:],f8[:,:],f8[:],int32[:])", nogil=True, fastmath=True)
def Hybrid_2(x, M, o, s):
    dim = x.shape[0]
    x = np.dot(np.ascontiguousarray(M), x - o[0:dim])
    x = x[s]
    fit = Griewank(6 * x[0:0.2*dim])
    fit += Weierstrass(0.005 * x[0.2*dim:0.4*dim])
    fit += Rosenbrock(0.02048 * x[0.4*dim:0.7*dim])
    fit += Expanded_Scaffer_F6(x[0.7*dim:dim])
    return fit + 1100


@numba.cfunc("f8(f8[:],f8[:,:],f8[:],int32[:])", nogil=True, fastmath=True)
def f11(x, M, o, s):
    return Hybrid_2(x, M, o, s)


@numba.njit("f8(f8[:])", nogil=True, fastmath=True)
def Ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(np.square(x)))) - np.exp(np.mean(np.cos(2*np.pi*x))) + 20 + np.e


@numba.njit("f8(f8[:],f8[:,:],f8[:],int32[:])", nogil=True, fastmath=True)
def Hybrid_3(x, M, o, s):
    dim = x.shape[0]
    x = np.dot(np.ascontiguousarray(M), x - o[0:dim])
    x = x[s]
    fit = Katsuura(0.05 * x[0:0.1*dim])
    fit += HappyCat(0.05 * x[0.1*dim:0.3*dim])
    fit += Expended_Griewank_plus_Rosenbrock(0.05 * x[0.3*dim:0.5*dim])
    fit += Schwefel(10 * x[0.5*dim:0.7*dim])
    fit += Ackley(x[0.7*dim:dim])
    return fit + 1200


@numba.cfunc("f8(f8[:],f8[:,:],f8[:],int32[:])", nogil=True, fastmath=True)
def f12(x, M, o, s):
    return Hybrid_3(x, M, o, s)


@numba.njit("f8[:](f8[:],f8[:],int64[:])", nogil=True, fastmath=True)
def composition_omega(x, o, sig):
    n = sig.shape[0]
    dim = x.shape[0]
    z = np.zeros(n)
    for i in range(n):
        z[i] = np.sum(np.square(x - o[i*dim : (i+1)*dim]))
    w = np.zeros(n)
    for i in range(n):
        if z[i] != 0:
            w[i] = pow(1.0 / z[i], 0.5) * np.exp(-z[i] / 2 / dim / sig[i])
        else:
            w[i] = 1e+99
    w_sum = np.sum(w)
    if w_sum == 0:
        return np.ones(n)
    return w / w_sum


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Composition_1(x, M, o):
    dim = x.shape[0]
    delta = np.array([10, 20, 30, 40, 50])
    w = composition_omega(x, o, np.square(delta))
    fit = w[0] * Rosenbrock(np.dot(np.ascontiguousarray(M[0:dim, 0:dim]), 0.02048 * (x - o[0:dim])))
    fit += w[1] * (High_Conditioned_Elliptic(np.dot(np.ascontiguousarray(M[dim:2*dim, 0:dim]), x - o[dim:2*dim])) * 1e-6 + 100)
    fit += w[2] * (Bent_Cigar(np.dot(np.ascontiguousarray(M[2*dim:3*dim, 0:dim]), x - o[2*dim:3*dim])) * 1e-26 + 200)
    fit += w[3] * (Discus(np.dot(np.ascontiguousarray(M[3*dim:4*dim, 0:dim]), x - o[3*dim:4*dim])) * 1e-6 + 300)
    fit += w[4] * (Schwefel(10 * (x - o[4*dim:5*dim])) * 1e-6 + 400)
    return fit + 1300


@numba.cfunc("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def f13(x, M, o):
    return Composition_1(x, M, o)


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Composition_2(x, M, o):
    dim = x.shape[0]
    delta = np.array([10, 30, 50])
    w = composition_omega(x, o, np.square(delta))
    fit = w[0] * Schwefel(np.dot(np.ascontiguousarray(M[0:dim, 0:dim]), 10 * (x - o[0:dim]))) * 0.25
    fit += w[1] * (Rastrigin(np.dot(np.ascontiguousarray(M[dim:2*dim, 0:dim]), 0.0512 * (x - o[dim:2*dim]))) + 100)
    fit += w[2] * (High_Conditioned_Elliptic(np.dot(np.ascontiguousarray(M[2*dim:3*dim, 0:dim]), x - o[2*dim:3*dim])) * 1e-7 + 200)
    return fit + 1400


@numba.cfunc("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def f14(x, M, o):
    return Composition_2(x, M, o)


@numba.njit("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def Composition_3(x, M, o):
    dim = x.shape[0]
    delta = np.array([10, 10, 10, 20, 20])
    w = composition_omega(x, o, np.square(delta))
    fit = w[0] * HGBat(np.dot(np.ascontiguousarray(M[0:dim, 0:dim]), 0.05 * (x - o[0:dim]))) * 10
    fit += w[1] * (Rastrigin(np.dot(np.ascontiguousarray(M[dim:2*dim, 0:dim]), 0.0512 * (x - o[dim:2*dim]))) * 10 + 100)
    fit += w[2] * (Schwefel(np.dot(np.ascontiguousarray(M[2*dim:3*dim, 0:dim]), 10 * (x - o[2*dim:3*dim]))) * 2.5 + 200)
    fit += w[3] * (Weierstrass(np.dot(np.ascontiguousarray(M[3*dim:4*dim, 0:dim]), 0.005 * (x - o[3*dim:4*dim]))) * 25 + 300)
    fit += w[4] * (High_Conditioned_Elliptic(np.dot(np.ascontiguousarray(M[4*dim:5*dim, 0:dim]), x - o[4*dim:5*dim])) * 1e-6 + 400)
    return fit + 1500


@numba.cfunc("f8(f8[:],f8[:,:],f8[:])", nogil=True, fastmath=True)
def f15(x, M, o):
    return Composition_3(x, M, o)


@numba.njit(nogil=True, fastmath=True)
def func_Simple(ind):

    f = numba.typed.Dict()
    f[1] = f1
    f[2] = f2
    f[3] = f3
    f[4] = f4
    f[5] = f5
    f[6] = f6
    f[7] = f7
    f[8] = f8
    f[9] = f9

    return f[ind]


@numba.njit(nogil=True, fastmath=True)
def func_Hybrid(ind):

    f = numba.typed.Dict()
    f[10] = f10
    f[11] = f11
    f[12] = f12

    return f[ind]


@numba.njit(nogil=True, fastmath=True)
def func_Composition(ind):

    f = numba.typed.Dict()
    f[13] = f13
    f[14] = f14
    f[15] = f15

    return f[ind]


@numba.njit(nogil=True, fastmath=True)
def func(ind, x, M, o, s):
    if 1 <= ind <= 9:
        return func_Simple(ind)(x, M, o)
    elif 10 <= ind <= 12:
        return func_Hybrid(ind)(x, M, o, s)
    else:
        return func_Composition(ind)(x, M, o)