""" Contains implementation of polar decomposition of Mueller matrices following:
    
    "Interpretation of Mueller matrices based on polar decomposition." 
    Lu and Chapman. JOSA A. 1996. https://doi.org/10.1364/JOSAA.13.001106

    Written by Kate, 11/2/20
"""

import math
import cmath
import numpy as np
import numpy.linalg as la

def polar_decomp(M):
    M_D = np.zeros((4,4))  # Diattenuator
    M_R = np.zeros((4,4))  # Retarder
    M_T = np.zeros((4,4))  # Depolarizer

    m_00 = M[0,0]

    # Unpolarized transmittance
    Tu = m_00

    # Diattenuation
    D = la.norm(M[0, 1:]) / m_00
    D_vec = (M[0, 1:] / m_00)[:, np.newaxis]
    D_hat = D_vec / la.norm(D_vec)
    
    # Polarizance
    P = la.norm(M[1:, 0]) / m_00
    P_vec = (M[1:, 0] / m_00)[:, np.newaxis]
    P_hat = P_vec / la.norm(P_vec)

    if D != 0:
        D_perp = math.sqrt(1 - D * D);
        m_d = D_perp * np.eye(3) + (1 - D_perp) * (D_hat @ D_hat.T)

        M_D[0 , 0 ] = 1
        M_D[0 , 1:] = D_vec.T
        M_D[1:, 0 ] = D_vec.squeeze()
        M_D[1:, 1:] = m_d
        M_D *= Tu
    else:
        M_D = Tu * np.eye(4)

    if la.det(M_D) != 0:
        P_tri = (P_vec - M[1:, 1:] @ D_vec) / (1 - D * D)

        M_T[0 , 0 ] = 1;
        M_T[1:, 0 ] = P_tri.squeeze()

        M_pri = M @ la.inv(M_D)
        m_pri = M_pri[1:, 1:]
        m_pri_xtr = m_pri @ m_pri.T
        e, _ = la.eig(m_pri_xtr)

        e1 = e[0]
        e2 = e[1]
        e3 = e[2]

        if e1 == 0 and e2 == 0 and e3 == 0:
            M_R = np.eye(4)

        elif la.det(m_pri) == 0:
            U, _, V = la.svd(m_pri)

            m_tri = math.sqrt(e1) * U[:,0] @ U[:,0].T + \
                    math.sqrt(e2) * U[:,1] @ U[:,1].T + \
                    math.sqrt(e3) * U[:,2] @ U[:,2].T 

            M_T[1:, 1:] = m_tri

            m_r = U[:,0] @ V[:,0].T + U[:,1] @ V[:,1].T + U[:,2] @ V[:,2].T

            M_R[0 , 0 ] = 1
            M_R[1:, 1:] = m_r

        else:
            m_tri = la.inv(m_pri_xtr + (math.sqrt(e1*e2) + math.sqrt(e2*e3) + math.sqrt(e3*e1)) * np.eye(3))
            m_tmp = (math.sqrt(e1) + math.sqrt(e2) + math.sqrt(e3)) * m_pri_xtr + math.sqrt(e1*e2*e3) * np.eye(3)
            m_tri = m_tri @ m_tmp

            if la.det(m_pri) < 0:
                m_tri *= -1

            M_T[1:, 1:] = m_tri

            M_R = la.inv(M_T) @ M_pri

    else:
        M_T[0 , 0 ] = 1
        M_T[1:, 1:] = P * np.eye(3)

        xPD = np.cross(P_hat.squeeze(), D_vec.squeeze())
        dPD = np.dot(P_hat.squeeze(), D_vec.squeeze())

        if dPD >= 1:
            R_vec = np.zeros(3)
            R_hat = np.zeros(3)
        else:
            R_vec = xPD / la.norm(xPD) * math.acos(dPD)
            R_hat = R_vec / la.norm(R_vec)

        R = la.norm(R_vec)

        if R == 0:
            M_R = np.eye(4)
        else:
            cosr = math.cos(R)
            cosr_ = 1 - cosr
            sinr = math.sin(R)

            r1 = R_hat[0]
            r2 = R_hat[1]
            r3 = R_hat[2]

            mr11 = cosr + r1 * r1 * cosr_
            mr12 = r1 * r2 * cosr_ + r3 * sinr
            mr13 = r1 * r3 * cosr_ - r2 * sinr
            mr21 = r2 * r1 * cosr_ - r3 * sinr
            mr22 = cosr + r2 * r2 * cosr_
            mr23 = r2 * r3 * cosr_ + r1 * sinr
            mr31 = r3 * r1 * cosr_ + r2 * sinr
            mr32 = r3 * r2 * cosr_ - r1 * sinr
            mr33 = cosr + r3 * r3 * cosr_

            m_r = np.array([[mr11, mr12, mr13],
                            [mr21, mr22, mr23],
                            [mr31, mr32, mr33]])

            M_R[0 , 0 ] = 1
            M_R[1:, 1:] = m_r

    return (M_D, M_R, M_T)


def phase_shift(wavelength, thickness, dior):
    """ 
    Calculate phase shift.

    * wavelength:    wavelength of light, nm
    * thickness:     thickness of tape, nm
    * dior:          birefringence of tape, unitless
    """
    return (2 * math.pi * dior * thickness) / wavelength


def polarizer_mueller(theta):
    """ Mueller matrix of a linear polarizer at angle theta. """
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)

    # Linear polarizer (analyzer, final layer)
    return 0.5 * np.array([[1., cos2t, sin2t, 0.],
                           [cos2t, cos2t * cos2t, sin2t * cos2t, 0.],
                           [sin2t, sin2t * cos2t, sin2t * sin2t, 0.],
                           [0., 0., 0., 0.]])


def waveplate_mueller(gamma, alpha):
    """ Mueller matrix of a waveplate with phase shift gamma at angle alpha. """
    cos2a = np.cos(2 * alpha)
    sin2a = np.sin(2 * alpha)
    cosg = np.cos(gamma)
    sing = np.sin(gamma)
    cosg_ = 1 - cosg

    w11 = cos2a * cos2a + sin2a * sin2a * cosg
    w12 = cos2a * sin2a * cosg_
    w13 = sin2a * sing
    w21 = cos2a * sin2a * cosg_
    w22 = cos2a * cos2a * cosg + sin2a * sin2a
    w23 = -cos2a * sing
    w31 = -sin2a * sing
    w32 = cos2a * sing
    w33 = cosg

    return np.array([[1.,  0.,  0.,  0.],
                     [0., w11, w12, w13],
                     [0., w21, w22, w23],
                     [0., w31, w32, w33]])


def diattenuator_mueller(m_00, m_01, m_02, m_03):
    M = np.array([m_00, m_01, m_02, m_03])
    M_D = np.zeros((4,4))  # Diattenuator

    # Unpolarized transmittance
    Tu = m_00

    # Diattenuation
    D = la.norm(M[1:]) / m_00
    D_vec = (M[1:] / m_00)[:, np.newaxis]
    D_hat = D_vec / la.norm(D_vec)

    if D != 0:
        D_perp = math.sqrt(1 - D * D)
        m_d = D_perp * np.eye(3) + (1 - D_perp) * (D_hat @ D_hat.T)

        M_D[0 , 0 ] = 1
        M_D[0 , 1:] = D_vec.T
        M_D[1:, 0 ] = D_vec.squeeze()
        M_D[1:, 1:] = m_d
        M_D *= Tu
    else:
        M_D = Tu * np.eye(4)

    return M_D

def depolarizer_mueller(alpha, beta, gamma):
    M_T = np.array([[1, 0, 0, 0],
                    [0, alpha, 0, 0],
                    [0, 0, beta, 0],
                    [0, 0, 0, gamma]])

    return M_T


def mie(radius, wavelength, ior_s, ior_m, mu, max_n = 0):
    x = 2. * math.pi * radius * ior_m / wavelength
    y = 2. * math.pi * radius * ior_s / wavelength
    
    # Empirical formula for maximum number of terms to sum
    if max_n == 0:
        max_n = int(math.ceil(abs(x) + 4.3 * math.pow(abs(x), 1./3.) + 1))

    Ax = np.zeros(max_n + 1, dtype=np.complex64)
    Ay = np.zeros(max_n + 1, dtype=np.complex64)
    
    # Calculate An for [0, max_n] by downward recurrence
    n = max_n - 1
    while n >= 0:
        
        kx_n = (n + 1.) / x
        ky_n = (n + 1.) / y

        Ax[n] = kx_n - (1. / (kx_n + Ax[n + 1]))
        Ay[n] = ky_n - (1. / (ky_n + Ay[n + 1]))
        
        n -= 1
    
    Bx_0 = 1j
    psi_times_zeta_0 = 0.5 * (1. - cmath.exp(2. * x * 1j))
    psi_over_zeta_0 = 0.5 * (1. - cmath.exp(-2. * x * 1j))
    
    pi_0 = 0.
    pi_1 = 1.

    s1 = 0.j
    s2 = 0.j
    
    accum_1 = 0.;
    accum_2 = 0.;
    
    # Calculate each term in infinite sum for s1 and s2
    n = 1
    while n <= max_n:
        
        n_over_x = n / x
        Ax_n = Ax[n]
        Ay_n = Ay[n]
        
        psi_times_zeta_n = psi_times_zeta_0 * (n_over_x - Ax[n - 1]) * (n_over_x - Bx_0)
        Bx_n = Ax_n + (1j / psi_times_zeta_n)
        psi_over_zeta_n = psi_over_zeta_0 * (Bx_n + n_over_x) / (Ax_n + n_over_x)

        a_n = psi_over_zeta_n * (ior_m * Ay_n - ior_s * Ax_n) / (ior_m * Ay_n - ior_s * Bx_n)
        b_n = psi_over_zeta_n * (ior_s * Ay_n - ior_m * Ax_n) / (ior_s * Ay_n - ior_s * Bx_n)
        
        if n == 1:
            pi_n = pi_1
            tau_n = mu
        else:
            pi_n = ((2. * n - 1.) / (n - 1.)) * mu * pi_1 - (n / (n - 1.)) * pi_0
            tau_n = n * mu * pi_n - (n + 1.) * pi_1
            
            # update terms for next iteration
            pi_0 = pi_1
            pi_1 = pi_n
        
        k = (2. * n + 1.) / (n * (n + 1.))
        
        # Calculate nth term of s1 and s2
        s1 += k * (a_n * pi_n + b_n * tau_n)
        s2 += k * (b_n * pi_n + a_n * tau_n)
        
        accum_1 += (2. * n + 1.) * ((a_n + b_n) / (ior_m * ior_m)).real
        accum_2 += (2. * n + 1.) * (abs(a_n)*abs(a_n) + abs(b_n)*abs(b_n))
        
        # Update terms for next iteration
        Bx_0 = Bx_n
        psi_times_zeta_0 = psi_times_zeta_n
        psi_over_zeta_0 = psi_over_zeta_n
        
        n += 1
        
    alpha = 4. * math.pi * radius * ior_m.imag / wavelength
    gamma = 1 if alpha == 0 else 2 * (1 + (alpha - 1) * math.exp(alpha)) / (alpha * alpha);
    
    wsq = wavelength * wavelength
    inv_2pi = 1. / (2. * math.pi)
    
    Ct = inv_2pi * accum_1 * wsq
    Cs = inv_2pi * accum_2 * wsq * math.exp(-4. * math.pi * radius * ior_m.imag / wavelength) / (gamma * abs(ior_m) * abs(ior_m))
    Nf = 4. * math.pi * accum_2

    return (s1, s2, Nf, Ct, Cs)

def squared_norm(c):
    return c.real * c.real + c.imag * c.imag

def mie_mueller(cost, radius, wavelength):

    S1, S2, Nf, _, _ = mie(
                            radius = radius,
                            wavelength = wavelength,
                            ior_m = 1.0 - 0j,
                            ior_s = 1.33 - 1e-8j,
                            mu = cost,
                            max_n = 150
                        )

    s11 = squared_norm(S1) + squared_norm(S2)
    s12 = squared_norm(S2) - squared_norm(S1)
    s33 = (S2 * S1.conj() + S1 * S2.conj()).real
    s34 = (1j * (S2 * S1.conj() - S1 * S2.conj())).real

    M = 0.5 / Nf * np.array([[s11, s12, 0, 0],
                             [s12, s11, 0, 0],
                             [0, 0, s33, -s34],
                             [0, 0, s34, s33]])

    return M

# Based on implementation in Mitsuba 2
def specular_reflection_mueller(cost_i, eta):
    eta_it = eta
    eta_ti = 1 / eta

    cost_t_sq = 1 - eta_ti * eta_ti * (-cost_i * cost_i + 1)

    cost_i_abs = abs(cost_i)
    cost_t = math.sqrt(cost_t_sq)

    a_s = (-eta_it * cost_t + cost_i_abs) / (eta_it * cost_t + cost_i_abs)
    a_p = (-eta_it * cost_i_abs + cost_t) / (eta_it * cost_i_abs + cost_t)

    if cost_t_sq > 0:
        cost_t *= -1 * np.sign(cost_i)

    r_s = abs(a_s * a_s)
    r_p = abs(a_p * a_p)
    a = .5 * (r_s + r_p)
    b = .5 * (r_s - r_p)
    c = math.sqrt(r_s * r_p)

    M = np.array([[a, b, 0, 0],
                  [b, a, 0, 0],
                  [0, 0, c, 0],
                  [0, 0, 0, c]])

    return M

def specular_transmission_mueller(cost_i, eta):
    eta_it = eta
    eta_ti = 1 / eta

    cost_t_sq = 1 - eta_ti * eta_ti * (-cost_i * cost_i + 1)

    cost_i_abs = abs(cost_i)
    cost_t = math.sqrt(cost_t_sq)

    a_s = (-eta_it * cost_t + cost_i_abs) / (eta_it * cost_t + cost_i_abs)
    a_p = (-eta_it * cost_i_abs + cost_t) / (eta_it * cost_i_abs + cost_t)

    if cost_t_sq > 0:
        cost_t *= -1 * np.sign(cost_i)

    factor = -eta_it * (cost_t / cost_i if abs(cost_i) > 1e-8 else 0)

    a_s_r = a_s + 1
    a_p_r = (1 - a_p) * eta_ti

    t_s = a_s_r * a_s_r
    t_p = a_p_r * a_p_r
    a = .5 * factor * (t_s + t_p)
    b = .5 * factor * (t_s - t_p)
    c = factor * math.sqrt(t_s * t_p)

    M = np.array([[a, b, 0, 0],
                  [b, a, 0, 0],
                  [0, 0, c, 0],
                  [0, 0, 0, c]])

    return M