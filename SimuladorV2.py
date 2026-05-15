import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

# ========= CONFIGURACIÓN DE PARÁMETROS =========
M_GRAVIT = 0.5
R_0 = 5.0

R_FINAL = 15.0 
NR = 500 * 4
radial_grid = np.linspace(0.0, R_FINAL, NR)
DELTA_R = radial_grid[1] - radial_grid[0]

dt = 0.001
RHO_VAL_0 = (3.0 * M_GRAVIT) / (4.0 * np.pi * R_0**3)
RHO_ATM = 1e-10 * RHO_VAL_0

# ========== FUNCIONES DE CONVERSIÓN (FÍSICA <-> CONSERVADAS) ==========

def phys_2_cons(rho, v, X):
    W = 1.0 / np.sqrt(1.0 - np.clip(v**2, 0, 0.9999999999999))
    D = X * rho * W
    S = rho * (W**2) * v
    tau = rho * (W**2) - D
    return D, S, tau

def cons_2_phys(D, S, tau, X):
    v = np.clip(S / (tau + D + 1e-20), -0.9999, 0.9999)
    W = 1.0 / np.sqrt(1.0 - v**2)
    rho = np.maximum(RHO_ATM, D / (X * W + 1e-20))
    return rho, v, W

# ========== FUNCIONES ANALÍTICAS (OPPENHEIMER-SNYDER) ==========

def t_of_eta_c(eta_c, M, xi_s):
    if abs(eta_c + math.pi) < 1e-12: return 0.0
    cos_xi = math.cos(xi_s)
    A = math.cos(0.5 * eta_c) / math.sqrt(cos_xi)
    if not -1.0 <= A <= 1.0: return float('nan')
    eta_s = -2.0 * math.acos(A)
    tan_half_eta_s = math.tan(0.5 * eta_s)
    tan_xi = math.tan(xi_s)
    log_arg = (tan_half_eta_s - tan_xi) / (tan_half_eta_s + tan_xi)
    if log_arg <= 0: return float('nan')
    return 2.0 * M * ((1.0/tan_xi)*(eta_s + math.pi + (0.5/math.sin(xi_s)**2)*(eta_s + math.pi - math.sin(eta_s))) + math.log(log_arg))

def eta_c_solver(t_target, M, xi_s):
    if t_target <= 1e-10: return -math.pi
    nodes = np.linspace(-math.pi, -1e-5, 1000)
    for i in range(len(nodes)-1):
        t1 = t_of_eta_c(nodes[i], M, xi_s)
        t2 = t_of_eta_c(nodes[i+1], M, xi_s)
        if (t1 - t_target) * (t2 - t_target) <= 0:
            return brentq(lambda ec: t_of_eta_c(ec, M, xi_s) - t_target, nodes[i], nodes[i+1])
    return -1e-5

def obtener_analitica_temporal(t, r_grid, M, R0):
    xi_s = math.asin(math.sqrt(2.0 * M / R0))
    etac = eta_c_solver(t, M, xi_s)
    cos2_nc2 = math.cos(etac / 2.0)**2
    R_t = R0 * (1.0 - cos2_nc2 / math.cos(xi_s))
    rho_an, alpha_an, v_an = np.full(NR, RHO_ATM), np.ones(NR), np.zeros(NR)
    alpha_o = (math.cos(xi_s)**3 - cos2_nc2) / np.power(abs(math.cos(xi_s) - cos2_nc2), 1.5)
    for i, r in enumerate(r_grid):
        if r <= R_t:
            chi = math.asin(np.clip((r / R_t) * math.sin(xi_s), 0.0, 1.0))
            diff = math.cos(chi) - cos2_nc2
            rho_an[i] = (3.0 * math.sin(xi_s)**6 / (32.0 * math.pi * M**2)) * np.power(math.cos(chi)/(diff+1e-15), 3)
            alpha_an[i] = alpha_o * diff / math.sqrt(abs(math.cos(chi)**3 - cos2_nc2) + 1e-15)
            v_an[i] = -math.cos(etac/2.0) * math.tan(chi) / math.sqrt(abs(diff) + 1e-15)
        elif r > R_t:
            alpha_an[i] = math.sqrt(max(1e-10, 1.0 - 2.0 * M / r))
    return rho_an, alpha_an, v_an, R_t

# ========== RECONSTRUCCIÓN MINMOD ==========

def minmod(a, b):
    # Minmod slope limiter function
    return 0.5*(np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

def reconstruct_primitive(u):
    du_forward = (u[1:] - u[:-1]) / DELTA_R
    slopes = np.zeros_like(u)
    
    slopes[1:-1] = minmod(du_forward[:-1], du_forward[1:])
    
    # reconstrucción en interfaces (j+1/2)
    u_L = u[:-1] + 0.5 * DELTA_R * slopes[:-1]
    u_R = u[1:] - 0.5 * DELTA_R * slopes[1:]
    
    return u_L, u_R

# ========== MÉTRICA Y CONDICIONES DE CONTORNO ==========

def BCs(D, S, tau, R_t):
    # r=0
    S[0] = 0.0
    D[0] = D[1]
    tau[0] = tau[1]

    # VACUUM BOUNDARY CONDITION: Impose vacuum strictly outside the star surface R(t)
    mask = radial_grid > R_t
    D[mask] = RHO_ATM * 1.0
    S[mask] = 0.0
    tau[mask] = 0.0

    # Outer boundary
    D[-2:], S[-2:], tau[-2:] = D[-3], S[-3], tau[-3]
    return D, S, tau

def get_metrica(D, S, tau, R_t):
    m = cumulative_trapezoid(4*np.pi*radial_grid**2 * (D + tau), radial_grid, initial=0)
    idx_s = np.searchsorted(radial_grid, R_t)
    if idx_s < NR: m[idx_s:] = M_GRAVIT
    
    r_safe = np.where(radial_grid > 1e-10, radial_grid, 1e-10)
    X = 1.0 / np.sqrt(np.clip(1.0 - 2.0*m/r_safe, 1e-10, None))
    
    rho, v, _ = cons_2_phys(D, S, tau, X)
    dphi_dr = X**2 * (m/r_safe**2 + 4*np.pi*radial_grid*(S*v))
    phi_raw = cumulative_trapezoid(dphi_dr, radial_grid, initial=0)
    
    phi_ext_val = 0.5 * np.log(np.clip(1.0 - 2.0*M_GRAVIT/R_t, 1e-10, None))
    idx_rt = np.clip(idx_s, 0, NR-1)
    phi = phi_raw + (phi_ext_val - phi_raw[idx_rt])
    
    mask_ext = radial_grid > R_t
    phi[mask_ext] = 0.5 * np.log(np.clip(1.0 - 2.0*M_GRAVIT/r_safe[mask_ext], 1e-10, None))
    return X, np.exp(phi), m

# ========== DERIVADAS TEMPORALES (RHS) ==========

def rhs(t, D, S, tau, R_t):
    X, alpha, m = get_metrica(D, S, tau, R_t)
    rho, v, W = cons_2_phys(D, S, tau, X)
    
    # RECONSTRUCCIÓN: Linear reconstruction of primitive variables
    rho_L, rho_R = reconstruct_primitive(rho)
    v_L, v_R = reconstruct_primitive(v)
    
    # Convert reconstructed primitive states back to conserved variables at interfaces
    W_L = 1.0 / np.sqrt(1.0 - np.clip(v_L**2, 0, 0.9999))
    W_R = 1.0 / np.sqrt(1.0 - np.clip(v_R**2, 0, 0.9999))
    X_inter = 0.5 * (X[:-1] + X[1:])
    
    D_L = X_inter * rho_L * W_L
    S_L = rho_L * W_L**2 * v_L
    tau_L = rho_L * W_L**2 - D_L
    
    D_R = X_inter * rho_R * W_R
    S_R = rho_R * W_R**2 * v_R
    tau_R = rho_R * W_R**2 - D_R
    
    # Evaluate fluxes at left and right reconstructed states
    f_D_L = D_L * v_L; f_S_L = S_L * v_L; f_tau_L = S_L - D_L * v_L
    f_D_R = D_R * v_R; f_S_R = S_R * v_R; f_tau_R = S_R - D_R * v_R
    
    A = radial_grid**2 * alpha / X
    A_inter = 0.5 * (A[1:] + A[:-1])
    
    # Flujo numérico: Mantenido sin Riemann solver (tipo Lax-Friedrichs)
    def get_num_flux(f_L, f_R, u_L, u_R, vel_L, vel_R):
        v_abs = np.maximum(np.abs(vel_L), np.abs(vel_R))
        return 0.5 * (f_L + f_R - 0.25 * v_abs * (u_R - u_L))

    nf_D = get_num_flux(f_D_L, f_D_R, D_L, D_R, v_L, v_R)
    nf_S = get_num_flux(f_S_L, f_S_R, S_L, S_R, v_L, v_R)
    nf_tau = get_num_flux(f_tau_L, f_tau_R, tau_L, tau_R, v_L, v_R)
    
    res = np.zeros((3, NR))
    r2 = np.where(radial_grid > 1e-10, radial_grid**2, 1e-10)
    
    for i, nf in enumerate([nf_D, nf_S, nf_tau]):
        res[i, 1:-1] = -(A_inter[1:] * nf[1:] - A_inter[:-1] * nf[:-1]) / (DELTA_R * r2[1:-1])
        
    source_S = (S*v - tau - D) * alpha * X * m / r2
    source_S[radial_grid < DELTA_R*1.5] = 0 
    res[1] += source_S
    
    return res

# ========== INICIALIZACIÓN Y BUCLE DE SIMULACIÓN ==========

rho_0, alpha_0, v_0, R_t0 = obtener_analitica_temporal(0, radial_grid, M_GRAVIT, R_0)
m_0 = (M_GRAVIT/R_0**3) * radial_grid**3
m_0[radial_grid > R_0] = M_GRAVIT
X_0 = 1.0 / np.sqrt(np.clip(1.0 - 2.0*m_0/np.where(radial_grid>0, radial_grid, 1e-10), 1e-10, None))

D, S, tau = phys_2_cons(rho_0, v_0, X_0)

t = 0.0
snapshots = [0.0, 3.0, 6.0, 9.0, 12.0]
colors = plt.cm.plasma(np.linspace(0, 0.8, len(snapshots)))
fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
s_idx = 0

print(f"Simulando Colapso OS... R0={R_0}, M={M_GRAVIT}")

while t <= max(snapshots) + dt:
    # Obtain current analytical surface radius
    _, _, _, Rt_a = obtener_analitica_temporal(t, radial_grid, M_GRAVIT, R_0)

    if s_idx < len(snapshots) and t >= snapshots[s_idx]:
        cur_t = snapshots[s_idx]
        rho_a, alpha_a, v_a, _ = obtener_analitica_temporal(cur_t, radial_grid, M_GRAVIT, R_0)
        X_n, alpha_n, _ = get_metrica(D, S, tau, Rt_a)
        rho_n, v_n, _ = cons_2_phys(D, S, tau, X_n)
        
        mask = (radial_grid >= 1*DELTA_R) & (radial_grid <= Rt_a + DELTA_R)
        print(mask)
        mask = radial_grid >= 0
        axs[0].plot(radial_grid[mask]/(2.0 * M_GRAVIT), rho_n[mask]/RHO_VAL_0, color=colors[s_idx], label=f't={cur_t}')
        axs[0].plot(radial_grid[mask]/(2.0 * M_GRAVIT), rho_a[mask]/RHO_VAL_0, '--', color=colors[s_idx], alpha=0.3)
        axs[1].plot(radial_grid[mask]/(2.0 * M_GRAVIT), v_n[mask], color=colors[s_idx])
        axs[1].plot(radial_grid[mask]/(2.0 * M_GRAVIT), v_a[mask], '--', color=colors[s_idx], alpha=0.3)
        axs[2].plot(radial_grid/(2.0 * M_GRAVIT), alpha_n, color=colors[s_idx])
        axs[2].plot(radial_grid/(2.0 * M_GRAVIT), alpha_a, '--', color=colors[s_idx], alpha=0.3)
        
        print(f"t={t:.2f} | R_surf={Rt_a:.3f} | rho_c={rho_n[2]/RHO_VAL_0:.3f}")
        s_idx += 1

    # Paso Runge-Kutta 2 (Updated to pass R_t into Boundary Conditions and RHS)
    D, S, tau = BCs(D, S, tau, Rt_a)
    k1 = rhs(t, D, S, tau, Rt_a)
    
    _, _, _, Rt_half = obtener_analitica_temporal(t + dt, radial_grid, M_GRAVIT, R_0)
    D1, S1, tau1 = BCs(D + dt*k1[0], S + dt*k1[1], tau + dt*k1[2], Rt_half)
    k2 = rhs(t + dt, D1, S1, tau1, Rt_half)
    
    D += 0.5*dt*(k1[0]+k2[0])
    S += 0.5*dt*(k1[1]+k2[1])
    tau += 0.5*dt*(k1[2]+k2[2])
    
    t += dt

# Formato de los ejes
axs[0].set_ylabel(r'$\rho / \rho_i$'); axs[0].legend()
axs[1].set_ylabel(r'$v$ (Velocidad)'); axs[1].grid(True, alpha=0.2)
axs[2].set_ylabel(r'$\alpha$ (Lapso)'); axs[2].set_xlabel(r'$R / 2M$')
axs[0].grid(True, alpha=0.2); axs[2].grid(True, alpha=0.2)
plt.suptitle("Simulación de Colapso Oppenheimer-Snyder")
plt.tight_layout()
plt.show()
