import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="LOADBANK-MASTER | Dimensionamento Barre Piastra Powerlock",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATABASE MATERIALI TELEMA DA TABELLA TECNICA ---
MATERIALI_DB = {
    "Rame elettrolitico (rho=0,0176 uOhm m)": {"K1": 1.27, "rho_20": 0.0176, "densita": 8890.0, "c_sp": 385.0},
    "Rame elettrolitico (rho=0,0178 uOhm m)": {"K1": 1.26, "rho_20": 0.0178, "densita": 8890.0, "c_sp": 385.0},
    "Alluminio purezza elettrica (rho=0,028 uOhm m)": {"K1": 1.02, "rho_20": 0.0280, "densita": 2700.0, "c_sp": 880.0},
    "Alluminio purezza elettrica (rho=0,029 uOhm m)": {"K1": 1.00, "rho_20": 0.0290, "densita": 2700.0, "c_sp": 880.0},
    "Lega d'alluminio (rho=0,0325 uOhm m)": {"K1": 0.94, "rho_20": 0.0325, "densita": 2700.0, "c_sp": 880.0},
    "Lega d'alluminio (rho=0,0385 uOhm m)": {"K1": 0.87, "rho_20": 0.0385, "densita": 2700.0, "c_sp": 880.0},
    "Lega d'alluminio (rho=0,0400 uOhm m)": {"K1": 0.85, "rho_20": 0.0400, "densita": 2700.0, "c_sp": 880.0},
    "Lega d'alluminio (rho=0,0580 uOhm m)": {"K1": 0.71, "rho_20": 0.0580, "densita": 2700.0, "c_sp": 880.0},
    "AISI 304": {"K1": 0.21, "rho_20": 0.7200, "densita": 8000.0, "c_sp": 500.0},
    "CuCr1Zr (rho=0,028 uOhm m)": {"K1": 1.01, "rho_20": 0.0280, "densita": 8910.0, "c_sp": 450.0}
}

ORIENT_ALTO = "Sormonto piegato verso l'ALTO (L dritta)"
ORIENT_BASSO = "Sormonto piegato verso il BASSO (L rovescia)"

st.title("LOADBANK-MASTER | Dimensionamento e Ripartizione Barre Powerlock")
st.markdown(
    "Calcolo termoelettrico alle differenze finite (2D FDM) per la verifica della ripartizione di corrente "
    "su piastre collettrice di banchi di carico e verifica delle sezioni secondo fattore **$K_1$**."
)

# --- SIDEBAR 1: MATERIALE E PIASTRA PRINCIPALE ---
st.sidebar.header("1. Materiale e Piastra Collettrice")
mat_scelto = st.sidebar.selectbox("Materiale Sbarre", list(MATERIALI_DB.keys()), index=0)
dati_mat = MATERIALI_DB[mat_scelto]
K1_mat = dati_mat["K1"]
rho_mat = dati_mat["rho_20"]
dens_mat = dati_mat["densita"]
csp_mat = dati_mat["c_sp"]

st.sidebar.caption(
    f"**Parametri di Progetto:** $K_1$ = **{K1_mat:.2f} A/mm²** | "
    f"$\\rho_{{20}}$ = **{rho_mat:.4f} µΩ·m** | $\\delta$ = **{dens_mat:.0f} kg/m³**"
)

B_piastra = st.sidebar.number_input("Larghezza Piastra B [mm]", min_value=50.0, max_value=1000.0, value=200.0, step=10.0)
H_piastra = st.sidebar.number_input("Altezza Piastra H [mm]", min_value=200.0, max_value=3000.0, value=1000.0, step=50.0)
s_piastra = st.sidebar.number_input("Spessore Piastra s [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0)

# --- SIDEBAR 2: GRIGLIA POWERLOCK ---
st.sidebar.header("2. Griglia Powerlock (Ingresso)")
I_totale = st.sidebar.number_input("Corrente Totale Nominale [A]", min_value=100.0, max_value=40000.0, value=3600.0, step=100.0)
X_col = st.sidebar.number_input("Colonne Powerlock (X)", min_value=1, max_value=10, value=2, step=1)
Y_row = st.sidebar.number_input("Righe Powerlock (Y)", min_value=1, max_value=25, value=9, step=1)
passo_x = st.sidebar.number_input("Passo Orizzontale Colonne [mm]", min_value=20.0, max_value=300.0, value=80.0, step=5.0)
passo_y = st.sidebar.number_input("Passo Verticale Righe [mm]", min_value=20.0, max_value=300.0, value=85.0, step=5.0)
y_offset_pl = st.sidebar.number_input("Quota Y Prima Riga dal Basso [mm]", min_value=10.0, max_value=2500.0, value=160.0, step=10.0)
diam_pl = st.sidebar.number_input("Diametro Impronta Powerlock [mm]", min_value=15.0, max_value=80.0, value=32.0, step=2.0)

# --- SIDEBAR 3: BARRE DI DERIVAZIONE A L ---
st.sidebar.header("3. Barre a L in Uscita")
w_L = st.sidebar.number_input("Larghezza Barre a L [mm]", min_value=20.0, max_value=600.0, value=120.0, step=10.0)
h_L = st.sidebar.number_input("Altezza Sormonto h_L [mm]", min_value=15.0, max_value=300.0, value=60.0, step=5.0)
s_L = st.sidebar.number_input("Spessore Barre a L s_L [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0)

L_A = st.sidebar.number_input(
    "Lunghezza Ramo A fino a Barra Equipotenziale [mm]",
    min_value=20.0, max_value=1500.0, value=120.0, step=10.0,
    help="Sviluppo lineare della Barra A dalla piega al punto di giunzione equipotenziale di valle."
)
L_B = st.sidebar.number_input(
    "Lunghezza Ramo B fino a Barra Equipotenziale [mm]",
    min_value=20.0, max_value=1500.0, value=120.0, step=10.0,
    help="Sviluppo lineare della Barra B dalla piega al punto di giunzione equipotenziale di valle."
)

verso_piega_A = st.sidebar.selectbox("Orientamento Sormonto Barra A (Bassa)", [ORIENT_ALTO, ORIENT_BASSO], index=0)
verso_piega_B = st.sidebar.selectbox("Orientamento Sormonto Barra B (Alta)", [ORIENT_ALTO, ORIENT_BASSO], index=1)


def limiti_ala(verso):
    if verso == ORIENT_ALTO:
        return s_L / 2.0, H_piastra - h_L + s_L / 2.0
    return h_L - s_L / 2.0, H_piastra - s_L / 2.0


def y_sormonto(y_ala, verso):
    if verso == ORIENT_ALTO:
        return y_ala - s_L / 2.0 + h_L / 2.0
    return y_ala + s_L / 2.0 - h_L / 2.0


min_y_A, max_y_A = limiti_ala(verso_piega_A)
min_y_B, max_y_B = limiti_ala(verso_piega_B)

if max_y_A <= min_y_A or max_y_B <= min_y_B:
    st.error("Errore geometrico: L'altezza del sormonto h_L supera l'altezza della piastra H.")
    st.stop()

y_ala_A = st.sidebar.slider(
    "Quota Y Uscita Barra A (Bassa) [mm]",
    min_value=float(min_y_A), max_value=float(max_y_A),
    value=float(min(max(30.0, min_y_A), max_y_A)), step=5.0
)
y_ala_B = st.sidebar.slider(
    "Quota Y Uscita Barra B (Alta) [mm]",
    min_value=float(min_y_B), max_value=float(max_y_B),
    value=float(min(max(500.0, min_y_B), max_y_B)), step=5.0
)

# Posizione geometrica del centro del sormonto di contatto
y_sorm_A = y_sormonto(y_ala_A, verso_piega_A)
y_sorm_B = y_sormonto(y_ala_B, verso_piega_B)
area_sormonto = w_L * h_L

# --- CONTROLLI MECCANICI DI SICUREZZA (DFM) ---
ingombro_x_pl = (X_col - 1) * passo_x + diam_pl
y_min_pl = y_offset_pl - diam_pl / 2.0
y_max_pl = y_offset_pl + (Y_row - 1) * passo_y + diam_pl / 2.0

errori_geom = []
if ingombro_x_pl > B_piastra:
    errori_geom.append(f"La griglia Powerlock è larga {ingombro_x_pl:.1f} mm e deborda dalla larghezza piastra B ({B_piastra:.1f} mm).")
if y_min_pl < 0 or y_max_pl > H_piastra:
    errori_geom.append(f"Le righe Powerlock occupano quote da Y={y_min_pl:.1f} mm a Y={y_max_pl:.1f} mm, uscendo dalla piastra H ({H_piastra:.1f} mm).")
if w_L > B_piastra:
    errori_geom.append(f"La larghezza delle barre a L ({w_L:.1f} mm) supera la larghezza della piastra ({B_piastra:.1f} mm).")
if abs(y_sorm_A - y_sorm_B) < h_L:
    errori_geom.append(
        f"Interferenza meccanica: I sormonti delle due barre si sovrappongono! "
        f"(Interasse = {abs(y_sorm_A - y_sorm_B):.1f} mm < altezza sormonto {h_L:.1f} mm)."
    )

if errori_geom:
    for err in errori_geom:
        st.error(err)
    st.stop()


# --- SOLUTORE ELETTRODINAMICO VETTORIALIZZATO ---
@st.cache_data(show_spinner=False)
def risolvi_busbar_2d(B, H, s, rho_uohm_m, I_tot, n_col, n_row, px, py, y0_pl, d_pl,
                      wL, hL, sL, lenA, lenB, y_sA, y_sB):
    rho = rho_uohm_m * 1e-3
    sigma = 1.0 / rho

    mesh_sz = 4.0 if (B * H) <= 600000 else 6.0
    nx = max(12, int(round(B / mesh_sz)))
    ny = max(24, int(round(H / mesh_sz)))
    dx = B / nx
    dy = H / ny
    x_c = (np.arange(nx) + 0.5) * dx
    y_c = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x_c, y_c)

    Gx = sigma * (s * dy) / dx
    Gy = sigma * (s * dx) / dy

    N = nx * ny
    idx_grid = np.arange(N).reshape((ny, nx))
    diag_main = np.zeros(N, dtype=np.float64)

    k_left = idx_grid[:, :-1].ravel()
    k_right = idx_grid[:, 1:].ravel()
    np.add.at(diag_main, k_left, Gx)
    np.add.at(diag_main, k_right, Gx)
    k_bot = idx_grid[:-1, :].ravel()
    k_top = idx_grid[1:, :].ravel()
    np.add.at(diag_main, k_bot, Gy)
    np.add.at(diag_main, k_top, Gy)

    # Iniezione nodale dai Powerlock
    I_vec = np.zeros(N, dtype=np.float64)
    x0_pl = (B - (n_col - 1) * px) / 2.0
    I_sing = I_tot / (n_col * n_row)
    r_pl = d_pl / 2.0
    coords_pl = []
    for r in range(n_row):
        yp = y0_pl + r * py
        for c in range(n_col):
            xp = x0_pl + c * px
            coords_pl.append((xp, yp))
            dist2 = (X - xp) ** 2 + (Y - yp) ** 2
            ind = np.flatnonzero(dist2 <= r_pl ** 2)
            if len(ind) == 0:
                ind = np.array([np.argmin(dist2)])
            I_vec[ind] += I_sing / len(ind)

    # Nodi delle impronte di sormonto
    x_min_L = (B - wL) / 2.0
    x_max_L = (B + wL) / 2.0
    mask_A = (X >= x_min_L) & (X <= x_max_L) & (Y >= y_sA - hL / 2.0) & (Y <= y_sA + hL / 2.0)
    mask_B = (X >= x_min_L) & (X <= x_max_L) & (Y >= y_sB - hL / 2.0) & (Y <= y_sB + hL / 2.0)
    ind_A = np.flatnonzero(mask_A)
    ind_B = np.flatnonzero(mask_B)

    # Modello in continuità di materiale su tutta la superficie del giunto
    area_giunto = wL * hL
    R_att = rho * (0.5 * (s + sL)) / area_giunto
    R_sorm = rho * hL / (3.0 * wL * sL)
    R_A = R_att + R_sorm + rho * lenA / (wL * sL)
    R_B = R_att + R_sorm + rho * lenB / (wL * sL)

    g_A = (1.0 / R_A) / len(ind_A)
    g_B = (1.0 / R_B) / len(ind_B)
    diag_main[ind_A] += g_A
    diag_main[ind_B] += g_B

    rows = np.concatenate([np.arange(N), k_left, k_right, k_bot, k_top])
    cols = np.concatenate([np.arange(N), k_right, k_left, k_top, k_bot])
    vals = np.concatenate([
        diag_main,
        np.full(len(k_left), -Gx), np.full(len(k_left), -Gx),
        np.full(len(k_bot), -Gy), np.full(len(k_bot), -Gy)
    ])
    G_csr = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    V = spla.spsolve(G_csr, I_vec)

    I_A = float(np.sum(V[ind_A] * g_A))
    I_B = float(np.sum(V[ind_B] * g_B))
    V_2d = V.reshape((ny, nx))
    err_perc = abs((I_A + I_B) - I_tot) / I_tot * 100.0

    dV_dy, dV_dx = np.gradient(V_2d, dy, dx)
    Jx = -sigma * dV_dx
    Jy = -sigma * dV_dy
    J_mag = np.sqrt(Jx ** 2 + Jy ** 2)
    profilo_Iy = np.sum(Jy, axis=1) * dx * s

    P_tot = float(I_vec @ V)
    P_rami_A = (I_A ** 2) * R_A
    P_rami_B = (I_B ** 2) * R_B
    P_piastra = P_tot - P_rami_A - P_rami_B

    return {
        "x": x_c, "y": y_c, "X": X, "Y": Y,
        "V_mV": V_2d * 1e3, "Jx": Jx, "Jy": Jy, "J_mag": J_mag,
        "I_A": I_A, "I_B": I_B, "err_perc": err_perc,
        "profilo_Iy": profilo_Iy, "coords_pl": coords_pl,
        "P_piastra": P_piastra, "P_A": P_rami_A, "P_B": P_rami_B,
        "R_A_uohm": R_A * 1e6, "R_B_uohm": R_B * 1e6, "R_att_uohm": R_att * 1e6
    }


res = risolvi_busbar_2d(
    B_piastra, H_piastra, s_piastra, rho_mat, I_totale,
    X_col, Y_row, passo_x, passo_y, y_offset_pl, diam_pl,
    w_L, h_L, s_L, L_A, L_B, y_sorm_A, y_sorm_B
)

# --- METRICHE DI SINTESI ---
p_A = res["I_A"] / I_totale * 100.0
p_B = res["I_B"] / I_totale * 100.0
sez_L = w_L * s_L
sez_piastra = B_piastra * s_piastra
J_L_A = res["I_A"] / sez_L
J_L_B = res["I_B"] / sez_L
J_sez_piastra_max = float(np.max(np.abs(res["profilo_Iy"]))) / sez_piastra
J_giunto_A = res["I_A"] / area_sormonto
J_giunto_B = res["I_B"] / area_sormonto

vol_piastra_m3 = B_piastra * H_piastra * s_piastra * 1e-9
vol_barre_m3 = w_L * ((h_L + L_A) + (h_L + L_B)) * s_L * 1e-9
massa_piastra_kg = vol_piastra_m3 * dens_mat
massa_totale_kg = (vol_piastra_m3 + vol_barre_m3) * dens_mat
P_tot = res["P_piastra"] + res["P_A"] + res["P_B"]

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Corrente Barra A (Bassa)",
    f"{res['I_A']:.0f} A ({p_A:.1f}%)",
    f"J: {J_L_A:.2f} A/mm² (K1={K1_mat:.2f})",
    delta_color="inverse" if J_L_A > K1_mat else "normal"
)
c2.metric(
    "Corrente Barra B (Alta)",
    f"{res['I_B']:.0f} A ({p_B:.1f}%)",
    f"J: {J_L_B:.2f} A/mm² (K1={K1_mat:.2f})",
    delta_color="inverse" if J_L_B > K1_mat else "normal"
)
c3.metric(
    "Densità Max Sezione Piastra",
    f"{J_sez_piastra_max:.2f} A/mm²",
    f"Soglia K1: {K1_mat:.2f} A/mm²",
    delta_color="inverse" if J_sez_piastra_max > K1_mat else "normal"
)
c4.metric(
    "Potenza Dissipata Totale",
    f"{P_tot:.1f} watt",
    f"Solo piastra: {res['P_piastra']:.1f} watt",
    delta_color="off"
)

st.markdown("---")

# --- GRAFICA A 4 VISTE SINCRONIZZATE IN SCALA 1:1 ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10), gridspec_kw={'width_ratios': [1.1, 1.1, 1.0, 1.0]})

# 1. Densità |J|
im1 = ax1.pcolormesh(res["X"], res["Y"], res["J_mag"], shading='auto', cmap='inferno',
                     vmin=0, vmax=max(K1_mat * 1.4, float(np.percentile(res["J_mag"], 99))))
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Densità |J| [A/mm²]")
ax1.streamplot(res["x"], res["y"], res["Jx"], res["Jy"], color='cyan', linewidth=0.7, density=0.8, arrowsize=0.7)
for (xp, yp) in res["coords_pl"]:
    ax1.add_patch(patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.2))
x_min_L = (B_piastra - w_L) / 2.0
for ys in (y_sorm_A, y_sorm_B):
    ax1.add_patch(patches.Rectangle((x_min_L, ys - h_L / 2.0), w_L, h_L, edgecolor='cyan', facecolor='none', linewidth=2.0))
ax1.text(B_piastra / 2.0, y_sorm_A, f"A: {p_A:.1f}%", color='white', weight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.75))
ax1.text(B_piastra / 2.0, y_sorm_B, f"B: {p_B:.1f}%", color='white', weight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.75))
ax1.set_title("1. Densità |J| (Scala 1:1)", fontsize=10, weight='bold')
ax1.set_xlabel("Larghezza X [mm]")
ax1.set_ylabel("Quota Y [mm]")
ax1.set_xlim(0, B_piastra)
ax1.set_ylim(0, H_piastra)
ax1.set_aspect('equal', adjustable='box')

# 2. Potenziale V [mV]
im2 = ax2.pcolormesh(res["X"], res["Y"], res["V_mV"], shading='auto', cmap='viridis')
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Potenziale [mV]")
for (xp, yp) in res["coords_pl"]:
    ax2.add_patch(patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.0, alpha=0.7))
for ys in (y_sorm_A, y_sorm_B):
    ax2.add_patch(patches.Rectangle((x_min_L, ys - h_L / 2.0), w_L, h_L, edgecolor='red', facecolor='none', linewidth=2.0))
ax2.set_title("2. Potenziale [mV] (Scala 1:1)", fontsize=10, weight='bold')
ax2.set_xlabel("Larghezza X [mm]")
ax2.set_xlim(0, B_piastra)
ax2.set_ylim(0, H_piastra)
ax2.set_aspect('equal', adjustable='box')

# 3. Flusso Assiale |Iy|
portata_limite_piastra = sez_piastra * K1_mat
ax3.plot(np.abs(res["profilo_Iy"]), res["y"], color='#d95f02', linewidth=2.2, label="|Iy(y)| in piastra")
ax3.axhline(y_sorm_A, color='blue', linestyle='--', linewidth=1.3, label=f"Sormonto A (Y={y_sorm_A:.0f} mm)")
ax3.axhline(y_sorm_B, color='green', linestyle='--', linewidth=1.3, label=f"Sormonto B (Y={y_sorm_B:.0f} mm)")
ax3.axvline(portata_limite_piastra, color='red', linestyle=':', linewidth=1.5, label=f"Portata K1 ({portata_limite_piastra:.0f} A)")
ax3.fill_betweenx(res["y"], 0, np.abs(res["profilo_Iy"]), color='#d95f02', alpha=0.2)
ax3.set_title("3. Flusso Assiale |Iy| [A]", fontsize=10, weight='bold')
ax3.set_xlabel("Corrente Assiale [A]")
ax3.set_ylim(0, H_piastra)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper right', fontsize=8)

# 4. Sezione Meccanica Y-Z (Forma reale a L)
L_DRAW_MAX = 150.0
prof_pl_disegno = diam_pl * 1.4
ax4.add_patch(patches.Rectangle((0, 0), s_piastra, H_piastra, facecolor='#b87333', edgecolor='black', linewidth=1.2))
for ys, ya, Lr, etichetta, colore, pc in (
        (y_sorm_A, y_ala_A, L_A, "A", 'blue', p_A), (y_sorm_B, y_ala_B, L_B, "B", 'green', p_B)):
    L_dis = min(Lr, L_DRAW_MAX)
    ax4.add_patch(patches.Rectangle((s_piastra, ys - h_L / 2.0), s_L, h_L, facecolor='#e68a00', edgecolor='black', linewidth=1.1))
    ax4.add_patch(patches.Rectangle((s_piastra + s_L, ya - s_L / 2.0), L_dis, s_L, facecolor='#e68a00', edgecolor='black', linewidth=1.1))
    nota = f"\nL = {Lr:.0f} mm" + (" (non in scala)" if Lr > L_DRAW_MAX else "")
    ax4.text(s_piastra + s_L + L_dis + 8, ya, f"{etichetta}: {pc:.1f}%\n({(res['I_A'] if etichetta == 'A' else res['I_B']):.0f} A){nota}",
             color=colore, fontsize=8, weight='bold', va='center')
y_pl_unici = sorted(set(yp for (_, yp) in res["coords_pl"]))
for yp in y_pl_unici:
    ax4.add_patch(patches.Rectangle((-prof_pl_disegno, yp - diam_pl / 2.0), prof_pl_disegno, diam_pl,
                                    facecolor='#333333', edgecolor='black', linewidth=0.9))
ax4.set_title("4. Sezione Laterale Y-Z", fontsize=10, weight='bold')
ax4.set_xlabel("Profondità Z [mm]")
ax4.set_xlim(-prof_pl_disegno - 30, s_piastra + s_L + L_DRAW_MAX + 150)
ax4.set_ylim(0, H_piastra)
ax4.set_aspect('equal', adjustable='box')
ax4.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- TABELLA DI VERIFICA FINALE ---
st.subheader("Report di Verifica Elettromeccanica e Termica")
st.table(pd.DataFrame({
    "Parametro Tecnico": [
        "Materiale e Coefficiente K1 (da Tabella)",
        "Massa Calcolata (Piastra / Piastra + Barre a L)",
        "Corrente Totale / Singolo Connettore Powerlock",
        "Ripartizione Ramo A (Inferiore)",
        "Ripartizione Ramo B (Superiore)",
        "Densità di Corrente Sezione Barre a L",
        "Densità Massima nella Sezione Piastra",
        "Densità di Corrente sull'Area di Sormonto",
        "Resistenza Elettrica dei Singoli Rami (A / B)",
        "Potenza Termica Dissipata (Piastra / Rami a L / Totale)",
        "Errore Numerico di Conservazione della Corrente"
    ],
    "Valore Calcolato": [
        f"{mat_scelto}  |  K1 = {K1_mat:.2f} A/mm²",
        f"Piastra: {massa_piastra_kg:.1f} kg  |  Totale: {massa_totale_kg:.1f} kg",
        f"{I_totale:.0f} A  |  {I_totale / (X_col * Y_row):.1f} A per Powerlock ({X_col * Y_row} poli)",
        f"{res['I_A']:.0f} A ({p_A:.1f} %)  |  Uscita a Y = {y_ala_A:.0f} mm (Sormonto centrato a Y = {y_sorm_A:.0f} mm)",
        f"{res['I_B']:.0f} A ({p_B:.1f} %)  |  Uscita a Y = {y_ala_B:.0f} mm (Sormonto centrato a Y = {y_sorm_B:.0f} mm)",
        f"Barra A: {J_L_A:.2f} A/mm²  |  Barra B: {J_L_B:.2f} A/mm²  (Sezione {sez_L:.0f} mm²)",
        f"{J_sez_piastra_max:.2f} A/mm²  (Sezione piastra {sez_piastra:.0f} mm²)",
        f"A: {J_giunto_A:.2f} A/mm²  |  B: {J_giunto_B:.2f} A/mm²  (Area sormonto {area_sormonto:.0f} mm²)",
        f"A: {res['R_A_uohm']:.2f} µΩ  |  B: {res['R_B_uohm']:.2f} µΩ  (Interfaccia sormonto {res['R_att_uohm']:.3f} µΩ)",
        f"{res['P_piastra']:.1f} watt  /  {(res['P_A'] + res['P_B']):.1f} watt  /  {P_tot:.1f} watt ({P_tot / 1000.0:.2f} chilowatt)",
        f"{res['err_perc']:.6f} %"
    ],
    "Esito e Indicazione di Conformità": [
        f"Densità: {dens_mat:.0f} kg/m³ | Calore specifico: {csp_mat:.0f} J/(kg·K)",
        "Calcolo massa netta semilavorati per distinta base (BOM)",
        "Iniezione bilanciata",
        "Bilanciato (45-55 %)" if 45.0 <= p_A <= 55.0 else ("Ramo più carico" if p_A > 55.0 else "Ramo scarico"),
        "Bilanciato (45-55 %)" if 45.0 <= p_B <= 55.0 else ("Ramo più carico" if p_B > 55.0 else "Ramo scarico"),
        "Entro K1" if max(J_L_A, J_L_B) <= K1_mat else f"SUPERAMENTO K1 ({K1_mat:.2f} A/mm²): Aumentare sezione L",
        "Entro K1" if J_sez_piastra_max <= K1_mat else f"SUPERAMENTO K1 ({K1_mat:.2f} A/mm²): Aumentare spessore piastra",
        "OK" if max(J_giunto_A, J_giunto_B) <= 0.40 else "Sollecitazione elevata sul sormonto (> 0.40 A/mm²)",
        "Include attraversamento, raccolta sormonto (1/3 h_L) e sviluppo L",
        "Dissipazione termica continua da evacuare con ventilazione forzata",
        "Controllo di convergenza Kirchhoff - Conforme"
    ]
}))
