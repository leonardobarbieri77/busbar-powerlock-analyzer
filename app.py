import streamlit as st
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

st.set_page_config(
    page_title="LOADBANK-MASTER | Calcolo Sbarre & Powerlock 2D",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATABASE MATERIALI DA TABELLA TECNICA ---
MATERIALI_DB = {
    "Rame elettrolitico (rho=0,0176 uOhm m)": {
        "K1": 1.27, "rho_20": 0.0176, "mu": 1.25663e-6, "densita": 8890.0, "c_sp": 385.0, "K_cw1": 1.000000, "alpha_std": 0.00393
    },
    "Rame elettrolitico (rho=0,0178 uOhm m)": {
        "K1": 1.26, "rho_20": 0.0178, "mu": 1.25663e-6, "densita": 8890.0, "c_sp": 385.0, "K_cw1": 0.994366, "alpha_std": 0.00393
    },
    "Alluminio purezza elettrica (rho=0,028 uOhm m)": {
        "K1": 1.02, "rho_20": 0.0280, "mu": 1.25666e-6, "densita": 2700.0, "c_sp": 880.0, "K_cw1": 0.660571, "alpha_std": 0.00403
    },
    "Alluminio purezza elettrica (rho=0,029 uOhm m)": {
        "K1": 1.00, "rho_20": 0.0290, "mu": 1.25666e-6, "densita": 2700.0, "c_sp": 880.0, "K_cw1": 0.649082, "alpha_std": 0.00403
    },
    "Lega d'alluminio (rho=0,0325 uOhm m)": {
        "K1": 0.94, "rho_20": 0.0325, "mu": 1.25664e-6, "densita": 2700.0, "c_sp": 880.0, "K_cw1": 0.0, "alpha_std": 0.00360
    },
    "Lega d'alluminio (rho=0,0385 uOhm m)": {
        "K1": 0.87, "rho_20": 0.0385, "mu": 1.25664e-6, "densita": 2700.0, "c_sp": 880.0, "K_cw1": 0.0, "alpha_std": 0.00360
    },
    "Lega d'alluminio (rho=0,0400 uOhm m)": {
        "K1": 0.85, "rho_20": 0.0400, "mu": 1.25664e-6, "densita": 2700.0, "c_sp": 880.0, "K_cw1": 0.0, "alpha_std": 0.00360
    },
    "Lega d'alluminio (rho=0,0580 uOhm m)": {
        "K1": 0.71, "rho_20": 0.0580, "mu": 1.25664e-6, "densita": 2700.0, "c_sp": 880.0, "K_cw1": 0.0, "alpha_std": 0.00360
    },
    "AISI 304": {
        "K1": 0.21, "rho_20": 0.7200, "mu": 1.28177e-6, "densita": 8000.0, "c_sp": 500.0, "K_cw1": 0.169020, "alpha_std": 0.00100
    },
    "CuCr1Zr (rho=0,028 uOhm m)": {
        "K1": 1.01, "rho_20": 0.0280, "mu": 1.25664e-6, "densita": 8910.0, "c_sp": 450.0, "K_cw1": 0.858106, "alpha_std": 0.00300
    }
}

st.title("Analisi Elettrica 2D e Layout Meccanico Piastra Collettrice Powerlock")
st.markdown(
    "**LOADBANK-MASTER Engineering Tool** — Calcolo alle differenze finite (2D FDM) della ripartizione "
    "di corrente, verifica del parametro materiale $K_1$ e schema costruttivo in scala reale $1:1$."
)

# --- SIDEBAR: 1. MATERIALE E GEOMETRIA PIASTRA ---
st.sidebar.header("1. Materiale e Geometria Piastra")
mat_scelto = st.sidebar.selectbox("Seleziona Materiale Sbarre", list(MATERIALI_DB.keys()), index=0)
dati_mat = MATERIALI_DB[mat_scelto]

K1_mat = dati_mat["K1"]
rho_20_mat = dati_mat["rho_20"]

st.sidebar.info(
    f"**Proprietà da Tabella:**\n"
    f"- Coefficiente $K_1$: **{K1_mat:.2f} A/mm²**\n"
    f"- Resistività $\\rho_{{20^\\circ\\text{{C}}}}$: **{rho_20_mat:.4f} $\\mu\\Omega\\cdot$m**\n"
    f"- Densità $\\delta$: **{dati_mat['densita']:.0f} kg/m³**"
)

T_operativa = st.sidebar.slider("Temperatura Operativa Sbarra [°C]", min_value=20, max_value=140, value=70, step=5)
alpha_tcr = st.sidebar.number_input(
    "Coeff. Temperatura TCR alpha_20 [1/K]",
    min_value=0.00000, max_value=0.01000, value=float(dati_mat["alpha_std"]), step=0.00010, format="%.5f",
    help="Valore standard proposto per la lega selezionata (impostare 0 per calcolo a 20 °C costante)."
)

B_piastra = st.sidebar.number_input("Larghezza Piastra B [mm]", min_value=50.0, max_value=1000.0, value=200.0, step=10.0)
H_piastra = st.sidebar.number_input("Altezza Piastra H [mm]", min_value=200.0, max_value=3000.0, value=1000.0, step=50.0)
s_piastra = st.sidebar.number_input("Spessore Piastra s [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0)

# --- SIDEBAR: 2. GRIGLIA POWERLOCK ---
st.sidebar.header("2. Griglia Connettori Powerlock")
I_totale = st.sidebar.number_input("Corrente Totale Iniettata [A]", min_value=100.0, max_value=30000.0, value=7200.0, step=100.0)
X_col = st.sidebar.number_input("Numero Colonne (X)", min_value=1, max_value=10, value=2, step=1)
Y_row = st.sidebar.number_input("Numero Righe (Y)", min_value=1, max_value=25, value=9, step=1)
passo_x = st.sidebar.number_input("Passo Orizzontale Colonne [mm]", min_value=20.0, max_value=300.0, value=80.0, step=5.0)
passo_y = st.sidebar.number_input("Passo Verticale Righe [mm]", min_value=20.0, max_value=300.0, value=85.0, step=5.0)
y_offset_pl = st.sidebar.number_input("Quota Y Prima Riga dal Basso [mm]", min_value=20.0, max_value=1000.0, value=160.0, step=10.0)
diam_pl = st.sidebar.number_input("Diametro Impronta Contatto Powerlock [mm]", min_value=10.0, max_value=80.0, value=32.0, step=2.0)

# --- SIDEBAR: 3. BARRE DI DERIVAZIONE A L ---
st.sidebar.header("3. Barre di Derivazione a L (Uscite)")
w_L = st.sidebar.number_input("Larghezza Barre a L [mm]", min_value=20.0, max_value=500.0, value=120.0, step=10.0)
h_L = st.sidebar.number_input("Altezza Sormonto Bullonato h_L [mm]", min_value=15.0, max_value=200.0, value=60.0, step=5.0)
s_L = st.sidebar.number_input("Spessore Barre a L s_L [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0)
ala_L = st.sidebar.number_input(
    "Lunghezza Ala a Sbalzo L_sbalzo [mm]",
    min_value=20.0, max_value=500.0, value=100.0, step=10.0,
    help="Determina sia la lunghezza del braccio perpendicolare nella vista laterale, sia la resistenza ohmica propria del tratto di barra a L."
)

tipo_rif_y = st.sidebar.radio(
    "Riferimento Quota Y inserita:",
    ["Mezzeria Ala Perpendicolare in Uscita (CAD)", "Mezzeria Sormonto Bullonato (h_L)"],
    index=0
)

orient_piega_A = st.sidebar.selectbox(
    "Posizione Ala Uscita Barra A rispetto al Sormonto",
    ["Bordo Inferiore del Sormonto", "Bordo Superiore del Sormonto"],
    index=0
)
orient_piega_B = st.sidebar.selectbox(
    "Posizione Ala Uscita Barra B rispetto al Sormonto",
    ["Bordo Inferiore del Sormonto", "Bordo Superiore del Sormonto"],
    index=0
)

y_input_A = st.sidebar.slider("Quota Y Barra A (Inferiore) [mm]", min_value=5.0, max_value=float(H_piastra - 5.0), value=50.0, step=5.0)
y_input_B = st.sidebar.slider("Quota Y Barra B (Superiore) [mm]", min_value=5.0, max_value=float(H_piastra - 5.0), value=500.0, step=5.0)

R_cont_puro_uohm = st.sidebar.number_input(
    "Resistenza Puro Contatto Bullonato [micro-ohm]",
    min_value=0.01, max_value=20.0, value=0.80, step=0.10,
    help="Resistenza di contatto dell'interfaccia imbullonata. A questa il software somma automaticamente la resistenza ohmica dell'ala a sbalzo."
)

st.sidebar.header("4. Risoluzione Mesh FDM")
passo_mesh = st.sidebar.select_slider("Dimensione Cella Mesh [mm]", options=[2.5, 4.0, 5.0, 8.0], value=4.0)


# --- CALCOLO COORDINATE EFFETTIVE DEL SORMONTO E DELL'ALA ---
def calcola_quote_L(y_in, hL, sL, modalita_rif, orientamento):
    if modalita_rif == "Mezzeria Sormonto Bullonato (h_L)":
        y_centro_sormonto = y_in
        if orientamento == "Bordo Inferiore del Sormonto":
            y_centro_ala = y_centro_sormonto - (hL / 2.0) + (sL / 2.0)
        else:
            y_centro_ala = y_centro_sormonto + (hL / 2.0) - (sL / 2.0)
    else:
        y_centro_ala = y_in
        if orientamento == "Bordo Inferiore del Sormonto":
            y_centro_sormonto = y_centro_ala + (hL / 2.0) - (sL / 2.0)
        else:
            y_centro_sormonto = y_centro_ala - (hL / 2.0) + (sL / 2.0)
    return y_centro_sormonto, y_centro_ala


y_sorm_A, y_ala_A = calcola_quote_L(y_input_A, h_L, s_L, tipo_rif_y, orient_piega_A)
y_sorm_B, y_ala_B = calcola_quote_L(y_input_B, h_L, s_L, tipo_rif_y, orient_piega_B)


# --- FUNZIONE SOLUTORE 2D FDM ---
def risolvi_piastra_2d(B, H, s, rho_20, alpha, T, I_tot, n_col, n_row, px, py, y0_pl, d_pl, wL, hL, sL, alaL, y_sA, y_sB, R_cont_uohm, d_mesh):
    rho_T = rho_20 * (1.0 + alpha * (T - 20.0))  # micro-ohm * m = ohm * mm^2 / m
    rho_ohm_mm = rho_T * 1e-3  # ohm * mm
    sigma = 1.0 / rho_ohm_mm  # S / mm

    # Resistenza propria dell'ala a sbalzo + resistenza di contatto
    R_ala_ohm = rho_ohm_mm * (alaL / (wL * sL))
    R_tot_ramo_ohm = (R_cont_uohm * 1e-6) + R_ala_ohm

    nx = max(12, int(round(B / d_mesh)))
    ny = max(24, int(round(H / d_mesh)))
    dx = B / nx
    dy = H / ny

    x_centers = (np.arange(nx) + 0.5) * dx
    y_centers = (np.arange(ny) + 0.5) * dy
    X_grid, Y_grid = np.meshgrid(x_centers, y_centers)

    Gx = sigma * (s * dy) / dx
    Gy = sigma * (s * dx) / dy

    N_nodi = nx * ny
    G_mat = sp.lil_matrix((N_nodi, N_nodi), dtype=np.float64)
    I_vec = np.zeros(N_nodi, dtype=np.float64)

    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            diag = 0.0
            if i > 0:
                G_mat[k, k - 1] = -Gx
                diag += Gx
            if i < nx - 1:
                G_mat[k, k + 1] = -Gx
                diag += Gx
            if j > 0:
                G_mat[k, k - nx] = -Gy
                diag += Gy
            if j < ny - 1:
                G_mat[k, k + nx] = -Gy
                diag += Gy
            G_mat[k, k] = diag

    larghezza_griglia_pl = (n_col - 1) * px
    x0_pl = (B - larghezza_griglia_pl) / 2.0
    n_tot_pl = n_col * n_row
    I_singolo_pl = I_tot / n_tot_pl
    raggio_pl = d_pl / 2.0

    coords_pl = []
    for r in range(n_row):
        yp = y0_pl + r * py
        for c in range(n_col):
            xp = x0_pl + c * px
            coords_pl.append((xp, yp))
            dist2 = (X_grid - xp) ** 2 + (Y_grid - yp) ** 2
            mask_pl = dist2 <= raggio_pl ** 2
            n_celle_pl = np.sum(mask_pl)
            if n_celle_pl == 0:
                idx_min = np.argmin(dist2)
                I_vec[idx_min] += I_singolo_pl
            else:
                indici_pl = np.flatnonzero(mask_pl)
                I_vec[indici_pl] += I_singolo_pl / n_celle_pl

    x_min_L = (B - wL) / 2.0
    x_max_L = (B + wL) / 2.0

    mask_A = (X_grid >= x_min_L) & (X_grid <= x_max_L) & (Y_grid >= y_sA - hL / 2.0) & (Y_grid <= y_sA + hL / 2.0)
    mask_B = (X_grid >= x_min_L) & (X_grid <= x_max_L) & (Y_grid >= y_sB - hL / 2.0) & (Y_grid <= y_sB + hL / 2.0)

    indici_A = np.flatnonzero(mask_A)
    indici_B = np.flatnonzero(mask_B)

    g_tot_barra = 1.0 / max(1e-9, R_tot_ramo_ohm)
    g_cella_A = g_tot_barra / max(1, len(indici_A))
    g_cella_B = g_tot_barra / max(1, len(indici_B))

    for idx in indici_A:
        G_mat[idx, idx] += g_cella_A
    for idx in indici_B:
        G_mat[idx, idx] += g_cella_B

    G_csr = G_mat.tocsr()
    V_sol = spla.spsolve(G_csr, I_vec)
    V_2d = V_sol.reshape((ny, nx))

    I_A = float(np.sum(V_sol[indici_A] * g_cella_A))
    I_B = float(np.sum(V_sol[indici_B] * g_cella_B))

    dV_dy, dV_dx = np.gradient(V_2d, dy, dx)
    Jx = -sigma * dV_dx
    Jy = -sigma * dV_dy
    J_mag = np.sqrt(Jx ** 2 + Jy ** 2)

    I_y_profile = np.sum(Jy, axis=1) * dx * s

    volume_cella = dx * dy * s
    potenza_piastra_watt = float(np.sum((J_mag ** 2) * rho_ohm_mm * volume_cella))
    potenza_ramo_A_watt = float((I_A ** 2) * R_tot_ramo_ohm)
    potenza_ramo_B_watt = float((I_B ** 2) * R_tot_ramo_ohm)

    return {
        "x": x_centers, "y": y_centers, "X": X_grid, "Y": Y_grid,
        "V_mV": V_2d * 1e3, "Jx": Jx, "Jy": Jy, "J_mag": J_mag,
        "I_A": I_A, "I_B": I_B, "I_y_profile": I_y_profile,
        "coords_pl": coords_pl,
        "potenza_piastra_watt": potenza_piastra_watt,
        "potenza_ramo_A_watt": potenza_ramo_A_watt,
        "potenza_ramo_B_watt": potenza_ramo_B_watt,
        "rho_T": rho_T,
        "R_ala_uohm": R_ala_ohm * 1e6,
        "R_tot_ramo_uohm": R_tot_ramo_ohm * 1e6
    }


# --- VALIDAZIONE GEOMETRICA ---
if (y_sorm_A - h_L / 2.0 < 0) or (y_sorm_A + h_L / 2.0 > H_piastra):
    st.warning("Attenzione: Il sormonto della Barra A sporge oltre i bordi inferiore/superiore della piastra collettrice.")
if (y_sorm_B - h_L / 2.0 < 0) or (y_sorm_B + h_L / 2.0 > H_piastra):
    st.warning("Attenzione: Il sormonto della Barra B sporge oltre i bordi inferiore/superiore della piastra collettrice.")

ris = risolvi_piastra_2d(
    B_piastra, H_piastra, s_piastra, rho_20_mat, alpha_tcr, T_operativa,
    I_totale, X_col, Y_row, passo_x, passo_y, y_offset_pl, diam_pl,
    w_L, h_L, s_L, ala_L, y_sorm_A, y_sorm_B, R_cont_puro_uohm, passo_mesh
)

# --- METRICHE DI RIEPILOGO ---
perc_A = (ris["I_A"] / I_totale) * 100.0
perc_B = (ris["I_B"] / I_totale) * 100.0
sezione_L = w_L * s_L
J_barra_A = ris["I_A"] / sezione_L
J_barra_B = ris["I_B"] / sezione_L
J_picco_piastra = float(np.max(ris["J_mag"]))
J_media_piastra = float(np.mean(ris["J_mag"]))
potenza_totale_watt = ris["potenza_piastra_watt"] + ris["potenza_ramo_A_watt"] + ris["potenza_ramo_B_watt"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Corrente Barra A (Inferiore)",
        value=f"{ris['I_A']:.1f} A ({perc_A:.1f}%)",
        delta=f"J sezione L: {J_barra_A:.2f} A/mm² (K1={K1_mat:.2f})",
        delta_color="inverse" if J_barra_A > K1_mat else "normal"
    )
with col2:
    st.metric(
        label="Corrente Barra B (Superiore)",
        value=f"{ris['I_B']:.1f} A ({perc_B:.1f}%)",
        delta=f"J sezione L: {J_barra_B:.2f} A/mm² (K1={K1_mat:.2f})",
        delta_color="inverse" if J_barra_B > K1_mat else "normal"
    )
with col3:
    st.metric(
        label="Picco Densità Piastra |J|",
        value=f"{J_picco_piastra:.2f} A/mm²",
        delta=f"Media: {J_media_piastra:.2f} A/mm² (Rif K1={K1_mat:.2f})",
        delta_color="inverse" if J_picco_piastra > K1_mat * 1.5 else "normal"
    )
with col4:
    st.metric(
        label="Potenza Dissipata Piastra",
        value=f"{ris['potenza_piastra_watt']:.1f} watt",
        delta=f"Sistema Totale: {potenza_totale_watt:.1f} watt",
        delta_color="off"
    )

st.markdown("---")

# --- GRAFICI IN SCALA GEOMETRICA REALE 1:1 (NESSUNO STIRAMENTO ASSE Y) ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(
    1, 4,
    figsize=(20, 10),
    gridspec_kw={'width_ratios': [1.0, 1.0, 1.1, 1.0]}
)

# 1. MAPPA DENSITÀ DI CORRENTE |J| (Scala 1:1)
im1 = ax1.pcolormesh(ris["X"], ris["Y"], ris["J_mag"], shading='auto', cmap='inferno', vmin=0, vmax=max(K1_mat * 1.5, J_picco_piastra))
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label("Densità |J| [A/mm²]", fontsize=9)
ax1.streamplot(ris["x"], ris["y"], ris["Jx"], ris["Jy"], color='cyan', linewidth=0.7, density=0.8, arrowsize=0.7)

if J_picco_piastra > K1_mat:
    ax1.contour(ris["X"], ris["Y"], ris["J_mag"], levels=[K1_mat], colors=['lime'], linewidths=1.5, linestyles='--')

for (xp, yp) in ris["coords_pl"]:
    circ = patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.3)
    ax1.add_patch(circ)

x_min_L = (B_piastra - w_L) / 2.0
rect_A = patches.Rectangle((x_min_L, y_sorm_A - h_L / 2.0), w_L, h_L, edgecolor='cyan', facecolor='none', linewidth=2.0)
rect_B = patches.Rectangle((x_min_L, y_sorm_B - h_L / 2.0), w_L, h_L, edgecolor='cyan', facecolor='none', linewidth=2.0)
ax1.add_patch(rect_A)
ax1.add_patch(rect_B)
ax1.axhline(y_ala_A, color='cyan', linestyle=':', linewidth=1.2)
ax1.axhline(y_ala_B, color='cyan', linestyle=':', linewidth=1.2)

ax1.text(B_piastra / 2.0, y_sorm_A, f"A: {perc_A:.1f}%", color='white', weight='bold', fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.75))
ax1.text(B_piastra / 2.0, y_sorm_B, f"B: {perc_B:.1f}%", color='white', weight='bold', fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.75))

ax1.set_title("1. Densità |J| [A/mm²] (Scala 1:1)", fontsize=10, weight='bold')
ax1.set_xlabel("Larghezza X [mm]")
ax1.set_ylabel("Quota Y [mm]")
ax1.set_xlim(0, B_piastra)
ax1.set_ylim(0, H_piastra)
ax1.set_aspect('equal', adjustable='box')

# 2. MAPPA POTENZIALE ELETTRICO V [mV] (Scala 1:1)
im2 = ax2.pcolormesh(ris["X"], ris["Y"], ris["V_mV"], shading='auto', cmap='viridis')
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label("Potenziale [mV]", fontsize=9)

for (xp, yp) in ris["coords_pl"]:
    circ2 = patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.1, alpha=0.8)
    ax2.add_patch(circ2)

rect_A2 = patches.Rectangle((x_min_L, y_sorm_A - h_L / 2.0), w_L, h_L, edgecolor='red', facecolor='none', linewidth=2.0)
rect_B2 = patches.Rectangle((x_min_L, y_sorm_B - h_L / 2.0), w_L, h_L, edgecolor='red', facecolor='none', linewidth=2.0)
ax2.add_patch(rect_A2)
ax2.add_patch(rect_B2)

ax2.set_title("2. Potenziale V [mV] (Scala 1:1)", fontsize=10, weight='bold')
ax2.set_xlabel("Larghezza X [mm]")
ax2.set_xlim(0, B_piastra)
ax2.set_ylim(0, H_piastra)
ax2.set_aspect('equal', adjustable='box')

# 3. PROFILO CARICO ASSIALE |Iy| LUNGO L'ALTEZZA
ax3.plot(np.abs(ris["I_y_profile"]), ris["y"], color='#d95f02', linewidth=2.2, label="|Iy(y)| in piastra")
ax3.axhline(y_ala_A, color='blue', linestyle='--', linewidth=1.4, label=f"Uscita A (Y={y_ala_A:.0f} mm)")
ax3.axhline(y_ala_B, color='green', linestyle='--', linewidth=1.4, label=f"Uscita B (Y={y_ala_B:.0f} mm)")
portata_k1_piastra = B_piastra * s_piastra * K1_mat
ax3.axvline(portata_k1_piastra, color='red', linestyle=':', linewidth=1.5, label=f"Portata K1 ({portata_k1_piastra:.0f} A)")
ax3.fill_betweenx(ris["y"], 0, np.abs(ris["I_y_profile"]), color='#d95f02', alpha=0.2)
ax3.set_title("3. Carico Sezione Verticale |Iy| [A]", fontsize=10, weight='bold')
ax3.set_xlabel("Corrente Assiale [A]")
ax3.set_ylabel("Quota Y [mm]")
ax3.set_ylim(0, H_piastra)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper right', fontsize=8)

# 4. VISTA LATERALE SEZIONE MECCANICA Y-Z IN SCALA REALE 1:1
prof_pl_schema = diam_pl * 1.4  # Proporzione schematica codolo Powerlock

# Piastra Collettrice Verticale
piastra_rett = patches.Rectangle((0, 0), s_piastra, H_piastra, facecolor='#b87333', edgecolor='black', linewidth=1.2)
ax4.add_patch(piastra_rett)

# Barra A: Sormonto verticale + Ala orizzontale a sbalzo
l_vert_A = patches.Rectangle((s_piastra, y_sorm_A - h_L / 2.0), s_L, h_L, facecolor='#e68a00', edgecolor='black', linewidth=1.2)
l_oriz_A = patches.Rectangle((s_piastra, y_ala_A - s_L / 2.0), ala_L, s_L, facecolor='#e68a00', edgecolor='black', linewidth=1.2)
ax4.add_patch(l_vert_A)
ax4.add_patch(l_oriz_A)

# Barra B: Sormonto verticale + Ala orizzontale a sbalzo
l_vert_B = patches.Rectangle((s_piastra, y_sorm_B - h_L / 2.0), s_L, h_L, facecolor='#e68a00', edgecolor='black', linewidth=1.2)
l_oriz_B = patches.Rectangle((s_piastra, y_ala_B - s_L / 2.0), ala_L, s_L, facecolor='#e68a00', edgecolor='black', linewidth=1.2)
ax4.add_patch(l_vert_B)
ax4.add_patch(l_oriz_B)

# Linee d'asse mezzeria uscita e mezzeria sormonto
ax4.axhline(y_ala_A, color='blue', linestyle='-.', linewidth=0.9)
ax4.axhline(y_ala_B, color='green', linestyle='-.', linewidth=0.9)

ax4.text(s_piastra + ala_L + 8, y_ala_A, f"Uscita A (Y={y_ala_A:.0f})\n{ris['I_A']:.0f} A ({perc_A:.1f}%)", color='blue', fontsize=8, weight='bold', va='center')
ax4.text(s_piastra + ala_L + 8, y_ala_B, f"Uscita B (Y={y_ala_B:.0f})\n{ris['I_B']:.0f} A ({perc_B:.1f}%)", color='green', fontsize=8, weight='bold', va='center')

# Powerlock sul lato sinistro (-Z) in scala 1:1
y_pl_unici = sorted(list(set([yp for (xp, yp) in ris["coords_pl"]])))
for yp in y_pl_unici:
    pl_box = patches.Rectangle((-prof_pl_schema, yp - diam_pl / 2.0), prof_pl_schema, diam_pl, facecolor='#3a3a3a', edgecolor='black', linewidth=0.9)
    ax4.add_patch(pl_box)

ax4.set_title("4. Vista Laterale Y-Z (Scala 1:1)", fontsize=10, weight='bold')
ax4.set_xlabel("Profondità Z [mm]")
ax4.set_xlim(-prof_pl_schema - 30, s_piastra + ala_L + 140)
ax4.set_ylim(0, H_piastra)
ax4.set_aspect('equal', adjustable='box')
ax4.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
st.pyplot(fig)

# --- TABELLA DI REPORT TECNICO ---
st.subheader("Riepilogo Grandezze Elettriche, Termiche e Parametri Materiale")
df_report = pd.DataFrame({
    "Parametro Tecnico": [
        "Materiale Selezionato",
        "Coefficiente Materiale K1 (Densità di riferimento)",
        "Resistività a 20 °C / Resistività Operativa a Caldo",
        "Resistenza Elettrica Singolo Ramo a L (Contatto + Sbalzo)",
        "Quota YBarra A (Mezzeria Sormonto / Mezzeria Ala Uscita)",
        "Quota YBarra B (Mezzeria Sormonto / Mezzeria Ala Uscita)",
        "Corrente Uscita Barra A (Inferiore)",
        "Corrente Uscita Barra B (Superiore)",
        "Densità di Corrente Sezione Barra A / Barra B",
        "Potenza Dissipata nella Piastra Collettrice",
        "Potenza Dissipata Giunto + Ala Sbalzo Barra A",
        "Potenza Dissipata Giunto + Ala Sbalzo Barra B",
        "Potenza Termica Totale Sistema"
    ],
    "Valore Calcolato": [
        mat_scelto,
        f"{K1_mat:.2f} A/mm² (K_cw1 = {dati_mat['K_cw1']})",
        f"{rho_20_mat:.4f} uOhm·m  -->  {ris['rho_T']:.4f} uOhm·m (a {T_operativa} °C)",
        f"{ris['R_tot_ramo_uohm']:.2f} micro-ohm ({R_cont_puro_uohm:.2f} contatto + {ris['R_ala_uohm']:.2f} ala sbalzo)",
        f"Sormonto: {y_sorm_A:.1f} mm  |  Ala Uscita: {y_ala_A:.1f} mm",
        f"Sormonto: {y_sorm_B:.1f} mm  |  Ala Uscita: {y_ala_B:.1f} mm",
        f"{ris['I_A']:.1f} A ({perc_A:.2f} %)",
        f"{ris['I_B']:.1f} A ({perc_B:.2f} %)",
        f"Barra A: {J_barra_A:.2f} A/mm²  |  Barra B: {J_barra_B:.2f} A/mm²",
        f"{ris['potenza_piastra_watt']:.2f} watt",
        f"{ris['potenza_ramo_A_watt']:.2f} watt",
        f"{ris['potenza_ramo_B_watt']:.2f} watt",
        f"{potenza_totale_watt:.2f} watt"
    ],
    "Verifica Tecnica (Standard Telema / CEI EN 61439-1)": [
        f"Densità: {dati_mat['densita']:.0f} kg/m³ | Calore spec.: {dati_mat['c_sp']:.0f} J/(kg·K)",
        "Legge di potenza K1 = 0,1778 * rho^(-0,487)",
        f"TCR applicato alpha_20 = {alpha_tcr:.5f} K^-1",
        f"Lunghezza sbalzo L = {ala_L:.0f} mm su sezione {w_L:.0f}x{s_L:.0f} mm",
        orient_piega_A,
        orient_piega_B,
        "Ramo scarico" if perc_A < 45 else ("Bilanciato" if perc_A <= 55 else "Ramo caricato"),
        "Ramo scarico" if perc_B < 45 else ("Bilanciato" if perc_B <= 55 else "Ramo caricato"),
        "OK (<= K1)" if max(J_barra_A, J_barra_B) <= K1_mat else "SUPERAMENTO FATTORE K1",
        "Calcolo integrale 2D effetto Joule",
        "P = I_A² * R_ramo",
        "P = I_B² * R_ramo",
        "Da smaltire tramite ventilazione quadro"
    ]
})
st.table(df_report)
