import streamlit as st
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

st.set_page_config(
    page_title="LOADBANK-MASTER | Ripartizione Corrente Busbar 2D",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATABASE MATERIALI TELEMA DA TABELLA EXCEL ---
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

st.title("LOADBANK-MASTER | Calcolo Ripartizione Corrente e Densità 2D")
st.markdown("Verifica della distribuzione di corrente su piastra collettrice con Powerlock multipli e derivazioni a L.")

# --- SIDEBAR 1: MATERIALE E PIASTRA ---
st.sidebar.header("1. Materiale e Piastra Principale")
mat_scelto = st.sidebar.selectbox("Materiale", list(MATERIALI_DB.keys()), index=0)
dati_mat = MATERIALI_DB[mat_scelto]
K1_mat = dati_mat["K1"]
rho_mat = dati_mat["rho_20"]

st.sidebar.caption(f"Fattore ammissibile K1: **{K1_mat:.2f} A/mm²** | Resistività: **{rho_mat:.4f} µΩ·m**")

B_piastra = st.sidebar.number_input("Larghezza Piastra B [mm]", min_value=50.0, max_value=1000.0, value=200.0, step=10.0)
H_piastra = st.sidebar.number_input("Altezza Piastra H [mm]", min_value=200.0, max_value=3000.0, value=1000.0, step=50.0)
s_piastra = st.sidebar.number_input("Spessore Piastra s [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0)

# --- SIDEBAR 2: GRIGLIA POWERLOCK ---
st.sidebar.header("2. Griglia Powerlock (Ingresso)")
I_totale = st.sidebar.number_input("Corrente Totale [A]", min_value=100.0, max_value=40000.0, value=7200.0, step=100.0)
X_col = st.sidebar.number_input("Colonne Powerlock (X)", min_value=1, max_value=6, value=2, step=1)
Y_row = st.sidebar.number_input("Righe Powerlock (Y)", min_value=1, max_value=20, value=9, step=1)
passo_x = st.sidebar.number_input("Passo Orizzontale Colonne [mm]", min_value=20.0, max_value=300.0, value=80.0, step=5.0)
passo_y = st.sidebar.number_input("Passo Verticale Righe [mm]", min_value=20.0, max_value=300.0, value=85.0, step=5.0)
y_offset_pl = st.sidebar.number_input("Quota Y Prima Riga dal Basso [mm]", min_value=10.0, max_value=1000.0, value=160.0, step=10.0)
diam_pl = st.sidebar.number_input("Diametro Ingombro Powerlock [mm]", min_value=15.0, max_value=80.0, value=32.0, step=2.0)

# --- SIDEBAR 3: BARRE A L ---
st.sidebar.header("3. Barre a L (Uscita)")
w_L = st.sidebar.number_input("Larghezza Barre a L [mm]", min_value=20.0, max_value=400.0, value=120.0, step=10.0)
h_L = st.sidebar.number_input("Altezza Sormonto Bullonato [mm]", min_value=15.0, max_value=200.0, value=60.0, step=5.0)
s_L = st.sidebar.number_input("Spessore Barre a L [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0)

y_barra_A = st.sidebar.slider("Quota Y Centro Barra A (Inferiore) [mm]", min_value=float(h_L / 2), max_value=float(H_piastra - h_L / 2), value=50.0, step=5.0)
y_barra_B = st.sidebar.slider("Quota Y Centro Barra B (Superiore) [mm]", min_value=float(h_L / 2), max_value=float(H_piastra - h_L / 2), value=500.0, step=5.0)

# Risoluzione mesh fissa per calcolo istantaneo
d_mesh = 4.0

# --- SOLUTORE ELETTRODINAMICO 2D FDM ---
def calcola_distribuzione(B, H, s, rho_uohm_m, I_tot, n_col, n_row, px, py, y0_pl, d_pl, wL, hL, yA, yB, mesh_sz):
    rho_ohm_mm = rho_uohm_m * 1e-3
    sigma = 1.0 / rho_ohm_mm

    nx = max(10, int(round(B / mesh_sz)))
    ny = max(20, int(round(H / mesh_sz)))
    dx = B / nx
    dy = H / ny

    x_c = (np.arange(nx) + 0.5) * dx
    y_c = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x_c, y_c)

    Gx = sigma * (s * dy) / dx
    Gy = sigma * (s * dx) / dy

    N = nx * ny
    G = sp.lil_matrix((N, N), dtype=np.float64)
    I = np.zeros(N, dtype=np.float64)

    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            diag = 0.0
            if i > 0:
                G[k, k - 1] = -Gx
                diag += Gx
            if i < nx - 1:
                G[k, k + 1] = -Gx
                diag += Gx
            if j > 0:
                G[k, k - nx] = -Gy
                diag += Gy
            if j < ny - 1:
                G[k, k + nx] = -Gy
                diag += Gy
            G[k, k] = diag

    # Iniezione corrente Powerlock
    x0_pl = (B - (n_col - 1) * px) / 2.0
    n_tot_pl = n_col * n_row
    I_sing = I_tot / n_tot_pl
    r_pl = d_pl / 2.0
    coords_pl = []

    for r in range(n_row):
        yp = y0_pl + r * py
        for c in range(n_col):
            xp = x0_pl + c * px
            coords_pl.append((xp, yp))
            dist2 = (X - xp) ** 2 + (Y - yp) ** 2
            mask = dist2 <= r_pl ** 2
            ind = np.flatnonzero(mask)
            if len(ind) == 0:
                ind = [np.argmin(dist2)]
            I[ind] += I_sing / len(ind)

    # Identificazione nodi sormonto Barre A e B
    x_min_L = (B - wL) / 2.0
    x_max_L = (B + wL) / 2.0
    mask_A = (X >= x_min_L) & (X <= x_max_L) & (Y >= yA - hL / 2.0) & (Y <= yA + hL / 2.0)
    mask_B = (X >= x_min_L) & (X <= x_max_L) & (Y >= yB - hL / 2.0) & (Y <= yB + hL / 2.0)
    ind_A = np.flatnonzero(mask_A)
    ind_B = np.flatnonzero(mask_B)

    # Conduttanza di scarico ideale di interfaccia
    g_scarico = 1e6
    for idx in ind_A:
        G[idx, idx] += g_scarico
    for idx in ind_B:
        G[idx, idx] += g_scarico

    G_csr = G.tocsr()
    V = spla.spsolve(G_csr, I)
    V_2d = V.reshape((ny, nx))

    # Correnti raccolte dai due rami
    I_A = float(np.sum(V[ind_A] * g_scarico))
    I_B = float(np.sum(V[ind_B] * g_scarico))

    # Bilanciamento conservativo di corrente
    somma_I = I_A + I_B
    if somma_I > 0:
        I_A = I_tot * (I_A / somma_I)
        I_B = I_tot * (I_B / somma_I)

    dV_dy, dV_dx = np.gradient(V_2d, dy, dx)
    Jx = -sigma * dV_dx
    Jy = -sigma * dV_dy
    J_mag = np.sqrt(Jx ** 2 + Jy ** 2)

    volume_cella = dx * dy * s
    potenza_watt = float(np.sum((J_mag ** 2) * rho_ohm_mm * volume_cella))
    profilo_Iy = np.sum(Jy, axis=1) * dx * s

    return {
        "x": x_c, "y": y_c, "X": X, "Y": Y,
        "V_mV": V_2d * 1e3, "Jx": Jx, "Jy": Jy, "J_mag": J_mag,
        "I_A": I_A, "I_B": I_B, "profilo_Iy": profilo_Iy,
        "coords_pl": coords_pl, "potenza_watt": potenza_watt
    }

res = calcola_distribuzione(
    B_piastra, H_piastra, s_piastra, rho_mat, I_totale,
    X_col, Y_row, passo_x, passo_y, y_offset_pl, diam_pl,
    w_L, h_L, y_barra_A, y_barra_B, d_mesh
)

# Metriche e verifiche
p_A = (res["I_A"] / I_totale) * 100.0
p_B = (res["I_B"] / I_totale) * 100.0
sez_L = w_L * s_L
J_L_A = res["I_A"] / sez_L
J_L_B = res["I_B"] / sez_L
J_max = float(np.max(res["J_mag"]))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Corrente Barra A (Bassa)", f"{res['I_A']:.0f} A ({p_A:.1f}%)", f"Densità L: {J_L_A:.2f} A/mm²")
c2.metric("Corrente Barra B (Alta)", f"{res['I_B']:.0f} A ({p_B:.1f}%)", f"Densità L: {J_L_B:.2f} A/mm²")
c3.metric("Picco Densità Piastra", f"{J_max:.2f} A/mm²", f"Limite K1: {K1_mat:.2f} A/mm²", delta_color="inverse" if J_max > K1_mat * 1.5 else "normal")
c4.metric("Dissipazione Piastra", f"{res['potenza_watt']:.1f} watt", f"{(res['potenza_watt']/1000.0):.2f} chilowatt", delta_color="off")

st.markdown("---")

# --- GRAFICA IN SCALA REALE 1:1 ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10), gridspec_kw={'width_ratios': [1.1, 1.1, 1.0, 0.9]})

# 1. MAPPA DENSITÀ |J|
im1 = ax1.pcolormesh(res["X"], res["Y"], res["J_mag"], shading='auto', cmap='inferno', vmin=0, vmax=max(K1_mat * 1.5, J_max))
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Densità [A/mm²]")
ax1.streamplot(res["x"], res["y"], res["Jx"], res["Jy"], color='cyan', linewidth=0.7, density=0.8, arrowsize=0.7)

for (xp, yp) in res["coords_pl"]:
    ax1.add_patch(patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.2))

x_min_L = (B_piastra - w_L) / 2.0
ax1.add_patch(patches.Rectangle((x_min_L, y_barra_A - h_L / 2.0), w_L, h_L, edgecolor='cyan', facecolor='none', linewidth=2.0))
ax1.add_patch(patches.Rectangle((x_min_L, y_barra_B - h_L / 2.0), w_L, h_L, edgecolor='cyan', facecolor='none', linewidth=2.0))
ax1.text(B_piastra / 2.0, y_barra_A, f"A: {p_A:.1f}%", color='white', weight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.75))
ax1.text(B_piastra / 2.0, y_barra_B, f"B: {p_B:.1f}%", color='white', weight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.75))

ax1.set_title("1. Densità |J| (Scala 1:1)", fontsize=10, weight='bold')
ax1.set_xlim(0, B_piastra)
ax1.set_ylim(0, H_piastra)
ax1.set_aspect('equal', adjustable='box')

# 2. POTENZIALE V
im2 = ax2.pcolormesh(res["X"], res["Y"], res["V_mV"], shading='auto', cmap='viridis')
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Potenziale [mV]")

for (xp, yp) in res["coords_pl"]:
    ax2.add_patch(patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.0, alpha=0.7))

ax2.add_patch(patches.Rectangle((x_min_L, y_barra_A - h_L / 2.0), w_L, h_L, edgecolor='red', facecolor='none', linewidth=2.0))
ax2.add_patch(patches.Rectangle((x_min_L, y_barra_B - h_L / 2.0), w_L, h_L, edgecolor='red', facecolor='none', linewidth=2.0))

ax2.set_title("2. Potenziale [mV] (Scala 1:1)", fontsize=10, weight='bold')
ax2.set_xlim(0, B_piastra)
ax2.set_ylim(0, H_piastra)
ax2.set_aspect('equal', adjustable='box')

# 3. CARICO VERTICALE |Iy|
ax3.plot(np.abs(res["profilo_Iy"]), res["y"], color='#d95f02', linewidth=2.2, label="|Iy(y)|")
ax3.axhline(y_barra_A, color='blue', linestyle='--', linewidth=1.2, label=f"Barra A ({y_barra_A:.0f} mm)")
ax3.axhline(y_barra_B, color='green', linestyle='--', linewidth=1.2, label=f"Barra B ({y_barra_B:.0f} mm)")
ax3.fill_betweenx(res["y"], 0, np.abs(res["profilo_Iy"]), color='#d95f02', alpha=0.2)
ax3.set_title("3. Flusso Assiale |Iy| [A]", fontsize=10, weight='bold')
ax3.set_xlabel("Amperaggio [A]")
ax3.set_ylim(0, H_piastra)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper right', fontsize=8)

# 4. SCHEMA LATERALE Y-Z (Scala 1:1)
sbalzo_disegno = 90.0
prof_pl_disegno = 45.0

# Piastra
ax4.add_patch(patches.Rectangle((0, 0), s_piastra, H_piastra, facecolor='#b87333', edgecolor='black', linewidth=1.2))

# Barre a L
ax4.add_patch(patches.Rectangle((s_piastra, y_barra_A - h_L / 2.0), s_L, h_L, facecolor='#e68a00', edgecolor='black'))
ax4.add_patch(patches.Rectangle((s_piastra, y_barra_A - s_L / 2.0), sbalzo_disegno, s_L, facecolor='#e68a00', edgecolor='black'))

ax4.add_patch(patches.Rectangle((s_piastra, y_barra_B - h_L / 2.0), s_L, h_L, facecolor='#e68a00', edgecolor='black'))
ax4.add_patch(patches.Rectangle((s_piastra, y_barra_B - s_L / 2.0), sbalzo_disegno, s_L, facecolor='#e68a00', edgecolor='black'))

ax4.text(s_piastra + sbalzo_disegno + 8, y_barra_A, f"A: {p_A:.1f}%", color='blue', fontsize=8, weight='bold', va='center')
ax4.text(s_piastra + sbalzo_disegno + 8, y_barra_B, f"B: {p_B:.1f}%", color='green', fontsize=8, weight='bold', va='center')

# Powerlock
y_pl_unici = sorted(list(set([yp for (xp, yp) in res["coords_pl"]])))
for yp in y_pl_unici:
    ax4.add_patch(patches.Rectangle((-prof_pl_disegno, yp - diam_pl / 2.0), prof_pl_disegno, diam_pl, facecolor='#333333', edgecolor='black'))

ax4.set_title("4. Schema Laterale Y-Z", fontsize=10, weight='bold')
ax4.set_xlim(-prof_pl_disegno - 25, s_piastra + sbalzo_disegno + 85)
ax4.set_ylim(0, H_piastra)
ax4.set_aspect('equal', adjustable='box')
ax4.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
st.pyplot(fig)

# --- TABELLA RIASSUNTIVA ---
st.subheader("Verifica Dimensionale e Termoelettrica")
st.table(pd.DataFrame({
    "Parametro": [
        "Materiale e Limite Ammissibile K1",
        "Corrente Totale / Singolo Powerlock",
        "Uscita Barra A (Inferiore)",
        "Uscita Barra B (Superiore)",
        "Densità Sezione Barre a L",
        "Potenza Termica Dissipata Piastra"
    ],
    "Valore": [
        f"{mat_scelto} (K1 = {K1_mat:.2f} A/mm²)",
        f"{I_totale:.0f} A  |  {I_totale/(X_col*Y_row):.1f} A per connettore ({X_col*Y_row} poli)",
        f"{res['I_A']:.1f} A ({p_A:.1f} %)",
        f"{res['I_B']:.1f} A ({p_B:.1f} %)",
        f"Barra A: {J_L_A:.2f} A/mm²  |  Barra B: {J_L_B:.2f} A/mm²",
        f"{res['potenza_watt']:.1f} watt ({(res['potenza_watt']/1000.0):.2f} chilowatt)"
    ],
    "Esito": [
        "Riferimento norma / tabella",
        "Iniezione uniforme",
        "Bilanciato" if 45 <= p_A <= 55 else ("Sovraccarico" if p_A > 55 else "Scarico"),
        "Bilanciato" if 45 <= p_B <= 55 else ("Sovraccarico" if p_B > 55 else "Scarico"),
        "OK" if max(J_L_A, J_L_B) <= K1_mat else "ATTENZIONE: Supera K1",
        "Da smaltire nel quadro per convezione/ventilazione"
    ]
}))
