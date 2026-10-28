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

st.title("LOADBANK-MASTER | Ripartizione Corrente e Layout Piastra Powerlock")
st.markdown(
    "Solutore alle differenze finite 2D vettorializzato per piastre collettrici con connettori **Powerlock** "
    "e derivazioni piegate a L. Verifica sezioni secondo parametro materiale **$K_1$**."
)

# --- SIDEBAR 1: MATERIALE E PIASTRA ---
st.sidebar.header("1. Materiale e Piastra Collettrice")
mat_scelto = st.sidebar.selectbox("Materiale Sbarre", list(MATERIALI_DB.keys()), index=0)
dati_mat = MATERIALI_DB[mat_scelto]
K1_mat = dati_mat["K1"]
rho_mat = dati_mat["rho_20"]
dens_mat = dati_mat["densita"]
csp_mat = dati_mat["c_sp"]

st.sidebar.caption(
    f"**Dati Tabella:** $K_1$ = **{K1_mat:.2f} A/mm²** | "
    f"$\\rho_{{20}}$ = **{rho_mat:.4f} µΩ·m** | $\\delta$ = **{dens_mat:.0f} kg/m³**"
)

B_piastra = st.sidebar.number_input("Larghezza Piastra B [mm]", min_value=50.0, max_value=1000.0, value=200.0, step=10.0)
H_piastra = st.sidebar.number_input("Altezza Piastra H [mm]", min_value=200.0, max_value=3000.0, value=1000.0, step=50.0)
s_piastra = st.sidebar.number_input("Spessore Piastra s [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0)

# --- SIDEBAR 2: GRIGLIA POWERLOCK ---
st.sidebar.header("2. Griglia Powerlock (Ingresso)")
I_totale = st.sidebar.number_input("Corrente Totale Iniettata [A]", min_value=100.0, max_value=40000.0, value=3600.0, step=100.0)
X_col = st.sidebar.number_input("Colonne Powerlock (X)", min_value=1, max_value=10, value=2, step=1)
Y_row = st.sidebar.number_input("Righe Powerlock (Y)", min_value=1, max_value=25, value=9, step=1)
passo_x = st.sidebar.number_input("Passo Orizzontale Colonne [mm]", min_value=20.0, max_value=300.0, value=80.0, step=5.0)
passo_y = st.sidebar.number_input("Passo Verticale Righe [mm]", min_value=20.0, max_value=300.0, value=85.0, step=5.0)
y_offset_pl = st.sidebar.number_input("Quota Y Prima Riga dal Basso [mm]", min_value=10.0, max_value=2500.0, value=160.0, step=10.0)
diam_pl = st.sidebar.number_input("Diametro Impronta Powerlock [mm]", min_value=15.0, max_value=80.0, value=32.0, step=2.0)

# --- SIDEBAR 3: BARRE PIEGATE A L (USCITA) ---
st.sidebar.header("3. Barre a L in Uscita")
w_L = st.sidebar.number_input("Larghezza Barre a L [mm]", min_value=20.0, max_value=600.0, value=120.0, step=10.0)
h_L = st.sidebar.number_input("Altezza Sormonto Bullonato h_L [mm]", min_value=15.0, max_value=300.0, value=60.0, step=5.0)
s_L = st.sidebar.number_input("Spessore Barre a L s_L [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0)
L_uscita = st.sidebar.number_input(
    "Lunghezza Barra a L in Uscita [mm]",
    min_value=20.0, max_value=1000.0, value=120.0, step=10.0,
    help="Lunghezza del tratto perpendicolare in uscita dopo la piega. Determina la resistenza ohmica propria di ciascuna barra a L."
)

verso_piega_A = st.sidebar.selectbox(
    "Orientamento Sormonto Barra A (Inferiore)",
    ["Sormonto piegato verso l'ALTO (L dritta)", "Sormonto piegato verso il BASSO (L rovescia)"],
    index=0
)
verso_piega_B = st.sidebar.selectbox(
    "Orientamento Sormonto Barra B (Superiore)",
    ["Sormonto piegato verso l'ALTO (L dritta)", "Sormonto piegato verso il BASSO (L rovescia)"],
    index=1
)

# Calcolo limiti sicuri per gli slider della quota Y dell'ala in uscita
min_y_A = (s_L / 2.0) if "ALTO" in verso_piega_A else (h_L - s_L / 2.0)
max_y_A = (H_piastra - h_L + s_L / 2.0) if "ALTO" in verso_piega_A else (H_piastra - s_L / 2.0)

min_y_B = (s_L / 2.0) if "ALTO" in verso_piega_B else (h_L - s_L / 2.0)
max_y_B = (H_piastra - h_L + s_L / 2.0) if "ALTO" in verso_piega_B else (H_piastra - s_L / 2.0)

if max_y_A <= min_y_A or max_y_B <= min_y_B:
    st.error("Errore geometrico: L'altezza del sormonto h_L supera l'altezza della piastra H.")
    st.stop()

val_default_A = float(min(max(30.0, min_y_A), max_y_A))
val_default_B = float(min(max(500.0, min_y_B), max_y_B))

y_ala_A = st.sidebar.slider(
    "Quota Y Mezzeria Uscita Barra A (Bassa) [mm]",
    min_value=float(min_y_A), max_value=float(max_y_A), value=val_default_A, step=5.0
)
y_ala_B = st.sidebar.slider(
    "Quota Y Mezzeria Uscita Barra B (Alta) [mm]",
    min_value=float(min_y_B), max_value=float(max_y_B), value=val_default_B, step=5.0
)

# Ricavo coordinate del centro del sormonto sulla piastra
y_sorm_A = (y_ala_A - s_L / 2.0 + h_L / 2.0) if "ALTO" in verso_piega_A else (y_ala_A + s_L / 2.0 - h_L / 2.0)
y_sorm_B = (y_ala_B - s_L / 2.0 + h_L / 2.0) if "ALTO" in verso_piega_B else (y_ala_B + s_L / 2.0 - h_L / 2.0)

# --- CONTROLLI GEOMETRICI RIGOROSI CON BLOCCO (st.stop) ---
ingombro_x_pl = (X_col - 1) * passo_x + diam_pl
y_min_pl = y_offset_pl - diam_pl / 2.0
y_max_pl = y_offset_pl + (Y_row - 1) * passo_y + diam_pl / 2.0

errori_geom = []
if ingombro_x_pl > B_piastra:
    errori_geom.append(f"La griglia Powerlock è larga {ingombro_x_pl:.1f} mm e deborda dalla larghezza piastra B ({B_piastra:.1f} mm).")
if y_min_pl < 0 or y_max_pl > H_piastra:
    errori_geom.append(f"Le righe Powerlock vanno da Y={y_min_pl:.1f} mm a Y={y_max_pl:.1f} mm, uscendo dall'altezza piastra H (0 - {H_piastra:.1f} mm).")
if w_L > B_piastra:
    errori_geom.append(f"La larghezza delle barre a L ({w_L:.1f} mm) supera la larghezza della piastra ({B_piastra:.1f} mm).")
if abs(y_sorm_A - y_sorm_B) < h_L:
    errori_geom.append(
        f"Collisione meccanica: I sormonti delle Barre A e B si sovrappongono! "
        f"(Distanza centri sormonto = {abs(y_sorm_A - y_sorm_B):.1f} mm < h_L = {h_L:.1f} mm)."
    )

if errori_geom:
    for err in errori_geom:
        st.error(err)
    st.stop()


# --- SOLUTORE ELETTRODINAMICO 2D FDM VETTORIALIZZATO E CACHATO ---
@st.cache_data(show_spinner=False)
def risolvi_busbar_2d(B, H, s, rho_uohm_m, I_tot, n_col, n_row, px, py, y0_pl, d_pl, wL, hL, sL, L_usc, y_sA, y_sB):
    rho_ohm_mm = rho_uohm_m * 1e-3
    sigma = 1.0 / rho_ohm_mm

    # Dimensione mesh adattiva per garantire velocità e risoluzione
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

    # Costruzione vettorializzata dei legami orizzontali (X) e verticali (Y)
    k_left = idx_grid[:, :-1].ravel()
    k_right = idx_grid[:, 1:].ravel()
    np.add.at(diag_main, k_left, Gx)
    np.add.at(diag_main, k_right, Gx)

    k_bot = idx_grid[:-1, :].ravel()
    k_top = idx_grid[1:, :].ravel()
    np.add.at(diag_main, k_bot, Gy)
    np.add.at(diag_main, k_top, Gy)

    # Vettore correnti iniettate dai Powerlock
    I_vec = np.zeros(N, dtype=np.float64)
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
            ind = np.flatnonzero(dist2 <= r_pl ** 2)
            if len(ind) == 0:
                ind = np.array([np.argmin(dist2)])
            I_vec[ind] += I_sing / len(ind)

    # Identificazione nodi impronta di sormonto Barra A e Barra B
    x_min_L = (B - wL) / 2.0
    x_max_L = (B + wL) / 2.0
    mask_A = (X >= x_min_L) & (X <= x_max_L) & (Y >= y_sA - hL / 2.0) & (Y <= y_sA + hL / 2.0)
    mask_B = (X >= x_min_L) & (X <= x_max_L) & (Y >= y_sB - hL / 2.0) & (Y <= y_sB + hL / 2.0)
    ind_A = np.flatnonzero(mask_A)
    ind_B = np.flatnonzero(mask_B)

    # Resistenza fisica di ciascun ramo a L: attraversamento spessore + tratto di barra in uscita L_usc
    R_attraversamento = rho_ohm_mm * (0.5 * (s + sL)) / (wL * hL)
    R_barra_uscita = rho_ohm_mm * L_usc / (wL * sL)
    R_ramo_ohm = R_attraversamento + R_barra_uscita
    G_ramo_tot = 1.0 / R_ramo_ohm

    g_cella_A = G_ramo_tot / len(ind_A)
    g_cella_B = G_ramo_tot / len(ind_B)

    diag_main[ind_A] += g_cella_A
    diag_main[ind_B] += g_cella_B

    # Assemblaggio diretto CSR senza loop
    rows = np.concatenate([np.arange(N), k_left, k_right, k_bot, k_top])
    cols = np.concatenate([np.arange(N), k_right, k_left, k_top, k_bot])
    vals = np.concatenate([
        diag_main,
        np.full(len(k_left), -Gx),
        np.full(len(k_left), -Gx),
        np.full(len(k_bot), -Gy),
        np.full(len(k_bot), -Gy)
    ])
    G_csr = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    # Risoluzione sistema lineare
    V = spla.spsolve(G_csr, I_vec)
    V_2d = V.reshape((ny, nx))

    # Correnti estratte reali (senza normalizzazione artificiale)
    I_A = float(np.sum(V[ind_A] * g_cella_A))
    I_B = float(np.sum(V[ind_B] * g_cella_B))
    errore_conservazione_perc = abs((I_A + I_B) - I_tot) / I_tot * 100.0

    # Campo densità di corrente
    dV_dy, dV_dx = np.gradient(V_2d, dy, dx)
    Jx = -sigma * dV_dx
    Jy = -sigma * dV_dy
    J_mag = np.sqrt(Jx ** 2 + Jy ** 2)

    profilo_Iy = np.sum(Jy, axis=1) * dx * s
    vol_cella = dx * dy * s
    potenza_piastra_watt = float(np.sum((J_mag ** 2) * rho_ohm_mm * vol_cella))
    potenza_barra_A_watt = float((I_A ** 2) * R_ramo_ohm)
    potenza_barra_B_watt = float((I_B ** 2) * R_ramo_ohm)

    return {
        "x": x_c, "y": y_c, "X": X, "Y": Y,
        "V_mV": V_2d * 1e3, "Jx": Jx, "Jy": Jy, "J_mag": J_mag,
        "I_A": I_A, "I_B": I_B, "err_perc": errore_conservazione_perc,
        "profilo_Iy": profilo_Iy, "coords_pl": coords_pl,
        "potenza_piastra_watt": potenza_piastra_watt,
        "potenza_barra_A_watt": potenza_barra_A_watt,
        "potenza_barra_B_watt": potenza_barra_B_watt,
        "R_ramo_uohm": R_ramo_ohm * 1e6
    }


res = risolvi_busbar_2d(
    B_piastra, H_piastra, s_piastra, rho_mat, I_totale,
    X_col, Y_row, passo_x, passo_y, y_offset_pl, diam_pl,
    w_L, h_L, s_L, L_uscita, y_sorm_A, y_sorm_B
)

# --- GRANDEZZE INGEGNERISTICHE DERIVATE ---
p_A = (res["I_A"] / I_totale) * 100.0
p_B = (res["I_B"] / I_totale) * 100.0
sez_L = w_L * s_L
sez_piastra = B_piastra * s_piastra
area_sormonto = w_L * h_L

J_L_A = res["I_A"] / sez_L
J_L_B = res["I_B"] / sez_L
J_sez_piastra_max = float(np.max(np.abs(res["profilo_Iy"]))) / sez_piastra
J_99_hotspot = float(np.percentile(res["J_mag"], 99))
J_contatto_max = max(res["I_A"], res["I_B"]) / area_sormonto

# Calcolo massa in kg e deriva termica con i parametri densità e calore specifico della tabella
vol_piastra_m3 = (B_piastra * H_piastra * s_piastra) * 1e-9
vol_due_L_m3 = 2.0 * (w_L * (h_L + L_uscita) * s_L) * 1e-9
massa_piastra_kg = vol_piastra_m3 * dens_mat
massa_totale_kg = (vol_piastra_m3 + vol_due_L_m3) * dens_mat

potenza_tot_watt = res["potenza_piastra_watt"] + res["potenza_barra_A_watt"] + res["potenza_barra_B_watt"]
gradiente_adiabatico_C_min = (potenza_tot_watt / (massa_totale_kg * csp_mat)) * 60.0

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Corrente Barra A (Bassa)",
    f"{res['I_A']:.0f} A ({p_A:.1f}%)",
    f"J barra A: {J_L_A:.2f} A/mm² (K1={K1_mat:.2f})",
    delta_color="inverse" if J_L_A > K1_mat else "normal"
)
c2.metric(
    "Corrente Barra B (Alta)",
    f"{res['I_B']:.0f} A ({p_B:.1f}%)",
    f"J barra B: {J_L_B:.2f} A/mm² (K1={K1_mat:.2f})",
    delta_color="inverse" if J_L_B > K1_mat else "normal"
)
c3.metric(
    "Densità Sezione Piastra",
    f"{J_sez_piastra_max:.2f} A/mm²",
    f"Hotspot 99° perc: {J_99_hotspot:.2f} A/mm²",
    delta_color="inverse" if J_sez_piastra_max > K1_mat else "normal"
)
c4.metric(
    "Potenza Dissipata Totale",
    f"{potenza_tot_watt:.1f} watt",
    f"Solo piastra: {res['potenza_piastra_watt']:.1f} watt",
    delta_color="off"
)

st.markdown("---")

# --- GRAFICA IN SCALA REALE 1:1 ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10), gridspec_kw={'width_ratios': [1.1, 1.1, 1.0, 1.0]})

# 1. MAPPA DENSITÀ |J|
im1 = ax1.pcolormesh(res["X"], res["Y"], res["J_mag"], shading='auto', cmap='inferno', vmin=0, vmax=max(K1_mat * 1.4, J_99_hotspot))
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Densità |J| [A/mm²]")
ax1.streamplot(res["x"], res["y"], res["Jx"], res["Jy"], color='cyan', linewidth=0.7, density=0.8, arrowsize=0.7)

for (xp, yp) in res["coords_pl"]:
    ax1.add_patch(patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.2))

x_min_L = (B_piastra - w_L) / 2.0
ax1.add_patch(patches.Rectangle((x_min_L, y_sorm_A - h_L / 2.0), w_L, h_L, edgecolor='cyan', facecolor='none', linewidth=2.0))
ax1.add_patch(patches.Rectangle((x_min_L, y_sorm_B - h_L / 2.0), w_L, h_L, edgecolor='cyan', facecolor='none', linewidth=2.0))
ax1.text(B_piastra / 2.0, y_sorm_A, f"A: {p_A:.1f}%", color='white', weight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.75))
ax1.text(B_piastra / 2.0, y_sorm_B, f"B: {p_B:.1f}%", color='white', weight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.75))

ax1.set_title("1. Densità |J| (Scala 1:1)", fontsize=10, weight='bold')
ax1.set_xlabel("Larghezza X [mm]")
ax1.set_ylabel("Quota Y [mm]")
ax1.set_xlim(0, B_piastra)
ax1.set_ylim(0, H_piastra)
ax1.set_aspect('equal', adjustable='box')

# 2. POTENZIALE V
im2 = ax2.pcolormesh(res["X"], res["Y"], res["V_mV"], shading='auto', cmap='viridis')
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Potenziale [mV]")

for (xp, yp) in res["coords_pl"]:
    ax2.add_patch(patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.0, alpha=0.7))

ax2.add_patch(patches.Rectangle((x_min_L, y_sorm_A - h_L / 2.0), w_L, h_L, edgecolor='red', facecolor='none', linewidth=2.0))
ax2.add_patch(patches.Rectangle((x_min_L, y_sorm_B - h_L / 2.0), w_L, h_L, edgecolor='red', facecolor='none', linewidth=2.0))

ax2.set_title("2. Potenziale [mV] (Scala 1:1)", fontsize=10, weight='bold')
ax2.set_xlabel("Larghezza X [mm]")
ax2.set_xlim(0, B_piastra)
ax2.set_ylim(0, H_piastra)
ax2.set_aspect('equal', adjustable='box')

# 3. CARICO ASSIALE |Iy|
portata_limite_piastra = sez_piastra * K1_mat
ax3.plot(np.abs(res["profilo_Iy"]), res["y"], color='#d95f02', linewidth=2.2, label="|Iy(y)| in piastra")
ax3.axhline(y_ala_A, color='blue', linestyle='--', linewidth=1.3, label=f"Uscita A (Y={y_ala_A:.0f} mm)")
ax3.axhline(y_ala_B, color='green', linestyle='--', linewidth=1.3, label=f"Uscita B (Y={y_ala_B:.0f} mm)")
ax3.axvline(portata_limite_piastra, color='red', linestyle=':', linewidth=1.5, label=f"Portata K1 ({portata_limite_piastra:.0f} A)")
ax3.fill_betweenx(res["y"], 0, np.abs(res["profilo_Iy"]), color='#d95f02', alpha=0.2)
ax3.set_title("3. Flusso Assiale |Iy| [A]", fontsize=10, weight='bold')
ax3.set_xlabel("Corrente Assiale [A]")
ax3.set_ylim(0, H_piastra)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper right', fontsize=8)

# 4. VISTA LATERALE SEZIONE Y-Z (VERA FORMA A L)
prof_pl_disegno = diam_pl * 1.4

# Piastra Collettrice Verticale
ax4.add_patch(patches.Rectangle((0, 0), s_piastra, H_piastra, facecolor='#b87333', edgecolor='black', linewidth=1.2))

# Barra A: Sormonto verticale + Ala perpendicolare posta all'estremità della L
ax4.add_patch(patches.Rectangle((s_piastra, y_sorm_A - h_L / 2.0), s_L, h_L, facecolor='#e68a00', edgecolor='black', linewidth=1.1))
ax4.add_patch(patches.Rectangle((s_piastra + s_L, y_ala_A - s_L / 2.0), L_uscita, s_L, facecolor='#e68a00', edgecolor='black', linewidth=1.1))

# Barra B: Sormonto verticale + Ala perpendicolare posta all'estremità della L
ax4.add_patch(patches.Rectangle((s_piastra, y_sorm_B - h_L / 2.0), s_L, h_L, facecolor='#e68a00', edgecolor='black', linewidth=1.1))
ax4.add_patch(patches.Rectangle((s_piastra + s_L, y_ala_B - s_L / 2.0), L_uscita, s_L, facecolor='#e68a00', edgecolor='black', linewidth=1.1))

ax4.text(s_piastra + s_L + L_uscita + 8, y_ala_A, f"A: {p_A:.1f}%\n({res['I_A']:.0f} A)", color='blue', fontsize=8, weight='bold', va='center')
ax4.text(s_piastra + s_L + L_uscita + 8, y_ala_B, f"B: {p_B:.1f}%\n({res['I_B']:.0f} A)", color='green', fontsize=8, weight='bold', va='center')

# Powerlock sul lato posteriore (-Z)
y_pl_unici = sorted(list(set([yp for (xp, yp) in res["coords_pl"]])))
for yp in y_pl_unici:
    ax4.add_patch(patches.Rectangle((-prof_pl_disegno, yp - diam_pl / 2.0), prof_pl_disegno, diam_pl, facecolor='#333333', edgecolor='black', linewidth=0.9))

ax4.set_title("4. Sezione Laterale Y-Z (Scala 1:1)", fontsize=10, weight='bold')
ax4.set_xlabel("Profondità Z [mm]")
ax4.set_xlim(-prof_pl_disegno - 30, s_piastra + s_L + L_uscita + 110)
ax4.set_ylim(0, H_piastra)
ax4.set_aspect('equal', adjustable='box')
ax4.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
st.pyplot(fig)

# --- TABELLA TECNICA COMPLETA ---
st.subheader("Report di Verifica Elettromeccanica e Termica")
st.table(pd.DataFrame({
    "Parametro Tecnico": [
        "Materiale e Coefficiente K1 (da Tabella)",
        "Massa Calcolata (Piastra / Sistema Completo con L)",
        "Corrente Totale Iniettata / Singolo Powerlock",
        "Corrente Ramo Barra A (Inferiore)",
        "Corrente Ramo Barra B (Superiore)",
        "Densità di Corrente Sezione Barre a L (w_L x s_L)",
        "Densità di Corrente Max Sezione Piastra / Hotspot 99° perc.",
        "Densità di Corrente Superficiale sul Sormonto Bullonato",
        "Resistenza Ohmica Calcolata per Singola Barra a L",
        "Potenza Termica Dissipata (Piastra / Barre a L / Totale)",
        "Errore Numerico Conservazione di Kirchhoff"
    ],
    "Valore Calcolato": [
        f"{mat_scelto}  |  K1 = {K1_mat:.2f} A/mm²",
        f"Piastra: {massa_piastra_kg:.2f} kg  |  Totale con 2 barre a L: {massa_totale_kg:.2f} kg",
        f"{I_totale:.0f} A totali  |  {I_totale / (X_col * Y_row):.1f} A per polo ({X_col * Y_row} Powerlock)",
        f"{res['I_A']:.1f} A ({p_A:.2f} %)  —  Uscita a Y = {y_ala_A:.0f} mm (Centro sormonto Y = {y_sorm_A:.0f} mm)",
        f"{res['I_B']:.1f} A ({p_B:.2f} %)  —  Uscita a Y = {y_ala_B:.0f} mm (Centro sormonto Y = {y_sorm_B:.0f} mm)",
        f"Barra A: {J_L_A:.2f} A/mm²  |  Barra B: {J_L_B:.2f} A/mm² (Sezione {sez_L:.0f} mm²)",
        f"Sezione Piastra: {J_sez_piastra_max:.2f} A/mm²  |  Hotspot 99°: {J_99_hotspot:.2f} A/mm²",
        f"{J_contatto_max:.3f} A/mm² (su area di contatto {area_sormonto:.0f} mm²)",
        f"{res['R_ramo_uohm']:.2f} micro-ohm (per L_uscita = {L_uscita:.0f} mm)",
        f"Piastra: {res['potenza_piastra_watt']:.1f} watt | Barre L: {(res['potenza_barra_A_watt'] + res['potenza_barra_B_watt']):.1f} watt | Totale: {potenza_tot_watt:.1f} watt ({potenza_tot_watt / 1000.0:.2f} chilowatt)",
        f"{res['err_perc']:.6f} %"
    ],
    "Esito e Conformità DFM": [
        f"Densità: {dens_mat:.0f} kg/m³ | Calore specifico: {csp_mat:.0f} J/(kg·K)",
        f"Deriva termica adiabatica iniziale: {gradiente_adiabatico_C_min:.2f} °C/min",
        "Ripartizione uniforme lato connettori",
        "OK (Bilanciato)" if 45.0 <= p_A <= 55.0 else ("Ramo più carico" if p_A > 55.0 else "Ramo più scarico"),
        "OK (Bilanciato)" if 45.0 <= p_B <= 55.0 else ("Ramo più carico" if p_B > 55.0 else "Ramo più scarico"),
        "OK (<= K1)" if max(J_L_A, J_L_B) <= K1_mat else f"SUPERAMENTO K1 ({K1_mat:.2f} A/mm²): Aumentare sezione L",
        "OK (<= K1)" if J_sez_piastra_max <= K1_mat else f"SUPERAMENTO K1 ({K1_mat:.2f} A/mm²): Aumentare spessore/larghezza piastra",
        "OK (<= 0.40 A/mm²)" if J_contatto_max <= 0.40 else "ATTENZIONE: Contatto bullonato sollecitato (> 0.40 A/mm²), aumentare h_L",
        "Calcolato da geometria L_uscita e resistività materiale",
        "Potenza da smaltire per convezione/ventilazione nel vano sbarre",
        "Solutore convergente" if res["err_perc"] < 0.01 else "Verificare mesh"
    ]
}))
