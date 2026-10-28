import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="LOADBANK-MASTER | Posizionamento barre in uscita",
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

# Parametri viteria per modalità opzionale (quando si disattiva la continuità del materiale)
VITI_DB = {
    "M8":  {"d_foro": 9.0,  "D_rond": 16.0, "coppia": 25.0, "d_nom": 8.0},
    "M10": {"d_foro": 11.0, "D_rond": 20.0, "coppia": 44.0, "d_nom": 10.0},
    "M12": {"d_foro": 13.5, "D_rond": 24.0, "coppia": 75.0, "d_nom": 12.0},
}

# Resistività superficiale specifica di pellicola [micro-ohm * mm^2] al variare della finitura
TRATTAMENTI_DB = {
    "Ideale (Solo costrizione geometrica cono, R_film = 0)": 0.0,
    "Argentato": 150.0,
    "Stagnato": 300.0,
    "Nichelato": 450.0,
    "Nudo (Spazzolato e ingrassato)": 600.0
}

K_ATTRITO = 0.20
ANGOLO_CONO_DEG = 30.0

ORIENT_ALTO = "Sormonto piegato verso l'ALTO (L dritta)"
ORIENT_BASSO = "Sormonto piegato verso il BASSO (L rovescia)"

st.title("LOADBANK-MASTER | Posizionamento e Ripartizione Barre in Uscita")
st.markdown(
    "Calcolo agli elementi finiti (2D FDM) della ripartizione di corrente tra due barre a L collegate "
    "a una piastra collettrice con **Powerlock**, con verifica delle sezioni secondo il fattore materiale **$K_1$**."
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

# --- SIDEBAR 3: BARRE A L ---
st.sidebar.header("3. Barre a L in Uscita")
w_L = st.sidebar.number_input("Larghezza Barre a L [mm]", min_value=20.0, max_value=600.0, value=120.0, step=10.0)
h_L = st.sidebar.number_input("Altezza Sormonto h_L [mm]", min_value=15.0, max_value=300.0, value=60.0, step=5.0)
s_L = st.sidebar.number_input("Spessore Barre a L s_L [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0)

L_A = st.sidebar.number_input(
    "Lunghezza Ramo A fino a barra equipotenziale [mm]",
    min_value=20.0, max_value=1500.0, value=120.0, step=10.0,
    help="Percorso della barra A dalla piega fino al nodo comune equipotenziale."
)
L_B = st.sidebar.number_input(
    "Lunghezza Ramo B fino a barra equipotenziale [mm]",
    min_value=20.0, max_value=1500.0, value=120.0, step=10.0,
    help="Percorso della barra B dalla piega fino al nodo comune equipotenziale."
)

verso_piega_A = st.sidebar.selectbox("Orientamento Sormonto Barra A (Inferiore)", [ORIENT_ALTO, ORIENT_BASSO], index=0)
verso_piega_B = st.sidebar.selectbox("Orientamento Sormonto Barra B (Superiore)", [ORIENT_ALTO, ORIENT_BASSO], index=1)


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
    st.error("Errore geometrico: l'altezza del sormonto h_L supera l'altezza della piastra H.")
    st.stop()

y_ala_A = st.sidebar.slider(
    "Quota Y Mezzeria Uscita Barra A (Bassa) [mm]",
    min_value=float(min_y_A), max_value=float(max_y_A),
    value=float(min(max(30.0, min_y_A), max_y_A)), step=5.0
)
y_ala_B = st.sidebar.slider(
    "Quota Y Mezzeria Uscita Barra B (Alta) [mm]",
    min_value=float(min_y_B), max_value=float(max_y_B),
    value=float(min(max(500.0, min_y_B), max_y_B)), step=5.0
)

# --- SIDEBAR 4: INTERFACCIA DI CONTATTO (CONTINUITÀ PREIMPOSTATA) ---
st.sidebar.header("4. Interfaccia Sormonto")
continuita_materiale = st.sidebar.checkbox(
    "Continuità del materiale (Preimpostato)",
    value=True,
    help="Se attivo, assume continuità metallica ideale sull'intera superficie del sormonto (w_L x h_L). "
         "Disattiva per modellare il cono di pressione delle viti e lo stato superficiale."
)

area_sormonto = w_L * h_L

if continuita_materiale:
    A_eff = area_sormonto
    R_film_ohm = 0.0
    descr_giunto = f"Continuità materiale sull'intero sormonto ({area_sormonto:.0f} mm²)"
    errore_viti = None
else:
    vite_scelta = st.sidebar.selectbox("Calibro Vite", list(VITI_DB.keys()), index=1)
    n_viti = st.sidebar.number_input("Numero di viti per giunto", min_value=1, max_value=24, value=4, step=1)
    trattamento = st.sidebar.selectbox("Trattamento superficiale", list(TRATTAMENTI_DB.keys()), index=0)

    dati_vite = VITI_DB[vite_scelta]
    coppia_Nm = dati_vite["coppia"]
    precarico_N = coppia_Nm / (K_ATTRITO * dati_vite["d_nom"] * 1e-3)

    t_cono = min(s_piastra, s_L)
    D_cono = dati_vite["D_rond"] + 2.0 * t_cono * np.tan(np.radians(ANGOLO_CONO_DEG))
    area_cono_vite = np.pi / 4.0 * (D_cono ** 2 - dati_vite["d_foro"] ** 2)
    A_eff = float(min(n_viti * area_cono_vite, area_sormonto))
    pressione_cono_MPa = precarico_N / area_cono_vite

    # Resistenza addizionale di pellicola/contatto in funzione della finitura e dell'area efficace
    rho_film_uohm_mm2 = TRATTAMENTI_DB[trattamento]
    R_film_ohm = (rho_film_uohm_mm2 / A_eff) * 1e-6

    st.sidebar.caption(
        f"Precarico: ~{precarico_N / 1000.0:.0f} kN/vite | Pressione cono: ~{pressione_cono_MPa:.0f} MPa | "
        f"Area efficace coni: {A_eff:.0f} mm²"
    )
    descr_giunto = (
        f"{n_viti} x {vite_scelta} ({trattamento}), area coni {A_eff:.0f} mm² su {area_sormonto:.0f} mm² "
        f"(Pressione ~{pressione_cono_MPa:.0f} MPa)"
    )
    if n_viti * (dati_vite["D_rond"] ** 2) > area_sormonto:
        errore_viti = f"{n_viti} viti {vite_scelta} con rondella da {dati_vite['D_rond']:.0f} mm non entrano nel sormonto {w_L:.0f} x {h_L:.0f} mm."
    else:
        errore_viti = None

# Posizione dei sormonti sulla piastra
y_sorm_A = y_sormonto(y_ala_A, verso_piega_A)
y_sorm_B = y_sormonto(y_ala_B, verso_piega_B)

# --- CONTROLLI GEOMETRICI CON BLOCCO ---
ingombro_x_pl = (X_col - 1) * passo_x + diam_pl
y_min_pl = y_offset_pl - diam_pl / 2.0
y_max_pl = y_offset_pl + (Y_row - 1) * passo_y + diam_pl / 2.0

errori_geom = []
if ingombro_x_pl > B_piastra:
    errori_geom.append(f"La griglia Powerlock è larga {ingombro_x_pl:.1f} mm e deborda dalla larghezza piastra B ({B_piastra:.1f} mm).")
if y_min_pl < 0 or y_max_pl > H_piastra:
    errori_geom.append(f"Le righe Powerlock vanno da Y={y_min_pl:.1f} mm a Y={y_max_pl:.1f} mm, fuori dall'altezza piastra H (0 - {H_piastra:.1f} mm).")
if w_L > B_piastra:
    errori_geom.append(f"La larghezza delle barre a L ({w_L:.1f} mm) supera la larghezza della piastra ({B_piastra:.1f} mm).")
if abs(y_sorm_A - y_sorm_B) < h_L:
    errori_geom.append(
        f"I sormonti delle barre A e B si sovrappongono (distanza centri = {abs(y_sorm_A - y_sorm_B):.1f} mm < h_L = {h_L:.1f} mm)."
    )
if errore_viti is not None:
    errori_geom.append(errore_viti)

if errori_geom:
    for err in errori_geom:
        st.error(err)
    st.stop()


# --- SOLUTORE 2D ALLE DIFFERENZE FINITE ---
@st.cache_data(show_spinner=False)
def risolvi_busbar_2d(B, H, s, rho_uohm_m, I_tot, n_col, n_row, px, py, y0_pl, d_pl,
                      wL, hL, sL, lenA, lenB, A_contatto, R_film_extra_ohm, y_sA, y_sB, mesh_sz, completo=True):
    rho = rho_uohm_m * 1e-3          # ohm * mm
    sigma = 1.0 / rho                # S / mm

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

    # Correnti iniettate dai Powerlock (distribuite sull'impronta circolare)
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

    # Nodi del sormonto delle barre A e B
    x_min_L = (B - wL) / 2.0
    x_max_L = (B + wL) / 2.0
    mask_A = (X >= x_min_L) & (X <= x_max_L) & (Y >= y_sA - hL / 2.0) & (Y <= y_sA + hL / 2.0)
    mask_B = (X >= x_min_L) & (X <= x_max_L) & (Y >= y_sB - hL / 2.0) & (Y <= y_sB + hL / 2.0)
    ind_A = np.flatnonzero(mask_A)
    ind_B = np.flatnonzero(mask_B)

    # Resistenza di ciascun ramo fino alla barra equipotenziale:
    # - attraversamento dello spessore nell'area di contatto (intera in continuità materiale, o coni vite)
    # - eventuale resistenza di film superficiale (0 in continuità materiale)
    # - raccolta corrente lungo il sormonto fino alla piega (1/3 h_L)
    # - sviluppo lineare del ramo L_A e L_B
    R_att = rho * (0.5 * (s + sL)) / A_contatto + R_film_extra_ohm
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
    if not completo:
        return {"I_A": I_A, "I_B": I_B}

    V_2d = V.reshape((ny, nx))
    err_perc = abs((I_A + I_B) - I_tot) / I_tot * 100.0

    dV_dy, dV_dx = np.gradient(V_2d, dy, dx)
    Jx = -sigma * dV_dx
    Jy = -sigma * dV_dy
    J_mag = np.sqrt(Jx ** 2 + Jy ** 2)
    profilo_Iy = np.sum(Jy, axis=1) * dx * s

    # Bilancio di potenza esatto in watt
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


MESH_SZ = 4.0 if (B_piastra * H_piastra) <= 600000 else 6.0

res = risolvi_busbar_2d(
    B_piastra, H_piastra, s_piastra, rho_mat, I_totale,
    X_col, Y_row, passo_x, passo_y, y_offset_pl, diam_pl,
    w_L, h_L, s_L, L_A, L_B, A_eff, R_film_ohm, y_sorm_A, y_sorm_B, MESH_SZ
)

# --- GRANDEZZE DERIVATE ---
p_A = res["I_A"] / I_totale * 100.0
p_B = res["I_B"] / I_totale * 100.0
sez_L = w_L * s_L
sez_piastra = B_piastra * s_piastra
J_L_A = res["I_A"] / sez_L
J_L_B = res["I_B"] / sez_L
J_sez_piastra_max = float(np.max(np.abs(res["profilo_Iy"]))) / sez_piastra
J_giunto_A = res["I_A"] / A_eff
J_giunto_B = res["I_B"] / A_eff

vol_piastra_m3 = B_piastra * H_piastra * s_piastra * 1e-9
vol_barre_m3 = w_L * ((h_L + L_A) + (h_L + L_B)) * s_L * 1e-9
massa_piastra_kg = vol_piastra_m3 * dens_mat
massa_totale_kg = (vol_piastra_m3 + vol_barre_m3) * dens_mat
P_tot = res["P_piastra"] + res["P_A"] + res["P_B"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Corrente Barra A (Bassa)", f"{res['I_A']:.0f} A ({p_A:.1f}%)",
          f"J barra A: {J_L_A:.2f} A/mm² (K1={K1_mat:.2f})",
          delta_color="inverse" if J_L_A > K1_mat else "normal")
c2.metric("Corrente Barra B (Alta)", f"{res['I_B']:.0f} A ({p_B:.1f}%)",
          f"J barra B: {J_L_B:.2f} A/mm² (K1={K1_mat:.2f})",
          delta_color="inverse" if J_L_B > K1_mat else "normal")
c3.metric("Densità Max Sezione Piastra", f"{J_sez_piastra_max:.2f} A/mm²",
          f"K1 = {K1_mat:.2f} A/mm²",
          delta_color="inverse" if J_sez_piastra_max > K1_mat else "normal")
c4.metric("Potenza Dissipata (20 °C)", f"{P_tot:.1f} watt",
          f"Solo piastra: {res['P_piastra']:.1f} watt", delta_color="off")

st.caption(
    f"**Modalità attiva:** {'Continuità del materiale sull’intero sormonto' if continuita_materiale else 'Giunto bullonato con coni di pressione'}. "
    "I due rami terminano su una barra equipotenziale comune."
)

st.markdown("---")

# --- GRAFICA (4 VISTE IN SCALA 1:1) ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10), gridspec_kw={'width_ratios': [1.1, 1.1, 1.0, 1.0]})

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

# --- SCANSIONE DELLA POSIZIONE ---
st.subheader("Scansione parametrica della posizione di una barra")
st.markdown(
    "Sposta una barra lungo la piastra tenendo ferma l'altra, individuando la quota $Y$ di pareggio ($50\\% / 50\\%$) "
    "e la densità di corrente risultante."
)
sc1, sc2 = st.columns(2)
barra_scan = sc1.radio("Barra da spostare", ["A", "B"], horizontal=True)
n_pos = sc2.slider("Numero di posizioni di calcolo", min_value=8, max_value=40, value=20, step=1)

chiave_scan = (barra_scan, n_pos, B_piastra, H_piastra, s_piastra, mat_scelto, I_totale, X_col, Y_row, passo_x,
               passo_y, y_offset_pl, diam_pl, w_L, h_L, s_L, L_A, L_B, A_eff, R_film_ohm, verso_piega_A, verso_piega_B,
               y_ala_A, y_ala_B)

if st.button("Calcola scansione"):
    if barra_scan == "A":
        lo, hi, verso_s = min_y_A, max_y_A, verso_piega_A
    else:
        lo, hi, verso_s = min_y_B, max_y_B, verso_piega_B
    posizioni = np.linspace(lo, hi, n_pos)
    pA, pB, jmax = [], [], []
    barra_prog = st.progress(0.0)
    for i, yp in enumerate(posizioni):
        ys_mobile = y_sormonto(yp, verso_s)
        ysA = ys_mobile if barra_scan == "A" else y_sorm_A
        ysB = ys_mobile if barra_scan == "B" else y_sorm_B
        if abs(ysA - ysB) < h_L:
            pA.append(np.nan)
            pB.append(np.nan)
            jmax.append(np.nan)
        else:
            r = risolvi_busbar_2d(B_piastra, H_piastra, s_piastra, rho_mat, I_totale, X_col, Y_row, passo_x,
                                  passo_y, y_offset_pl, diam_pl, w_L, h_L, s_L, L_A, L_B, A_eff, R_film_ohm, ysA, ysB,
                                  6.0, completo=False)
            pA.append(r["I_A"] / I_totale * 100.0)
            pB.append(r["I_B"] / I_totale * 100.0)
            jmax.append(max(r["I_A"], r["I_B"]) / sez_L)
        barra_prog.progress((i + 1) / len(posizioni))
    barra_prog.empty()
    st.session_state["scansione"] = {
        "chiave": chiave_scan, "pos": posizioni, "pA": np.array(pA), "pB": np.array(pB), "jmax": np.array(jmax),
        "barra": barra_scan
    }

scan = st.session_state.get("scansione")
if scan is not None and scan["chiave"] == chiave_scan:
    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(14, 4.5))
    y_corrente = y_ala_A if scan["barra"] == "A" else y_ala_B
    bx1.plot(scan["pos"], scan["pA"], color='blue', marker='o', markersize=3, label="Barra A")
    bx1.plot(scan["pos"], scan["pB"], color='green', marker='o', markersize=3, label="Barra B")
    bx1.axhline(50.0, color='gray', linestyle='--', linewidth=1.0, label="Pareggio 50%")
    bx1.axvline(y_corrente, color='red', linestyle=':', linewidth=1.2, label="Posizione attuale")
    bx1.set_xlabel(f"Quota Y mezzeria uscita barra {scan['barra']} [mm]")
    bx1.set_ylabel("Quota di corrente [%]")
    bx1.set_title("Ripartizione A / B [%]", fontsize=10, weight='bold')
    bx1.grid(True, linestyle='--', alpha=0.5)
    bx1.legend(fontsize=8)

    bx2.plot(scan["pos"], scan["jmax"], color='#d95f02', marker='o', markersize=3, label="Barra più carica")
    bx2.axhline(K1_mat, color='red', linestyle=':', linewidth=1.5, label=f"K1 = {K1_mat:.2f} A/mm²")
    bx2.axvline(y_corrente, color='red', linestyle=':', linewidth=1.2)
    bx2.set_xlabel(f"Quota Y mezzeria uscita barra {scan['barra']} [mm]")
    bx2.set_ylabel("Densità nella barra [A/mm²]")
    bx2.set_title("Densità nella barra più carica", fontsize=10, weight='bold')
    bx2.grid(True, linestyle='--', alpha=0.5)
    bx2.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
elif scan is not None:
    st.info("I parametri geometrici o elettrici sono cambiati: premi di nuovo «Calcola scansione».")

# --- REPORT ---
st.subheader("Report di Verifica Elettromeccanica")
st.table(pd.DataFrame({
    "Parametro": [
        "Materiale e coefficiente K1",
        "Massa (piastra / piastra + 2 barre)",
        "Corrente totale / singolo Powerlock",
        "Ramo A (inferiore)",
        "Ramo B (superiore)",
        "Densità nella sezione delle barre a L",
        "Densità massima nella sezione della piastra",
        "Modello interfaccia sormonto",
        "Densità media sull'area efficace di sormonto",
        "Resistenza di ciascun ramo (A / B)",
        "Potenza dissipata a 20 °C (piastra / rami / totale)",
        "Errore di conservazione della corrente"
    ],
    "Valore": [
        f"{mat_scelto}  |  K1 = {K1_mat:.2f} A/mm²",
        f"{massa_piastra_kg:.1f} kg  /  {massa_totale_kg:.1f} kg",
        f"{I_totale:.0f} A  |  {I_totale / (X_col * Y_row):.1f} A per Powerlock ({X_col * Y_row})",
        f"{res['I_A']:.0f} A ({p_A:.1f} %)  |  ala a Y = {y_ala_A:.0f} mm, sormonto centrato a Y = {y_sorm_A:.0f} mm  |  L_A = {L_A:.0f} mm",
        f"{res['I_B']:.0f} A ({p_B:.1f} %)  |  ala a Y = {y_ala_B:.0f} mm, sormonto centrato a Y = {y_sorm_B:.0f} mm  |  L_B = {L_B:.0f} mm",
        f"A: {J_L_A:.2f} A/mm²  |  B: {J_L_B:.2f} A/mm²  (sezione {sez_L:.0f} mm²)",
        f"{J_sez_piastra_max:.2f} A/mm²  (sezione {sez_piastra:.0f} mm²)",
        descr_giunto,
        f"A: {J_giunto_A:.2f} A/mm²  |  B: {J_giunto_B:.2f} A/mm²",
        f"A: {res['R_A_uohm']:.2f} µΩ  |  B: {res['R_B_uohm']:.2f} µΩ  (attraversamento interfaccia {res['R_att_uohm']:.3f} µΩ)",
        f"{res['P_piastra']:.1f} watt  /  {res['P_A'] + res['P_B']:.1f} watt  /  {P_tot:.1f} watt ({P_tot / 1000.0:.2f} chilowatt)",
        f"{res['err_perc']:.6f} %"
    ],
    "Indicazione": [
        f"Valori da tabella (Calore spec. {csp_mat:.0f} J/(kg·K))",
        "Piastra e barre; esclusi Powerlock e viteria",
        "Iniezione uniforme sui connettori",
        ("Ramo più carico" if p_A > 55.0 else ("Ramo più scarico" if p_A < 45.0 else "Bilanciato (45-55 %)")),
        ("Ramo più carico" if p_B > 55.0 else ("Ramo più scarico" if p_B < 45.0 else "Bilanciato (45-55 %)")),
        "Entro K1" if max(J_L_A, J_L_B) <= K1_mat else f"Supera K1 ({K1_mat:.2f} A/mm²): aumentare la sezione delle barre",
        "Entro K1" if J_sez_piastra_max <= K1_mat else f"Supera K1 ({K1_mat:.2f} A/mm²): aumentare spessore o larghezza",
        "Preimpostato in continuità materiale; deselezionare in sidebar per analisi viteria",
        "Riferimento sul sormonto attivo",
        "Include attraversamento, raccolta sormonto (1/3 h_L) e lunghezza ramo L",
        "Da smaltire per convezione/ventilazione quadro",
        "Controllo numerico" + (" - OK" if res["err_perc"] < 0.01 else " - verificare la mesh")
    ]
}))
