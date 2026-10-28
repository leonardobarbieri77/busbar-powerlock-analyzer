import streamlit as st
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

st.set_page_config(
    page_title="Calcolatore Busbar & Powerlock - Analisi FDM 2D + Vista Laterale",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Analisi Elettrica 2D e Layout Meccanico Piastra Collettrice Powerlock")
st.markdown(
    "**LOADBANK-MASTER Specialist Tool** — Verifica termoelettrica agli elementi finiti (2D FDM), "
    "ripartizione di corrente sui rami a L e schema costruttivo in sezione laterale $Y\text{-}Z$ per la carpenteria."
)

# --- SIDEBAR: PARAMETRI DI INPUT ---
st.sidebar.header("1. Geometria e Materiale Piastra")
B_piastra = st.sidebar.number_input("Larghezza Piastra B [mm]", min_value=50.0, max_value=1000.0, value=200.0, step=10.0)
H_piastra = st.sidebar.number_input("Altezza Piastra H [mm]", min_value=200.0, max_value=3000.0, value=1000.0, step=50.0)
s_piastra = st.sidebar.number_input("Spessore Piastra s [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0)
T_operativa = st.sidebar.slider("Temperatura Operativa Rame [°C]", min_value=20, max_value=120, value=70, step=5)
J_limite = st.sidebar.number_input("Densità Ammissibile Progetto [A/mm²]", min_value=0.5, max_value=5.0, value=1.5, step=0.1)

st.sidebar.header("2. Griglia Connettori Powerlock")
I_totale = st.sidebar.number_input("Corrente Totale Iniettata [A]", min_value=100.0, max_value=30000.0, value=7200.0, step=100.0)
X_col = st.sidebar.number_input("Numero Colonne (X)", min_value=1, max_value=10, value=2, step=1)
Y_row = st.sidebar.number_input("Numero Righe (Y)", min_value=1, max_value=25, value=9, step=1)
passo_x = st.sidebar.number_input("Passo Orizzontale Colonne [mm]", min_value=20.0, max_value=300.0, value=80.0, step=5.0)
passo_y = st.sidebar.number_input("Passo Verticale Righe [mm]", min_value=20.0, max_value=300.0, value=85.0, step=5.0)
y_offset_pl = st.sidebar.number_input("Quota Y Prima Riga dal Basso [mm]", min_value=20.0, max_value=1000.0, value=160.0, step=10.0)
diam_pl = st.sidebar.number_input("Diametro Impronta Contatto Powerlock [mm]", min_value=10.0, max_value=80.0, value=32.0, step=2.0)
profondita_pl = st.sidebar.number_input("Ingombro Posteriore Connettore Powerlock [mm]", min_value=20.0, max_value=200.0, value=75.0, step=5.0)

st.sidebar.header("3. Barre di Derivazione a L (Uscite)")
w_L = st.sidebar.number_input("Larghezza Barre a L [mm]", min_value=20.0, max_value=500.0, value=120.0, step=10.0)
h_L = st.sidebar.number_input("Altezza Sormonto Sovrapposizione [mm]", min_value=10.0, max_value=200.0, value=60.0, step=5.0)
s_L = st.sidebar.number_input("Spessore Barre a L [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0)
ala_L = st.sidebar.number_input("Lunghezza Ala Orizzontale a Sbalzo [mm]", min_value=20.0, max_value=300.0, value=100.0, step=10.0)

y_barra_A = st.sidebar.slider(
    "Quota Y Centro Barra A (Inferiore) [mm]",
    min_value=float(h_L / 2),
    max_value=float(H_piastra - h_L / 2),
    value=50.0,
    step=5.0
)
y_barra_B = st.sidebar.slider(
    "Quota Y Centro Barra B (Superiore) [mm]",
    min_value=float(h_L / 2),
    max_value=float(H_piastra - h_L / 2),
    value=500.0,
    step=5.0
)

R_cont_uohm = st.sidebar.number_input(
    "Resistenza Contatto + Ramo per Barra [micro-ohm]",
    min_value=0.01,
    max_value=50.0,
    value=1.00,
    step=0.10,
    help="Resistenza di contatto della giunzione bullonata (tipica 0.5 - 1.5 micro-ohm con rondelle Belleville) più impedenza del ramo di derivazione a valle."
)

st.sidebar.header("4. Risoluzione Griglia Calcolo")
passo_mesh = st.sidebar.select_slider("Dimensione Cella Mesh [mm]", options=[2.5, 4.0, 5.0, 8.0, 10.0], value=5.0)


# --- FUNZIONE SOLUTORE 2D FDM ---
def risolvi_piastra_2d(B, H, s, T, I_tot, n_col, n_row, px, py, y0_pl, d_pl, wL, hL, yA, yB, R_uohm, d_mesh):
    rho_20 = 0.017241  # ohm * mm^2 / m
    alpha_20 = 0.00393  # 1 / K
    rho_T = rho_20 * (1.0 + alpha_20 * (T - 20.0))
    rho_ohm_mm = rho_T * 1e-3
    sigma = 1.0 / rho_ohm_mm

    nx = max(10, int(round(B / d_mesh)))
    ny = max(20, int(round(H / d_mesh)))
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

    mask_A = (X_grid >= x_min_L) & (X_grid <= x_max_L) & (Y_grid >= yA - hL / 2.0) & (Y_grid <= yA + hL / 2.0)
    mask_B = (X_grid >= x_min_L) & (X_grid <= x_max_L) & (Y_grid >= yB - hL / 2.0) & (Y_grid <= yB + hL / 2.0)

    indici_A = np.flatnonzero(mask_A)
    indici_B = np.flatnonzero(mask_B)

    R_ohm = max(1e-9, R_uohm * 1e-6)
    g_tot_barra = 1.0 / R_ohm
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
    potenza_contatto_A_watt = float((I_A ** 2) * R_ohm)
    potenza_contatto_B_watt = float((I_B ** 2) * R_ohm)

    return {
        "x": x_centers,
        "y": y_centers,
        "X": X_grid,
        "Y": Y_grid,
        "V_mV": V_2d * 1e3,
        "Jx": Jx,
        "Jy": Jy,
        "J_mag": J_mag,
        "I_A": I_A,
        "I_B": I_B,
        "I_y_profile": I_y_profile,
        "coords_pl": coords_pl,
        "potenza_piastra_watt": potenza_piastra_watt,
        "potenza_contatto_A_watt": potenza_contatto_A_watt,
        "potenza_contatto_B_watt": potenza_contatto_B_watt,
        "rho_T": rho_T
    }


# Validazione geometrica
max_x_pl = (X_col - 1) * passo_x + diam_pl
max_y_pl = y_offset_pl + (Y_row - 1) * passo_y + diam_pl / 2.0

if max_x_pl > B_piastra:
    st.error(f"Errore DFM: L'ingombro orizzontale dei Powerlock ({max_x_pl:.1f} mm) supera la larghezza piastra ({B_piastra:.1f} mm).")
if max_y_pl > H_piastra:
    st.error(f"Errore DFM: L'ingombro verticale dei Powerlock ({max_y_pl:.1f} mm) supera l'altezza piastra ({H_piastra:.1f} mm).")
if w_L > B_piastra:
    st.error("Errore DFM: La larghezza delle barre a L supera la larghezza della piastra collettrice.")

ris = risolvi_piastra_2d(
    B_piastra, H_piastra, s_piastra, T_operativa,
    I_totale, X_col, Y_row, passo_x, passo_y, y_offset_pl, diam_pl,
    w_L, h_L, y_barra_A, y_barra_B, R_cont_uohm, passo_mesh
)

# Calcolo metriche
perc_A = (ris["I_A"] / I_totale) * 100.0
perc_B = (ris["I_B"] / I_totale) * 100.0
sezione_L = w_L * s_L
J_barra_A = ris["I_A"] / sezione_L
J_barra_B = ris["I_B"] / sezione_L
J_picco_piastra = float(np.max(ris["J_mag"]))
J_media_piastra = float(np.mean(ris["J_mag"]))
potenza_totale_watt = ris["potenza_piastra_watt"] + ris["potenza_contatto_A_watt"] + ris["potenza_contatto_B_watt"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Corrente Barra A (Inferiore)",
        value=f"{ris['I_A']:.1f} A ({perc_A:.1f}%)",
        delta=f"Densità: {J_barra_A:.2f} A/mm²",
        delta_color="inverse" if J_barra_A > J_limite else "normal"
    )
with col2:
    st.metric(
        label="Corrente Barra B (Superiore)",
        value=f"{ris['I_B']:.1f} A ({perc_B:.1f}%)",
        delta=f"Densità: {J_barra_B:.2f} A/mm²",
        delta_color="inverse" if J_barra_B > J_limite else "normal"
    )
with col3:
    st.metric(
        label="Densità di Picco nella Piastra",
        value=f"{J_picco_piastra:.2f} A/mm²",
        delta=f"Media: {J_media_piastra:.2f} A/mm² (Lim: {J_limite:.1f})",
        delta_color="inverse" if J_picco_piastra > J_limite * 1.5 else "normal"
    )
with col4:
    st.metric(
        label="Potenza Dissipata Piastra",
        value=f"{ris['potenza_piastra_watt']:.1f} watt",
        delta=f"Totale con giunti: {potenza_totale_watt:.1f} watt",
        delta_color="off"
    )

st.markdown("---")

# --- VISUALIZZAZIONE GRAFICA COMPLETA (4 SOTTOPLOT INCLUSA VISTA LATERALE) ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(
    1, 4,
    figsize=(22, 9),
    gridspec_kw={'width_ratios': [1.3, 1.3, 1.0, 1.1]}
)

# 1. MAPPA DENSITÀ DI CORRENTE |J|
im1 = ax1.pcolormesh(ris["X"], ris["Y"], ris["J_mag"], shading='auto', cmap='inferno', vmin=0, vmax=max(J_limite * 1.5, J_picco_piastra))
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label("Densità di Corrente |J| [A/mm²]", fontsize=9)
ax1.streamplot(ris["x"], ris["y"], ris["Jx"], ris["Jy"], color='cyan', linewidth=0.7, density=0.9, arrowsize=0.8)

if J_picco_piastra > J_limite:
    ax1.contour(ris["X"], ris["Y"], ris["J_mag"], levels=[J_limite], colors=['lime'], linewidths=1.5, linestyles='--')

for (xp, yp) in ris["coords_pl"]:
    circ = patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.2)
    ax1.add_patch(circ)

x_min_L = (B_piastra - w_L) / 2.0
rect_A = patches.Rectangle((x_min_L, y_barra_A - h_L / 2.0), w_L, h_L, edgecolor='cyan', facecolor='none', linewidth=2.0)
rect_B = patches.Rectangle((x_min_L, y_barra_B - h_L / 2.0), w_L, h_L, edgecolor='cyan', facecolor='none', linewidth=2.0)
ax1.add_patch(rect_A)
ax1.add_patch(rect_B)
ax1.text(B_piastra / 2.0, y_barra_A, f"BARRA A\n{perc_A:.1f}%", color='white', weight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.75))
ax1.text(B_piastra / 2.0, y_barra_B, f"BARRA B\n{perc_B:.1f}%", color='white', weight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.75))

ax1.set_title("1. Mappa Densità |J| e Flusso", fontsize=11, weight='bold')
ax1.set_xlabel("Larghezza X [mm]")
ax1.set_ylabel("Quota Verticale Y [mm]")
ax1.set_xlim(0, B_piastra)
ax1.set_ylim(0, H_piastra)

# 2. MAPPA POTENZIALE ELETTRICO V [mV]
im2 = ax2.pcolormesh(ris["X"], ris["Y"], ris["V_mV"], shading='auto', cmap='viridis')
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label("Potenziale Elettrico [mV]", fontsize=9)

for (xp, yp) in ris["coords_pl"]:
    circ2 = patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.0, alpha=0.7)
    ax2.add_patch(circ2)

rect_A2 = patches.Rectangle((x_min_L, y_barra_A - h_L / 2.0), w_L, h_L, edgecolor='red', facecolor='none', linewidth=2.0)
rect_B2 = patches.Rectangle((x_min_L, y_barra_B - h_L / 2.0), w_L, h_L, edgecolor='red', facecolor='none', linewidth=2.0)
ax2.add_patch(rect_A2)
ax2.add_patch(rect_B2)

ax2.set_title("2. Distribuzione Potenziale [mV]", fontsize=11, weight='bold')
ax2.set_xlabel("Larghezza X [mm]")
ax2.set_xlim(0, B_piastra)
ax2.set_ylim(0, H_piastra)

# 3. PROFILO CARICO ASSIALE |Iy|
ax3.plot(np.abs(ris["I_y_profile"]), ris["y"], color='#d95f02', linewidth=2.2, label="|Iy(y)|")
ax3.axhline(y_barra_A, color='blue', linestyle='--', linewidth=1.4, label=f"Barra A ({y_barra_A:.0f} mm)")
ax3.axhline(y_barra_B, color='green', linestyle='--', linewidth=1.4, label=f"Barra B ({y_barra_B:.0f} mm)")
ax3.axvline(B_piastra * s_piastra * J_limite, color='red', linestyle=':', linewidth=1.4, label=f"Limite ({B_piastra * s_piastra * J_limite:.0f} A)")
ax3.fill_betweenx(ris["y"], 0, np.abs(ris["I_y_profile"]), color='#d95f02', alpha=0.2)
ax3.set_title("3. Carico Assiale |Iy| [A]", fontsize=11, weight='bold')
ax3.set_xlabel("Corrente Sezione [A]")
ax3.set_ylim(0, H_piastra)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper right', fontsize=8)

# 4. VISTA LATERALE SEZIONE MECCANICA (Y-Z)
# La piastra collettrice si trova tra Z = 0 e Z = s_piastra.
# Le L sporgono sul lato destro (+Z), i Powerlock sono inseriti dal retro a sinistra (-Z).

# Disegno Piastra Collettrice principale (rame)
piastra_rett = patches.Rectangle((0, 0), s_piastra, H_piastra, facecolor='#b87333', edgecolor='black', linewidth=1.5, label='Piastra Rame')
ax4.add_patch(piastra_rett)

# Barre a L: Tratto verticale di sovrapposizione + Ala orizzontale a sbalzo
# Barra A (Inferiore)
l_vert_A = patches.Rectangle((s_piastra, y_barra_A - h_L / 2.0), s_L, h_L, facecolor='#d9822b', edgecolor='black', linewidth=1.2)
l_oriz_A = patches.Rectangle((s_piastra, y_barra_A - h_L / 2.0), ala_L, s_L, facecolor='#d9822b', edgecolor='black', linewidth=1.2)
ax4.add_patch(l_vert_A)
ax4.add_patch(l_oriz_A)

# Barra B (Superiore)
l_vert_B = patches.Rectangle((s_piastra, y_barra_B - h_L / 2.0), s_L, h_L, facecolor='#d9822b', edgecolor='black', linewidth=1.2)
l_oriz_B = patches.Rectangle((s_piastra, y_barra_B - h_L / 2.0), ala_L, s_L, facecolor='#d9822b', edgecolor='black', linewidth=1.2)
ax4.add_patch(l_vert_B)
ax4.add_patch(l_oriz_B)

# Tratteggio quote e frecce uscite di potenza
ax4.annotate(
    f"Uscita A\n{perc_A:.1f}%",
    xy=(s_piastra + ala_L, y_barra_A - h_L / 2.0 + s_L / 2.0),
    xytext=(s_piastra + ala_L + 25, y_barra_A),
    arrowprops=dict(arrowstyle="->", color="blue", lw=1.8),
    fontsize=9, weight='bold', color='blue', va='center'
)
ax4.annotate(
    f"Uscita B\n{perc_B:.1f}%",
    xy=(s_piastra + ala_L, y_barra_B - h_L / 2.0 + s_L / 2.0),
    xytext=(s_piastra + ala_L + 25, y_barra_B),
    arrowprops=dict(arrowstyle="->", color="green", lw=1.8),
    fontsize=9, weight='bold', color='green', va='center'
)

# Disegno Powerlock sul retro (-Z)
y_pl_unici = sorted(list(set([yp for (xp, yp) in ris["coords_pl"]])))
for yp in y_pl_unici:
    pl_box = patches.Rectangle((-profondita_pl, yp - diam_pl / 2.0), profondita_pl, diam_pl, facecolor='#333333', edgecolor='white', linewidth=0.8)
    ax4.add_patch(pl_box)
    ax4.plot([-profondita_pl - 15, -profondita_pl], [yp, yp], color='red', linewidth=1.5)

ax4.text(-profondita_pl / 2.0, y_pl_unici[-1] + diam_pl, "POWERLOCK\n(Ingressi)", color='black', fontsize=8, weight='bold', ha='center')

# Quote e linee asse bulloneria
ax4.axvline(0, color='gray', linestyle=':', linewidth=0.8)
ax4.axvline(s_piastra, color='gray', linestyle=':', linewidth=0.8)

ax4.set_title("4. Schema Meccanico (Sezione Y-Z)", fontsize=11, weight='bold')
ax4.set_xlabel("Profondità Z [mm]")
ax4.set_xlim(-profondita_pl - 40, s_piastra + ala_L + 70)
ax4.set_ylim(0, H_piastra)
ax4.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
st.pyplot(fig)

# --- TABELLA DI REPORT TECNICO ---
st.subheader("Riepilogo Grandezze Elettriche, Termiche e Verifica Sezioni")
df_report = pd.DataFrame({
    "Parametro Tecnico": [
        "Corrente Totale Iniettata (Powerlock)",
        "Corrente per Singolo Connettore Powerlock",
        "Corrente Uscita Barra A (Inferiore)",
        "Corrente Uscita Barra B (Superiore)",
        "Densità di Corrente Sezione Barra A",
        "Densità di Corrente Sezione Barra B",
        "Densità di Corrente Picco Locale Piastra (Current Crowding)",
        "Potenza Dissipata nella Piastra Collettrice",
        "Potenza Dissipata Contatto + Ramo Barra A",
        "Potenza Dissipata Contatto + Ramo Barra B",
        "Potenza Termica Totale Sistema"
    ],
    "Valore Calcolato": [
        f"{I_totale:.1f} A",
        f"{I_totale / (X_col * Y_row):.1f} A ({X_col * Y_row} connettori)",
        f"{ris['I_A']:.1f} A ({perc_A:.2f} %)",
        f"{ris['I_B']:.1f} A ({perc_B:.2f} %)",
        f"{J_barra_A:.2f} A/mm² (Sezione {w_L:.0f}x{s_L:.0f} mm)",
        f"{J_barra_B:.2f} A/mm² (Sezione {w_L:.0f}x{s_L:.0f} mm)",
        f"{J_picco_piastra:.2f} A/mm²",
        f"{ris['potenza_piastra_watt']:.2f} watt",
        f"{ris['potenza_contatto_A_watt']:.2f} watt",
        f"{ris['potenza_contatto_B_watt']:.2f} watt",
        f"{potenza_totale_watt:.2f} watt"
    ],
    "Stato Verifica (DFM / CEI EN 61439-1)": [
        "Dato di targa",
        "Carico simmetrico per polo",
        "Ramo scarico" if perc_A < 45 else ("Bilanciato" if perc_A <= 55 else "Ramo sovraccarico"),
        "Ramo scarico" if perc_B < 45 else ("Bilanciato" if perc_B <= 55 else "Ramo sovraccarico"),
        "OK" if J_barra_A <= J_limite else "SUPERAMENTO SOGLIA",
        "OK" if J_barra_B <= J_limite else "SUPERAMENTO SOGLIA",
        "OK" if J_picco_piastra <= J_limite * 1.5 else "HOTSPOT LOCALE SUL BORDO BARRA L",
        f"Resistività a {T_operativa} °C: {ris['rho_T']:.5f} ohm·mm²/m",
        f"R_eq impostata: {R_cont_uohm:.2f} micro-ohm",
        f"R_eq impostata: {R_cont_uohm:.2f} micro-ohm",
        "Da smaltire per convezione/ventilazione quadro"
    ]
})
st.table(df_report)
