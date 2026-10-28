import streamlit as st
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

# Configurazione pagina Streamlit
st.set_page_config(
    page_title="Calcolatore Busbar & Powerlock - Analisi FDM 2D",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Analisi Ripartizione Corrente e Densità 2D su Piastra Collettrice")
st.markdown(
    "**Modello alle Differenze Finite (2D FDM)** per piastre di distribuzione in rame con "
    "matrice di connettori **Powerlock** ($X$ colonne $\\times$ $Y$ righe) e barre di derivazione a L."
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

st.sidebar.header("3. Barre di Derivazione a L (Uscite)")
w_L = st.sidebar.number_input("Larghezza Barre a L [mm]", min_value=20.0, max_value=500.0, value=120.0, step=10.0)
h_L = st.sidebar.number_input("Altezza Sormonto Contatto L [mm]", min_value=10.0, max_value=200.0, value=60.0, step=5.0)
s_L = st.sidebar.number_input("Spessore Barre a L [mm]", min_value=3.0, max_value=30.0, value=10.0, step=1.0)

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
    help="Includere la resistenza di contatto della giunzione bullonata (tipica 0.5 - 1.5 micro-ohm) più la resistenza del tratto di barra a L fino al nodo comune. Impostare 0.01 per vincolo equipotenziale ideale."
)

st.sidebar.header("4. Risoluzione Griglia Calcolo")
passo_mesh = st.sidebar.select_slider("Dimensione Cella Mesh [mm]", options=[2.5, 4.0, 5.0, 8.0, 10.0], value=5.0)


# --- FUNZIONE SOLUTORE 2D FDM ---
def risolvi_piastra_2d(B, H, s, T, I_tot, n_col, n_row, px, py, y0_pl, d_pl, wL, hL, yA, yB, R_uohm, d_mesh):
    # Proprietà fisiche Rame Cu-ETP (CW004A)
    rho_20 = 0.017241  # ohm * mm^2 / m
    alpha_20 = 0.00393  # 1 / K
    rho_T = rho_20 * (1.0 + alpha_20 * (T - 20.0))  # ohm * mm^2 / m
    rho_ohm_mm = rho_T * 1e-3  # ohm * mm
    sigma = 1.0 / rho_ohm_mm  # S / mm

    # Creazione griglia
    nx = max(10, int(round(B / d_mesh)))
    ny = max(20, int(round(H / d_mesh)))
    dx = B / nx
    dy = H / ny

    x_centers = (np.arange(nx) + 0.5) * dx
    y_centers = (np.arange(ny) + 0.5) * dy
    X_grid, Y_grid = np.meshgrid(x_centers, y_centers)

    # Conduttanze tra celle adiacenti
    Gx = sigma * (s * dy) / dx
    Gy = sigma * (s * dx) / dy

    N_nodi = nx * ny
    G_mat = sp.lil_matrix((N_nodi, N_nodi), dtype=np.float64)
    I_vec = np.zeros(N_nodi, dtype=np.float64)

    # Costruzione matrice conduttanze interne della piastra
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

    # Coordinate dei Powerlock (centrati simmetricamente lungo X)
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
                # Fallback sulla cella più vicina se il diametro è inferiore alla mesh
                idx_min = np.argmin(dist2)
                I_vec[idx_min] += I_singolo_pl
            else:
                indici_pl = np.flatnonzero(mask_pl)
                I_vec[indici_pl] += I_singolo_pl / n_celle_pl

    # Impronta Barra A e Barra B (centrate orizzontalmente sulla piastra)
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

    # Risoluzione del sistema lineare sparso
    G_csr = G_mat.tocsr()
    V_sol = spla.spsolve(G_csr, I_vec)
    V_2d = V_sol.reshape((ny, nx))

    # Calcolo correnti estratte dalle Barre A e B
    I_A = float(np.sum(V_sol[indici_A] * g_cella_A))
    I_B = float(np.sum(V_sol[indici_B] * g_cella_B))

    # Calcolo campo Elettrico e Densità di Corrente J [A/mm^2]
    dV_dy, dV_dx = np.gradient(V_2d, dy, dx)
    Jx = -sigma * dV_dx
    Jy = -sigma * dV_dy
    J_mag = np.sqrt(Jx ** 2 + Jy ** 2)

    # Calcolo profilo corrente longitudinale lungo l'altezza Y
    I_y_profile = np.sum(Jy, axis=1) * dx * s

    # Calcolo potenza dissipata per effetto Joule nella piastra (in watt)
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


# --- VALIDAZIONE GEOMETRICA ED ESECUZIONE ---
max_x_pl = (X_col - 1) * passo_x + diam_pl
max_y_pl = y_offset_pl + (Y_row - 1) * passo_y + diam_pl / 2.0

if max_x_pl > B_piastra:
    st.error(f"Attenzione: L'ingombro orizzontale dei Powerlock ({max_x_pl:.1f} mm) supera la larghezza della piastra ({B_piastra:.1f} mm).")
if max_y_pl > H_piastra:
    st.error(f"Attenzione: L'ingombro verticale dei Powerlock ({max_y_pl:.1f} mm) supera l'altezza della piastra ({H_piastra:.1f} mm).")
if w_L > B_piastra:
    st.error("Attenzione: La larghezza delle barre a L supera la larghezza della piastra collettrice.")

ris = risolvi_piastra_2d(
    B_piastra, H_piastra, s_piastra, T_operativa,
    I_totale, X_col, Y_row, passo_x, passo_y, y_offset_pl, diam_pl,
    w_L, h_L, y_barra_A, y_barra_B, R_cont_uohm, passo_mesh
)

# --- METRICHE PRINCIPALI DI RIEPILOGO ---
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
        delta=f"Densità L: {J_barra_A:.2f} A/mm²",
        delta_color="inverse" if J_barra_A > J_limite else "normal"
    )
with col2:
    st.metric(
        label="Corrente Barra B (Superiore)",
        value=f"{ris['I_B']:.1f} A ({perc_B:.1f}%)",
        delta=f"Densità L: {J_barra_B:.2f} A/mm²",
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

# --- VISUALIZZAZIONE GRAFICA 2D ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8), gridspec_kw={'width_ratios': [1.3, 1.3, 1.0]})

# GRAFICO 1: Mappa Densità di Corrente |J| e Linee di Flusso
im1 = ax1.pcolormesh(ris["X"], ris["Y"], ris["J_mag"], shading='auto', cmap='inferno', vmin=0, vmax=max(J_limite * 1.5, J_picco_piastra))
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label("Densità di Corrente |J| [A/mm²]", fontsize=10)

# Sovrapposizione linee di flusso della corrente
ax1.streamplot(ris["x"], ris["y"], ris["Jx"], ris["Jy"], color='cyan', linewidth=0.7, density=1.0, arrowsize=0.8)

# Contorno critica J > J_limite
if J_picco_piastra > J_limite:
    ax1.contour(ris["X"], ris["Y"], ris["J_mag"], levels=[J_limite], colors=['lime'], linewidths=1.5, linestyles='--')

# Disegno Powerlock e Barre a L
for (xp, yp) in ris["coords_pl"]:
    circ = patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.2)
    ax1.add_patch(circ)

x_min_L = (B_piastra - w_L) / 2.0
rect_A = patches.Rectangle((x_min_L, y_barra_A - h_L / 2.0), w_L, h_L, edgecolor='cyan', facecolor='none', linewidth=2.0, linestyle='-')
rect_B = patches.Rectangle((x_min_L, y_barra_B - h_L / 2.0), w_L, h_L, edgecolor='cyan', facecolor='none', linewidth=2.0, linestyle='-')
ax1.add_patch(rect_A)
ax1.add_patch(rect_B)
ax1.text(B_piastra / 2.0, y_barra_A, f"BARRA A\n{perc_A:.1f}%", color='white', weight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
ax1.text(B_piastra / 2.0, y_barra_B, f"BARRA B\n{perc_B:.1f}%", color='white', weight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

ax1.set_title("Mappa Densità |J| [A/mm²] e Linee di Corrente", fontsize=12, weight='bold')
ax1.set_xlabel("Larghezza X [mm]")
ax1.set_ylabel("Quota Verticale Y [mm]")
ax1.set_aspect('equal')
ax1.set_xlim(0, B_piastra)
ax1.set_ylim(0, H_piastra)

# GRAFICO 2: Mappa del Potenziale Elettrico V [mV]
im2 = ax2.pcolormesh(ris["X"], ris["Y"], ris["V_mV"], shading='auto', cmap='viridis')
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label("Potenziale Elettrico Nodo [mV]", fontsize=10)

for (xp, yp) in ris["coords_pl"]:
    circ2 = patches.Circle((xp, yp), radius=diam_pl / 2.0, edgecolor='white', facecolor='none', linewidth=1.0, alpha=0.7)
    ax2.add_patch(circ2)

rect_A2 = patches.Rectangle((x_min_L, y_barra_A - h_L / 2.0), w_L, h_L, edgecolor='red', facecolor='none', linewidth=2.0)
rect_B2 = patches.Rectangle((x_min_L, y_barra_B - h_L / 2.0), w_L, h_L, edgecolor='red', facecolor='none', linewidth=2.0)
ax2.add_patch(rect_A2)
ax2.add_patch(rect_B2)

ax2.set_title("Distribuzione Potenziale Elettrico [mV]", fontsize=12, weight='bold')
ax2.set_xlabel("Larghezza X [mm]")
ax2.set_ylabel("Quota Verticale Y [mm]")
ax2.set_aspect('equal')
ax2.set_xlim(0, B_piastra)
ax2.set_ylim(0, H_piastra)

# GRAFICO 3: Profilo della Corrente Verticale lungo l'Altezza Y
ax3.plot(np.abs(ris["I_y_profile"]), ris["y"], color='#d95f02', linewidth=2.5, label="|Iy(y)| lungo la sbarra")
ax3.axhline(y_barra_A, color='blue', linestyle='--', linewidth=1.5, label=f"Quota Barra A ({y_barra_A:.0f} mm)")
ax3.axhline(y_barra_B, color='green', linestyle='--', linewidth=1.5, label=f"Quota Barra B ({y_barra_B:.0f} mm)")
ax3.axvline(B_piastra * s_piastra * J_limite, color='red', linestyle=':', linewidth=1.5, label=f"Portata Limite Piastra ({B_piastra * s_piastra * J_limite:.0f} A)")
ax3.fill_betweenx(ris["y"], 0, np.abs(ris["I_y_profile"]), color='#d95f02', alpha=0.2)
ax3.set_title("Carico Longitudinale Sbarra |Iy| [A]", fontsize=12, weight='bold')
ax3.set_xlabel("Corrente Assiale nella Sezione [A]")
ax3.set_ylabel("Quota Verticale Y [mm]")
ax3.set_ylim(0, H_piastra)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend(loc='upper right', fontsize=9)

plt.tight_layout()
st.pyplot(fig)

# --- TABELLA DI REPORT TECNICO E DFM ---
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
