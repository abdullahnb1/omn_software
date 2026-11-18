import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches

# ==========================================
# 1. PARAMETRELER VE HARİTA (10x10 Metre)
# ==========================================
RESOLUTION = 0.05  # 1 Piksel = 5 cm
MAX_RANGE_PX = 20  # 1000 mm / 50 mm = 20 Piksel (Maksimum Sensör Menzili)

grid_size = 200
map_grid = np.zeros((grid_size, grid_size))

# Dış Duvarlar ve Odalar
map_grid[0:4, :] = 1; map_grid[-4:, :] = 1; map_grid[:, 0:4] = 1; map_grid[:, -4:] = 1
map_grid[0:90, 80:84] = 1; map_grid[110:130, 80:84] = 1
map_grid[30:50, 30:50] = 1 # Masa
map_grid[125:129, 80:150] = 1; map_grid[125:129, 170:200] = 1
map_grid[150:185, 110:114] = 1 # Uzun Dolap

# ==========================================
# 2. ROBOT VE SENSÖR MODELİ (40x40 Sekizgen)
# ==========================================
R_SIZE = 8.0 # 8 piksel = 40cm
C_SIZE = 2.4 # Chamfer

corners_local = np.array([
    [R_SIZE/2 - C_SIZE, R_SIZE/2], [-(R_SIZE/2 - C_SIZE), R_SIZE/2],
    [-R_SIZE/2, R_SIZE/2 - C_SIZE], [-R_SIZE/2, -(R_SIZE/2 - C_SIZE)],
    [-(R_SIZE/2 - C_SIZE), -R_SIZE/2], [R_SIZE/2 - C_SIZE, -R_SIZE/2],
    [R_SIZE/2, -(R_SIZE/2 - C_SIZE)], [R_SIZE/2, R_SIZE/2 - C_SIZE]
])

sensors = [
    ("Ön", 0, R_SIZE/2, 90), ("Ön-Sol", -(R_SIZE/2 - C_SIZE/2), R_SIZE/2 - C_SIZE/2, 135),
    ("Ön-Sağ", R_SIZE/2 - C_SIZE/2, R_SIZE/2 - C_SIZE/2, 45), ("Sol", -R_SIZE/2, 0, 180),
    ("Sağ", R_SIZE/2, 0, 0), ("Arka", 0, -R_SIZE/2, 270)
]

def draw_robot_and_sensors(ax, rx, ry, ryaw):
    """Robotu ve Sensör Işınlarını Çizer"""
    yaw_rad = np.radians(ryaw)
    rot_matrix = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad)],
                           [np.sin(yaw_rad),  np.cos(yaw_rad)]])

    corners_world = np.dot(corners_local, rot_matrix.T)
    corners_world[:, 0] += rx; corners_world[:, 1] += ry
    poly = patches.Polygon(corners_world, closed=True, facecolor='black', edgecolor='cyan', linewidth=1.5, zorder=5)
    ax.add_patch(poly)

    for name, sx, sy, angle in sensors:
        sx_w = (sx * np.cos(yaw_rad) - sy * np.sin(yaw_rad)) + rx
        sy_w = (sx * np.sin(yaw_rad) + sy * np.cos(yaw_rad)) + ry
        ray_ang = np.radians(ryaw + angle)

        hit_x, hit_y = sx_w, sy_w
        for step in range(MAX_RANGE_PX):
            test_x = sx_w + step * np.cos(ray_ang)
            test_y = sy_w + step * np.sin(ray_ang)
            if not (0 <= int(test_x) < grid_size and 0 <= int(test_y) < grid_size) or map_grid[int(test_y), int(test_x)] == 1:
                break
            hit_x, hit_y = test_x, test_y

        ax.plot([sx_w, hit_x], [sy_w, hit_y], color='lime', linewidth=2, zorder=4)
        ax.scatter(hit_x, hit_y, color='white', s=15, zorder=6)

# ==========================================
# 3. SENARYO VERİLERİ VE ISI HARİTALARI
# ==========================================
robot_poses = [(130, 150, 90), (130, 170, 90), (130, 185, 90)]

heatmaps = []
# T=0 (İkilem - Alt Duvar vs Üst Dolap)
h0 = np.zeros_like(map_grid, dtype=float); h0[40:80, 60] = 1; h0[140:175, 130] = 1
heatmaps.append(gaussian_filter(h0, sigma=5))

# T=1 (Hareket Ediyor - İkilem Devam Ediyor)
h1 = np.zeros_like(map_grid, dtype=float); h1[60:90, 60] = 1; h1[160:185, 130] = 1
heatmaps.append(gaussian_filter(h1, sigma=4))

# T=2 (%100 Eminlik - Kapı Bulundu)
h2 = np.zeros_like(map_grid, dtype=float); h2[185, 130] = 100
heatmaps.append(gaussian_filter(h2, sigma=4))

# ==========================================
# 4. GÖRSELLEŞTİRME (2 Satır x 3 Sütun)
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
plt.subplots_adjust(hspace=0.2, wspace=0.1)

time_titles = ["T=0: Kör Uyanış", "T=1: İleri Hareket", "T=2: SİMETRİ KIRILMASI"]

for c in range(3):
    rx, ry, ryaw = robot_poses[c]

    # --- ÜST SATIR: ROBOTUN BEYNİ (TÜM HARİTA ISI HARİTASI) ---
    ax_top = axes[0, c]
    ax_top.imshow(heatmaps[c], cmap='inferno', origin='lower')
    walls = np.ma.masked_where(map_grid == 0, map_grid)
    ax_top.imshow(walls, cmap='Greys', origin='lower', alpha=0.3)

    # Gerçek Konumu Temsili Belirtme
    rect = patches.Rectangle((rx-6, ry-6), 12, 12, linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--')
    ax_top.add_patch(rect)

    ax_top.set_title(time_titles[c], fontweight='bold', fontsize=14, pad=15)
    if c == 0: ax_top.set_ylabel("ROBOTUN BEYNİ\n(Global Heatmap)", fontweight='bold', fontsize=12, labelpad=20)
    ax_top.set_xticks([]); ax_top.set_yticks([])

    # --- ALT SATIR: ROBOTUN GÖZÜ (LOKAL SENSÖR OKUMASI) ---
    ax_bot = axes[1, c]
    ax_bot.imshow(map_grid, cmap='Greys', origin='lower')
    draw_robot_and_sensors(ax_bot, rx, ry, ryaw)

    # Kamerayı robotun etrafına zoomla
    ax_bot.set_xlim(rx - 30, rx + 30)
    ax_bot.set_ylim(ry - 30, ry + 30)

    if c == 0: ax_bot.set_ylabel("ROBOTUN GÖZÜ\n(Lokal Sensör Işınları)", fontweight='bold', fontsize=12, labelpad=20)
    ax_bot.set_xticks([]); ax_bot.set_yticks([])

# Özel Vurgular (Sadece T=2 anı için)
axes[0, 2].text(rx-60, ry-30, "Yanlış ihtimal\nsilindi!\n%100 EMİNLİK", color='cyan', fontsize=12, fontweight='bold')
axes[1, 2].text(rx+10, ry+10, "Sağ-Ön sensör 1 metre\nboşluğa fırladı!\n(İmza Okundu)", color='lime', fontsize=11, fontweight='bold')

plt.suptitle("Monte Carlo Lokalizasyonunda 'Eşleştirme' Anı (ME407 Omniwheel Araç)", fontsize=18, fontweight='bold', y=0.95)
plt.show()