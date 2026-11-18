# Path bulma

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev

# ==========================================
# 1. PARAMETRELER (40x40 cm Araç)
# ==========================================
RESOLUTION = 0.05  # 1 piksel = 5 cm
ROBOT_WIDTH = 0.40
ROBOT_LENGTH = 0.40
hard_limit_px = (ROBOT_WIDTH / 2) / RESOLUTION

# ==========================================
# 2. KOMPLEKS HARİTA ÜRETİMİ (10x10 Metre Ev)
# ==========================================
grid_size = 200
map_grid = np.zeros((grid_size, grid_size))

# Dış Duvarlar
map_grid[0:200, 0:3] = 1; map_grid[0:200, 197:200] = 1
map_grid[0:3, 0:200] = 1; map_grid[197:200, 0:200] = 1

# Oda 1 (Sol Alt) ve Duvarı
map_grid[0:90, 80:85] = 1
# Kapı boşluğu Y: 90 - 110 arası açık
map_grid[110:130, 80:85] = 1

# Oda 2 (Üst Katman) ve Yatay Duvarı
map_grid[125:130, 80:150] = 1
# Kapı boşluğu X: 150 - 170 arası açık
map_grid[125:130, 170:200] = 1

# Mobilyalar ve İç Engeller
map_grid[30:50, 30:50] = 1   # Oda 1'in ortasında masa
map_grid[150:185, 110:115] = 1 # Üst odada odayı ikiye bölen uzun dolap
map_grid[150:155, 110:160] = 1 # Dolabın L uzantısı (Labirent etkisi)

# ==========================================
# 3. COSTMAP (Maliyet Haritası)
# ==========================================
print("Harita ve Costmap oluşturuluyor...")
distance_map = distance_transform_edt(1 - map_grid)
costmap = np.zeros_like(distance_map, dtype=float)

for row in range(grid_size):
    for col in range(grid_size):
        dist = distance_map[row, col]
        if dist <= hard_limit_px:
            costmap[row, col] = np.inf
        else:
            # Üstel merkez çekimi (Koridoru ortalamak için)
            costmap[row, col] = 200 * np.exp(-0.4 * (dist - hard_limit_px))

# ==========================================
# 4. A-STAR ALGORİTMASI
# ==========================================
def astar(costmap, start, goal):
    neighbors = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        current_f, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size):
                continue
            if costmap[neighbor[0], neighbor[1]] == np.inf:
                continue

            move_cost = np.sqrt(dx**2 + dy**2)
            tentative_g = g_score[current] + move_cost + costmap[neighbor[0], neighbor[1]]

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h_score = np.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                heapq.heappush(open_set, (tentative_g + h_score, neighbor))
    return None

start_node = (20, 20)   # Sol alt odanın köşesi
goal_node = (180, 140)  # Üst odadaki dolabın arkası

print("A* rotayı hesaplıyor (Bu işlem 1-2 saniye sürebilir)...")
path = astar(costmap, start_node, goal_node)
path_y, path_x = zip(*path)

# ==========================================
# 5. PATH SMOOTHING VE YÖNELİM
# ==========================================
print("Rota pürüzsüzleştiriliyor ve araç footprint'i yerleştiriliyor...")
path_coords = np.array([path_x, path_y])
tck, u = splprep(path_coords, s=80.0) # Keskin köşeleri kırmak için yumuşatma
u_new = np.linspace(0, 1.0, 600)
smooth_x, smooth_y = splev(u_new, tck)

# Yönelim (Yaw / Teğet) hesabı
dx = np.gradient(smooth_x)
dy = np.gradient(smooth_y)
headings = np.arctan2(dy, dx)

# ==========================================
# 6. GÖRSELLEŞTİRME (Robot Footprint'i ile)
# ==========================================
fig, ax = plt.subplots(figsize=(12, 12))

# Haritayı arka plan olarak çiz
masked_costmap = np.ma.masked_invalid(costmap)
ax.imshow(masked_costmap, cmap='jet', origin='lower', alpha=0.5)
ax.imshow(map_grid, cmap='Greys', origin='lower', alpha=0.7)

# Yumuşatılmış Spline Rotası
ax.plot(smooth_x, smooth_y, color='white', linewidth=2, linestyle='--', label='İzlenecek Rota')
ax.scatter(start_node[1], start_node[0], color='lime', s=150, marker='*', label='Başlangıç', zorder=5)
ax.scatter(goal_node[1], goal_node[0], color='magenta', s=150, marker='*', label='Hedef', zorder=5)

# Belirli aralıklarla robot şasisini rotaya çizme
step = 40  # Robotu her 40 noktada bir çiz
L = ROBOT_LENGTH / RESOLUTION
W = ROBOT_WIDTH / RESOLUTION

# Sadece label kargaşası olmaması için ilkine label vereceğiz
first_robot_drawn = False

for i in range(0, len(smooth_x), step):
    x, y, yaw = smooth_x[i], smooth_y[i], headings[i]

    # Robot köşeleri (Merkeze göre, X: ileri, Y: sol)
    corners = np.array([
        [L/2, W/2],   # Sol Ön
        [L/2, -W/2],  # Sağ Ön
        [-L/2, -W/2], # Sağ Arka
        [-L/2, W/2]   # Sol Arka
    ])

    # Dönüş Matrisi (Rotation Matrix)
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])

    # Köşeleri döndür ve haritadaki (x, y) konumuna ötele
    rot_corners = np.dot(corners, rot.T)
    rot_corners[:, 0] += x
    rot_corners[:, 1] += y

    # 40x40 Kutu çizimi
    lbl = 'Araç Şasisi (40x40)' if not first_robot_drawn else ""
    poly = patches.Polygon(rot_corners, closed=True, edgecolor='lime', facecolor='black', linewidth=1.5, alpha=0.8, label=lbl)
    ax.add_patch(poly)

    # Aracın yönünü belli eden Kırmızı Ön Yüz
    front_lbl = 'Aracın Önü (Kırmızı)' if not first_robot_drawn else ""
    front_x = [rot_corners[0, 0], rot_corners[1, 0]]
    front_y = [rot_corners[0, 1], rot_corners[1, 1]]
    ax.plot(front_x, front_y, color='red', linewidth=3, zorder=4, label=front_lbl)

    first_robot_drawn = True

ax.set_title('Kompleks Kat Planında 40x40 Omniwheel Aracın Navigasyonu\n(Aracın ön yüzü sürekli yola bakacak şekilde dönüyor)')
ax.set_xlabel('X Ekseni (Piksel)'); ax.set_ylabel('Y Ekseni (Piksel)')
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()