import argparse
import os
import time
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import map_coordinates, zoom
from scipy.optimize import minimize

# ---------------------------------------------------------
# 0. 全域配置
# ---------------------------------------------------------
OUTPUT_DIR = "hbsr_final_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_image(img_arr, filename):
    """將浮點數影像轉為 uint8 並儲存為 PNG"""
    u8 = np.clip(img_arr, 0, 255).astype(np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(u8), os.path.join(OUTPUT_DIR, filename))

def load_processed_image(path, target_size=(256, 256)):
    """讀取影像並進行標準化處理 (Resample + Normalize)"""
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(names)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(path)
    
    arr = sitk.GetArrayFromImage(image)
    # 若是 3D，取中間切片
    if arr.ndim == 3: arr = arr[arr.shape[0] // 2]
    
    # 簡單縮放 (Bilinear)
    zoom_factor = [t / s for t, s in zip(target_size, arr.shape)]
    arr = zoom(arr, zoom_factor, order=1)
    
    # 正規化至 0-255
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    return arr.astype(np.float32)

# ---------------------------------------------------------
# 1. 核心類別：Hierarchical B-Spline Registrar
# ---------------------------------------------------------
class HBSRRegistrar:
    def __init__(self, fixed, moving):
        self.fixed = fixed
        self.moving = moving
        self.H, self.W = fixed.shape
        
        # 狀態紀錄
        self.mi_history = []
        self.tre_history = []
        self.roi_history = []
        self.spacings = []
        
        # 預對齊後的參考影像
        self.moving_aligned = None

    def _get_deformation_field(self, grid_y, grid_x):
        """
        利用 Bicubic 插值將稀疏的控制點 (Grid) 放大成緻密的變形場 (Deformation Field)。
        這模擬了 B-Spline 的平滑特性。
        """
        scale_y = self.H / grid_y.shape[0]
        scale_x = self.W / grid_x.shape[0]
        dy = zoom(grid_y, scale_y, order=3, mode='nearest')[:self.H, :self.W]
        dx = zoom(grid_x, scale_x, order=3, mode='nearest')[:self.H, :self.W]
        return dy, dx

    def _warp(self, image, dy, dx):
        """
        Backward Mapping Warping:
        Output(y, x) = Input(y + dy, x + dx)
        """
        y, x = np.mgrid[0:self.H, 0:self.W]
        coords = np.array([y + dy, x + dx])
        return map_coordinates(image, coords, order=1, mode='nearest')

    def _cost_function(self, params, grid_shape, mask_indices, base_dy, base_dx):
        """
        目標函數 (Negative Mutual Information)。
        關鍵修正：這裡計算的是 (累積變形場 + 當前增量)。
        """
        # 1. 解析優化器傳入的參數 (增量 Delta)
        if mask_indices is None:
            grid = params.reshape(2, *grid_shape)
        else:
            grid = np.zeros((2, *grid_shape))
            n = len(params) // 2
            grid[0][mask_indices] = params[:n]
            grid[1][mask_indices] = params[n:]

        delta_dy, delta_dx = self._get_deformation_field(grid[0], grid[1])
        
        # 2. 疊加：Total = Base (History) + Delta (Current Optimization)
        total_dy = base_dy + delta_dy
        total_dx = base_dx + delta_dx
        
        # 3. 變形 (始終對 moving_aligned 進行操作，避免插值誤差累積)
        warped = self._warp(self.moving_aligned, total_dy, total_dx)
        
        # 4. 計算 Histogram-based MI
        bins = 64
        hist_2d, _, _ = np.histogram2d(self.fixed.ravel(), warped.ravel(), bins=bins)
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1); py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        
        # Negative MI (Minimize this to Maximize MI)
        cost = -np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
        
        self.mi_history.append(cost)
        return cost

    def run(self, pts_f, pts_m, grid_spacings=[48, 32, 16]):
        self.spacings = grid_spacings
        print(f"--- 開始配準 (Initial TRE: {np.mean(np.linalg.norm(pts_f - pts_m, axis=1)):.2f} px) ---")
        
        # ---------------- Step 1: 預對齊 (Pre-alignment) ----------------
        # 修正：Shift Vector = Moving - Fixed
        # 因為 Warp 是 Backward Mapping: coord = grid + displacement
        # 要把 Moving 移到 Fixed 的位置，Displacement 必須是 (Moving座標 - Fixed座標)
        if len(pts_f) > 0:
            shift = np.mean(pts_m, axis=0) - np.mean(pts_f, axis=0)
        else:
            shift = np.zeros(2)
            
        # 產生對齊後的基準圖
        self.moving_aligned = self._warp(self.moving, np.full((self.H, self.W), shift[0]), np.full((self.H, self.W), shift[1]))
        
        # 紀錄 TRE
        current_tre = np.mean(np.linalg.norm((pts_f + shift) - pts_m, axis=1))
        self.tre_history.append(current_tre) # Start
        print(f"Pre-alignment Shift: {shift}, TRE: {current_tre:.2f} px")

        # ---------------- Step 2: 分層優化 (Hierarchical) ----------------
        # 初始化累積變形場
        total_dy = np.zeros((self.H, self.W))
        total_dx = np.zeros((self.H, self.W))
        warped = self.moving_aligned.copy()

        start_time = time.time()

        for level, spacing in enumerate(grid_spacings):
            gh, gw = self.H // spacing + 1, self.W // spacing + 1
            print(f"\nLevel {level+1} (Spacing: {spacing})")

            # --- A. ROI 選擇邏輯 (根據地標誤差) ---
            mask = None
            if level == 0:
                print("  Mode: Global Registration (Full Grid)")
            else:
                # 計算目前 Fixed 地標經過變形後，預測會對應到 Moving 的哪裡
                # Predict = Fixed + Shift + Deformation
                curr_dy = map_coordinates(total_dy, [pts_f[:,0], pts_f[:,1]], order=1)
                curr_dx = map_coordinates(total_dx, [pts_f[:,0], pts_f[:,1]], order=1)
                pred_pts = pts_f + shift + np.stack([curr_dy, curr_dx], axis=1)
                
                # 計算誤差
                diffs = np.linalg.norm(pred_pts - pts_m, axis=1)
                print(f"  Avg Landmark Error: {np.mean(diffs):.2f} px")
                
                # 篩選誤差大的區域 (Threshold = 2.0 px)
                active = np.where(diffs > 2.0)[0]
                if len(active) == 0:
                    print("  Converged. Skipping this level.")
                    self.roi_history.append(None)
                    continue
                
                # 建立 Mask
                mask = np.zeros((gh, gw), dtype=bool)
                for idx in active:
                    gy, gx = int(pts_f[idx,0]/spacing), int(pts_f[idx,1]/spacing)
                    # 擴張 3x3 區域
                    mask[max(0, gy-1):min(gh, gy+2), max(0, gx-1):min(gw, gx+2)] = True
                print(f"  Mode: ROI Refinement ({np.sum(mask)}/{gh*gw} grids active)")
            
            self.roi_history.append(mask)

            # --- B. 執行優化 (Powell) ---
            # 初始參數為 0 (代表從目前的 total_dy/dx 開始微調)
            num_params = 2 * (np.sum(mask) if mask is not None else gh * gw)
            initial_params = np.zeros(num_params)
            
            # 傳入 total_dy/dx 作為 base
            res = minimize(
                self._cost_function, 
                initial_params,
                args=((gh, gw), mask, total_dy, total_dx), 
                method='Powell', 
                options={'maxiter': 5, 'disp': True}
            )

            # --- C. 更新累積變形場 ---
            if mask is None:
                grid = res.x.reshape(2, gh, gw)
            else:
                grid = np.zeros((2, gh, gw))
                n = len(res.x)//2
                grid[0][mask] = res.x[:n]
                grid[1][mask] = res.x[n:]
            
            delta_dy, delta_dx = self._get_deformation_field(grid[0], grid[1])
            
            # 修正：正確累加變形場
            total_dy += delta_dy
            total_dx += delta_dx
            
            # 更新顯示用的影像
            warped = self._warp(self.moving_aligned, total_dy, total_dx)
            
            # 紀錄本層結束後的 TRE
            fin_dy = map_coordinates(total_dy, [pts_f[:,0], pts_f[:,1]], order=1)
            fin_dx = map_coordinates(total_dx, [pts_f[:,0], pts_f[:,1]], order=1)
            fin_pred = pts_f + shift + np.stack([fin_dy, fin_dx], axis=1)
            self.tre_history.append(np.mean(np.linalg.norm(fin_pred - pts_m, axis=1)))

        print(f"\nTotal Time: {time.time() - start_time:.2f}s")
        return warped, total_dy, total_dx, shift, self.moving_aligned

# ---------------------------------------------------------
# 2. 視覺化模組 (包含進階圖表)
# ---------------------------------------------------------
class Visualizer:
    @staticmethod
    def plot_roi_strategy(fixed, spacings, rois):
        """繪製每一層的 ROI 策略 (論文 Fig 1 概念)"""
        num = len(spacings)
        fig, axs = plt.subplots(1, num, figsize=(4*num, 4))
        if num == 1: axs = [axs]
        
        for i, (sp, mask) in enumerate(zip(spacings, rois)):
            axs[i].imshow(fixed, cmap='gray')
            axs[i].set_title(f"Level {i+1} (Spacing: {sp})")
            # 畫出 B-Spline 網格線
            h, w = fixed.shape
            for y in range(0, h, sp): axs[i].axhline(y, c='cyan', alpha=0.1, lw=0.5)
            for x in range(0, w, sp): axs[i].axvline(x, c='cyan', alpha=0.1, lw=0.5)
            
            # 畫出 Active Region (紅框)
            if mask is not None:
                ys, xs = np.where(mask)
                for gy, gx in zip(ys, xs):
                    axs[i].add_patch(Rectangle((gx*sp, gy*sp), sp, sp, ec='r', fc='r', alpha=0.3))
                axs[i].text(5, h-10, f"Active: {np.sum(mask)}", c='white', backgroundcolor='red', fontsize=8)
            else:
                axs[i].text(5, h-10, "Global (All)", c='white', backgroundcolor='blue', fontsize=8)
            axs[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "01_ROI_strategy.png"))
        plt.close()

    @staticmethod
    def plot_overlay(fixed, moving, name="overlay"):
        """假色疊圖: Fixed(Green) vs Moving(Magenta)"""
        norm_f = np.clip(fixed/255, 0, 1)
        norm_m = np.clip(moving/255, 0, 1)
        # R=Moving, G=Fixed, B=Moving -> Magenta vs Green
        rgb = np.stack([norm_m, norm_f, norm_m], axis=-1)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(rgb)
        plt.title(f"{name} (G:Fixed, M:Moving)")
        plt.axis('off')
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"))
        plt.close()

    @staticmethod
    def plot_vectors(dy, dx, name="vectors"):
        """變形向量場 (Vector Field)"""
        step = 15
        h, w = dy.shape
        y, x = np.mgrid[0:h:step, 0:w:step]
        # 下採樣以避免箭頭過密
        vy, vx = dy[0:h:step, 0:w:step], dx[0:h:step, 0:w:step]
        mag = np.sqrt(vy**2 + vx**2)
        
        plt.figure(figsize=(6, 6))
        # 使用 jet colormap 表示變形強度
        plt.quiver(x, y, vx, vy, mag, cmap='jet', angles='xy', scale_units='xy', scale=0.8)
        plt.gca().invert_yaxis()
        plt.colorbar(label='Displacement (px)')
        plt.title("Deformation Vector Field")
        plt.axis('off')
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"))
        plt.close()

    @staticmethod
    def plot_contours(fixed, moving, name="contours"):
        """輪廓比較圖"""
        plt.figure(figsize=(6, 6))
        plt.imshow(fixed, cmap='gray', alpha=0.5)
        # 紅色: Target, 綠色: Result
        plt.contour(fixed, levels=[50], colors='red', linewidths=1, alpha=0.8)
        plt.contour(moving, levels=[50], colors='#00FF00', linewidths=1, alpha=0.8)
        
        # 製作 Legend
        plt.plot([], [], c='red', label='Target (Fixed)')
        plt.plot([], [], c='#00FF00', label='Result (Warped)')
        plt.legend()
        plt.axis('off')
        plt.title("Contour Comparison")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"))
        plt.close()
    
    @staticmethod
    def plot_grid_deformation(img, dy, dx, name="grid_deformation"):
        """在影像上疊加變形後的網格"""
        step = 20
        h, w = img.shape
        y, x = np.mgrid[0:h:step, 0:w:step]
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        
        # 加上位移後的網格座標
        def_y = y + dy[0:h:step, 0:w:step]
        def_x = x + dx[0:h:step, 0:w:step]
        
        plt.plot(def_x, def_y, 'cyan', alpha=0.4, lw=0.8)
        plt.plot(def_x.T, def_y.T, 'cyan', alpha=0.4, lw=0.8)
        plt.axis('off')
        plt.title("Deformed Grid")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"))
        plt.close()

    @staticmethod
    def plot_tre(history):
        """繪製 TRE 收斂曲線"""
        plt.figure(figsize=(6, 4))
        plt.plot(history, 'o-', c='purple', lw=2)
        plt.title("Registration Accuracy (TRE)")
        plt.ylabel("Mean Landmark Error (px)")
        plt.xlabel("Stage")
        plt.xticks(range(len(history)), ["Start"] + [f"Level {i+1}" for i in range(len(history)-1)])
        for i, v in enumerate(history): 
            plt.text(i, v+0.5, f"{v:.2f}", ha='center')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "02_TRE_chart.png"))
        plt.close()

    @staticmethod
    def plot_mi(history):
        """繪製 MI 優化歷程 (Optimization History)"""
        plt.figure(figsize=(8, 4))
        plt.plot(history, linewidth=1.5)
        plt.title("Optimization History (Cost Function)")
        plt.ylabel("Negative Mutual Information")
        plt.xlabel("Iteration")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "00_MI_history.png"))
        plt.close()

# ---------------------------------------------------------
# 3. 輔助功能：地標管理
# ---------------------------------------------------------
def get_landmarks(fixed, moving, path):
    """載入或建立地標檔案"""
    if os.path.exists(path):
        print(f"從檔案載入地標: {path}")
        d = np.loadtxt(path)
        return d[:, :2], d[:, 2:]
    
    print("未發現地標檔，啟動手動標記介面...")
    print("操作：左圖點擊 -> 右圖點擊 (成對)。按 Enter 完成並儲存。")
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(fixed, cmap='gray'); ax[0].set_title("Fixed (MRI)")
    ax[1].imshow(moving, cmap='gray'); ax[1].set_title("Moving (CT)")
    p1, p2 = [], []
    
    def onclick(event):
        if event.inaxes == ax[0]:
            p1.append((event.ydata, event.xdata))
            ax[0].plot(event.xdata, event.ydata, 'ro')
            ax[0].text(event.xdata, event.ydata, str(len(p1)), color='yellow')
        elif event.inaxes == ax[1]:
            p2.append((event.ydata, event.xdata))
            ax[1].plot(event.xdata, event.ydata, 'go')
            ax[1].text(event.xdata, event.ydata, str(len(p2)), color='yellow')
        fig.canvas.draw()
        
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    p1, p2 = np.array(p1), np.array(p2)
    if len(p1) > 0 and len(p1) == len(p2):
        np.savetxt(path, np.hstack([p1, p2]), header="yf xf ym xm", fmt='%.4f')
        print(f"地標已儲存至 {path}")
    else:
        print("警告：未標記或標記不成對，將不儲存。")
        
    return p1, p2

# ---------------------------------------------------------
# 4. 主程式入口
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HBSR Registration Tool")
    parser.add_argument('--mri', required=True, help="Path to Fixed Image")
    parser.add_argument('--ct', required=True, help="Path to Moving Image")
    parser.add_argument('--lm', default='landmarks.txt', help="Path to landmarks file")
    args = parser.parse_args()

    print("=== 初始化 HBSR 配準流程 ===")
    
    # 1. 載入與前處理
    fixed = load_processed_image(args.mri)
    moving = load_processed_image(args.ct)
    save_image(fixed, "source_fixed.png")
    save_image(moving, "source_moving.png")

    # 2. 取得地標
    pts_f, pts_m = get_landmarks(fixed, moving, args.lm)
    if len(pts_f) == 0:
        print("錯誤：無可用地標，程式終止。")
        return

    # 3. 執行配準
    registrar = HBSRRegistrar(fixed, moving)
    warped, total_dy, total_dx, shift, moving_aligned = registrar.run(pts_f, pts_m)

    # 4. 產生報告
    print("\n=== 產生視覺化報告 ===")
    save_image(warped, "result_warped.png")
    save_image(np.abs(fixed - warped)*2, "result_diff.png")
    
    vis = Visualizer()
    
    # 效能與策略圖表
    vis.plot_mi(registrar.mi_history)
    vis.plot_roi_strategy(fixed, registrar.spacings, registrar.roi_history)
    vis.plot_tre(registrar.tre_history)
    
    # 配準前後對比 (疊圖 & 輪廓)
    vis.plot_overlay(fixed, moving_aligned, "overlay_before") # Before (但已經過 Pre-align)
    vis.plot_overlay(fixed, warped, "overlay_after")
    vis.plot_contours(fixed, warped, "contour_compare")
    
    # B-Spline 變形分析 (向量場 & 網格)
    vis.plot_vectors(total_dy, total_dx, "deformation_vectors")
    vis.plot_grid_deformation(warped, total_dy, total_dx, "grid_deformation")

    print(f"全部完成！所有結果已儲存於目錄: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()