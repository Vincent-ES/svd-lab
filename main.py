# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
import os

# ========== 讀取圖片 ==========
img_org = cv.imread('Cybercab.jpg')
if img_org is None:
    print("錯誤：找不到 Cybercab.jpg，請確認圖片路徑")
    exit()

print('原始圖片形狀 (高度, 寬度, 通道數):', img_org.shape)
height, width, channels = img_org.shape
print(f'解析度: {width} x {height} x {channels}')

# ========== 定義 SVD 壓縮函數 ==========
def svd_compression(img, k):
    """
    對彩色圖像進行 SVD 壓縮。
    對 R、G、B 三個通道分別進行 SVD 分解，保留前 k 個奇異值來重建圖像。
    
    參數:
        img: 輸入的彩色圖像 (numpy array, shape: m x n x 3)
        k:   保留的奇異值數量
    回傳:
        壓縮後的圖像 (numpy array, shape: m x n x 3)
    """
    res_image = np.zeros_like(img)  # 與原圖相同的 uint8 型態
    for i in range(img.shape[2]):  # 迭代 B, G, R 三個通道
        U, Sigma, VT = np.linalg.svd(img[:, :, i])  # SVD 分解
        # 保留前 k 個奇異值，重建壓縮後的圖像通道
        res_image[:, :, i] = U[:, :k].dot(np.diag(Sigma[:k])).dot(VT[:k, :])

    return res_image


# ========== 計算 MSE 和 PSNR ==========
def calculate_mse(original, compressed):
    """計算均方誤差 (Mean Squared Error)"""
    return np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)

def calculate_psnr(original, compressed):
    """計算峰值信噪比 (Peak Signal-to-Noise Ratio)"""
    mse = calculate_mse(original, compressed)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255.0 ** 2) / mse)


# ========== 計算壓縮率 ==========
def calculate_compression_ratio(m, n, c, k):
    """
    計算 SVD 壓縮的壓縮比。
    原始資料量: m * n * c
    壓縮後資料量 (每個通道儲存 U_k, Sigma_k, VT_k): c * (m*k + k + k*n)
    """
    original_size = m * n * c
    compressed_size = c * (m * k + k + k * n)
    ratio = original_size / compressed_size
    return original_size, compressed_size, ratio


# ========== 在圖片上加標籤的輔助函數 ==========
def add_label(img, text, position='top-left', font_scale=1.0, color=(255, 255, 255)):
    """在圖片上添加文字標籤（帶半透明背景）"""
    labeled = img.copy()
    font = cv.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv.getTextSize(text, font, font_scale, thickness)
    
    if position == 'top-left':
        x, y = 10, 10
    elif position == 'top-center':
        x = (labeled.shape[1] - tw) // 2
        y = 10
    
    # 畫半透明背景
    overlay = labeled.copy()
    cv.rectangle(overlay, (x - 5, y - 5), (x + tw + 10, y + th + baseline + 10), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, labeled, 0.4, 0, labeled)
    # 畫文字
    cv.putText(labeled, text, (x, y + th + 5), font, font_scale, color, thickness, cv.LINE_AA)
    return labeled


# ========== 執行 SVD 壓縮 ==========
k_values = [544, 100, 50, 25]
results = {}

print('\n' + '='*60)
print('SVD 圖像壓縮分析結果')
print('='*60)

for k in k_values:
    compressed = svd_compression(img_org, k)
    mse = calculate_mse(img_org, compressed)
    psnr = calculate_psnr(img_org, compressed)
    orig_size, comp_size, ratio = calculate_compression_ratio(height, width, channels, k)
    
    results[k] = {
        'image': compressed,
        'mse': mse,
        'psnr': psnr,
        'orig_size': orig_size,
        'comp_size': comp_size,
        'ratio': ratio
    }
    
    print(f'\n--- k = {k} ---')
    print(f'  MSE  (均方誤差):      {mse:.4f}')
    print(f'  PSNR (峰值信噪比):    {psnr:.2f} dB')
    print(f'  原始資料量:           {orig_size:,} 個數值')
    print(f'  壓縮後資料量:         {comp_size:,} 個數值')
    print(f'  壓縮比:               {ratio:.2f}x')
    print(f'  壓縮率:               {(1 - 1/ratio)*100:.1f}%')


# ========== 產生基礎 2x2 拼接圖（含標籤）==========
labeled_images = []
for k in k_values:
    img_k = results[k]['image']
    label = f'k={k} | PSNR={results[k]["psnr"]:.1f}dB | Ratio={results[k]["ratio"]:.1f}x'
    labeled = add_label(img_k, label, position='top-left', font_scale=0.7)
    labeled_images.append(labeled)

row1 = np.hstack((labeled_images[0], labeled_images[1]))
row2 = np.hstack((labeled_images[2], labeled_images[3]))
combined = np.vstack((row1, row2))

# 儲存結果
cv.imwrite('result_combined.png', combined)
print('\n\n已儲存 2x2 拼接結果圖: result_combined.png')

# 儲存各個壓縮結果
for k in k_values:
    filename = f'result_k{k}.png'
    cv.imwrite(filename, results[k]['image'])
    print(f'已儲存 k={k} 壓縮結果: {filename}')

# 儲存原圖的 PNG 版本
cv.imwrite('original.png', img_org)
print('已儲存原圖: original.png')


# ========== 產生奇異值分佈圖（用文字表格呈現）==========
print('\n\n' + '='*60)
print('各 k 值壓縮比較表')
print('='*60)
print(f'{"k 值":<8} {"MSE":<15} {"PSNR (dB)":<15} {"壓縮比":<10} {"壓縮率":<10}')
print('-'*60)
for k in k_values:
    r = results[k]
    print(f'{k:<8} {r["mse"]:<15.4f} {r["psnr"]:<15.2f} {r["ratio"]:<10.2f} {(1-1/r["ratio"])*100:<10.1f}%')


# ========== 顯示結果 ==========
cv.imshow('SVD Compression Results', combined)
cv.waitKey(0)
cv.destroyAllWindows()

print('\n程式執行完畢')
