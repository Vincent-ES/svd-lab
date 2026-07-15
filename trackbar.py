"""
進階加分項：利用 OpenCV Trackbar 動態調整 k 值
拖動滑桿即可即時觀察不同奇異值數量下的壓縮效果
"""
import numpy as np
import cv2 as cv

# 讀取圖片
img_org = cv.imread('Cybercab.jpg')
if img_org is None:
    print("錯誤：找不到 Cybercab.jpg")
    exit()

# 縮小圖片以加快 SVD 計算速度
scale = 0.5
img_small = cv.resize(img_org, None, fx=scale, fy=scale)
h, w, c = img_small.shape
print(f'使用縮小圖片: {w}x{h}x{c}')

# 預先計算 SVD（只需計算一次）
svd_data = []
for i in range(c):
    U, Sigma, VT = np.linalg.svd(img_small[:, :, i])
    svd_data.append((U, Sigma, VT))
    print(f'通道 {i} SVD 完成，奇異值數量: {len(Sigma)}')

max_k = min(h, w)

def reconstruct(k):
    """用前 k 個奇異值重建圖像"""
    if k < 1:
        k = 1
    res = np.zeros_like(img_small)
    for i in range(c):
        U, Sigma, VT = svd_data[i]
        res[:, :, i] = U[:, :k].dot(np.diag(Sigma[:k])).dot(VT[:k, :])
    return res

def on_trackbar(k):
    """Trackbar 回調函數"""
    if k < 1:
        k = 1
    compressed = reconstruct(k)
    
    # 計算 PSNR
    mse = np.mean((img_small.astype(np.float64) - compressed.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else float('inf')
    
    # 計算壓縮比
    orig = h * w * c
    comp = c * (h * k + k + k * w)
    ratio = orig / comp
    
    # 並排顯示原圖與壓縮圖
    display = np.hstack((img_small, compressed))
    
    # 加上資訊文字
    info = f'k={k}/{max_k}  PSNR={psnr:.1f}dB  Ratio={ratio:.2f}x'
    cv.putText(display, 'Original', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.putText(display, f'SVD k={k}', (w + 10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.putText(display, info, (10, h - 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv.imshow('SVD Trackbar', display)

# 建立視窗和 Trackbar
cv.namedWindow('SVD Trackbar', cv.WINDOW_AUTOSIZE)
cv.createTrackbar('k value', 'SVD Trackbar', 50, max_k, on_trackbar)

# 初始顯示
on_trackbar(50)

print('=================================')
print('拖動上方滑桿可以即時調整奇異值 k 值。')
print('請按鍵盤上的 [q] 鍵或 [ESC] 鍵結束程式')
print('=================================')

while True:
    key = cv.waitKey(50) & 0xFF
    if key == 27 or key == ord('q'):  # ESC 或 q 鍵
        break

cv.destroyAllWindows()
