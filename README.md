# Landmark_and_Intensity-Based_Images_Elastic_Registration_Using_Hierarchical_B-Splines

[https://rire.insight-journal.org/download_data](https://rire.insight-journal.org/download_data)

`
python main.py --mri "./mr_T1/patient_109_mr_T1.mhd" --ct "./ct/patient_109_ct.mhd"
`

## 與論文描述不同的地方：

### 優化器：Powell vs. Marquardt-Levenberg (ML)

論文：使用 Marquardt-Levenberg 優化器。ML 需要計算 Cost Function 的梯度 (Gradient) 與 Hessian 矩陣 。
程式碼：使用 Powell 方法。

原因：論文中提到的梯度計算需要推導 B-Spline 與 Parzen Window 的解析解導數 ，這在純 Python (scipy) 中極難實作且運算緩慢。Powell 是一種無梯度 (Gradient-free) 方法，非常適合處理 Histogram-based MI 這種不連續或難微分的函數。

影響：Powell 收斂速度可能比 ML 慢，但在這個規模的應用中，結果是等效的，且實作更穩健。

### B-Spline 數學模型：Bicubic Interpolation vs. Tensor Product Sum

論文：使用嚴格的 B-Spline 基底函數 (Tensor Product) 求和公式。
程式碼：使用 scipy.ndimage.zoom (order=3)。

原因：zoom 的 order=3 是 雙三次插值 (Bicubic Interpolation)。在數學上，Cubic B-Spline 與 Bicubic Interpolation 非常相似（都是三次多項式），且 zoom 是經過高度優化的 C 語言實作，速度遠快於在 Python 中手寫 B-Spline 基底函數求和。

影響：對最終變形場的平滑度與拓撲性質影響微乎其微，是工程上極佳的近似替代。

## 額外多做的
預對齊 (Pre-alignment)

論文：論文暗示這是一個標準步驟，但未詳述具體公式 。
程式碼：明確實作了 shift = Mean(Moving) - Mean(Fixed) 的剛性平移。這能顯著提高後續 B-Spline 優化的成功率，避免落入局部極值。
