# ☕ The Great American Coffee Taste Test 2024 - Data Exploration & ML Prediction

## 📌 專案簡介 (Project Overview)
本專案基於 Kaggle 上的「The Great American Coffee Taste Test 2024」資料集，對 4,042 位美國咖啡飲用者的問卷數據進行探索性資料分析 (EDA) 與機器學習預測。
除了將龐大的表格數據轉化為直觀的儀表板 (Dashboard) 外，更導入了機器學習模型，不僅預測消費者的咖啡喜好，還透過特徵重要性 (Feature Importance) 剖析了影響味覺偏好的關鍵潛在因素。

## ▶ 核心技術與資料處理 (Core Technologies)
- **資料前處理與視覺化 (EDA)**: 
  - 使用 `pandas` 與 `seaborn` 清理字串標籤並建立 2x2 的動態子圖表，解決字元重疊與排版問題。
- **機器學習與可解釋性 AI (Machine Learning & XAI)**: 
  - 使用 `scikit-learn` 建立 **Random Forest Classifier (隨機森林分類器)**。
  - 將類別變數進行 One-Hot Encoding，訓練 AI 預測消費者的盲測喜好。
  - 提取並視覺化特徵重要性 (Feature Importance)，解釋 AI 的決策邏輯。

## 📊 數據洞察與分析結果 (Data Insights & Results)
透過視覺化圖表與 AI 模型分析，我們得出以下關鍵結論：

1. **基本消費輪廓**:
   - 受訪者以 25-34 歲為主，絕大多數人每天飲用 2 杯咖啡。
   - 盲測結果顯示 **Coffee D** 獲得壓倒性的最高票數。
   - 無論是居家辦公 (WFH) 還是實體進公司 (In Person)，大多數人都習慣「在家裡」喝咖啡，顯示家用市場的強大潛力。
2. **影響喜好的關鍵因素 (來自 AI 特徵重要性分析)**:
   - AI 在預測咖啡喜好時，最重要的判斷依據為 **「是否每天喝 2 杯 (cups_2)」**。
   - 令人意外的是，**「工作型態 (WFH vs In Person)」** 對於味覺偏好的影響力，甚至大於「35-44歲」等特定年齡層次。這為未來的精準行銷提供了數據支撐。

## 🛠️ 環境配置與執行 (How to Run)
1. 安裝核心套件：
   ```bash
   python -m pip install pandas matplotlib seaborn scikit-learn

2. 確保 24_Coffee_Taste_Test.csv 與程式碼位於同一目錄下。

3. 執行視覺化儀表板程式碼：
    ```bash
    coffee_1.py

4. 執行機器學習與特徵重要性分析程式碼：
    ```bash
    coffee_2.py

5. 執行 coffee_1.py 後的視覺化儀表板結果，可參考專案中的圖片檔案：
    ```bash
    Result1_The Great American Coffee Taste Test 2024 - Data Exploration.png

6. 執行 coffee_2.py 後的機器學習分析結果，可參考專案中的圖片檔案：
    ```bash
    Result2_Top 10 Most Important Factors in Choosing a Coffee.png