
# DSAI HW1 - Electricity Forecasting

## Overview

* In this HW, we will implement an algorithm to predict the operating reserve (備轉容量) of electrical power. Given a time series electricity data to predict the value of the operating reserve value of each day during 2021/03/23 ~ 2021/03/29.
## 虛擬環境
### 使用 Anoconda 進行套件管理
* Create virtual environment
```
$ conda create --name DSAI-HW1
```
* Activate virtual environment
```
$ conda activate DSAI-HW1
```
## 套件版本及安裝資訊
### python 版本
* python version : 3.8.8
### 套件版本
* numpy==1.19.2
* pandas==1.2.3
* scikit-learn==0.24.1
* matplotlib==3.3.4
* tensorflow==2.4.1
* keras==2.4.3
### 安裝方法
```bash 
$ conda install --yes --file requirements.txt
```

## 實做細節
### app.py 參數
```python=
parser.add_argument('--training',
                   default='training_data.csv',
                   help='input training data file name(.csv)')

parser.add_argument('--testing',
                   default=False,
                   help='input trained model file name(.h5)')

parser.add_argument('--output',
                    default='submission.csv',
                    help='output file name(.csv)')
```
- 預設不使用參數，會進行 training，以 training_data.csv 來當作訓練資料、並使用此次 train 好的 model 預測
- 新增 testing 參數，預設為 False、若被指派為 True，則直接 load 表現較好的 model ，並預測未來備轉容量
    - 用法

```bash 
$ python app.py --testing True
```
- 測試
```bash 
$ python app.py --training training_data.csv --output submission.csv
```
### Training Data
- 主要來自助教提供的 [台灣電力公司_過去電力供需資訊](https://data.gov.tw/dataset/19995)、及 [台灣電力公司_本年度每日尖峰備轉容量率](https://data.gov.tw/dataset/25850)，主要包含從 2019/1/1 - 2021/3/20 的資料，欄位只保留以下：
    - 日期
    - 備轉容量
    - 備轉容量率
- 此外，我利用 http://www.tianqihoubao.com/lishi/taibei.html 提供的台北市氣溫歷史資料和備轉容量進行作圖，三者資料曲線呈高度正相關，因此也把台北市氣溫資訊加入其中
![](https://i.imgur.com/XU5wxiD.png)
- 為何選擇台北市來代表台灣氣溫資料？

-> 考量資料取得難度及人口密度，台北市的資料相對容易取得，且台北市、新北市、桃園市等人口數遠突破 200 萬的直轄接在鄰近區域（氣溫差異較小），且北台灣區域(臺北市、新北市、基隆市、宜蘭縣、桃園市、新竹縣、新竹市)人口總數也突破千萬（[資料來源](https://zh.wikipedia.org/wiki/%E5%8C%97%E8%87%BA%E7%81%A3)），，佔台灣總人口近半，因此具有相當的代表性。

### Preprocessing
- 以下備轉容量曲線可看出，資料雖然有劇烈起伏，但長期來看約以一年為一個波動週期，因此日期及月份佔有一定的重要性

![](https://i.imgur.com/UElKkwq.png)
- 加上上述所提及的資料，csv 欄位變成以下所示
    - 欄位 : 
        - 日
        - 月
        - 周內的天數（禮拜一 = 0, 禮拜天 = 6）
        - 備轉容量(MW)
        - 備轉容量率(%)
        - 當日台北市高溫(°C)
        - 當日台北市低溫(°C)
- 一部分資料截圖

![](https://i.imgur.com/CudvN8m.png)

- 正規化資料至 0-1 區間
```python
data_norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
```
    
### Model
- 使用 keras.layers 中的 LSTM 來當作基本的模型框架，主要使用動機是因為 LSTM 適合處理時序性的資料預測，常用做股價預測、油價預測或天氣預測等等
- Input : 為 60 天的歷史資料
- Output : input 60 天資料後 7 天的預測資料
- Model structure
    - 使用兩層 LSTM 來訓練
```python=
model = Sequential()
model.add(LSTM(64, input_shape=(shape[1], shape[2]), return_sequences = True))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(TimeDistributed(Dense(units = 1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
# output 7-days prediction of 備轉容量(MW)
model.add(Dense(7))
model.compile(loss="mse", optimizer="adam", metrics=['mse'])
model.summary()
```
- Summary
```bash 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 60, 64)            18432     
_________________________________________________________________
dropout (Dropout)            (None, 60, 64)            0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 60, 64)            33024     
_________________________________________________________________
dropout_1 (Dropout)          (None, 60, 64)            0         
_________________________________________________________________
time_distributed (TimeDistri (None, 60, 1)             65        
_________________________________________________________________
flatten (Flatten)            (None, 60)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                3904      
_________________________________________________________________
dense_2 (Dense)              (None, 7)                 455       
=================================================================
```
### 成果
- loss/MSE 曲線
    - val_mse 降至 0.001 左右
![](https://i.imgur.com/g9sJiJL.png)

- 預測最近 10 筆資料結果
    - 黃色線為實際值
    - 藍色線為預測值
    - 綠色線為絕對值（誤差）
![](https://i.imgur.com/FWpUP0Q.png)
- submission.csv
```csv 
date,operating_reserve(MW)
20210323,3070.62
20210324,3158.32
20210325,3068.95
20210326,3094.21
20210327,3077.08
20210328,3158.96
20210329,3013.34
```
