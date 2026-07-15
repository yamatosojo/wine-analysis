import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# データを読み込む
df = pd.read_csv('data/wine.csv')

# 説明変数
X = df[['alcohol', 'volatile acidity', 'sulphates']]

# 目的変数
y = df['quality']

# 学習用データとテスト用データに分割
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 線形回帰モデルを作成
model = LinearRegression()

# モデルを学習
model.fit(X_train, y_train)

# テストデータを予測
y_pred = model.predict(X_test)

# 評価結果を表示
print("平均二乗誤差:", mean_squared_error(y_test, y_pred))
print("決定係数:", r2_score(y_test, y_pred))
print("回帰係数:", model.coef_)
print("切片:", model.intercept_)

# 実際の品質と予測した品質を散布図で表示
plt.scatter(y_test, y_pred)

# 比較用の直線
plt.plot([3, 8], [3, 8])

plt.xlabel("Actual quality")
plt.ylabel("Predicted quality")
plt.title("Linear Regression: Actual vs Predicted")

plt.tight_layout()

# wine-analysis直下に画像を保存
plt.savefig('linear_regression.png')

plt.show()