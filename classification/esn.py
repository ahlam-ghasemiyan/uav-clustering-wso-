# classification/esn.py
import numpy as np
import os
import joblib
import sys
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# افزودن مسیر جاری
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from mpa import MarinePredatorsAlgorithm
except ImportError:
    print("❌ Error: فایل 'mpa.py' پیدا نشد.")
    exit(1)

# ==========================================
# 1. کلاس شبکه ESN (اصلاح شده)
# ==========================================
class ESN:
    def __init__(self, input_dim, reservoir_size=1000, spectral_radius=0.9, sparsity=0.1, leak_rate=0.5, random_state=42):
        self.input_dim = input_dim
        self.reservoir_size = int(reservoir_size)
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.rng = np.random.RandomState(random_state)
        
        self._initialize_weights()

    def _initialize_weights(self):
        # وزن ورودی (Win)
        self.Win = self.rng.uniform(-1, 1, size=(self.reservoir_size, self.input_dim))
        
        # وزن مخزن (W)
        W = self.rng.uniform(-1, 1, size=(self.reservoir_size, self.reservoir_size))
        mask = self.rng.rand(*W.shape) > self.sparsity
        W[mask] = 0
        
        # تنظیم شعاع طیفی
        try:
            eigenvalues = np.linalg.eigvals(W)
            max_eig = np.max(np.abs(eigenvalues))
            if max_eig > 0:
                W *= (self.spectral_radius / max_eig)
        except:
            pass 
        self.W = W

    def transform(self, X):
        """
        تبدیل ویژگی‌ها به فضای حالت.
        نکته مهم: برای جلوگیری از سرریز حافظه در دیتاست‌های بزرگ، محاسبات ماتریسی بهینه شده‌اند.
        """
        # فرمول: State = (1-a)*State + a*Tanh(Win*Input + W*State)
        # برای تصاویر (غیر سری زمانی)، State اولیه صفر است، پس W*State حذف می‌شود.
        # فرمول ساده شده: State = a * Tanh(Win * Input)
        
        input_projection = np.dot(X, self.Win.T)
        states = self.leak_rate * np.tanh(input_projection)
        return states

    def fit(self, X_train, y_train, alpha=1.0):
        train_states = self.transform(X_train)
        
        # استفاده از Ridge Classifier
        self.readout = RidgeClassifier(alpha=alpha)
        self.readout.fit(train_states, y_train)
        return self

    def predict(self, X_test):
        test_states = self.transform(X_test)
        return self.readout.predict(test_states)

# ==========================================
# 2. بدنه اصلی اجرا
# ==========================================
def main():
    FEATS_DIR = "out_capsnet"
    
    print(f"📂 Loading features from '{FEATS_DIR}'...")
    try:
        X_train = np.load(os.path.join(FEATS_DIR, 'X_feats_train.npy'))
        y_train = np.load(os.path.join(FEATS_DIR, 'y_train.npy'))
        X_val = np.load(os.path.join(FEATS_DIR, 'X_feats_val.npy'))
        y_val = np.load(os.path.join(FEATS_DIR, 'y_val.npy'))
    except FileNotFoundError:
        print("❌ Error: فایل‌ها پیدا نشدند. ابتدا train_capsnet.py را اجرا کنید.")
        return

    # اصلاح ابعاد y اگر به صورت (N, 1) باشد
    if y_train.ndim > 1:
        if y_train.shape[1] > 1: # One-hot
            y_train = np.argmax(y_train, axis=1)
            y_val = np.argmax(y_val, axis=1)
        else: # (N, 1) -> (N,)
            y_train = y_train.ravel()
            y_val = y_val.ravel()

    # 🔥 گام حیاتی: نرمال‌سازی داده‌ها (Fixing the 6% Accuracy)
    print("⚖️  Normalizing features (MinMax Scaling)...")
    scaler = MinMaxScaler(feature_range=(-1, 1)) # ESN دوست دارد ورودی بین -1 و 1 باشد
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    print(f"✅ Data Ready: Train={X_train.shape}, Val={X_val.shape}")

    # تابع هدف MPA
    def objective_function(params):
        # Decode parameters
        res_size = int(params[0])
        spec_rad = params[1]
        spars = params[2]
        leak = params[3]
        reg_alpha = params[4] # اضافه کردن پارامتر آلفا به بهینه‌سازی
        
        try:
            model = ESN(input_dim=X_train.shape[1],
                        reservoir_size=res_size,
                        spectral_radius=spec_rad,
                        sparsity=spars,
                        leak_rate=leak)
            
            model.fit(X_train, y_train, alpha=reg_alpha)
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            return acc
        except Exception as e:
            return 0.0

    print("\n🦈 Running MPA Optimization...")
    
    # محدودیت‌ها (اضافه شدن alpha بین 0.01 تا 10.0)
    bounds = [
        (500, 3000),   # Reservoir Size
        (0.1, 1.5),    # Spectral Radius
        (0.01, 0.5),   # Sparsity
        (0.1, 0.99),   # Leak Rate
        (0.01, 10.0)   # Ridge Alpha
    ]
    
    optimizer = MarinePredatorsAlgorithm(
        objective_func=objective_function,
        bounds=bounds,
        pop_size=10,    # جمعیت کمتر برای سرعت بیشتر
        max_iter=15,    # تعداد دور
        maximize=True
    )
    
    best_params, best_acc = optimizer.optimize()
    
    print("\n" + "="*40)
    print(f"🏆 Best Accuracy Found: {best_acc*100:.2f}%")
    print(f"⚙️  Best Parameters: {best_params}")
    print("="*40)

    # آموزش مدل نهایی
    print("\n💾 Training Final Model...")
    final_model = ESN(input_dim=X_train.shape[1],
                      reservoir_size=int(best_params[0]),
                      spectral_radius=best_params[1],
                      sparsity=best_params[2],
                      leak_rate=best_params[3])
    
    final_model.fit(X_train, y_train, alpha=best_params[4])
    
    final_preds = final_model.predict(X_val)
    print("\n📄 Classification Report:")
    print(classification_report(y_val, final_preds))
    
    # ذخیره مدل و اسکالر
    joblib.dump({'model': final_model, 'scaler': scaler}, os.path.join(FEATS_DIR, "final_esn_model.pkl"))
    print("✅ Model saved successfully.")

if __name__ == "__main__":
    main()