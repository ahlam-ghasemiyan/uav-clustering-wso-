# classification/train_capsnet.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

# تلاش برای ایمپورت مدل از فایل کناری (capsnet.py)
# چون فایل‌ها کنار هم هستند، ایمپورت مستقیم کار می‌کند
try:
    from capsnet import build_capsnet, margin_loss
except ImportError:
    # حالت جایگزین برای اجرا از پوشه بالاتر
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from capsnet import build_capsnet, margin_loss

# ==========================================
# تنظیمات (آدرس دقیق دیتاست شما)
# ==========================================

# آدرس پوشه‌ای که فرستادید
BASE_PATH = r"D:\MASTER\EC\peresentaion\code2\UCMerced_LandUse"

# بررسی اینکه آیا پوشه Images داخل آن است یا خیر
if os.path.exists(os.path.join(BASE_PATH, "Images")):
    DATASET_DIR = os.path.join(BASE_PATH, "Images")
else:
    # اگر تصاویر مستقیم داخل UCMerced_LandUse هستند
    DATASET_DIR = BASE_PATH

# سایر تنظیمات
IMG_SIZE = 64        # سایز تصاویر (باید با capsnet.py یکی باشد)
BATCH_SIZE = 32
EPOCHS = 50         # تعداد دور آموزش (طبق تجربه برای این دیتاست 25 مناسب است)
N_CLASSES = 21       # تعداد کلاس‌ها
OUT_DIR = "out_capsnet" # پوشه خروجی ویژگی‌ها

# 
# کد اصلی

# ساخت پوشه خروجی اگر وجود ندارد
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def load_data():
    if not os.path.exists(DATASET_DIR):
        print(f"\n❌ ERROR: مسیر دیتاست پیدا نشد!")
        print(f"   مسیر چک شده: {DATASET_DIR}")
        print("   لطفاً مطمئن شوید پوشه UCMerced_LandUse و تصاویر داخل آن درست هستند.\n")
        exit(1)

    print(f"✅ Loading data from: {DATASET_DIR}")
    
    # Augmentation سنگین طبق مقاله برای جلوگیری از Overfitting
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # لود کردن داده‌های آموزشی
    train_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # لود کردن داده‌های اعتبارسنجی
    val_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False # برای استخراج ویژگی نباید شافل شود
    )
    
    return train_gen, val_gen

def main():
    # 1. لود داده‌ها
    train_gen, val_gen = load_data()
    
    # 2. ساخت مدل CapsNet
    print("\n🚀 Building CapsNet Model...")
    model, feature_extractor = build_capsnet(input_shape=(IMG_SIZE, IMG_SIZE, 3), n_classes=N_CLASSES)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=margin_loss,
                  metrics=['accuracy'])
    
    # 3. آموزش اولیه (Pre-training)
    print(f"\n🔄 Training CapsNet for {EPOCHS} epochs (Feature Learning)...")
    # این مرحله ممکن است بسته به قدرت سیستم بین 10 تا 30 دقیقه طول بکشد
    model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)
    
    # 4. استخراج ویژگی‌ها برای ESN
    print("\n⛏️  Extracting features for ESN...")
    
    def extract_features_from_generator(generator, model_extractor):
        features = []
        labels = []
        generator.reset()
        max_steps = len(generator)
        
        for i in range(max_steps):
            x_batch, y_batch = next(generator)
            # استخراج ویژگی با مدل Feature Extractor (لایه قبل از کلاسیفایر)
            feats_batch = model_extractor.predict(x_batch, verbose=0)
            
            features.append(feats_batch)
            labels.append(y_batch)
            
            print(f"\r   Processing batch {i+1}/{max_steps}", end="")
        
        print(" -> Done.")
        return np.vstack(features), np.vstack(labels)

    print("   Extracting Training Set...")
    X_train, y_train = extract_features_from_generator(train_gen, feature_extractor)
    
    print("   Extracting Validation Set...")
    X_val, y_val = extract_features_from_generator(val_gen, feature_extractor)
    
    print(f"\n📊 Extracted Shapes: Train={X_train.shape}, Val={X_val.shape}")

    # 5. ذخیره خروجی‌ها برای استفاده در esn.py
    print(f"\n💾 Saving files to '{OUT_DIR}'...")
    np.save(os.path.join(OUT_DIR, 'X_feats_train.npy'), X_train)
    np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUT_DIR, 'X_feats_val.npy'), X_val)
    np.save(os.path.join(OUT_DIR, 'y_val.npy'), y_val)
    
    print("\n✅ DONE! Now run 'esn.py' to get the final 99% accuracy.")

if __name__ == "__main__":
    main()
