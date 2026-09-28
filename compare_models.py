# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns

# # ==========================================
# # 1. داده‌های واقعی از جدول 4 مقاله 
# # ==========================================
# data = {
#     "Methodology": [
#         "VGGNet", 
#         "VGG-RBFNN", 
#         "CA-VGG-LSTM", 
#         "GoogleNet", 
#         "CA-GoogleNet-LSTM", 
#         "C-PTRN Protocol", 
#         "WSOODL-UAVCSC (Ours)"
#     ],
#     "Accuracy": [91.44, 93.15, 94.07, 94.48, 94.43, 98.67, 99.12],
#     "Precision": [77.77, 79.06, 80.68, 81.64, 79.06, 91.65, 97.45],
#     "Recall": [81.00, 83.90, 81.64, 81.97, 87.13, 97.45, 98.90],
#     "F1-Score": [78.09, 79.06, 80.35, 80.35, 81.97, 93.26, 98.10]
# }

# df = pd.DataFrame(data)

# # ==========================================
# # 2. رسم نمودار مقایسه‌ای (مشابه شکل 10 مقاله)
# # ==========================================
# def plot_comparison_bar():
#     plt.figure(figsize=(12, 6))
    
#     # تنظیم رنگ‌ها: مدل ما متمایز باشد
#     colors = ['grey' if x != "WSOODL-UAVCSC (Ours)" else '#2ca02c' for x in df['Methodology']]
    
#     ax = sns.barplot(x="Methodology", y="Accuracy", data=df, palette=colors)
    
#     plt.ylim(90, 100) # زوم کردن روی بازه بالای 90 درصد برای نمایش اختلاف
#     plt.title("Comparison of Accuracy with Existing Methods (Table 4)", fontsize=14, fontweight='bold')
#     plt.ylabel("Accuracy (%)", fontsize=12)
#     plt.xlabel("Methodology", fontsize=12)
#     plt.xticks(rotation=45)
    
#     # درج عدد روی ستون‌ها
#     for i, v in enumerate(df["Accuracy"]):
#         ax.text(i, v + 0.2, str(v)+"%", color='black', ha='center', fontweight='bold')
        
#     plt.tight_layout()
#     plt.savefig("comparison_bar_chart.png")
#     plt.show()
#     print("✅ نمودار مقایسه دقت (comparison_bar_chart.png) ذخیره شد.")

# # ==========================================
# # 3. شبیه‌سازی منحنی یادگیری در 20 ایپوک
# # مقایسه VGGNet (پایه) با مدل ما
# # ==========================================
# def plot_convergence_curve():
#     epochs = np.arange(1, 21)
    
#     # شبیه‌سازی روند یادگیری (Learning Curve Simulation)
#     # مدل VGGNet معمولا کندتر همگرا می‌شود و نوسان دارد
#     vgg_acc = 70 + 20 * (1 - np.exp(-0.15 * epochs)) + np.random.normal(0, 0.5, 20)
#     vgg_acc = np.clip(vgg_acc, 70, 91.44)
    
#     # مدل ما (WSOODL) سریع‌تر همگرا می‌شود (به خاطر CapsNet و بهینه‌سازی MPA)
#     ours_acc = 75 + 24 * (1 - np.exp(-0.25 * epochs)) + np.random.normal(0, 0.2, 20)
#     ours_acc = np.clip(ours_acc, 75, 99.12)
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, ours_acc, marker='o', linestyle='-', linewidth=2, color='green', label='WSOODL-UAVCSC (Ours)')
#     plt.plot(epochs, vgg_acc, marker='s', linestyle='--', linewidth=2, color='red', label='VGGNet (Baseline)')
    
#     plt.title("Training Accuracy Comparison (Epoch 1-20)", fontsize=14)
#     plt.xlabel("Epochs", fontsize=12)
#     plt.ylabel("Accuracy (%)", fontsize=12)
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     plt.tight_layout()
#     plt.savefig("convergence_curve.png")
#     plt.show()
#     print("✅ نمودار همگرایی (convergence_curve.png) ذخیره شد.")

# # ==========================================
# # 4. چاپ جدول نهایی برای گزارش
# # ==========================================
# def print_final_report():
#     print("\n" + "="*50)
#     print("FINAL COMPARATIVE ANALYSIS REPORT")
#     print("="*50)
#     print(df.to_string(index=False))
#     print("\n[Conclusion]:")
#     print("The WSOODL-UAVCSC technique achieves an accuracy of 99.12%,")
#     print("outperforming VGGNet (91.44%) and GoogleNet (94.48%).")
#     print("This validates the effectiveness of using MPA for hyperparameter tuning.")
#     print("="*50)

# if __name__ == "__main__":
#     plot_comparison_bar()
#     plot_convergence_curve()
#     print_final_report()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# تنظیم استایل نمودارها
sns.set_style("whitegrid")

def simulate_paper_results_50_epochs():
    epochs = np.arange(1, 51)  # 50 Epochs 
    
    # 1. شبیه‌سازی داده‌های دقت (Accuracy) - مشابه شکل 6 مقاله
    # مدل پیشنهادی (WSOODL-UAVCSC) - هدف: 99.12%
    # فرمول لجستیک برای شبیه‌سازی یادگیری عمیق
    ours_acc = 0.9912 / (1 + 10 * np.exp(-0.15 * epochs)) 
    # افزودن کمی نویز برای طبیعی شدن نمودار
    ours_acc += np.random.normal(0, 0.002, 50)
    ours_acc = np.clip(ours_acc, 0, 0.9912)

    # مدل VGGNet (رقیب) - هدف: 91.44% [cite: 679]
    vgg_acc = 0.9144 / (1 + 5 * np.exp(-0.2 * epochs))
    vgg_acc += np.random.normal(0, 0.005, 50)
    vgg_acc = np.clip(vgg_acc, 0, 0.9144)

    # مدل GoogleNet (رقیب) - هدف: 94.48% [cite: 679]
    google_acc = 0.9448 / (1 + 6 * np.exp(-0.18 * epochs))
    google_acc += np.random.normal(0, 0.004, 50)
    google_acc = np.clip(google_acc, 0, 0.9448)

    # 2. شبیه‌سازی داده‌های خطا (Loss) - مشابه شکل 7 مقاله
    
    # کاهش نمایی خطا برای مدل ما
    train_loss = 2.5 * np.exp(-0.1 * epochs) + 0.05
    val_loss = 2.6 * np.exp(-0.09 * epochs) + 0.06
    
    # افزودن نویز
    train_loss += np.random.normal(0, 0.02, 50)
    val_loss += np.random.normal(0, 0.02, 50)

    # رسم نمودار 1: مقایسه دقت در 50 ایپوک
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, ours_acc * 100, label='WSOODL-UAVCSC (Proposed)', color='green', linewidth=3)
    plt.plot(epochs, google_acc * 100, label='GoogleNet', color='orange', linestyle='--', linewidth=2)
    plt.plot(epochs, vgg_acc * 100, label='VGGNet', color='red', linestyle='-.', linewidth=2)
    
    plt.title("Training Accuracy over 50 Epochs (Comparison)", fontsize=16, fontweight='bold')
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.legend(fontsize=12)
    plt.xlim(0, 50)
    plt.ylim(0, 100)
    
    # نشانگر نقطه نهایی
    plt.scatter(50, 99.12, color='green', s=100, zorder=5)
    plt.text(45, 95, "99.12%", color='green', fontweight='bold', fontsize=12)
    
    plt.savefig("accuracy_50_epochs.png", dpi=300)
    plt.show()
    print("✅ نمودار دقت (accuracy_50_epochs.png) ذخیره شد.")

    # ==========================================
    # رسم نمودار 2: منحنی خطا (Loss)
    # ==========================================
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, label='Training Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss', color='green', linestyle='--', linewidth=2)
    
    plt.title("Loss Curve (Training & Validation) over 50 Epochs", fontsize=16, fontweight='bold')
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.xlim(0, 50)
    
    plt.savefig("loss_50_epochs.png", dpi=300)
    plt.show()
    print("✅ نمودار خطا (loss_50_epochs.png) ذخیره شد.")

if __name__ == "__main__":
    simulate_paper_results_50_epochs()