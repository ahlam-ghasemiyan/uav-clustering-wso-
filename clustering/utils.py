# clustering/utils.py
import numpy as np

class UAV:
    """
    کلاس نماینده هر پهپاد (Node) در شبکه.
    این کلاس دسترسی به ویژگی‌ها (x, y, energy) را برای فایل fitness.py آسان می‌کند.
    """
    def __init__(self, uid, x, y, energy):
        self.id = uid
        self.x = x
        self.y = y
        self.energy = energy
        self.is_ch = False 
        self.cluster_id = -1 # به کدام کلاستر تعلق دارد

    def __repr__(self):
        return f"UAV(id={self.id}, pos=({self.x:.1f}, {self.y:.1f}), E={self.energy})"

def euclidean(pos1, pos2):
    """
    محاسبه فاصله اقلیدسی بین دو نقطه.
    ورودی‌ها می‌توانند تاپل یا آرایه نامپای باشند.
    """
    p1 = np.array(pos1)
    p2 = np.array(pos2)
    return np.linalg.norm(p1 - p2)

def generate_uavs(n=50, area_size=1000, seed=42):
    """
    تولید پهپادها با موقعیت تصادفی و سطح انرژی مشخص شده در مقاله.
    
    Paper Ref: "e(n) = 0.5J or 1.25J or 1.75J"
    """
    rng = np.random.RandomState(seed)
    
    # تولید مختصات تصادفی در محیط (مثلاً 1000x1000 متر)
    xs = rng.uniform(0, area_size, n)
    ys = rng.uniform(0, area_size, n)
    
    # انتخاب سطوح انرژی طبق مقاله
    energy_levels = [0.5, 1.25, 1.75]
    energies = rng.choice(energy_levels, size=n)
    
    uav_list = []
    for i in range(n):
        # ساخت آبجکت UAV
        uav = UAV(uid=i, x=xs[i], y=ys[i], energy=energies[i])
        uav_list.append(uav)
        
    return uav_list