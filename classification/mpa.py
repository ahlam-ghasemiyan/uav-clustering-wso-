# classification/mpa.py
import numpy as np
from scipy.special import gamma

def levy_flight(n, dim, beta=1.5):
    """
    تولید گام‌های پروازی لوی (Lévy Flight) برای جستجوی اکتشافی.
    """
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * np.random.randn(n, dim) * sigma
    v = np.random.randn(n, dim)
    step = u / np.abs(v) ** (1 / beta)
    return step

class MarinePredatorsAlgorithm:
    def __init__(self, objective_func, bounds, pop_size=20, max_iter=50, maximize=True):
        """
        objective_func: تابعی که پارامترها را می‌گیرد و دقت (Accuracy) را برمی‌گرداند.
        bounds: لیست تاپل‌ها [(min, max), ...]
        maximize: اگر True باشد، به دنبال بیشترین مقدار (دقت) می‌گردد.
        """
        self.func = objective_func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.maximize = maximize
        
        # مقداردهی اولیه جمعیت (Prey)
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.prey = np.random.uniform(self.lb, self.ub, (pop_size, self.dim))
        
        # حافظه بهترین‌ها
        self.fitness = np.full(pop_size, -np.inf if maximize else np.inf)
        self.elite = np.zeros(self.dim)
        self.elite_score = -np.inf if maximize else np.inf

    def optimize(self):
        print(f"Starting MPA Optimization (Pop={self.pop_size}, Iter={self.max_iter})...")
        
        # ارزیابی اولیه
        self._evaluate_population()
        
        stepsize = np.zeros((self.pop_size, self.dim))
        
        # حلقه اصلی تکرار
        for t in range(self.max_iter):
            # محاسبه ضریب تطبیقی CF طبق فرمول مقاله
            CF = (1 - (t / self.max_iter)) ** (2 * (t / self.max_iter))
            
            # بردار گام تصادفی براونی و لوی
            RB = np.random.randn(self.pop_size, self.dim) # Brownian
            RL = levy_flight(self.pop_size, self.dim)     # Lévy
            
            # کپی جمعیت قبلی برای مقایسه
            new_prey = self.prey.copy()
            
            for i in range(self.pop_size):
                # -------------------------------------------------
                # Phase 1: High Velocity (Exploration) - 1/3 اول
                # -------------------------------------------------
                if t < self.max_iter / 3:
                    stepsize[i, :] = RB[i, :] * (self.elite - RB[i, :] * self.prey[i, :])
                    new_prey[i, :] = self.prey[i, :] + 0.5 * stepsize[i, :]

                # -------------------------------------------------
                # Phase 2: Unit Velocity (Transition) - 1/3 دوم
                # -------------------------------------------------
                elif t < 2 * self.max_iter / 3:
                    if i < self.pop_size / 2:
                        # نیمه اول جمعیت: Exploration (Lévy)
                        stepsize[i, :] = RL[i, :] * (self.elite - RL[i, :] * self.prey[i, :])
                        new_prey[i, :] = self.prey[i, :] + 0.5 * stepsize[i, :]
                    else:
                        # نیمه دوم جمعیت: Exploitation (Brownian)
                        stepsize[i, :] = RB[i, :] * (RB[i, :] * self.elite - self.prey[i, :])
                        new_prey[i, :] = self.elite + 0.5 * CF * stepsize[i, :]

                # -------------------------------------------------
                # Phase 3: Low Velocity (Exploitation) - 1/3 آخر
                # -------------------------------------------------
                else:
                    stepsize[i, :] = RL[i, :] * (RL[i, :] * self.elite - self.prey[i, :])
                    new_prey[i, :] = self.elite + 0.5 * CF * stepsize[i, :]

            # -------------------------------------------------
            # اعمال FADs (Fish Aggregating Devices)
            # پرش ناگهانی برای فرار از لوکال مینیمم
            # -------------------------------------------------
            FADs = 0.2
            r = np.random.rand()
            if r <= FADs:
                for i in range(self.pop_size):
                    U = np.where(np.random.rand(self.dim) < FADs, 1, 0)
                    random_prey = np.random.uniform(self.lb, self.ub, self.dim)
                    new_prey[i, :] = new_prey[i, :] + CF * (self.lb + np.random.rand(self.dim) * (self.ub - self.lb)) * U
            else:
                # اصلاح جزئی بر اساس همسایگان
                r1, r2 = np.random.rand(), np.random.rand()
                for i in range(self.pop_size):
                    idx1 = np.random.randint(0, self.pop_size)
                    idx2 = np.random.randint(0, self.pop_size)
                    new_prey[i, :] = new_prey[i, :] + (FADs * (1 - r1) + r1) * (new_prey[idx1, :] - new_prey[idx2, :])

            # بررسی مرزها
            new_prey = np.clip(new_prey, self.lb, self.ub)
            
            # ارزیابی و آپدیت (فقط اگر نتیجه بهتر شد آپدیت کن - Memory Saving)
            self._update_population_greedy(new_prey)
            
            print(f"MPA Iter {t+1}/{self.max_iter}: Best Accuracy = {self.elite_score:.4f}")

        return self.elite, self.elite_score

    def _evaluate_population(self):
        for i in range(self.pop_size):
            score = self.func(self.prey[i])
            self.fitness[i] = score
            
            if self.maximize:
                if score > self.elite_score:
                    self.elite_score = score
                    self.elite = self.prey[i].copy()
            else:
                if score < self.elite_score:
                    self.elite_score = score
                    self.elite = self.prey[i].copy()

    def _update_population_greedy(self, new_prey):
        for i in range(self.pop_size):
            score = self.func(new_prey[i])
            
            # آپدیت حریصانه: فقط اگر جواب بهتر بود جایگزین کن
            improve = False
            if self.maximize:
                if score > self.fitness[i]:
                    improve = True
            else:
                if score < self.fitness[i]:
                    improve = True
            
            if improve:
                self.fitness[i] = score
                self.prey[i] = new_prey[i].copy()
                
                # آپدیت Elite
                if self.maximize:
                    if score > self.elite_score:
                        self.elite_score = score
                        self.elite = new_prey[i].copy()
                else:
                    if score < self.elite_score:
                        self.elite_score = score
                        self.elite = new_prey[i].copy()