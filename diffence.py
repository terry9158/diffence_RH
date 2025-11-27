import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from diffusers import DDPMPipeline
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from tqdm.auto import tqdm
from PIL import Image

# 設定隨機亂數種子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.manual_seed(SEED)

# @title 2. 執行 DIFFENCE 系統 (Run Implementation)

# @markdown ### 選擇資料集 (Select Dataset)
# @markdown 注意：CelebA 可能需要手動上傳資料，建議先測試 CIFAR-10。
DATASET_NAME = "CIFAR-10" # @param ["CIFAR-10", "CIFAR-100", "SVHN"]

# @markdown ### 實驗參數 (Parameters)
EVAL_SIZE = 200          # @param {type:"integer"} 最終畫圖用的樣本數 (為了速度設為 1000，可改大)
CALIBRATION_SIZE = 100    # @param {type:"integer"} 校準用的樣本數
EPOCHS = 10                # @param {type:"integer"} Target Model 訓練次數 (為了演示設為 5，論文建議更多)
N_RECONSTRUCTIONS = 200    # @param {type:"integer"} 每張圖重建次數
T_STEPS = 160             # @param {type:"integer"} 擴散步數 (T)
BATCH_SIZE = 128

# ==========================================
# 1. 資料集準備
# ==========================================
def get_dataset(dataset_name):
    print(f"正在準備資料集: {dataset_name} ...")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_root = './data'

    if dataset_name == "CIFAR-10":
        full_data = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        member_indices = list(range(25000))
        non_member_indices = list(range(25000, 50000))
        num_classes = 10
    elif dataset_name == "CIFAR-100":
        full_data = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
        member_indices = list(range(25000))
        non_member_indices = list(range(25000, 50000))
        num_classes = 100
    elif dataset_name == "SVHN":
        full_data = torchvision.datasets.SVHN(root=data_root, split='train', download=True, transform=transform)
        member_indices = list(range(5000))
        non_member_indices = list(range(5000, len(full_data)))
        num_classes = 10
    else:
        raise ValueError("不支援的資料集或需手動下載")

    member_set = torch.utils.data.Subset(full_data, member_indices)
    non_member_set = torch.utils.data.Subset(full_data, non_member_indices)
    return member_set, non_member_set, num_classes

# ==========================================
# 2. 模型定義
# ==========================================
def get_target_model(num_classes):
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

def get_diffusion_model():
    print("載入擴散模型 (google/ddpm-cifar10-32)...")
    pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    pipeline.to(DEVICE)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline

# ==========================================
# 3. 訓練 Target Model
# ==========================================
def train_target_model(model, loader, epochs):
    print(f"開始訓練 Target Model (共 {epochs} epochs)...")
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        scheduler.step()
        print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f} | Acc {100.*correct/total:.2f}%")
    return model

# ==========================================
# 4. DIFFENCE 核心邏輯
# ==========================================
def calculate_logit(model, x):
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        max_conf, _ = torch.max(probs, dim=1)     #改過的地方，嘗試不用softmax---------------------------------------------
        max_conf = torch.clamp(max_conf, 1e-6, 1 - 1e-6)
        logits = torch.log(max_conf / (1 - max_conf))
    return logits

def diffusion_reconstruct(pipeline, x, n_recons, t_steps):
    # x: [1, C, H, W] -> Output: [1, N, C, H, W]
    # 節省顯存，將輸入重複 N 次組成一個 Batch 處理
    x_repeated = x.repeat(n_recons, 1, 1, 1) # [N, C, H, W]

    # 1. 加噪 (Forward)
    noise = torch.randn_like(x_repeated)
    timesteps = torch.full((n_recons,), t_steps, device=DEVICE, dtype=torch.long)
    noisy_images = pipeline.scheduler.add_noise(x_repeated, noise, timesteps)

    # 2. 去噪 (Reverse)
    curr_img = noisy_images
    # 只需要從 t_steps 倒數回去
    relevant_timesteps = [t for t in pipeline.scheduler.timesteps if t < t_steps]

    for t in tqdm(relevant_timesteps, desc="Reconstructing", leave=False):
        t_batch = torch.full((n_recons,), t, device=DEVICE, dtype=torch.long)
        with torch.no_grad():
            model_out = pipeline.unet(curr_img, t_batch).sample
            curr_img = pipeline.scheduler.step(model_out, t, curr_img).prev_sample

    return curr_img.unsqueeze(0) # [1, N, C, H, W]

def get_candidates(target_model, original_x, reconstructions):
    target_model.eval()
    with torch.no_grad():
        orig_pred = target_model(original_x).argmax(dim=1).item()

        recons_flat = reconstructions.squeeze(0) # [N, C, H, W]
        preds = target_model(recons_flat).argmax(dim=1)

        # 篩選預測一致的
        mask = (preds == orig_pred)
        if mask.sum() == 0: return []

        valid_recons = recons_flat[mask]
        phis = calculate_logit(target_model, valid_recons)
        return phis.cpu().tolist()

def calibrate_scenario_1(target_model, pipeline, mem_loader, non_mem_loader):
    print("執行 Scenario 1 校準 (尋找最佳區間)...")

    def collect(loader, limit):
        candidates_pool = []
        count = 0
        for x, _ in tqdm(loader, desc="Collecting", total=limit):
            if count >= limit: break
            x = x.to(DEVICE)
            if x.dim() == 3: x = x.unsqueeze(0)
            recons = diffusion_reconstruct(pipeline, x, N_RECONSTRUCTIONS, T_STEPS)
            cands = get_candidates(target_model, x, recons)
            candidates_pool.append(cands)
            count += 1
        return candidates_pool

    mem_cands = collect(mem_loader, CALIBRATION_SIZE)
    non_mem_cands = collect(non_mem_loader, CALIBRATION_SIZE)

    # Grid Search
    #---------------------------------------------------------------------------
    #all_vals = [v for sub in mem_cands + non_mem_cands for v in sub]
    #if not all_vals: return (-100, 100) # Fallback

    #min_v, max_v = min(all_vals), max(all_vals)
    mem_vals = [v for sub in mem_cands for v in sub]
    non_vals = [v for sub in non_mem_cands for v in sub]    #grid search嘗試
    
    min_v = min(mem_vals)
    max_v = max(non_vals)
    #----------------------------------------------------------------------------
    steps = np.linspace(min_v, max_v, 20)
    best_js = float('inf')
    best_interval = (min_v, max_v)

    print(f"Logit 範圍: [{min_v:.2f}, {max_v:.2f}]，開始 Grid Search...")

    for i in range(len(steps)):
        for j in range(i+1, len(steps)):
            low, high = steps[i], steps[j]

            # 模擬選擇過程
            def select(pool):
                res = []
                for cands in pool:
                    if not cands: continue
                    in_range = [c for c in cands if low <= c <= high]
                    if in_range:
                        res.append(np.random.choice(in_range))
                    else:
                        # 選最近的
                        closest = min(cands, key=lambda x: min(abs(x-low), abs(x-high)))
                        res.append(closest)
                return res

            sel_m = select(mem_cands)
            sel_n = select(non_mem_cands)

            if len(sel_m) < 10 or len(sel_n) < 10: continue

            # 計算 JS Divergence
            bins = np.linspace(min_v, max_v, 50)
            h_m, _ = np.histogram(sel_m, bins=bins, density=True)
            h_n, _ = np.histogram(sel_n, bins=bins, density=True)
            js = jensenshannon(h_m + 1e-10, h_n + 1e-10)

            if js < best_js:
                best_js = js
                best_interval = (low, high)

    print(f"最佳區間: {best_interval} (JS Div: {best_js:.4f})")
    return best_interval

def run_inference(loader, label, limit):
        logits = []
        count = 0
        for x, _ in tqdm(loader, desc=f"Processing {label}", total=limit):
            if count >= limit: break
            x = x.to(DEVICE)
            if x.dim() == 3: x = x.unsqueeze(0)

            recons = diffusion_reconstruct(diffusion_pipeline, x, N_RECONSTRUCTIONS, T_STEPS)
            cands = get_candidates(target_model, x, recons)

            if not cands: continue

            in_range = [c for c in cands if l_min <= c <= l_max]
            if in_range:
                logits.append(np.random.choice(in_range))
            else:
                closest = min(cands, key=lambda v: min(abs(v-l_min), abs(v-l_max)))
                logits.append(closest)
            count += 1
        return logits

def calculate_logit_undefended(model, x):
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            max_conf, _ = torch.max(probs, dim=1)     #嘗試不用softmax-----------------------------------
            # 避免 log(0)
            max_conf = torch.clamp(max_conf, 1e-6, 1 - 1e-6)
            logits = torch.log(max_conf / (1 - max_conf))

        return logits.cpu().item()

def run_undefended_inference(model, loader, label, limit):
        logits = []
        count = 0
        for x, _ in tqdm(loader, desc=f"Processing {label}", total=limit):
            if count >= limit: break
            x = x.to(DEVICE)
            if x.dim() == 3: x = x.unsqueeze(0)

            l = calculate_logit_undefended(model, x)
            logits.append(l)
            count += 1
        return logits

# ==========================================
# 5. 主程式執行
# ==========================================
if __name__ == "__main__":
    print(f"環境設定完成。使用裝置: {DEVICE}")
    # A. 準備資料與模型
    mem_set, non_mem_set, num_classes = get_dataset(DATASET_NAME)
    train_loader = torch.utils.data.DataLoader(mem_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    # 評估用 Loader (Batch Size 1 以方便擴散重建)
    mem_loader_eval = torch.utils.data.DataLoader(mem_set, batch_size=1, shuffle=True)
    non_mem_loader_eval = torch.utils.data.DataLoader(non_mem_set, batch_size=1, shuffle=True)

    target_model = get_target_model(num_classes)
    diffusion_pipeline = get_diffusion_model()

    # B. 訓練 Target Model
    target_model = train_target_model(target_model, train_loader, epochs=EPOCHS)

    # C. 校準
    interval = calibrate_scenario_1(target_model, diffusion_pipeline, mem_loader_eval, non_mem_loader_eval)

    # D. 最終推論與繪圖
    print("開始最終推論並繪圖 (這需要一點時間)...")
    l_min, l_max = interval


    mem_logits = run_inference(mem_loader_eval, "Members", EVAL_SIZE)
    non_mem_logits = run_inference(non_mem_loader_eval, "Non-Members", EVAL_SIZE)

    # 繪圖
    plt.figure(figsize=(10, 6))
    plt.hist(mem_logits, bins=50, alpha=0.6, label='Training Samples (Members)', color='blue')
    plt.hist(non_mem_logits, bins=50, alpha=0.6, label='Test Samples (Non-Members)', color='orange')
    plt.xlabel("Logit of the Confidences")
    plt.ylabel("#Samples")
    plt.title(f"Prediction Distribution Gap ({DATASET_NAME} - Scenario 1)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    save_path = f"diffence_scenario1{DATASET_NAME}.png"
    plt.savefig(save_path, dpi=300)
    print(f"圖片已儲存至: {save_path}")
    # ==========================================
    # 6. 未被保護執行
    # ==========================================



    # C. 直接推論 (Undefended)
    print("開始計算 Undefended Logits...")
    mem_logits = run_undefended_inference(target_model, mem_loader_eval, "Members", EVAL_SIZE)
    non_mem_logits = run_undefended_inference(target_model, non_mem_loader_eval, "Non-Members", EVAL_SIZE)

    # D. 繪圖
    plt.figure(figsize=(10, 6))
    plt.hist(mem_logits, bins=50, alpha=0.6, label='Training Samples (Members)', color='blue')
    plt.hist(non_mem_logits, bins=50, alpha=0.6, label='Test Samples (Non-Members)', color='orange')
    plt.xlabel("Logit of the Confidences")
    plt.ylabel("#Samples")
    plt.title(f"Prediction Distribution Gap (Undefended - {DATASET_NAME})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 儲存圖片
    save_path = f"undefended{DATASET_NAME}.png"
    plt.savefig(save_path, dpi=300)
    print(f"圖片已儲存至: {save_path}")

    plt.show()

