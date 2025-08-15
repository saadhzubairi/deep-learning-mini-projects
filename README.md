# Jailbreaking Deep Models: Adversarial Attacks on ResNet-34 & DenseNet-121

## Overview
This project implements **pixel-wise ($L_\infty$)** and **patch-based ($L_0$)** adversarial attacks on **ResNet-34** and tests transferability to **DenseNet-121**.  
We find that small, imperceptible perturbations can cause **>90% Top-1 accuracy drops**.

---

## Attacks Implemented
- **FGSM** – Single-step $L_\infty$
- **PGD** – Multi-step $L_\infty$
- **Patch Attack (PGD + $L_0$)** – Saliency-guided, targeted, momentum-optimized

---

## Key Results
| Attack     | ε     | ResNet-34 Top-1 | DenseNet-121 Top-1 | Drop (ResNet) |
|------------|-------|-----------------|--------------------|---------------|
| Clean      | -     | 76.00%          | 74.20%             | -             |
| FGSM       | 0.02  | **6.00%**       | 33.00%             | **-92.1%**    |
| PGD        | 0.02  | 57.40%          | 71.40%             | -24.5%        |
| Patch      | 0.50  | 11.60%          | 31.20%             | -84.7%        |

---

## Install
```bash
git clone https://github.com/<user>/adversarial-attacks-resnet34.git
cd adversarial-attacks-resnet34
pip install -r requirements.txt
```

### Usage
#### Baseline
```
python evaluate_clean.py --model resnet34 --data_path /path/to/data
```

#### FGSM Attack
```
python attack.py --attack fgsm --epsilon 0.02
```
#### PGD Attack
```
python attack.py --attack pgd --epsilon 0.02 --steps 10 --step_size 0.002
```
#### Patch Attack
```
python attack.py --attack patch --epsilon 0.5 --patch_size 32 --steps 60 --saliency --momentum 0.9 --target least_likely
```


### Lessons Learned

Saving adversarial examples as .pt tensors preserves subtle perturbations — PNG/JPEG degrades them.

FGSM can outperform poorly tuned PGD.

Patch attacks work best with saliency-based targeting + momentum.