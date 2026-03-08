# GradCAM Analysis Report

This report presents the GradCAM analysis results for a trained EfficientNet-B0 model and a randomized classifier (sanity check). The analysis focuses on understanding where the model's attention is directed across different retinal zones: the central 45-degree fundus, three peripheral rings, and artifact regions. Attention distribution and density are reported for each class.

## Trained Model Analysis

### Attention Distribution (% of FOV attention)

| Class   | fundus_45 | periph_1 | periph_2 | periph_3 | artifact |
| :------ | :-------- | :------- | :------- | :------- | :------- |
| AMD     | 8.7%      | 24.8%    | 28.6%    | 31.4%    | 15.3%    |
| DR      | 14.9%     | 33.3%    | 26.6%    | 21.4%    | 7.7%     |
| Healthy | 18.0%     | 41.9%    | 29.0%    | 10.6%    | 1.6%     |
| PM      | 21.8%     | 44.7%    | 24.4%    | 8.5%     | 2.8%     |
| RD      | 12.0%     | 36.8%    | 36.1%    | 14.3%    | 3.3%     |
| RVO     | 13.7%     | 36.3%    | 31.9%    | 16.6%    | 9.9%     |
| Uveitis | 16.5%     | 37.6%    | 30.0%    | 14.4%    | 5.9%     |

### Attention Density (1.0 = uniform, >1.0 = disproportionate)

| Class   | fundus_45 | periph_1 | periph_2 | periph_3 |
| :------ | :-------- | :------- | :------- | :------- |
| AMD     | 1.37      | 1.09     | 0.78     | 1.08     |
| DR      | 2.41      | 1.50     | 0.72     | 0.73     |
| Healthy | 3.20      | 2.06     | 0.83     | 0.31     |
| PM      | 3.86      | 2.20     | 0.70     | 0.25     |
| RD      | 2.11      | 1.81     | 1.03     | 0.42     |
| RVO     | 2.46      | 1.80     | 0.92     | 0.48     |
| Uveitis | 2.93      | 1.85     | 0.86     | 0.42     |

---


## Sanity Check (Randomized Classifier)

### Attention Distribution (% of FOV attention)

| Class   | fundus_45 | periph_1 | periph_2 | periph_3 | artifact |
| :------ | :-------- | :------- | :------- | :------- | :------- |
| AMD     | 11.1%     | 33.9%    | 32.0%    | 18.3%    | 20.6%    |
| DR      | 14.7%     | 34.6%    | 29.4%    | 18.1%    | 8.9%     |
| Healthy | 19.7%     | 37.9%    | 24.0%    | 15.6%    | 5.1%     |
| PM      | 10.5%     | 29.1%    | 34.2%    | 23.3%    | 7.8%     |
| RD      | 10.7%     | 34.4%    | 37.6%    | 15.9%    | 4.6%     |
| RVO     | 11.1%     | 32.3%    | 33.7%    | 20.3%    | 10.9%    |
| Uveitis | 11.1%     | 30.8%    | 36.2%    | 19.6%    | 7.6%     |

### Attention Density (1.0 = uniform, >1.0 = disproportionate)

| Class   | fundus_45 | periph_1 | periph_2 | periph_3 |
| :------ | :-------- | :------- | :------- | :------- |
| AMD     | 1.75      | 1.50     | 0.87     | 0.63     |
| DR      | 2.38      | 1.56     | 0.80     | 0.61     |
| Healthy | 3.50      | 1.87     | 0.69     | 0.46     |
| PM      | 1.86      | 1.43     | 0.98     | 0.69     |
| RD      | 1.89      | 1.69     | 1.07     | 0.46     |
| RVO     | 1.99      | 1.60     | 0.97     | 0.59     |
| Uveitis | 1.98      | 1.51     | 1.03     | 0.58     |

---

## Sanity Check Comparison (fundus_45 density)

| Class   | Trained | Random | Diff  |
| :------ | :------ | :----- | :---- |
| AMD     | 1.37    | 1.75   | -0.37 |
| DR      | 2.41    | 2.38   | +0.03 |
| Healthy | 3.20    | 3.50   | -0.30 |
| PM      | 3.86    | 1.86   | +2.00 |
| RD      | 2.11    | 1.89   | +0.22 |
| RVO     | 2.46    | 1.99   | +0.47 |
| Uveitis | 2.93    | 1.98   | +0.95 |
