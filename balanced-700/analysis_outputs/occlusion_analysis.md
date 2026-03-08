# Occlusion Sensitivity Analysis Report

## Per-Class Accuracy by Occlusion Condition

| Class | Full Image | Central 45° | Periphery Only | Mask Ring 1 | Mask Ring 2 | Mask Ring 3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMD | 99.0% | 2.0% | 99.0% | 95.0% | 87.0% | 91.0% |
| DR | 98.0% | 17.0% | 70.0% | 72.0% | 92.0% | 98.0% |
| Healthy | 90.0% | 0.0% | 97.0% | 66.0% | 68.0% | 98.0% |
| PM | 96.0% | 91.0% | 76.0% | 96.0% | 94.0% | 90.0% |
| RD | 93.0% | 11.0% | 80.0% | 54.0% | 55.0% | 76.0% |
| RVO | 95.0% | 5.0% | 93.0% | 51.0% | 82.0% | 93.0% |
| Uveitis | 94.0% | 62.0% | 78.0% | 82.0% | 83.0% | 70.0% |
| **OVERALL** | **95.0%** | **26.9%** | **84.7%** | **73.7%** | **80.1%** | **88.0%** |

## Accuracy Drop from Baseline
*Negative values indicate that removing the region hurts performance.*

| Class | Central 45° | Periphery Only | Mask Ring 1 | Mask Ring 2 | Mask Ring 3 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| AMD | -97.0% | +0.0% | -4.0% | -12.0% | -8.0% |
| DR | -81.0% | -28.0% | -26.0% | -6.0% | +0.0% |
| Healthy | -90.0% | +7.0% | -24.0% | -22.0% | +8.0% |
| PM | -5.0% | -20.0% | +0.0% | -2.0% | -6.0% |
| RD | -82.0% | -13.0% | -39.0% | -38.0% | -17.0% |
| RVO | -90.0% | -2.0% | -44.0% | -13.0% | -2.0% |
| Uveitis | -32.0% | -16.0% | -12.0% | -11.0% | -24.0% |
| **OVERALL** | **-68.1%** | **-10.3%** | **-21.3%** | **-14.9%** | **-7.0%** |

## Mean Confidence on True Class

| Class | Full Image | Central 45° | Periphery Only | Mask Ring 1 | Mask Ring 2 | Mask Ring 3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMD | 0.924 | 0.133 | 0.930 | 0.790 | 0.769 | 0.735 |
| DR | 0.866 | 0.174 | 0.599 | 0.571 | 0.810 | 0.869 |
| Healthy | 0.790 | 0.074 | 0.824 | 0.509 | 0.620 | 0.865 |
| PM | 0.863 | 0.555 | 0.644 | 0.818 | 0.815 | 0.782 |
| RD | 0.806 | 0.151 | 0.670 | 0.428 | 0.481 | 0.645 |
| RVO | 0.871 | 0.096 | 0.827 | 0.400 | 0.692 | 0.814 |
| Uveitis | 0.850 | 0.364 | 0.668 | 0.649 | 0.707 | 0.606 |

## Key Finding: Peripheral Importance Score
*Calculated as: Baseline Accuracy - Central-only Accuracy. Higher drop indicates greater reliance on the periphery.*

| Class | Baseline | Central | Drop | Importance |
| :--- | ---: | ---: | ---: | :--- |
| AMD | 99.0% | 2.0% | +97.0% | HIGH |
| Healthy | 90.0% | 0.0% | +90.0% | HIGH |
| RVO | 95.0% | 5.0% | +90.0% | HIGH |
| RD | 93.0% | 11.0% | +82.0% | HIGH |
| DR | 98.0% | 17.0% | +81.0% | HIGH |
| Uveitis | 94.0% | 62.0% | +32.0% | HIGH |
| PM | 96.0% | 91.0% | +5.0% | LOW |

### Ranked by Peripheral Importance
1. **AMD**: +97.0%
2. **Healthy**: +90.0%
3. **RVO**: +90.0%
4. **RD**: +82.0%
5. **DR**: +81.0%
6. **Uveitis**: +32.0%
7. **PM**: +5.0%