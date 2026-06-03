# Sprint 4K Sonuç Notları: Image-Level Balanced Sampler

Sprint 4K, Sprint 4J'deki cached-feature balanced sampler deneyi sonrasında aynı fikri CNN
fine-tuning aşamasında test etti. Amaç, az örnekli sınıfların görüntülerini train sırasında daha
sık göstererek fine-tuned representation'ın macro-F1 tarafında iyileşip iyileşmediğini ölçmekti.

## Deney Kurulumu

- Feature source: `finetuned_balanced_sampler`.
- Backbone: ResNet50 diagnostic.
- Unfreeze policy: `layer3 + layer4`.
- Loss: plain cross-entropy.
- Class weights: kapalı.
- Train sampler: image-level class-balanced `WeightedRandomSampler`.
- Augmentation: crop-free hafif flip, rotation ve color jitter.
- Model seçimi: validation macro-F1.
- Seed: `42`.

Splitler Sprint 1'deki lesion-aware train/validation/test ayrımıyla aynen korundu. Sampler yalnızca
train split içinde çalıştı; validation ve test dağılımları değiştirilmedi. Bu nedenle deney yeni
bir split üretmedi ve cross-split lesion leakage riskini artırmadı.

## Ana Metrikler

| Koşu | Best val epoch | Best val macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Sprint 4K ResNet50 balanced sampler | `14` | `0.672` | `0.756` | `0.657` | `0.772` |

Validation macro-F1 epoch 14'te `0.672` seviyesine çıktı. Ancak test macro-F1 `0.657` ve
weighted-F1 `0.772` seviyesinde kaldı. Bu sonuç, önceki en güçlü Sprint 4 çizgisinin altında olduğu
için deney full üç-backbone matrix'e eskale edilmedi.

## Karşılaştırmalı Yorum

| Deney | Test macro-F1 |
|---|---:|
| Sprint 4D weighted + `tta_rot4` | `0.733` |
| Sprint 4G ensemble | `0.707` |
| Canonical Sprint 4 three-backbone concat | `0.706` |
| Sprint 4C weighted identity | `0.699` |
| Sprint 4J balanced concat diagnostic | `0.690` |
| Sprint 4K ResNet50 balanced sampler | `0.657` |

Sprint 4K validation tarafında iyileşme sinyali verse de test split üzerinde genelleme artışı
sağlamadı. Train accuracy'nin yüksek, validation/test performansının daha sınırlı kalması, image
level oversampling'in az örnekli sınıflardaki sınırlı görsel varyasyonu daha sık tekrar ederek
overfitting etkisi yaratabileceğini düşündürür.

## Sonuç

Sprint 4K ana sonuç olarak kullanılmamalıdır. Final raporda kısa bir negatif diagnostic deney
olarak konumlandırılabilir: class-balanced sampling, cached-feature MLP aşamasında da image-level
fine-tuning aşamasında da canonical Sprint 4 / Sprint 4D sonucunu geçmedi.

Bu sonuç SMOTE yapmama kararını destekler. Az örnekli sınıflara daha fazla ağırlık vermek tek
başına yeterli olmadı; bu projede en güvenilir iyileşme train-time oversampling yerine
geometry-safe inference-time averaging, yani Sprint 4D TTA çizgisinden geldi.
