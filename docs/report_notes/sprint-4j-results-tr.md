# Sprint 4J Sonuç Notları: Balanced Sampler Diagnostic

Sprint 4J, SMOTE yerine daha savunulabilir bir class-aware yaklaşımı hızlıca test etmek için
yapıldı. Deney yeni image, sentetik feature veya yeni split üretmedi. Mevcut canonical Sprint 4
`finetuned` feature cache'leri sabit tutuldu; yalnızca cached-feature MLP training sırasında train
mini-batch'leri `WeightedRandomSampler` ile sınıf dengeli örneklendi.

## Deney Kurulumu

- Feature source: `finetuned`.
- Kombinasyonlar: ResNet50 + MobileNetV2 + EfficientNetB0 concat ve weighted.
- Train sampling: `class_balanced`.
- Class weights: kapalı. Sampler zaten class balance uyguladığı için loss ağırlıklarıyla üst üste
bindirilmedi.
- Validation/test loader'ları gerçek split dağılımında bırakıldı.
- Seçim metriği yorumu: validation macro-F1.

## Sonuçlar

| Koşu | Train sampling | Class weighting | Best val macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---|---:|---:|---:|---:|---:|
| Sprint 4 canonical concat | shuffle | true | `0.655` | `0.811` | `0.706` | `0.813` |
| Sprint 4C weighted identity | shuffle | true | `0.680` | `0.803` | `0.699` | `0.809` |
| Sprint 4J balanced concat | class-balanced | false | `0.660` | `0.804` | `0.690` | `0.809` |
| Sprint 4J balanced weighted | class-balanced | false | `0.663` | `0.799` | `0.668` | `0.797` |

## Yorum

Balanced sampling, cached-feature MLP aşamasında yeni bir iyileşme sinyali üretmedi. Concat
koşusunda validation macro-F1 canonical concat'e göre çok az yükseldi (`0.655` -> `0.660`), ancak
test macro-F1 `0.706`'dan `0.690`'a düştü. Weighted koşuda düşüş daha belirgindi.

Bu nedenle Sprint 4J, Colab'da image-level balanced-sampler fine-tuning'e geçmek için yeterince
promising değildir. SMOTE yapmama kararı korunmalı; balanced sampling ise final raporda kısa bir
diagnostic negatif sonuç olarak anılabilir.
