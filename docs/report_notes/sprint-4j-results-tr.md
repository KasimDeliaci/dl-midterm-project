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

| Koşu | Train sampling | Class weighting | Best val macro-F1 | Test accuracy | Macro precision | Macro recall | Test macro-F1 | Test weighted-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Sprint 4 canonical concat | shuffle | true | `0.655` | `0.811` | `0.724` | `0.702` | `0.706` | `0.813` |
| Sprint 4C weighted identity | shuffle | true | `0.680` | `0.803` | `0.679` | `0.725` | `0.699` | `0.809` |
| Sprint 4J balanced concat | class-balanced | false | `0.660` | `0.804` | `0.706` | `0.687` | `0.690` | `0.809` |
| Sprint 4J balanced weighted | class-balanced | false | `0.663` | `0.799` | `0.718` | `0.634` | `0.668` | `0.797` |

## Yorum

Balanced sampling, cached-feature MLP aşamasında yeni bir iyileşme sinyali üretmedi. Concat
koşusunda validation macro-F1 canonical concat'e göre çok az yükseldi (`0.655` -> `0.660`), ancak
test macro-F1 `0.706`'dan `0.690`'a düştü. Weighted koşuda düşüş daha belirgindi.

Bu nedenle Sprint 4J, Colab'da image-level balanced-sampler fine-tuning'e geçmek için yeterince
promising değildir. SMOTE yapmama kararı korunmalı; balanced sampling ise final raporda kısa bir
diagnostic negatif sonuç olarak anılabilir.

## Literatürle Bağlantı

Class imbalance literatürde gerçek bir problemdir; Gessert et al. balanced sampling ve
class-specific weighting gibi stratejileri tartışır. Bu yüzden Sprint 4J'nin sorusu doğrudur. Ancak
Sprint 4J'de sampling yalnızca cached-feature MLP aşamasında değiştirildi; CNN feature'ları zaten
sabit olduğu için az sınıflara ait yeni görsel temsil öğrenilemedi.

Sonuç bu sınırlamayı gösterir. Balanced sampler validation macro-F1'i biraz oynattı, fakat test
macro-F1 ve weighted-F1'i düşürdü. Bu, minority class örneklerini daha sık göstermek ile gerçek
genelleme arasında fark olduğunu gösterir. SMOTE benzeri sentetik yaklaşımlara mesafeli durmak da
bu nedenle savunulabilir: HAM10000'de az sınıflar hem küçük hem görsel olarak heterojendir; sentetik
veya aşırı tekrar edilen feature'lar validation'a uyum sağlayıp testte genelleşmeyebilir.
