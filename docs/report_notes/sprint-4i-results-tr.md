# Sprint 4I Sonuç Notları: Geometry-Safe TTA Refinement

Sprint 4I, yeni bir eğitim koşusu değil; mevcut en güçlü Sprint 4C weighted fusion modeline
uygulanan inference-time bir TTA refinement çalışmasıdır. Amaç, lezyon geometrisini bozmadan tam
çerçeveyi koruyan deterministik dönüşümlerin macro-F1'i artırıp artırmadığını ölçmekti.

## Deney Kurulumu

- Model: Sprint 4C üç-backbone weighted fusion MLP.
- Backbone checkpointleri: canonical Sprint 4 fine-tuned ResNet50, MobileNetV2 ve EfficientNetB0.
- TTA politikası: `tta_d4_8`.
- Dönüşümler: identity, 90/180/270 derece rotasyonlar, horizontal mirror ve mirrored rotasyonlar.
- Crop, affine warp, color jitter veya stochastic augmentation kullanılmadı.
- Policy seçimi validation macro-F1 ile yapıldı; test sonucu policy seçmek için kullanılmadı.

## Ana Metrikler

| Split | Policy | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|---:|---:|
| Validation | identity | `0.804` | `0.678` | `0.687` | `0.678` | `0.810` |
| Validation | `tta_d4_8` | `0.826` | `0.724` | `0.718` | `0.717` | `0.831` |
| Test | identity | `0.803` | `0.679` | `0.725` | `0.699` | `0.809` |
| Test | `tta_d4_8` | `0.813` | `0.720` | `0.737` | `0.727` | `0.816` |

D4 TTA, Sprint 4C weighted modelin identity inference sonucuna göre test macro-F1'i `+0.028`
artırdı. Ancak Sprint 4D'deki daha basit `tta_rot4` policy hâlâ en iyi sonuçtur:

| Koşu | Test macro-F1 |
|---|---:|
| Sprint 4D weighted + `tta_rot4` | `0.733` |
| Sprint 4I weighted + `tta_d4_8` | `0.727` |
| Sprint 4C weighted identity | `0.699` |

Accuracy ve weighted-F1 tarafında da D4 TTA identity inference'ın üstündedir: test accuracy
`0.803 -> 0.813`, macro precision `0.679 -> 0.720`, macro recall `0.725 -> 0.737` ve weighted-F1
`0.809 -> 0.816`.

## Sınıf Bazlı Etki

Test split üzerinde D4 TTA özellikle bazı az örnekli sınıflarda F1'i artırdı:

| Sınıf | Identity F1 | D4 TTA F1 | Değişim | Support |
|---|---:|---:|---:|---:|
| akiec | `0.613` | `0.692` | `+0.079` | 52 |
| bcc | `0.705` | `0.723` | `+0.018` | 71 |
| bkl | `0.632` | `0.653` | `+0.020` | 167 |
| df | `0.667` | `0.634` | `-0.033` | 20 |
| nv | `0.899` | `0.901` | `+0.002` | 1004 |
| mel | `0.567` | `0.556` | `-0.011` | 167 |
| vasc | `0.810` | `0.927` | `+0.117` | 21 |

Bu tablo, TTA'nın bazı az örnekli sınıflarda stabilite sağlayabildiğini ama tüm sınıflarda
monoton iyileşme üretmediğini gösterir. `df` ve `vasc` support düşük olduğu için bu sınıflardaki
değişimler final raporda dikkatli yorumlanmalıdır.

## Yorum

Sprint 4I, “daha fazla TTA view daha iyi sonuç verir” varsayımını doğrulamadı. D4 TTA, identity
inference'a göre belirgin iyileşme sağladı; fakat Sprint 4D'deki `tta_rot4` policy daha yüksek
test macro-F1 verdi. Bu nedenle final performans iddiası Sprint 4D weighted + `tta_rot4` sonucuna
dayanmalıdır.

Sprint 4I'nin rapordaki değeri, inference-time geometry averaging'in faydalı olduğunu ikinci kez
göstermesi ve aynı zamanda TTA politikasının validation-gated seçilmesi gerektiğini vurgulamasıdır.

## Literatürle Bağlantı

Sprint 4I, literatürdeki ensemble mantığının inference-time augmentation karşılığı olarak
yorumlanabilir. Liu et al. (2024)'te model-level ensemble sınırlı ama gerçek accuracy artışı
sağlarken, Sprint 4I aynı modelin farklı deterministik görünümlerini ortalayarak macro-F1'i artırdı.
Bu, training-time augmentation'dan farklıdır: yeni temsil öğrenmek yerine karar sınırındaki görüntü
dönüşümü kaynaklı oynaklığı azaltır.

Ancak D4'ün `tta_rot4`'ten düşük kalması da önemlidir. Daha fazla view, daha fazla bilgi anlamına
gelmeyebilir; bazı mirrored dönüşümler modelin öğrendiği dermoskopik ipuçlarını stabilize etmek
yerine belirsizleştirebilir. Bu yüzden raporda TTA sonucu "augmentation her zaman iyidir" şeklinde
değil, "validation ile seçilmiş geometry-safe averaging bu pipeline'da en iyi iyileştirmeyi verdi"
şeklinde yazılmalıdır.
