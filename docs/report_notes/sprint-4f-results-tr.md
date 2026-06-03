# Sprint 4F Sonuç Notları: Augmented Fine-Tuning

Sprint 4F, Sprint 4D'de test-time augmentation'ın fayda göstermesinden sonra, benzer dayanıklılık
sinyalinin eğitim aşamasında kontrollü image-level augmentation ile elde edilip edilemeyeceğini
test etti. Deney ayrı `finetuned_augmented` feature source'u altında yürütüldü; canonical Sprint 4
`finetuned` checkpoint ve cache'leri overwrite edilmedi.

## Deney Tasarımı

ResNet50, MobileNetV2 ve EfficientNetB0 için yine sadece son anlamlı CNN blokları fine-tune edildi.
Eğitimde train-split class weights, validation macro-F1 checkpoint seçimi ve seed `42` kullanıldı.
Augmentation policy yatay/dikey flip, random resized crop, sınırlı rotation/affine ve hafif color
jitter içerdi. Feature extraction aşaması deterministik eval transform ile yapıldı; stochastic
augmentation cache üretimine taşınmadı.

Sprint 4F matrix'i bilinçli olarak küçük tutuldu:

- 3 single-backbone MLP;
- three-backbone concat;
- three-backbone weighted.

## Ana Sonuçlar

| Model | Test accuracy | Macro precision | Macro recall | Test macro-F1 | Test weighted-F1 |
|---|---:|---:|---:|---:|---:|
| ResNet50 single | `0.746` | `0.561` | `0.648` | `0.589` | `0.761` |
| MobileNetV2 single | `0.730` | `0.551` | `0.619` | `0.575` | `0.743` |
| EfficientNetB0 single | `0.721` | `0.537` | `0.648` | `0.579` | `0.739` |
| Three-backbone concat | `0.786` | `0.643` | `0.661` | `0.645` | `0.790` |
| Three-backbone weighted | `0.776` | `0.599` | `0.639` | `0.615` | `0.775` |

Bu sonuçlar canonical Sprint 4 three-backbone concat macro-F1 `0.706` ve Sprint 4D weighted +
`tta_rot4` macro-F1 `0.733` seviyelerinin altında kaldı.

## Yorum

Sprint 4F'nin düşük sonucu, TTA başarısının doğrudan training-time augmentation başarısına
çevrilemeyeceğini gösteriyor. TTA, eğitilmiş modeli değiştirmeden birkaç deterministik görünümün
tahminini ortalarken; Sprint 4F, backbone feature space'ini augmentation altında yeniden öğrendi.
Bu policy dermoskopik görüntülerde lezyon sınırı, renk dağılımı veya küçük sınıf sinyalini fazla
bozmuş olabilir.

Bu nedenle Sprint 4F negatif ama değerli bir sonuçtur: "daha fazla augmentation" tek başına
macro-F1'i yükseltmedi. Sonraki deneyler için daha kontrollü policy, validation-gated TTA,
multi-seed doğrulama veya mimari/loss düzeyinde daha belirgin değişiklikler daha mantıklıdır.

## Literatürle Bağlantı

Hu ve Yang (2023) gibi HAM10000 çalışmalarında augmentation ve imbalance-aware training yüksek F1
sonuçlarıyla birlikte raporlanır. Ancak bu, augmentation'ın her protokolde otomatik iyileşme
sağlayacağı anlamına gelmez. O çalışmalarda augmentation, custom mimari ve training recipe ile
birlikte değerlendirilir; bizim Sprint 4F deneyimizde ise backbone seti ve cached-feature fusion
protokolü sabit tutuldu.

Bu ayrım raporda özellikle belirtilmelidir. Sprint 4F'nin düşüşü, "augmentation işe yaramaz" değil,
"bu lezyon-aware split ve bu fixed-backbone/fusion pipeline içinde random resized crop + affine
tarzı train-time augmentation feature space'i bozmuş olabilir" anlamına gelir. Dermoskopik
görüntülerde renk tonu, sınır, asimetri ve küçük dokusal ipuçları sınıflar için önemlidir; agresif
geometrik veya crop tabanlı değişimler minority-class sinyalini azaltabilir.

Bu yüzden Sprint 4D/4I TTA ile Sprint 4F aynı kefeye konmamalıdır. TTA, modeli değiştirmeden
tahminleri stabilize eder; Sprint 4F ise modeli augmentation altında yeniden optimize eder. Bizim
sonuçlarımızda birincisi macro-F1'i artırmış, ikincisi düşürmüştür.

## Üretilen Çıktılar

Başlıca tablolar:

- `artifacts/report_assets/tables/single_backbone_finetuned_augmented_results.csv`
- `artifacts/report_assets/tables/finetuned_augmented_all_results.csv`
- `artifacts/report_assets/tables/finetuned_augmented_per_class_f1.csv`
- `artifacts/report_assets/tables/finetuned_augmented_fusion_gain_summary.csv`
- `artifacts/report_assets/tables/finetuned_augmented_fusion_weight_summary.csv`

Başlıca görseller:

- `artifacts/report_assets/figures/finetuned_augmented_single_backbone_f1.png`
- `artifacts/report_assets/figures/finetuned_augmented_fusion_comparison.png`
- `artifacts/report_assets/figures/finetuned_augmented_concat_vs_weighted.png`
- `artifacts/report_assets/figures/finetuned_augmented_per_class_f1_heatmap.png`
- `artifacts/report_assets/figures/finetuned_augmented_best_confusion_matrix.png`
- `artifacts/report_assets/figures/finetuned_augmented_learned_fusion_weights.png`

Büyük checkpoint, feature cache ve `model.pt` dosyaları git dışında kalır; Drive mirror ve lokal
gitignored artifact klasörlerinde korunur.
