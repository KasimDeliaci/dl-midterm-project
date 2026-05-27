# Sprint 3 Sonuç Yorumu: Frozen Feature Fusion

Bu not, Sprint 3 sonunda lokalde üretilen frozen cached-feature fusion artifact'lerine dayanır.
Colab smoke-test çıktıları bu değerlendirmeye dahil edilmemiştir. Ana kaynaklar:

- `artifacts/report_assets/tables/frozen_all_results.csv`
- `artifacts/report_assets/tables/fusion_gain_summary.csv`
- `artifacts/report_assets/tables/fusion_weight_summary.csv`
- `artifacts/report_assets/tables/per_class_f1_frozen.csv`
- `artifacts/report_assets/figures/frozen_fusion_comparison.png`
- `artifacts/report_assets/figures/concat_vs_weighted.png`
- `artifacts/report_assets/figures/fusion_gain_macro_f1.png`
- `artifacts/report_assets/figures/per_class_f1_frozen_heatmap.png`
- `artifacts/report_assets/figures/frozen_best_confusion_matrix.png`
- `artifacts/report_assets/figures/learned_fusion_weights.png`

Bu sprintin amacı klinik tanı iddiası üretmek değil, HAM10000 üzerinde benchmark dermoscopic image classification için frozen feature fusion davranışını ölçmektir.

## Ne Yapıldı?

Sprint 3'te raw image üzerinden yeni feature extraction yapılmadı. Sprint 2'de üretilen frozen ResNet50, MobileNetV2 ve EfficientNetB0 `.pt` cache dosyaları ana input olarak kullanıldı. Her fusion run öncesinde ilgili backbone cache'lerinin `image_id`, label, label index ve split sırası birebir hizalı olduğu doğrulandı.

Toplam 8 fusion run üretildi:

| Combination | Fusion methods |
| --- | --- |
| ResNet50 + MobileNetV2 | concat, weighted |
| ResNet50 + EfficientNetB0 | concat, weighted |
| MobileNetV2 + EfficientNetB0 | concat, weighted |
| ResNet50 + MobileNetV2 + EfficientNetB0 | concat, weighted |

Concatenation fusion feature vektörlerini yan yana birleştirdi. Weighted fusion ise her backbone feature'ını 512-dimensional ortak projection space'e indirdi ve global learnable softmax weights ile birleştirdi. MLP classifier, Sprint 2 ile aynı genel training/evaluation protokolünü kullandı: seed 42, train split'ten class weighting, validation macro-F1 early stopping ve test split üzerinde final raporlama.

## Ana Sonuçlar

Sprint 3 fusion sonuçları, Sprint 2 default single-backbone baseline'larıyla birlikte 11-run frozen matrix olarak toplandı.

| Run | Fusion | Accuracy | Macro-F1 | Weighted-F1 |
| --- | --- | ---: | ---: | ---: |
| ResNet50 + EfficientNetB0 | concat | 0.746 | 0.595 | 0.760 |
| ResNet50 + MobileNetV2 + EfficientNetB0 | concat | 0.751 | 0.576 | 0.761 |
| MobileNetV2 + EfficientNetB0 | weighted | 0.719 | 0.562 | 0.738 |
| ResNet50 + MobileNetV2 | concat | 0.743 | 0.543 | 0.751 |
| ResNet50 + EfficientNetB0 | weighted | 0.685 | 0.532 | 0.711 |
| ResNet50 | none | 0.712 | 0.531 | 0.730 |
| MobileNetV2 + EfficientNetB0 | concat | 0.732 | 0.531 | 0.744 |
| ResNet50 + MobileNetV2 | weighted | 0.710 | 0.520 | 0.726 |
| ResNet50 + MobileNetV2 + EfficientNetB0 | weighted | 0.732 | 0.519 | 0.748 |
| EfficientNetB0 | none | 0.697 | 0.506 | 0.720 |
| MobileNetV2 | none | 0.668 | 0.468 | 0.692 |

En iyi Sprint 3 frozen fusion sonucu ResNet50 + EfficientNetB0 concat oldu: test macro-F1 0.595. Bu, Sprint 2 default single-backbone baseline'ındaki en iyi macro-F1 olan ResNet50 0.531'e göre +0.064 macro-F1 artış anlamına gelir.

Three-backbone concat da güçlü sonuç verdi: macro-F1 0.576 ve accuracy 0.751. Ancak en yüksek macro-F1'i üretmedi. Bu, daha fazla backbone eklemenin otomatik olarak daha iyi class-balanced performans vermediğini gösterir.

## Concat ve Weighted Fusion Karşılaştırması

Bu single-seed frozen matrix'te concat fusion genel olarak weighted fusion'dan daha güçlü oldu. En iyi concat run 0.595 macro-F1 üretirken en iyi weighted run MobileNetV2 + EfficientNetB0 ile 0.562 macro-F1 üretti.

Weighted fusion'ın zayıf kalması beklenmedik değildir. Weighted fusion her backbone'u 512-dimensional projection'a indirip global ağırlıklı toplam alır. Bu daha yorumlanabilir ve kompakt bir temsil verir, fakat concat fusion feature bilgisini daha az sıkıştırır. HAM10000 gibi fine-grained ve imbalanced bir benchmark'ta bu ekstra kapasite macro-F1 için faydalı olmuş görünüyor.

## Learned Fusion Weights

Weighted fusion run'larında softmax sonrası weight toplamı her run için 1.0 olarak doğrulandı.

| Weighted run | Learned weights |
| --- | --- |
| ResNet50 + MobileNetV2 | ResNet50 0.506, MobileNetV2 0.494 |
| ResNet50 + EfficientNetB0 | ResNet50 0.568, EfficientNetB0 0.432 |
| MobileNetV2 + EfficientNetB0 | MobileNetV2 0.575, EfficientNetB0 0.425 |
| ResNet50 + MobileNetV2 + EfficientNetB0 | ResNet50 0.364, MobileNetV2 0.341, EfficientNetB0 0.295 |

Weights tamamen tek backbone'a collapse etmedi. Pairwise ResNet50 içeren weighted run'larda ResNet50 biraz daha yüksek ağırlık aldı. MobileNetV2 + EfficientNetB0 weighted run'ında MobileNetV2 ağırlığı daha yüksek olmasına rağmen bu sonuç tek başına MobileNetV2 feature'ının daha iyi olduğu anlamına gelmez; projection katmanları ve MLP birlikte optimize edildiği için weights katkı göstergesi olarak yorumlanmalıdır, kesin feature kalitesi sıralaması olarak değil.

## Sınıf Bazlı Gözlemler

En iyi model olan ResNet50 + EfficientNetB0 concat, ResNet50 single baseline'a göre bazı minority ve orta-support sınıflarda belirgin iyileşme sağladı:

| Class | ResNet50 F1 | ResNet50 + EfficientNetB0 concat F1 |
| --- | ---: | ---: |
| `akiec` | 0.488 | 0.529 |
| `bcc` | 0.451 | 0.555 |
| `bkl` | 0.506 | 0.558 |
| `df` | 0.250 | 0.480 |
| `nv` | 0.857 | 0.875 |
| `mel` | 0.446 | 0.476 |
| `vasc` | 0.720 | 0.694 |

En dikkat çekici iyileşme `df` sınıfında görüldü: 0.250'den 0.480'e çıktı. Ancak `df` test support'u yalnızca 20 olduğu için bu artış yüksek belirsizlikle yorumlanmalıdır. `bcc`, `bkl`, `akiec` ve `mel` sınıflarında da daha dengeli bir kazanım var. `vasc` ise ResNet50 single baseline'da daha güçlü kaldı.

## Rapor İçin Ana Mesaj

Sprint 3'ün ana bulgusu şudur: frozen feature fusion, default single-backbone baseline'a göre macro-F1'i iyileştirdi; fakat bu iyileşme en güçlü şekilde concat fusion ile geldi. ResNet50 + EfficientNetB0 concat, hem macro-F1 hem weighted-F1 açısından frozen matrix'in en iyi modelidir.

Discussion bölümünde vurgulanması gereken tradeoff: weighted fusion daha yorumlanabilir ve kompakt bir 512-dimensional temsil üretir, fakat bu sprintte concat kadar güçlü değildir. Learned weights rapora katkı analizi sağlar; ancak tek-seed sonuçlar ve küçük sınıf support'ları nedeniyle kesin mimari üstünlük iddiası yapılmamalıdır.

Sprint 4'e geçmeden önce ana risk, bu sonuçların tek seed ile üretilmiş olmasıdır. En iyi birkaç frozen fusion run'ı ileride ek seed'lerle tekrar etmek, fine-tuning sonuçlarıyla karşılaştırmadan önce varyansı anlamaya yardımcı olabilir.

## Report-Ready Artifact List

Küçük rapor artifact'leri:

- `artifacts/report_assets/tables/frozen_all_results.csv`
- `artifacts/report_assets/tables/fusion_gain_summary.csv`
- `artifacts/report_assets/tables/fusion_weight_summary.csv`
- `artifacts/report_assets/tables/per_class_f1_frozen.csv`
- `artifacts/report_assets/figures/frozen_fusion_comparison.png`
- `artifacts/report_assets/figures/concat_vs_weighted.png`
- `artifacts/report_assets/figures/fusion_gain_macro_f1.png`
- `artifacts/report_assets/figures/per_class_f1_frozen_heatmap.png`
- `artifacts/report_assets/figures/frozen_best_confusion_matrix.png`
- `artifacts/report_assets/figures/learned_fusion_weights.png`
- `artifacts/report_assets/figures/fusion_runs/`

Büyük cache dosyaları, run klasörleri ve `model.pt` checkpoint'leri git dışında kalmalıdır.
