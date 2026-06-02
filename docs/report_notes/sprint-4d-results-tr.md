# Sprint 4D Sonuç Notları

Sprint 4D, tamamlanmış fine-tuned fusion modelleri üzerinde eğitim yapmadan uygulanan bir
test-time augmentation (TTA) deneyi olarak tasarlandı. Amaç, sabit Sprint 1 lesion-aware split
korunurken, inference sırasında aynı görüntünün deterministik geometrik görünümlerinden elde edilen
olasılıkları ortalamanın benchmark dermoscopic image classification performansını iyileştirip
iyileştirmediğini ölçmekti.

Bu deneyde model veya TTA politikası test metriklerine göre seçilmedi. Önce validation split üzerinde
`identity`, `tta_flip4` ve `tta_rot4` politikaları karşılaştırıldı. Test split yalnızca validation
gate geçildikten sonra, seçilen politika için bir kez çalıştırıldı. Bu nedenle Sprint 4D sonuçları
post-hoc ama validation-gated bir inference audit olarak raporlanmalıdır.

Önemli artifact notu: May 29 tarihli canonical Sprint 4 concat MLP checkpoint dosyası
(`model.pt`) Drive mirror içinde bulunmadığı için Sprint 4D’de bu modelin exact checkpoint’i değil,
mevcut fine-tuned feature cache’lerden yerelde yeniden üretilmiş bir concat MLP checkpoint’i
kullanıldı. Bu checkpoint Sprint 4 ana sonucu yerine geçmez; yalnızca Sprint 4D TTA yolunu
çalıştırmak için kullanılan local inference artifact olarak belirtilmelidir.

## Validation Bulguları

Validation aşamasında iki test-eligible model değerlendirildi:

- Regenerated Sprint 4 concat: `resnet50 + mobilenet_v2 + efficientnet_b0`, concat fusion.
- Sprint 4C weighted: `resnet50 + mobilenet_v2 + efficientnet_b0`, learned weighted fusion.

Regenerated Sprint 4 concat modelinde validation macro-F1:

- `identity`: `0.660`
- `tta_flip4`: `0.695`
- `tta_rot4`: `0.695`

Sprint 4C weighted modelinde validation macro-F1:

- `identity`: `0.678`
- `tta_flip4`: `0.703`
- `tta_rot4`: `0.707`

Validation gate iki model için de `tta_rot4` politikasını seçti. Her iki durumda da macro-F1 artışı
`+0.005` eşiğinin üstündeydi ve accuracy / weighted-F1 düşüşü gözlenmedi.

## Test Bulguları

Test aşamasında yalnızca identity ve validation-selected `tta_rot4` politikaları çalıştırıldı.

Regenerated Sprint 4 concat checkpoint için:

- test macro-F1: `0.685 -> 0.690` (`+0.005`)
- accuracy: `0.804 -> 0.814`
- weighted-F1: `0.806 -> 0.815`

Bu artış pozitif olmakla birlikte küçük ve noise-level sınırına yakındır. Ayrıca exact May 29
checkpoint yerine regenerated checkpoint kullanıldığı için bu sonuç canonical Sprint 4 sonucunun
yerine yazılmamalıdır.

Sprint 4C weighted checkpoint için:

- test macro-F1: `0.699 -> 0.733` (`+0.034`)
- accuracy: `0.803 -> 0.815`
- weighted-F1: `0.809 -> 0.818`

Bu sonuç Sprint 4D’nin en güçlü bulgusudur. Validation gate ile seçilen `tta_rot4`, Sprint 4C
weighted fusion modelinde test macro-F1’i belirgin biçimde artırdı.

## Per-Class Yorum

Sprint 4C weighted + `tta_rot4` test sonucunda en belirgin F1 artışları şu sınıflarda görüldü:

- `vasc`: `+0.117`
- `akiec`: `+0.098`
- `bkl`: `+0.034`
- `bcc`: `+0.033`

`df`, `mel` ve `nv` için F1 küçük düşüş gösterdi. Özellikle `df` ve `vasc` support düşük olduğu için
bu sınıflardaki değişimler dikkatli yorumlanmalıdır. Buna rağmen macro-F1 artışı yalnızca tek bir
düşük-support sınıfa dayanmıyor; `akiec`, `bcc` ve `bkl` gibi daha farklı sınıflarda da iyileşme var.

## Maliyet

`tta_rot4` dört view kullandığı için inference süresi identity’ye göre yaklaşık 4-5 kat arttı.
Bu nedenle Sprint 4D sonucu performans/maliyet trade-off’u olarak raporlanmalıdır: daha yüksek
macro-F1, ama daha pahalı inference.

## Report-ready Artifactler

Ana tablolar:

- `artifacts/report_assets/tables/sprint4d/sprint4d_validation_results.csv`
- `artifacts/report_assets/tables/sprint4d/sprint4d_test_results.csv`
- `artifacts/report_assets/tables/sprint4d/sprint4d_decision_log.csv`
- `artifacts/report_assets/tables/sprint4d/sprint4d_delta_vs_identity.csv`
- `artifacts/report_assets/tables/sprint4d/sprint4d_per_class_f1_gain.csv`
- `artifacts/report_assets/tables/sprint4d/sprint4d_runtime_summary.csv`

Ana görseller:

- `artifacts/report_assets/figures/sprint4d/sprint4d_val_policy_comparison.png`
- `artifacts/report_assets/figures/sprint4d/sprint4d_validation_macro_f1_delta.png`
- `artifacts/report_assets/figures/sprint4d/sprint4d_test_macro_f1_delta.png`
- `artifacts/report_assets/figures/sprint4d/sprint4d_per_class_f1_gain_heatmap.png`
- `artifacts/report_assets/figures/sprint4d/sprint4d_runtime_multiplier.png`

Prediction dump dosyaları `artifacts/runs/sprint4d_tta/predictions/` altında tutuldu ve git dışında
kalmalıdır.
