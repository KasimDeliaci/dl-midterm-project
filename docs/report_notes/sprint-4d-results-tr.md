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
- macro precision: `0.679 -> 0.727`
- macro recall: `0.725 -> 0.743`
- weighted-F1: `0.809 -> 0.818`

Bu sonuç Sprint 4D’nin en güçlü bulgusudur. Validation gate ile seçilen `tta_rot4`, Sprint 4C
weighted fusion modelinde test macro-F1’i belirgin biçimde artırdı. Artış yalnızca recall
genişlemesi değildir; macro precision, accuracy ve weighted-F1 de birlikte yükselmiştir.

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

## Literatürle Bağlantı ve Yorum

Sprint 4D'nin en güçlü sonucu, yeni bir model eğitmeden macro-F1'i `0.699`'dan `0.733`'e
taşımasıdır. Bu, augmentation fikrinin bu projede en faydalı biçiminin training-time perturbation
değil, inference-time geometry averaging olduğunu düşündürür. Dermoskopik görüntülerde lezyon
orientasyonu çoğu zaman sınıf etiketinin kendisi değildir; bu yüzden rotation tabanlı TTA farklı
görünümlerdeki tahmin gürültüsünü azaltabilir.

Bu bulgu literatürdeki ensemble/fusion sonuçlarıyla da uyumludur. Liu et al. (2024), farklı modellerin
oylama/stacking ile sınırlı ama gerçek kazançlar sağlayabildiğini raporlar. Sprint 4D'de ise ensemble
model seviyesinde değil, aynı modelin farklı geometrik görünümleri seviyesinde yapılmıştır. Bu daha
ucuz ve daha kontrollü bir ensemble türüdür: training split değişmez, yeni checkpoint üretilmez,
yalnızca inference maliyeti artar.

Yine de Sprint 4D sonucu state-of-the-art iddiası değildir. Roy et al. (2024) ve Haque et al. (2026)
daha yüksek F1/macro-F1 değerleri raporlar; ancak custom attention/fusion mimarileri ve farklı
eğitim protokolleri kullanırlar. Sprint 4D'nin rapordaki değeri, fixed-CNN-feature projesinde
validation-gated TTA'nın en tutarlı macro-F1 iyileştirmesini sağlamasıdır.

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
