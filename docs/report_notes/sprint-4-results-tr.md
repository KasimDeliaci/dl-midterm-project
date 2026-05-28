# Sprint 4 Sonuç Notu: Fine-Tuning ve Fine-Tuned Feature Matrix

Bu not Sprint 4 implementasyon durumunu ve full Colab run tamamlandığında doldurulacak sonuç
yorumlarını izlemek için oluşturuldu. Klinik teşhis iddiası yoktur; çalışma HAM10000 üzerinde
benchmark dermoscopic image classification deneyidir.

## Mevcut Durum

Fine-tuning altyapısı implement edildi ve lokal `/tmp` smoke run ile doğrulandı. Canonical Sprint 4
sonuçları henüz üretilmedi; full fine-tuning ve fine-tuned MLP/fusion matrix Colab GPU üzerinde
çalıştırılmalıdır. Colab smoke-test veya küçük lokal smoke çıktıları Sprint 3 canonical frozen
sonuçlarıyla karıştırılmayacaktır.

## Fine-Tuning Policy

| Backbone | Unfrozen block/stage | Head |
| --- | --- | --- |
| ResNet50 | `layer4` | `fc` |
| MobileNetV2 | `features[16]`, `features[17]`, `features[18]` | `classifier` |
| EfficientNetB0 | `features[7]`, `features[8]` | `classifier` |

Tüm önceki CNN katmanları frozen kalır. Full run için ImageNet pretrained ağırlıklar kullanılır.
Class weights yalnızca train split label dağılımından hesaplanır. Model selection validation
macro-F1 ile yapılır; test metric hyperparameter veya checkpoint seçiminde kullanılmaz.

## Beklenen Canonical Artifact'ler

- `artifacts/features/ham10000/finetuned/resnet50/`
- `artifacts/features/ham10000/finetuned/mobilenet_v2/`
- `artifacts/features/ham10000/finetuned/efficientnet_b0/`
- `artifacts/report_assets/tables/finetuned_all_results.csv`
- `artifacts/report_assets/tables/frozen_vs_finetuned_summary.csv`
- `artifacts/report_assets/tables/finetuned_per_class_f1.csv`
- `artifacts/report_assets/tables/finetuning_gain_summary.csv`
- `artifacts/report_assets/tables/finetuned_fusion_weight_summary.csv`
- `artifacts/report_assets/figures/frozen_vs_finetuned_macro_f1.png`
- `artifacts/report_assets/figures/finetuned_fusion_comparison.png`
- `artifacts/report_assets/figures/finetuned_concat_vs_weighted.png`
- `artifacts/report_assets/figures/finetuning_gain_macro_f1.png`
- `artifacts/report_assets/figures/finetuned_per_class_f1_heatmap.png`
- `artifacts/report_assets/figures/finetuned_best_confusion_matrix.png`
- `artifacts/report_assets/figures/finetuned_learned_fusion_weights.png`

## Sonuç Yorumu İçin Doldurulacak Alanlar

Full Colab run tamamlandıktan sonra bu bölüm doldurulmalıdır:

| Soru | Cevap |
| --- | --- |
| En iyi fine-tuned backbone hangisi? | TBD |
| En iyi fine-tuned fusion run hangisi? | TBD |
| Frozen Sprint 3 best macro-F1 olan ResNet50 + EfficientNetB0 concat `0.595` üstüne çıkıldı mı? | TBD |
| Fine-tuning en çok hangi sınıflara yardım etti? | TBD |
| Concat mı weighted mı daha güçlü kaldı? | TBD |
| Learned fusion weights nasıl dağıldı? | TBD |
| Ek runtime karşılığında macro-F1 gain anlamlı mı? | TBD |

## Dikkat Edilecek Yorum Sınırları

- Tek-seed sonuçlar kesin mimari üstünlük iddiası olarak sunulmamalıdır.
- `df` ve `vasc` gibi düşük-support sınıflardaki değişimler yüksek belirsizlikle yorumlanmalıdır.
- Accuracy ve weighted-F1 çoğunluk sınıfı etkisiyle iyimser görünebilir; ana yorum macro-F1 ve
  per-class F1 üzerinden yapılmalıdır.
- Büyük checkpoints, feature cache `.pt` dosyaları ve run folder'ları git dışında kalmalıdır.
