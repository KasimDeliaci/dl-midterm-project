# Sprint 4 Sonuç Notu: Fine-Tuning ve Fine-Tuned Feature Matrix

Bu not Sprint 4 full Colab run sonuçlarını ve frozen Sprint 3 karşılaştırmasını özetler. Klinik
teşhis iddiası yoktur; çalışma HAM10000 üzerinde benchmark dermoscopic image classification
deneyidir.

## Mevcut Durum

Full Sprint 4 run Colab GPU üzerinde tamamlandı ve Drive'daki `dl-midterm-artifacts` çıktıları
lokale indirildi. Fine-tuned feature cache'leri, checkpoint'ler, 11-run fine-tuned MLP/fusion matrix
sonuçları ve report-ready asset'ler local repo altına senkronlandı. Frozen-vs-fine-tuned
karşılaştırması local canonical Sprint 3 frozen run'larıyla yeniden üretildi.

## Fine-Tuning Policy

| Backbone | Unfrozen block/stage | Head |
| --- | --- | --- |
| ResNet50 | `layer4` | `fc` |
| MobileNetV2 | `features[16]`, `features[17]`, `features[18]` | `classifier` |
| EfficientNetB0 | `features[7]`, `features[8]` | `classifier` |

Tüm önceki CNN katmanları frozen kalır. Full run için ImageNet pretrained ağırlıklar kullanılır.
Class weights yalnızca train split label dağılımından hesaplanır. Model selection validation
macro-F1 ile yapılır; test metric hyperparameter veya checkpoint seçiminde kullanılmaz.

## Canonical Artifact'ler

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

Cross-check:

- Feature cache shape/alignment doğru: train `6981`, val `1532`, test `1502` satır.
- Feature dimensions: ResNet50 `2048`, MobileNetV2 `1280`, EfficientNetB0 `1280`.
- Fine-tuned matrix report table `11` satır içeriyor: 3 single, 3 pairwise concat, 3 pairwise
  weighted, 1 three-backbone concat, 1 three-backbone weighted.
- Weighted fusion weight sum tüm weighted run'larda `1.0`.
- Fine-tuned run folders içinde `model.pt` yok; checkpoints ve feature `.pt` dosyaları git dışında
  kalmalı.

## Sonuç Özeti

| Soru | Cevap |
| --- | --- |
| En iyi fine-tuned backbone hangisi? | ResNet50 single-backbone: macro-F1 `0.658`, accuracy `0.776`, weighted-F1 `0.785`. |
| En iyi fine-tuned fusion run hangisi? | ResNet50 + MobileNetV2 + EfficientNetB0 concat: macro-F1 `0.706`, accuracy `0.811`, weighted-F1 `0.813`. |
| Frozen Sprint 3 best macro-F1 olan ResNet50 + EfficientNetB0 concat `0.595` üstüne çıkıldı mı? | Evet. En iyi fine-tuned fusion `0.706` macro-F1 ile yaklaşık `+0.111` mutlak artış sağladı. Aynı üç-backbone concat karşılığı frozen `0.576` idi; fine-tuned karşılığı `+0.130` gain verdi. |
| Fine-tuning en çok hangi sınıflara yardım etti? | En iyi fine-tuned run'da F1 gain: `vasc +0.214`, `bcc +0.194`, `df +0.171`, `akiec +0.136`, `mel +0.106`, `bkl +0.067`, `nv +0.020`. |
| Concat mı weighted mı daha güçlü kaldı? | En iyi sonuç concat oldu (`r50+mnv2+effb0 concat`, macro-F1 `0.706`). Weighted fusion da güçlüydü; en iyi weighted `r50+effb0 weighted` macro-F1 `0.698`. |
| Learned fusion weights nasıl dağıldı? | Weighted run'larda ağırlıklar softmax sonrası normalize kaldı. Örnek: `r50+effb0 weighted` ResNet50 `0.507`, EfficientNetB0 `0.493`; `r50+mnv2 weighted` ResNet50 `0.539`, MobileNetV2 `0.461`. |
| Ek runtime karşılığında macro-F1 gain anlamlı mı? | Tek-seed sınırıyla birlikte, Sprint 4 fine-tuning frozen matrix'e göre belirgin macro-F1 artışı gösterdi. Son raporda ek compute maliyeti ve tek-seed belirsizliği açıkça belirtilmeli. |

## Dikkat Edilecek Yorum Sınırları

- Tek-seed sonuçlar kesin mimari üstünlük iddiası olarak sunulmamalıdır.
- `df` ve `vasc` gibi düşük-support sınıflardaki değişimler yüksek belirsizlikle yorumlanmalıdır.
- Accuracy ve weighted-F1 çoğunluk sınıfı etkisiyle iyimser görünebilir; ana yorum macro-F1 ve
  per-class F1 üzerinden yapılmalıdır.
- Büyük checkpoints, feature cache `.pt` dosyaları ve run folder'ları git dışında kalmalıdır.
