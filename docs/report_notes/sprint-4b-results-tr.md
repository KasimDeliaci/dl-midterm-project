# Sprint 4B Sonuç Notu: Class-Aware Fine-Tuning Extension

Bu not, Sprint 4 canonical `finetuned` sonuçlarına karşı Sprint 4B class-aware extension
çalışmasının özetidir. Sprint 4B canonical sonucu değiştirmez; `finetuned_classaware` ve
`finetuned_deeper` ayrı exploratory feature source olarak tutulur.

## Çalıştırılan Deneyler

Sprint 4B Colab koşusunda şu çıktılar üretildi ve lokal `artifacts/` altına senkronize edildi:

- `finetuned_classaware`: ResNet50, MobileNetV2 ve EfficientNet-B0 için class-balanced focal loss
  ile fine-tuning, feature cache ve single-backbone MLP screening.
- `finetuned_classaware`: 3 single, 3 pairwise concat, 3 pairwise weighted, 1 three-backbone concat
  ve 1 three-backbone weighted olmak üzere 11 MLP matrix run.
- `finetuned_deeper`: ResNet50 için `layer3 + layer4 + fc` unfreeze probe ve single-backbone MLP.

Lokal asset refresh canonical Sprint 4 run klasörleriyle tekrar çalıştırıldı. Böylece Colab'da
canonical run'lar bulunmadığı için atlanan Sprint 4B karşılaştırma tabloları lokal olarak üretildi.

## Ana Sonuç

Class-aware fine-tuning bazı tekil modellerde ve bazı fusion kombinasyonlarında küçük kazançlar
verdi, fakat canonical Sprint 4'ün en iyi fusion sonucunu geçemedi. Bu nedenle Sprint 4B'nin ana
bulgusu şudur:

> Class-aware objective, standalone macro-F1'i yer yer iyileştirse de daha güçlü fusion sonucuna
> dönüşmedi. Fusion başarımı yalnızca backbone kalitesine değil, backboneların tamamlayıcı ve farklı
> hata örüntüleri üretmesine de bağlı görünüyor.

## Single-Backbone Screening

Canonical Sprint 4 single-backbone sonuçlarıyla karşılaştırma:

| Candidate | Canonical macro-F1 | Sprint 4B macro-F1 | Gain | Accuracy gain | Weighted-F1 gain |
|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 class-aware | 0.5957 | 0.6190 | +0.0232 | -0.0140 | -0.0063 |
| MobileNetV2 class-aware | 0.5750 | 0.5917 | +0.0168 | +0.0173 | +0.0119 |
| ResNet50 deeper probe | 0.6578 | 0.6885 | +0.0307 | +0.0153 | +0.0115 |

Class-aware ResNet50 ayrıca full class-aware comparison içinde yer aldı:

- Canonical ResNet50 macro-F1: `0.6578`
- Class-aware ResNet50 macro-F1: `0.6482`
- Test macro-F1 farkı: `-0.0095`

Bu yüzden class-aware kaybın ResNet50 tarafında yararlı olmadığı, deeper unfreeze probe'un ise
ResNet50 için daha umut verici olduğu görülüyor.

## Full Class-Aware Fusion Matrix

En iyi class-aware full matrix sonucu:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| `r50+mnv2 concat` | 0.8209 | 0.6946 | 0.8164 |

Canonical Sprint 4 ile matched fusion karşılaştırmasında önemli satırlar:

| Fusion | Canonical macro-F1 | Class-aware macro-F1 | Gain |
|---|---:|---:|---:|
| `r50+mnv2 weighted` | 0.6512 | 0.6738 | +0.0226 |
| `mnv2+effb0 concat` | 0.6016 | 0.6209 | +0.0194 |
| `mnv2+effb0 weighted` | 0.5940 | 0.6124 | +0.0184 |
| `r50+mnv2 concat` | 0.6853 | 0.6946 | +0.0093 |
| `r50+mnv2+effb0 weighted` | 0.6868 | 0.6781 | -0.0087 |
| `r50+effb0 concat` | 0.6787 | 0.6519 | -0.0268 |
| `r50+mnv2+effb0 concat` | 0.7059 | 0.6762 | -0.0297 |
| `r50+effb0 weighted` | 0.6977 | 0.6628 | -0.0348 |

Canonical Sprint 4'ün en iyi overall sonucu `r50+mnv2+effb0 concat` macro-F1 `0.7059` iken,
class-aware matrix'in en iyi sonucu `r50+mnv2 concat` macro-F1 `0.6946` seviyesinde kaldı.

## Yorum

Bu sonuç class-aware yaklaşımın tamamen anlamsız olduğunu göstermez; EfficientNet-B0,
MobileNetV2 ve bazı MobileNet/EfficientNet fusion kombinasyonlarında sinyal var. Ancak sonuçlar
canonical Sprint 4'e göre daha iyi bir nihai model üretmedi.

Muhtemel açıklama:

- Class-aware loss minority-class davranışını bazı backbonelarda iyileştirirken feature uzayını
  fusion için daha az tamamlayıcı hale getirmiş olabilir.
- Backbonelar benzer class-aware karar sınırlarına itilmişse, single macro-F1 artışı fusion gain'e
  dönüşmeyebilir.
- ResNet50 class-aware single sonucu canonical ResNet50'den düşük kaldığı için ResNet50 içeren bazı
  fusionlar aşağı çekilmiş olabilir.
- Weighted fusion, class-aware feature dağılımı ve validation gürültüsüne concat'ten daha hassas
  davranmış olabilir.

Bu nedenle raporda Sprint 4B için en savunulabilir sonuç:

> Class-aware fine-tuning, bazı zayıf/orta backbone sonuçlarını iyileştirdi ancak canonical
> three-backbone concat fusion'ı geçemedi. Better standalone macro-F1 is not sufficient for better
> fusion; representation complementarity and calibration remain important.

## Literatürle Bağlantı

Class imbalance, HAM10000 literatüründe merkezi bir problemdir. Gessert et al. gibi çalışmalar
class-specific weighting, balanced sampling ve class-sensitive metriklerin önemini vurgular; Hu ve
Yang (2023) de imbalance-aware eğitim ve augmentation ile yüksek F1 raporlar. Sprint 4B bu çizgiyle
uyumludur: class-aware objective denemek metodolojik olarak doğru bir sorudur.

Ancak Sprint 4B'nin sonucu, imbalance-aware loss'un otomatik olarak daha iyi final fusion üretmediğini
gösterir. Bu da literatürdeki yüksek skorların yalnızca "class balancing var" diye açıklanamayacağını
hatırlatır. Custom attention, mimari kapasite, preprocessing, augmentation, split protokolü ve
metadata kullanımı gibi etkenler de sonuçları belirler. Bizim kontrollü protokolümüzde class-aware
loss bazı single/backbone kombinasyonlarında macro-F1'i artırdı, fakat üç-backbone concat'in
tamamlayıcı feature dengesini bozmuş olabilir.

Rapor dili bu yüzden şöyle olmalıdır: Sprint 4B, literatürdeki imbalance-aware motivasyonu test
eder; fakat bu proje koşullarında class-aware objective'in macro-F1/weighted-F1 dengesini canonical
Sprint 4'ten daha iyi hale getirmediğini gösterir.

## Artifact'ler

Senkronize edilen büyük artifact grupları:

- `artifacts/features/ham10000/finetuned_classaware/`
- `artifacts/features/ham10000/finetuned_deeper/`
- `artifacts/checkpoints/finetuned_classaware_backbones/`
- `artifacts/checkpoints/finetuned_deeper_backbones/`
- `artifacts/runs/20260531_*_finetuned_classaware_*`
- `artifacts/runs/20260531_*_finetuned_deeper_*`

Report-ready tablolar:

- `artifacts/report_assets/tables/sprint4b_classaware_all_results.csv`
- `artifacts/report_assets/tables/sprint4b_classaware_vs_canonical_fusion_summary.csv`
- `artifacts/report_assets/tables/sprint4b_screening_results.csv`
- `artifacts/report_assets/tables/sprint4b_vs_canonical_single_backbone.csv`
- `artifacts/report_assets/tables/sprint4b_per_class_f1_gain.csv`
- `artifacts/report_assets/tables/sprint4b_deeper_all_results.csv`

Report-ready figürler:

- `artifacts/report_assets/figures/sprint4b_classaware_fusion_comparison.png`
- `artifacts/report_assets/figures/sprint4b_classaware_concat_vs_weighted.png`
- `artifacts/report_assets/figures/sprint4b_classaware_learned_fusion_weights.png`
- `artifacts/report_assets/figures/sprint4b_best_confusion_matrix.png`
- `artifacts/report_assets/figures/sprint4b_test_macro_f1_vs_canonical.png`
- `artifacts/report_assets/figures/sprint4b_per_class_f1_gain_heatmap.png`

## Verification

Lokal doğrulama:

- İndirilen Sprint 4B feature, checkpoint ve run klasörleri lokal kopyalarla `diff -qr` üzerinden
  eşleşti.
- Beklenen Sprint 4B CSV/PNG report asset'leri mevcut ve boş değil.
- Row count kontrolleri geçti:
  - `sprint4b_classaware_all_results.csv`: 11
  - `sprint4b_classaware_vs_canonical_fusion_summary.csv`: 11
  - `sprint4b_screening_results.csv`: 6
  - `sprint4b_vs_canonical_single_backbone.csv`: 3
  - `sprint4b_deeper_all_results.csv`: 1
- `uv run ruff check src scripts tests` geçti.
- `uv run pytest` geçti: 32 test.
