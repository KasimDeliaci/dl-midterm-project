# Sprint 4C Sonuç Notu: Fine-Tuned MLP/Fusion Hyperparameter Search

Sprint 4C, canonical Sprint 4 `finetuned` feature cache'lerini sabit tutarak final MLP/fusion
classifier ayarlarının sonucu ne kadar etkilediğini test eder. Bu sprintte yeni CNN fine-tuning,
dataset split değişikliği veya test-time augmentation yapılmadı.

Ana soru:

> Sabit fine-tuned feature cache'leri üzerinde daha geniş bir MLP/fusion hyperparameter search,
> validation-selected macro-F1'i artırıyor mu, yoksa canonical Sprint 4 sonucu final classifier
> tuning'e karşı robust mu?

## Protokol

Selection metric her aşamada `best_val_macro_f1` olarak tutuldu. Test metrikleri otomatik export
edildi, ancak aday seçimi için kullanılmadı.

Canonical Sprint 4 referansı:

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| `r50+mnv2+effb0 concat` | 0.811 | 0.724 | 0.702 | 0.706 | 0.813 |

## Stage A: Screening

Stage A, 72 lokal cached-feature MLP/fusion run olarak tamamlandı:

- 24 single-backbone run.
- 24 concat fusion run.
- 24 weighted fusion run.

Stage A'nın en iyi validation-selected sonucu:

| Combination | Candidate | Val macro-F1 | Accuracy | Macro precision | Macro recall | Test macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `r50+mnv2+effb0 weighted` | `w_cw_adamw_low_lr_p512` | 0.680 | 0.802 | 0.679 | 0.724 | 0.699 | 0.808 |

Bu aday matched canonical weighted baseline'a göre validation macro-F1'i `+0.034` artırdı. Bu,
önceden belirlenen Stage B eşiğini geçtiği için full-matrix expansion çalıştırıldı.

Stage A'da en iyi profiller:

- Non-weighted: `nocw_adamw_base`, `cw_adamw_small`.
- Weighted: `w_cw_adamw_low_lr_p512`, `w_cw_adamw_base_p1024`.

## Stage B: Full Matrix

Stage B, Stage A'da seçilen profillerle canonical 11-combination matrix üzerinde 22 run olarak
tamamlandı:

- 6 single-backbone run.
- 8 concat fusion run.
- 8 weighted fusion run.

Stage B validation-selected sıralamasında en iyi aday yine üç-backbone weighted fusion oldu:

| Combination | Candidate | Val macro-F1 | Accuracy | Macro precision | Macro recall | Test macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `r50+mnv2+effb0 weighted` | `w_cw_adamw_low_lr_p512` | 0.680 | 0.802 | 0.679 | 0.724 | 0.699 | 0.808 |
| `r50+mnv2+effb0 concat` | `nocw_adamw_base` | 0.660 | 0.812 | 0.712 | 0.675 | 0.688 | 0.807 |
| `r50+effb0 concat` | `cw_adamw_small` | 0.659 | 0.788 | 0.654 | 0.711 | 0.675 | 0.794 |
| `r50` | `nocw_adamw_base` | 0.653 | 0.807 | 0.734 | 0.652 | 0.684 | 0.798 |

Matched canonical karşılaştırmada en önemli validation gain'ler:

| Combination | Candidate | Val gain | Test macro-F1 gain | Accuracy gain | Weighted-F1 gain |
|---|---|---:|---:|---:|---:|
| `r50+mnv2+effb0 weighted` | `w_cw_adamw_low_lr_p512` | +0.034 | +0.012 | +0.001 | +0.006 |
| `mnv2+effb0 concat` | `nocw_adamw_base` | +0.034 | +0.050 | +0.090 | +0.063 |
| `r50+effb0 concat` | `cw_adamw_small` | +0.022 | -0.004 | -0.005 | -0.004 |
| `mnv2+effb0 weighted` | `w_cw_adamw_low_lr_p512` | +0.019 | +0.047 | +0.043 | +0.036 |

Önemli ayrım: `r50+mnv2+effb0 concat / cw_adamw_small` test macro-F1 tarafında `0.707`
görünüyor, yani canonical best `0.706` ile aynı seviyede ve çok az üzerinde. Ancak validation
macro-F1'i `0.651` olduğu için bu aday protokole göre seçilmedi. Bu sonucu yeni best olarak
sunmak test-set shopping olur.

## Per-Class Etki

Validation-selected en iyi aday olan `r50+mnv2+effb0 weighted / w_cw_adamw_low_lr_p512`, matched
canonical weighted baseline'a göre test per-class F1'de şu ana hareketleri verdi:

| Class | F1 gain |
|---|---:|
| `bkl` | +0.042 |
| `df` | +0.035 |
| `mel` | +0.031 |
| `akiec` | -0.001 |
| `vasc` | -0.001 |
| `nv` | -0.002 |
| `bcc` | -0.018 |

Kazanç tek bir düşük-support sınıfa bağlı değil; `bkl`, `df` ve `mel` tarafında birlikte geliyor.
Yine de bu, canonical overall best concat sonucunu test macro-F1 açısından geçmeye yetmedi.

## Learned Fusion Weights

En iyi weighted fusion adayında global softmax ağırlıkları dengeli kaldı:

| Backbone | Weight |
|---|---:|
| `resnet50` | 0.336 |
| `mobilenet_v2` | 0.320 |
| `efficientnet_b0` | 0.344 |

Bu, modelin tek bir backbone'u baskın seçmediğini; üç temsilin de benzer katkı aldığını gösterir.

## Sonuç

Sprint 4C, final classifier/fusion tuning'in validation metric üzerinde anlamlı etki yaratabildiğini
gösterdi. Özellikle weighted fusion, canonical weighted baseline'a göre belirgin validation gain
verdi ve testte de matched baseline'dan daha iyi çıktı.

Ancak Sprint 4C, canonical Sprint 4 overall best sonucu olan `r50+mnv2+effb0 concat` modelini
protokole uygun şekilde geçemedi:

- En iyi validation-selected aday test macro-F1 `0.699` ile canonical best `0.706` altında kaldı.
- Testte `0.707` görünen concat adayının validation sinyali zayıf olduğu için final best olarak
seçilmemesi gerekir.

Bu nedenle Sprint 4C'nin rapor sonucu şudur:

> Canonical Sprint 4 sonucu yalnızca zayıf MLP hyperparameter seçiminden kaynaklanmıyor. Final-stage
> tuning bazı matched kombinasyonlarda önemli iyileşme sağlıyor, fakat genel en iyi Sprint 4 concat
> fusion sonucunu validation-selected ve test-audit protokolü altında güvenilir biçimde aşmıyor.

## Literatürle Bağlantı

Mahbod et al. (2025), frozen veya sabit feature temsilleri üzerine MLP/probe eğitmenin dermoskopik
sınıflandırmada ciddi bir baseline olabileceğini gösterdiği için Sprint 4C'nin sorusu literatürle
doğrudan uyumludur: feature extractor sabitken classifier ve fusion head ayarları sonucu ne kadar
değiştirir? Bizim bulgumuz, MLP/fusion tuning'in özellikle validation macro-F1 üzerinde etkili
olduğunu, ancak testte canonical concat sonucunu güvenilir biçimde geçmediğini gösterir.

Bu sonuç final raporda iki açıdan değerlidir. Birincisi, yüksek literatür skorları yalnızca backbone
seçiminden değil, classifier head, regularization, validation protokolü ve selection discipline
gibi ayrıntılardan da etkilenir. İkincisi, testte daha yüksek görünen ama validation-selected olmayan
bir adayın "best" seçilmemesi, bizim protokolümüzün test-set shopping yapmadığını gösterir. Bu
nokta, split ayrıntıları belirsiz preprintlerle kıyas yaparken önemli bir metodolojik üstünlüktür.

## Artifact'ler

Stage A:

- Tables: `artifacts/report_assets/tables/sprint4c/sprint4c_stage_a/`
- Figures: `artifacts/report_assets/figures/sprint4c/sprint4c_stage_a/`
- Runs: `artifacts/runs/sprint4c_hparam_search/sprint4c_stage_a/`

Stage B:

- Tables: `artifacts/report_assets/tables/sprint4c/sprint4c_stage_b/`
- Figures: `artifacts/report_assets/figures/sprint4c/sprint4c_stage_b/`
- Runs: `artifacts/runs/sprint4c_hparam_search/sprint4c_stage_b/`

Run artifact klasörleri model/checkpoint içerdiği için Git'e eklenmemelidir. CSV/PNG report assets
proje rapor politikasına göre küçük artifact olarak değerlendirilebilir.
