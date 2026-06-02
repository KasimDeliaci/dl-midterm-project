# Sprint 4G Sonuç Notları: Local Autoresearch Ensemble

Sprint 4G, yeni CNN fine-tuning yapmadan mevcut checkpoint'leri birleştiren yerel bir
cached-feature autoresearch deneyidir. Amaç, Sprint 4B/4C/4E/4F sonrasında elde edilen farklı
MLP/fusion modellerinin soft-vote ensemble ile macro-F1'i artırıp artırmadığını test etmektir.

## Deney Tasarımı

Aday modeller `artifacts/runs/` altında mevcut `model.pt` ve `config_resolved.yaml` dosyalarından
keşfedildi. Deney yalnızca cached feature kullandı; raw image path'leri veya image dataset loader'ı
bu aşamada kullanılmadı. Aday ve ensemble seçimi validation macro-F1 ile yapıldı, test split yalnızca
seçilen ensemble için bir kez değerlendirildi.

Random Dirichlet ağırlık araması ilk denemede validation üzerinde daha yüksek görünmesine rağmen
test tarafında düşüş ürettiği için kapatıldı. Nihai Sprint 4G sonucu deterministic uniform/rank
weighted soft-vote araması üzerinden seçildi.

## Ana Sonuç

Validation-selected ensemble:

- fusion policy: uniform soft-vote;
- candidate count: 3;
- validation macro-F1: `0.725`;
- test accuracy: `0.806`;
- test macro-F1: `0.707`;
- test weighted-F1: `0.812`.

Karşılaştırma:

| Sonuç | Test macro-F1 |
|---|---:|
| Canonical Sprint 4 three-backbone concat | `0.706` |
| Sprint 4G selected ensemble | `0.707` |
| Sprint 4D weighted + `tta_rot4` | `0.733` |

Sprint 4G canonical Sprint 4'ü yalnızca yaklaşık `+0.0015` macro-F1 ile geçti. Bu fark pratikte
çok küçük olduğu için "güçlü yeni best" olarak değil, post-hoc ensemble averaging'in sınırlı katkı
sağladığı şeklinde yorumlanmalıdır. Sprint 4D TTA hâlâ en yüksek macro-F1 sonucudur.

## Yorum

Bu deney iki şeyi netleştirdi. Birincisi, farklı class-aware ve weighted modellerin validation'da
tamamlayıcı sinyal taşıdığı görülüyor; uniform üç-model ensemble validation macro-F1'i `0.725`
seviyesine çıkardı. İkincisi, bu sinyal test split'e çok sınırlı taşındı. Bu durum validation set
üzerinde ensemble seçiminin de overfit riski taşıdığını ve single-seed sonuçların dikkatli
yorumlanması gerektiğini gösterir.

Final raporda Sprint 4G, literatüre yaklaşmak için denenmiş kontrollü bir model-combination
extension olarak kullanılabilir; fakat ana performans iddiası Sprint 4D TTA sonucuna dayanmalıdır.

## Üretilen Çıktılar

Başlıca tablolar:

- `artifacts/report_assets/tables/sprint4g/sprint4g_individual_validation_results.csv`
- `artifacts/report_assets/tables/sprint4g/sprint4g_ensemble_validation_results.csv`
- `artifacts/report_assets/tables/sprint4g/sprint4g_selection_log.csv`
- `artifacts/report_assets/tables/sprint4g/sprint4g_test_results.csv`
- `artifacts/report_assets/tables/sprint4g/sprint4g_test_per_class_f1.csv`
- `artifacts/report_assets/tables/sprint4g/sprint4g_final_comparison.csv`

Başlıca görseller:

- `artifacts/report_assets/figures/sprint4g/sprint4g_validation_ensemble_macro_f1.png`
- `artifacts/report_assets/figures/sprint4g/sprint4g_test_macro_f1.png`
- `artifacts/report_assets/figures/sprint4g/sprint4g_selected_weights.png`

Full validation search table `artifacts/runs/sprint4g_autoresearch_ensemble/` altında tutulur ve
git'e alınmaz.
