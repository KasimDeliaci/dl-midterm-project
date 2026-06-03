# Sprint 4E Sonuç Notları: Fusion Training Diagnostic

Sprint 4E, Sprint 4C ve Sprint 4D sonrasında üç fine-tuned CNN özelliğinin nasıl
birleştirildiğini inceleyen yerel bir cached-feature diagnostik çalışmasıdır. Bu aşamada CNN
backbone'ları yeniden fine-tune edilmedi; deneyler yalnızca mevcut `finetuned` feature cache'leri
üzerinden yürütülen MLP/fusion eğitimlerini kapsadı. Böylece lesion-aware train/validation/test
split sabit tutuldu ve raw image path'leri fusion aşamasında kullanılmadı.

## Amaç

Bu deneyin ana sorusu, learned weighted fusion'ın concat fusion'a göre neden tutarlı şekilde daha
iyi davranmadığını anlamaktı. Özellikle feature ölçeği, projection kapasitesi, global ağırlıkların
sınıf bazlı tamamlayıcılığı temsil edip edememesi ve class imbalance ile loss seçiminin etkisi
test edildi.

## Deney Tasarımı

Toplam 14 üç-backbone fusion adayı çalıştırıldı:

- concat baseline: `none`, `standardize_per_backbone`, `l2_per_backbone`
- global weighted fusion: farklı normalizasyon, projection dimension ve regularization ayarları
- per-class weighted fusion: sınıf başına backbone ağırlığı öğrenen varyantlar
- loss varyantları: class-weighted cross entropy, focal loss ve label smoothing

Normalizasyon istatistikleri yalnızca train split üzerinde fit edildi ve validation/test
split'lerine aynı istatistikler uygulandı. Model seçimi validation macro-F1 ile yapıldı; test
metrikleri candidate seçimi için kullanılmadı.

## Ana Bulgular

Validation tarafında en yüksek macro-F1 `perclass_l2_p512` adayıyla elde edildi:

- validation macro-F1: `0.683`
- validation accuracy: `0.816`
- validation weighted-F1: `0.819`

Global weighted baseline olan `weighted_none_p512_low_lr` validation macro-F1 `0.680` verdi.
Bu sonuç per-class weighting'in validation üzerinde küçük bir sinyal taşıdığını gösterdi, ancak
pre-registered test gate bu aday yerine accuracy/weighted-F1 koşullarını da sağlayan concat
normalizasyon adaylarını test'e taşıdı.

Validation-gated test sonuçları:

| Aday | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---:|---:|---:|
| `concat_standardize_base` | `0.790` | `0.691` | `0.798` |
| `concat_l2_base` | `0.787` | `0.683` | `0.795` |

Bu sonuçlar canonical Sprint 4 concat test macro-F1 `0.706` ve Sprint 4D weighted + `tta_rot4`
test macro-F1 `0.733` seviyelerini geçmedi.

## Yorum

Sprint 4E'nin en önemli katkısı doğrudan skor artışı değil, fusion aşamasındaki davranışı daha
açık hale getirmesidir. Per-backbone standardization concat validation macro-F1'i `0.639`'dan
`0.662`'ye çıkardı. Bu, fine-tuned feature cache'leri arasında ölçek farkı olduğunu ve concat MLP
eğitiminin bundan etkilendiğini düşündürür.

Buna rağmen normalizasyon, projection kapasitesi ve per-class ağırlıklandırma tek başına test
performansını canonical Sprint 4 ve Sprint 4D sonuçlarının üzerine taşımadı. Bu nedenle final
raporda Sprint 4E, "skor kovalamaya yönelik başarısız bir ek deney" yerine, weighted-vs-concat
farkını açıklamaya yardımcı olan kontrollü bir fusion-stage diagnostic olarak sunulmalıdır.

Per-class yorumlarda dikkatli olunmalıdır. `df` ve `vasc` gibi düşük destekli sınıflarda F1
değişimleri birkaç örnekten güçlü biçimde etkilenebilir. Bu yüzden sınıf bazlı heatmap ve gain
tabloları genel eğilimi desteklemek için kullanılmalı, tek başına klinik ya da güçlü sınıf bazlı
sonuç iddiasına dönüştürülmemelidir.

## Fusion Neden Beklendiği Kadar İyi Gelmedi?

Literatür fusion'ı destekler ama garanti etmez. Roy et al. (2024) gibi çalışmalar custom
attention/fusion bloklarıyla yüksek F1 raporlar; Mahbod et al. (2025) frozen/fine-tuned feature
ve prediction fusion'ın faydalı olabileceğini gösterir. Buna karşılık Akter et al. (2023) stacking
modellerinin güçlü single backbonelardan düşük kalabildiğini raporlar. Sprint 4E tam bu ikinci
uyarıyı açıklamaya yardımcı oldu.

Bizim weighted fusion tasarımımız global backbone ağırlıkları öğrenir. Bu, yorumlanabilir ve kompakt
bir temsil sağlar; ancak HAM10000'de hata örüntüleri sınıf bazlı değişebilir. Bir backbone `vasc`
veya `bcc` için faydalı olurken başka bir backbone `mel` veya `bkl` için daha iyi olabilir. Global
ağırlık bu sınıf bazlı tamamlayıcılığı tam temsil edemez. Per-class weighting validation'da küçük
sinyal verdi, fakat testte canonical concat'i geçemedi; bu da daha esnek fusion'ın küçük validation
setine overfit olabileceğini gösterir.

Bu nedenle "fusion neden iyi gelmedi?" sorusunun cevabı skor düşüklüğü değil, kapasite ve
genelleme trade-off'udur. Concat daha yüksek boyutlu ve daha az sıkıştırılmış feature bilgisini MLP'ye
bırakır. Weighted fusion ise daha düzenli ve yorumlanabilir ama bilgi sıkıştırır. Bizim single-seed,
lesion-aware protokolümüzde macro-F1 için en iyi denge concat tarafında kaldı.

## Üretilen Çıktılar

Başlıca tablolar:

- `artifacts/report_assets/tables/sprint4e/sprint4e_validation_results.csv`
- `artifacts/report_assets/tables/sprint4e/sprint4e_selection_log.csv`
- `artifacts/report_assets/tables/sprint4e/sprint4e_test_results.csv`
- `artifacts/report_assets/tables/sprint4e/sprint4e_feature_scale_summary.csv`
- `artifacts/report_assets/tables/sprint4e/sprint4e_concat_weighted_audit.csv`
- `artifacts/report_assets/tables/sprint4e/sprint4e_per_class_gap_summary.csv`

Başlıca görseller:

- `artifacts/report_assets/figures/sprint4e/sprint4e_feature_norms_by_backbone.png`
- `artifacts/report_assets/figures/sprint4e/sprint4e_validation_macro_f1_by_candidate.png`
- `artifacts/report_assets/figures/sprint4e/sprint4e_concat_vs_weighted_gap.png`
- `artifacts/report_assets/figures/sprint4e/sprint4e_learned_weights_audit.png`
- `artifacts/report_assets/figures/sprint4e/sprint4e_per_class_f1_gain_heatmap.png`
- `artifacts/report_assets/figures/sprint4e/sprint4e_best_confusion_matrix.png`

Run klasörleri ve `model.pt` checkpoint'leri `artifacts/runs/sprint4e_fusion_diagnostic/`
altında kaldı ve git'e alınmamalıdır.
