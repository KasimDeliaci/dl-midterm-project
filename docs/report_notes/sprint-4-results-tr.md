# Sprint 4 Sonuç Notu: Fine-Tuning ve Fine-Tuned Feature Matrix

Bu not Sprint 4 fine-tuning deneylerini, fine-tuned feature matrix sonuçlarını ve Sprint 3 frozen
matrix ile karşılaştırmayı özetler. Çalışma klinik teşhis iddiası taşımaz; sonuçlar HAM10000
üzerinde benchmark dermoscopic image classification bağlamında yorumlanmalıdır.

## Kısa Özet

Sprint 4, ImageNet pretrained ResNet50, MobileNetV2 ve EfficientNetB0 backbone'larının son anlamlı
CNN bloklarını fine-tune ederek Sprint 3'teki frozen feature fusion protokolünü tekrarlar. Fine-tuned
feature cache'leri üretildikten sonra aynı cached-feature MLP/fusion matrix çalıştırılmıştır:

- 3 single-backbone MLP,
- 3 pairwise concat fusion,
- 3 pairwise weighted fusion,
- 1 three-backbone concat fusion,
- 1 three-backbone weighted fusion.

En iyi Sprint 4 sonucu `r50+mnv2+effb0 concat` modelidir: accuracy `0.811`, macro-F1 `0.706`,
weighted-F1 `0.813`. Bu sonuç, Sprint 3'teki en iyi frozen model olan `r50+effb0 concat`
macro-F1 `0.595` değerinin üstüne çıkmıştır.

## Deney Protokolü

Sprint 4, Sprint 1'de üretilen lesion-aware train/validation/test split'i sabit tutar. Test seti
model veya hyperparameter seçiminde kullanılmaz. Fine-tuned CNN checkpoint seçimi validation
macro-F1 ile yapılır; downstream MLP eğitiminde de early stopping validation macro-F1'e dayanır.

Fine-tuning politikası kontrollü ve konservatiftir:

| Backbone | Trainable CNN block/stage | Trainable head |
| --- | --- | --- |
| ResNet50 | `layer4` | `fc` |
| MobileNetV2 | `features[16]`, `features[17]`, `features[18]` | `classifier` |
| EfficientNetB0 | `features[7]`, `features[8]` | `classifier` |

Tüm önceki CNN katmanları frozen kalır. Fine-tuning sırasında train split label dağılımından
hesaplanan class weights kullanılır. Bu tercih, HAM10000'in class imbalance yapısı nedeniyle
accuracy yerine macro-F1 odaklı değerlendirme ile uyumludur.

## Üretilen Artifact'ler

Canonical Sprint 4 feature cache'leri:

- `artifacts/features/ham10000/finetuned/resnet50/`
- `artifacts/features/ham10000/finetuned/mobilenet_v2/`
- `artifacts/features/ham10000/finetuned/efficientnet_b0/`

Report-ready tablolar:

- `artifacts/report_assets/tables/finetuned_all_results.csv`
- `artifacts/report_assets/tables/frozen_vs_finetuned_summary.csv`
- `artifacts/report_assets/tables/finetuned_per_class_f1.csv`
- `artifacts/report_assets/tables/finetuning_gain_summary.csv`
- `artifacts/report_assets/tables/finetuned_fusion_weight_summary.csv`

Report-ready figürler:

- `artifacts/report_assets/figures/frozen_vs_finetuned_macro_f1.png`
- `artifacts/report_assets/figures/finetuned_fusion_comparison.png`
- `artifacts/report_assets/figures/finetuned_concat_vs_weighted.png`
- `artifacts/report_assets/figures/finetuning_gain_macro_f1.png`
- `artifacts/report_assets/figures/finetuned_per_class_f1_heatmap.png`
- `artifacts/report_assets/figures/finetuned_best_confusion_matrix.png`
- `artifacts/report_assets/figures/finetuned_learned_fusion_weights.png`

Yerel cross-check:

- Feature cache satır sayıları split'lerle eşleşir: train `6981`, val `1532`, test `1502`.
- Feature dimensions: ResNet50 `2048`, MobileNetV2 `1280`, EfficientNetB0 `1280`.
- Fine-tuned matrix `11` canonical run içerir.
- Weighted fusion ağırlıkları her weighted run'da softmax sonrası `1.0` toplamına sahiptir.
- Büyük checkpoint, feature `.pt` ve run artifact'leri git dışında kalmalıdır.

## Ana Sonuçlar

Single-backbone fine-tuned sonuçlarında en güçlü model ResNet50 olmuştur:

| Backbone | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| ResNet50 | `0.776` | `0.658` | `0.785` |
| EfficientNetB0 | `0.749` | `0.596` | `0.760` |
| MobileNetV2 | `0.734` | `0.575` | `0.750` |

Fine-tuned fusion matrix içinde en iyi sonuç üç-backbone concatenation modelinden gelmiştir:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| `r50+mnv2+effb0 concat` | `0.811` | `0.706` | `0.813` |
| `r50+effb0 weighted` | `0.794` | `0.698` | `0.802` |
| `r50+mnv2+effb0 weighted` | `0.800` | `0.687` | `0.802` |
| `r50+mnv2 concat` | `0.786` | `0.685` | `0.795` |
| `r50+effb0 concat` | `0.792` | `0.679` | `0.798` |

Bu sonuçlar iki ana bulgu verir. Birincisi, fine-tuning genel olarak frozen feature extraction'a
göre daha güçlü temsil üretmiştir. İkincisi, Sprint 3 frozen matrix'te en iyi kombinasyon pairwise
`r50+effb0 concat` iken Sprint 4'te en iyi model üç-backbone concat olmuştur. Bu, fine-tuning
sonrası MobileNetV2 temsilinin de ensemble/fusion bağlamında daha faydalı hale geldiğini düşündürür.

## Frozen-vs-Fine-Tuned Karşılaştırması

Sprint 3'te en iyi frozen sonuç `r50+effb0 concat` ile macro-F1 `0.595` idi. Sprint 4'te en iyi
fine-tuned sonuç `r50+mnv2+effb0 concat` ile macro-F1 `0.706` değerine ulaştı. Bu, en iyi model
seviyesinde yaklaşık `+0.111` mutlak macro-F1 artışıdır.

Matched configuration bazında en büyük macro-F1 artışı three-backbone weighted fusion run'ında
gözlendi:

| Matched model | Frozen macro-F1 | Fine-tuned macro-F1 | Gain |
| --- | ---: | ---: | ---: |
| `r50+mnv2+effb0 weighted` | `0.519` | `0.687` | `+0.168` |
| `r50+effb0 weighted` | `0.532` | `0.698` | `+0.166` |
| `r50+mnv2 concat` | `0.543` | `0.685` | `+0.142` |
| `r50+mnv2 weighted` | `0.520` | `0.651` | `+0.131` |
| `r50+mnv2+effb0 concat` | `0.576` | `0.706` | `+0.130` |

Weighted fusion en iyi genel modeli üretmemiştir; buna rağmen fine-tuning sonrası weighted
konfigürasyonların da belirgin biçimde iyileştiği görülür. En iyi weighted Sprint 4 modeli
`r50+effb0 weighted` olup macro-F1 `0.698` değerine ulaşmıştır.

## Per-Class Bulgular

En iyi fine-tuned model olan `r50+mnv2+effb0 concat`, matched frozen three-backbone concat modeline
göre tüm sınıflarda F1 artışı göstermiştir:

| Class | Support | Frozen F1 | Fine-tuned F1 | Gain |
| --- | ---: | ---: | ---: | ---: |
| `vasc` | `21` | `0.636` | `0.850` | `+0.214` |
| `bcc` | `71` | `0.532` | `0.726` | `+0.194` |
| `df` | `20` | `0.476` | `0.647` | `+0.171` |
| `akiec` | `52` | `0.500` | `0.636` | `+0.136` |
| `mel` | `167` | `0.444` | `0.550` | `+0.106` |
| `bkl` | `167` | `0.560` | `0.627` | `+0.067` |
| `nv` | `1004` | `0.885` | `0.905` | `+0.020` |

Bu tablo fine-tuning kazanımının yalnızca çoğunluk sınıfı `nv` ile sınırlı olmadığını gösterir.
Bununla birlikte `df` ve `vasc` sınıflarının test support'u sırasıyla yalnızca `20` ve `21` olduğu
için bu sınıflardaki F1 değişimleri yüksek belirsizlikle yorumlanmalıdır. Birkaç örneklik doğru veya
yanlış tahmin farkı bu sınıfların F1 skorunu belirgin biçimde değiştirebilir.

## Learned Fusion Weights

Weighted fusion run'larında öğrenilen backbone ağırlıkları tek bir backbone'a çökmedi:

| Weighted model | Learned weights |
| --- | --- |
| `r50+mnv2 weighted` | ResNet50 `0.539`, MobileNetV2 `0.461` |
| `r50+effb0 weighted` | ResNet50 `0.507`, EfficientNetB0 `0.493` |
| `mnv2+effb0 weighted` | MobileNetV2 `0.509`, EfficientNetB0 `0.491` |
| `r50+mnv2+effb0 weighted` | ResNet50 `0.388`, MobileNetV2 `0.299`, EfficientNetB0 `0.313` |

Bu dağılım, weighted fusion'ın her backbone'dan sinyal kullandığını gösterir. Ancak concatenation
fusion, tam feature vektörlerini koruduğu için Sprint 4'te en güçlü final operator olarak kalmıştır.

## Literatürle Konumlandırma

Sprint 4 sonucu literatürdeki yüksek HAM10000 skorlarıyla doğrudan "daha iyi/daha kötü" şeklinde
okunmamalıdır. Hu ve Yang (2023), Roy et al. (2024) ve Haque et al. (2026) gibi çalışmalar daha
yüksek accuracy/F1 değerleri raporlar; ancak bunlar çoğunlukla özel attention modülleri, custom
mimariler, daha ağır augmentation/balancing stratejileri veya farklı backbone aileleri kullanır.
Birçok güncel preprintte split ayrıntısı ve lesion-level leakage kontrolü de bizim protokolümüz
kadar açık değildir. Bu nedenle Sprint 4'ün güçlü tarafı state-of-the-art iddiası değil, aynı
lesion-aware split üzerinde frozen, fine-tuned, concat ve weighted fusion koşullarını kontrollü
karşılaştırmasıdır.

Bu bağlamda Sprint 4'ün macro-F1 `0.706`, weighted-F1 `0.813` ve accuracy `0.811` sonucu kötü bir
benchmark sonucu değildir. Liu et al. (2024), pretrained model ensemble'larında yaklaşık `0.82-0.83`
accuracy bandına çıktıklarını raporlar; bizim en iyi canonical fine-tuned concat modelimiz accuracy
olarak bu banda yaklaşır, fakat macro-F1'i ayrıca raporladığı için sınıf dengesizliğini daha açık
gösterir. Haque et al. (2026) gibi daha yeni çalışmalar macro-F1 de raporladığından iyi bir güncel
bağlam sağlar; ancak EfficientNetV2-L/attention/XAI gibi daha güçlü ve farklı bir sistem kullandığı
için birebir baseline değildir.

Fusion yorumu da literatürle uyumludur. Roy et al. (2024) ve Mahbod et al. (2025) farklı temsil
kaynaklarını birleştirmenin faydalı olabileceğini desteklerken, Akter et al. (2023) stacking/fusion
modellerinin güçlü tekil backbonelardan daha düşük kalabildiğini gösterir. Bizim sonuçlarımız bu iki
tarafı birlikte doğrular: fine-tuned üç-backbone concat en iyi sonucu üretmiştir, ancak weighted
fusion ve sonraki fusion diagnostic'leri "daha fazla model" veya "öğrenilen ağırlık" eklemenin tek
başına garanti iyileştirme sağlamadığını göstermiştir.

## Sprint 4A-K Genel Sentez

Sprint 4 sonrasında yapılan A-K extension çizgisi, tek bir skoru kovalamaktan çok hangi müdahalenin
hangi tür katkı verdiğini anlamak için kullanıldı. Ana performans iddiası hâlâ validation-gated ve
test-audit protokolüne göre seçilen Sprint 4D TTA sonucuna dayanmalıdır; diğer deneyler bu sonucu
açıklayan veya sınırlarını gösteren ablation/diagnostic çalışmalarıdır.

| Deney | En güçlü raporlanabilir koşu | Accuracy | Macro-F1 | Weighted-F1 | Rapor yorumu |
|---|---|---:|---:|---:|---|
| Sprint 4 canonical | Three-backbone concat | `0.811` | `0.706` | `0.813` | Fine-tuning + concat, frozen fusion'a göre ana kazanım. |
| Sprint 4B | Class-aware `r50+mnv2 concat` | `0.821` | `0.695` | `0.816` | Class-aware loss bazı modelleri iyileştirdi ama canonical best'i geçmedi. |
| Sprint 4C | Validation-selected weighted fusion | `0.802` | `0.699` | `0.808` | MLP/fusion tuning etkili, fakat test-selected olmayan adayları best saymıyoruz. |
| Sprint 4D | Weighted fusion + `tta_rot4` | `0.815` | `0.733` | `0.818` | En iyi genel sonuç; inference maliyeti artıyor. |
| Sprint 4E | `concat_standardize_base` | `0.790` | `0.691` | `0.798` | Fusion davranışını açıkladı; skor artışı getirmedi. |
| Sprint 4F | Augmented three-backbone concat | `0.786` | `0.645` | `0.790` | Train-time augmentation bu pipeline'da feature space'i zayıflattı. |
| Sprint 4G | Uniform soft-vote ensemble | `0.806` | `0.707` | `0.812` | Ensemble sinyali çok küçük; ana sonucu değiştirmiyor. |
| Sprint 4H | Targeted three-backbone concat | `0.768` | `0.643` | `0.779` | Minority recall artsa da genel denge bozuldu. |
| Sprint 4I | Weighted fusion + `tta_d4_8` | `0.813` | `0.727` | `0.816` | TTA faydasını doğruladı; daha fazla view rot4'ü geçmedi. |
| Sprint 4J | Cached-feature balanced concat | `0.804` | `0.690` | `0.809` | MLP-stage balanced sampling genelleme sağlamadı. |
| Sprint 4K | ResNet50 image-level balanced sampler | `0.756` | `0.657` | `0.772` | Image-level oversampling de tek başına yeterli olmadı. |

Bu tablo raporun tartışma bölümünde üç ana mesajı destekler. Birincisi, macro-F1 ile weighted-F1
birlikte okunmalıdır: weighted-F1 çoğunluk sınıfı `nv` nedeniyle daha yüksek kalır, macro-F1 ise az
sınıflardaki hataları görünür yapar. İkincisi, literatürdeki fusion/ensemble başarıları bizim
sonuçlarımızı motive eder ama birebir garanti etmez; bizim kontrollü ablation'larda en güvenilir
fusion operatörü concat, en güvenilir ek iyileştirme ise TTA olmuştur. Üçüncüsü, class-aware loss,
balanced sampling ve train-time augmentation gibi makul fikirler validation'da sinyal verebilse de
lesion-aware test split üzerinde genelleme sağlamadığında ana sonuç olarak sunulmamalıdır.

## Yorum Sınırları

- Sonuçlar single-seed (`seed=42`) çalıştırmalardır. Bu nedenle küçük model farkları kesin mimari
  üstünlük iddiası olarak sunulmamalıdır.
- Macro-F1 ana yorum metriği olarak kalmalıdır. Accuracy ve weighted-F1 çoğunluk sınıfı etkisiyle
  daha yüksek görünebilir.
- `df` ve `vasc` gibi düşük-support sınıflardaki per-class gain'ler raporda dikkatli ve sınırlı bir
  dille anlatılmalıdır.
- Çalışma benchmark image classification sonucudur; klinik tanı, karar desteği veya gerçek hasta
  kullanımı iddiası yapılmamalıdır.
- Büyük checkpoints, feature cache `.pt` dosyaları ve run folder'ları git dışında kalmalıdır.

## Sprint 4B Extension Gerekçesi

Sprint 4, assignment için gerekli frozen-vs-fine-tuned karşılaştırmasını tamamlamıştır. Buna rağmen
weighted-F1 `0.813` seviyesine çıkarken macro-F1 `0.706` seviyesinde kalmıştır. Bu fark, çoğunluk
sınıfı performansının güçlü olduğunu fakat balanced/minority-class performansında hâlâ iyileştirme
alanı bulunduğunu gösterir.

Bu nedenle Sprint 4B, Sprint 4'ün yerine geçmeyecek ayrı bir exploratory extension olarak
planlanmıştır. Amaç, aynı cached-feature MLP/fusion protokolünü koruyarak class-aware fine-tuning
objective'inin macro-F1 ve minority-class davranışını iyileştirip iyileştirmediğini test etmektir.
İlk aşama `finetuned_classaware` screening olacaktır; deeper unfreezing yalnızca ResNet50 üzerinde
kontrollü bir probe olarak denenmelidir.
