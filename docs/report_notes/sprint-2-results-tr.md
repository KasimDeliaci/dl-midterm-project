# Sprint 2 Sonuç Yorumu: Frozen Feature Extraction ve Single-Backbone MLP

Bu not, Sprint 2 sonunda lokalde üretilen artifact'lere dayanır. Colab'da üretilen smoke-test çıktıları bu değerlendirmeye dahil edilmemiştir. Ana kaynaklar:

- `artifacts/report_assets/tables/single_backbone_frozen_results.csv`
- `artifacts/report_assets/tables/single_backbone_frozen_per_class_f1.csv`
- `artifacts/report_assets/tables/mlp_hparam_search/mlp_hparam_v1_full/mlp_hyperparam_search_results.csv`
- `artifacts/report_assets/tables/mlp_hparam_search/mlp_hparam_v1_full/mlp_hyperparam_best_by_backbone.csv`
- `artifacts/report_assets/tables/mlp_hparam_search/mlp_hparam_v1_full/mlp_hyperparam_per_class_f1.csv`
- `artifacts/report_assets/figures/frozen_single_backbone_f1.png`
- `artifacts/report_assets/figures/mlp_hparam_search/mlp_hparam_v1_full/`

Bu sprintin amacı klinik tanı iddiası üretmek değil, HAM10000 üzerinde benchmark dermoscopic image classification deneyleri için sağlam bir frozen-feature baseline oluşturmaktır.

## Ne Yapıldı?

Sprint 2'de üç ImageNet-pretrained CNN backbone'u classifier head'leri çıkarılarak frozen feature extractor olarak kullanıldı: ResNet50, MobileNetV2 ve EfficientNetB0. HAM10000 görüntüleri Sprint 1'de üretilmiş lesion-aware train/validation/test split'leriyle yüklendi. Preprocessing bütün backbone'lar için TorchVision pretrained weight beklentileriyle uyumlu tutuldu: RGB, 224x224 resize ve ImageNet normalization.

Backbone ağırlıkları donduruldu. Bu nedenle CNN'ler bu sprintte HAM10000 üzerinde fine-tune edilmedi; sadece sabit görsel temsil vektörleri üretildi. Her görüntü için backbone embedding'i cache'e yazıldı. Daha sonra bu cache'lenmiş feature tensor'ları üzerinden küçük bir MLP classifier eğitildi. MLP mimarisi, input feature dimension'a bağlı olarak `input_dim -> 512 -> 256 -> 7` yapısındadır. Ara katmanlarda linear layer, batch normalization, ReLU ve dropout kullanıldı. Default dropout `0.3`, default optimizer AdamW, learning rate `0.001`, weight decay `0.0001`, seed `42` ve early stopping metriği validation macro-F1 olarak ayarlandı.

Feature boyutları beklenen mimari farklarını yansıtır:

- ResNet50: 2048-dimensional global average pooled feature
- MobileNetV2: 1280-dimensional pooled feature
- EfficientNetB0: 1280-dimensional pooled feature

ResNet50 feature dimension'ın daha büyük olması modelin son convolutional stage kanal sayısından kaynaklanır. Bu durum ResNet50'nin "iki kat daha iyi" olduğu anlamına gelmez; yalnızca MLP'ye daha geniş bir temsil vektörü verdiği anlamına gelir.

## Ana Single-Backbone Baseline Sonuçları

İlk frozen single-backbone baseline setinde class weighting açık tutuldu. Bu, HAM10000'in ciddi class imbalance yapısı nedeniyle özellikle macro-F1 yorumunu daha anlamlı kılmak için seçildi. Class weighting yalnızca train split üzerinden hesaplandı; validation/test dağılımından ağırlık öğrenilmedi.

| Backbone | Feature dim | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| ResNet50 | 2048 | 0.712 | 0.531 | 0.730 |
| EfficientNetB0 | 1280 | 0.697 | 0.506 | 0.720 |
| MobileNetV2 | 1280 | 0.668 | 0.468 | 0.692 |

Bu ilk karşılaştırmada ResNet50 en iyi macro-F1 skorunu verdi. EfficientNetB0 ikinci sırada kaldı; MobileNetV2 ise daha hafif backbone olmasına rağmen bu frozen-feature kurulumunda daha zayıf temsil üretti. Accuracy ve weighted-F1 değerleri macro-F1'den belirgin şekilde yüksek görünüyor. Bu beklenen bir sonuçtur çünkü test split içinde `nv` sınıfı çok baskındır. Weighted-F1, sınıf support'larıyla ağırlıklandığı için çoğunluk sınıfındaki iyi performanstan daha fazla etkilenir. Macro-F1 ise her sınıfa eşit ağırlık verir ve az örnekli sınıflardaki hataları daha görünür yapar.

Bu nedenle Sprint 2 için ana yorum metriği macro-F1 olmalıdır. Accuracy tek başına yanıltıcıdır; yüksek accuracy çoğunluk sınıfını iyi tahmin etmekten gelebilir.

## Sınıf Bazlı Gözlemler

Sınıf bazında en belirgin ayrım `nv` ile az örnekli sınıflar arasındadır. `nv` sınıfı bütün modellerde en güçlü sınıftır:

- ResNet50 `nv` F1: 0.857
- MobileNetV2 `nv` F1: 0.827
- EfficientNetB0 `nv` F1: 0.849

Bu sonuç şaşırtıcı değildir; `nv` hem train hem test dağılımında en yüksek support'a sahiptir. Model, çoğunluk sınıfından çok daha fazla örnek gördüğü için bu sınıfta daha istikrarlı karar sınırı öğrenir.

`vasc` sınıfı az örnekli olmasına rağmen single-backbone baseline'larda görece iyi sonuç verdi:

- ResNet50 `vasc` F1: 0.720
- EfficientNetB0 `vasc` F1: 0.667
- MobileNetV2 `vasc` F1: 0.489

Bu muhtemelen `vasc` örneklerinin dermoscopic görsel paterni bakımından diğer bazı sınıflardan daha ayırt edilebilir olmasına bağlıdır. Ancak test support sadece 21 olduğu için bu sınıfta skorlar yüksek varyanslı yorumlanmalıdır.

En zayıf sınıf `df` oldu:

- ResNet50 `df` F1: 0.250
- MobileNetV2 `df` F1: 0.308
- EfficientNetB0 `df` F1: 0.154

`df` test support'u yalnızca 20 olduğu için bu sınıfta hem öğrenme hem ölçüm belirsizliği yüksektir. Sprint 1'de de görüldüğü gibi `df` split oranı lesion-aware grouping nedeniyle ideal 70/15/15 oranından sapmıştı. Bu leakage prevention açısından kabul edilen bir tradeoff'tu; Sprint 2 sonuçlarında da `df` performansının oynak ve düşük olması bu bağlamla uyumludur.

`mel`, `bkl`, `bcc` ve `akiec` sınıfları orta-düşük performans bandında kaldı. Özellikle `mel` sınıfı benchmark açısından önemlidir çünkü majority class olmayan ama görsel olarak karışabilen bir sınıftır. Baseline `mel` F1 skorları yaklaşık 0.407-0.448 aralığındadır. Bu, frozen ImageNet feature'larının HAM10000 domain-specific ayrımlarını sınırlı ölçüde yakalayabildiğini gösterir.

## Class Weighting ve Hyperparameter Search Gözlemleri

Sprint 2b'deki lokal hyperparameter search, ana Sprint 2 baseline'ı değiştirmekten çok MLP training ayarlarının etkisini anlamak için kullanıldı. Denenen değişkenler:

- class weighting açık/kapalı
- dropout: 0.1, 0.3, 0.5
- optimizer: AdamW, Adam, SGD momentum
- backbone: ResNet50, MobileNetV2, EfficientNetB0

Backbone başına en iyi lokal hyperparameter sonuçları:

| Backbone | En iyi ayar | Class weighting | Optimizer | Dropout | Accuracy | Macro-F1 | Weighted-F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| EfficientNetB0 | `cw_sgd_d03` | True | SGD | 0.3 | 0.692 | 0.560 | 0.718 |
| ResNet50 | `nocw_adamw_d03` | False | AdamW | 0.3 | 0.748 | 0.543 | 0.746 |
| MobileNetV2 | `cw_adamw_d03` | True | AdamW | 0.3 | 0.668 | 0.468 | 0.692 |

Bu aramada en yüksek test macro-F1, EfficientNetB0 + class weighting + SGD momentum kombinasyonundan geldi: macro-F1 0.560. Bu sonuç, ilk default baseline'daki ResNet50 üstünlüğünün mutlak olmadığını gösterir. MLP training ayarları değiştiğinde EfficientNetB0 daha iyi macro-F1 verebilmektedir.

Class weighting etkisi tek yönlü değildir. Genel örüntü şöyledir:

- Class weighting açıkken minority class recall genellikle artar.
- Bunun bedeli çoğu zaman precision ve overall accuracy tarafında düşüştür.
- Class weighting kapalıyken model çoğunluk sınıfına daha fazla yaslanır; accuracy ve weighted-F1 yükselebilir, fakat macro-recall ve az örnekli sınıfların yakalanması zayıflayabilir.

ResNet50 için en iyi hyperparameter run class weighting kapalıyken geldi. Bu run accuracy 0.748 ve weighted-F1 0.746 ile güçlüdür; `nv` F1 0.887'ye çıkmıştır. Fakat macro-recall 0.546 seviyesindedir. Buna karşılık EfficientNetB0 + class weighting + SGD run'ı accuracy olarak daha düşük olsa da macro-recall 0.618'e çıkar ve macro-F1 0.560 ile en iyi dengeyi verir. Bu yüzden proje hedefi macro-F1 ve class-balanced yorum ise EfficientNetB0 + weighted SGD dikkate değer bir adaydır; accuracy odaklı yorum yapılacaksa ResNet50 no-class-weight AdamW daha güçlü görünür.

## Sonuçlar İyi mi, Beklenen mi?

Bu sprint için sonuçlar beklenen aralıktadır. Frozen ImageNet feature extraction, HAM10000 gibi domain-specific ve imbalanced bir dermoscopic dataset üzerinde genellikle güçlü ama sınırlı bir baseline üretir. CNN backbone'lar ImageNet'te doğal görüntüler üzerinde öğrenildiği için genel kenar, doku ve renk temsilleri taşır; ancak dermatolojik sınıflar arasındaki ince görsel farklara doğrudan optimize edilmemiştir. Bu nedenle macro-F1'in 0.47-0.56 bandında kalması şaşırtıcı değildir.

Accuracy'nin 0.69-0.76 bandına çıkması da tek başına modelin iyi dengelendiği anlamına gelmez. HAM10000 test split'inde `nv` support'u 1004 iken bazı sınıfların support'u 20-71 aralığındadır. Bu dağılımda majority class başarısı overall accuracy'yi yukarı taşır. Bu yüzden raporda "best model" seçimi yapılırken primary metric olarak macro-F1 kullanılmalıdır.

Sprint 2'nin asıl değeri, nihai performans rekoru kırmasından çok şudur:

- Veri sızıntısı olmayan lesion-aware split üstünde ölçüm yapıldı.
- Üç backbone için tekrarlanabilir frozen feature cache üretildi.
- MLP classifier pipeline'ı cache üzerinden hızlı deney yapılabilir hale geldi.
- Macro-F1, per-class F1 ve confusion matrix çıktıları rapora hazır üretildi.
- Class weighting'in ve optimizer/dropout seçimlerinin metric tradeoff'ları görünür hale geldi.

Bu altyapı Sprint 3 için doğrudan kullanılabilir. Pairwise fusion ve three-backbone fusion deneylerinde artık soru şudur: birden fazla frozen representation birleştirildiğinde az örnekli ve görsel olarak karışan sınıfların macro-F1'i anlamlı şekilde iyileşiyor mu?

## Rapor İçin Ana Mesaj

Raporun Results bölümünde vurgulanması gereken ana bulgu: default single-backbone frozen baseline'da ResNet50 en yüksek macro-F1'i üretmiştir, ancak hyperparameter search EfficientNetB0'nun class-weighted SGD ile daha iyi macro-F1 dengesine ulaşabildiğini göstermiştir. Bu, backbone karşılaştırmasının yalnızca feature extractor mimarisine değil, classifier training ayarlarına da duyarlı olduğunu gösterir.

Discussion bölümünde vurgulanması gereken ana yorum: high-level accuracy ve weighted-F1 çoğunluk sınıfı nedeniyle iyimser görünebilir; macro-F1 ve per-class F1, HAM10000 için daha doğru yorum yüzeyi sağlar. `df`, `mel`, `akiec`, `bcc` ve `bkl` sınıflarındaki sınırlı performans, frozen ImageNet feature'larının domain-specific ayrımlar için yeterli olmadığını ve sonraki sprintlerde fusion/fine-tuning denemelerinin gerekçesini oluşturduğunu gösterir.

## Report-Ready Artifact List

Sprint 2 raporuna doğrudan bağlanabilecek küçük artifact'ler:

- `artifacts/report_assets/tables/single_backbone_frozen_results.csv`
- `artifacts/report_assets/tables/single_backbone_frozen_per_class_f1.csv`
- `artifacts/report_assets/figures/frozen_single_backbone_f1.png`
- `artifacts/report_assets/tables/mlp_hparam_search/mlp_hparam_v1_full/mlp_hyperparam_best_by_backbone.csv`
- `artifacts/report_assets/tables/mlp_hparam_search/mlp_hparam_v1_full/mlp_hyperparam_search_results.csv`
- `artifacts/report_assets/tables/mlp_hparam_search/mlp_hparam_v1_full/mlp_hyperparam_per_class_f1.csv`
- `artifacts/report_assets/figures/mlp_hparam_search/mlp_hparam_v1_full/mlp_hyperparam_search_macro_f1.png`
- `artifacts/report_assets/figures/mlp_hparam_search/mlp_hparam_v1_full/best_resnet50_nocw_adamw_d03_confusion_matrix.png`
- `artifacts/report_assets/figures/mlp_hparam_search/mlp_hparam_v1_full/best_resnet50_nocw_adamw_d03_training_curve.png`
- `artifacts/report_assets/figures/mlp_hparam_search/mlp_hparam_v1_full/best_efficientnet_b0_cw_sgd_d03_confusion_matrix.png`
- `artifacts/report_assets/figures/mlp_hparam_search/mlp_hparam_v1_full/best_efficientnet_b0_cw_sgd_d03_training_curve.png`
- `artifacts/report_assets/figures/mlp_hparam_search/mlp_hparam_v1_full/best_mobilenet_v2_cw_adamw_d03_confusion_matrix.png`
- `artifacts/report_assets/figures/mlp_hparam_search/mlp_hparam_v1_full/best_mobilenet_v2_cw_adamw_d03_training_curve.png`

