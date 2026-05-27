# Sprint 1 Sonuç Yorumu: HAM10000 Dataset Audit ve Lesion-Aware Split

Bu not, Sprint 1 sonunda lokalde üretilen HAM10000 audit ve split artifact'lerine dayanır. Ana kaynaklar:

- `docs/DATASET_AUDIT.md`
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`
- `artifacts/report_assets/tables/class_distribution.csv`
- `artifacts/report_assets/tables/split_class_distribution.csv`
- `artifacts/report_assets/figures/class_distribution.png`
- `artifacts/report_assets/figures/split_class_distribution.png`

Bu proje klinik teşhis iddiası üretmez. Sprint 1'in amacı, sonraki modelleme sprintleri için güvenilir, leakage-free ve raporlanabilir bir benchmark dermoscopic image classification veri temeli oluşturmaktır.

## Ne Yapıldı?

Sprint 1'de HAM10000 dataset'i lokal olarak audit edildi ve train/validation/test split'leri üretildi. Metadata dosyası `data/metadata/HAM10000_metadata.csv` altında okundu; raw image dosyaları `data/raw` altında recursive şekilde image ID ile eşleştirildi. Label alanı HAM10000 `dx` değerlerinden canonical yedi sınıfa normalize edildi:

- `akiec`
- `bcc`
- `bkl`
- `df`
- `mel`
- `nv`
- `vasc`

Audit sırasında image ID tekrarları, missing image dosyaları, missing label durumları ve lesion ID varlığı kontrol edildi. Ardından split üretimi image-level random split yerine lesion-aware grouping ile yapıldı. Bu karar önemlidir çünkü HAM10000 metadata'sında aynı lesion'a ait birden fazla image bulunabilir. Aynı lesion'ın hem train hem test tarafında görünmesi model performansını yapay şekilde iyileştirebilir. Bu nedenle split üretiminde `lesion_id` grupları split sınırlarını geçmeyecek şekilde ayrıldı.

## Audit Sonucu

HAM10000 lokal audit sonucu temizdir:

| Kontrol | Sonuç |
| --- | ---: |
| Metadata rows | 10,015 |
| Unique image IDs | 10,015 |
| Duplicate image IDs | 0 |
| Missing images | 0 |
| Missing labels | 0 |
| Lesion ID available | True |
| Unique lesion IDs | 7,470 |

Bu sonuçlar, kullanılan lokal veri kopyasının standart HAM10000 benchmark büyüklüğüyle uyumlu olduğunu gösterir. Missing image veya missing label bulunmaması, Sprint 2 ve sonrası için dataset kaynaklı blocking problem olmadığını gösterir.

## Sınıf Dağılımı

HAM10000 dağılımı ciddi şekilde dengesizdir:

| Label | Count | Percent |
| --- | ---: | ---: |
| `akiec` | 327 | 3.27 |
| `bcc` | 514 | 5.13 |
| `bkl` | 1,099 | 10.97 |
| `df` | 115 | 1.15 |
| `mel` | 1,113 | 11.11 |
| `nv` | 6,705 | 66.95 |
| `vasc` | 142 | 1.42 |

En büyük sınıf `nv` tüm dataset'in yaklaşık üçte ikisini oluşturur. Buna karşılık `df` ve `vasc` sınıfları çok küçüktür. Bu dengesizlik, sonraki modelleme sprintleri için doğrudan metrik seçimini etkiler. Accuracy tek başına yeterli değildir; model çoğunluk sınıfını iyi tahmin ederek yüksek accuracy alabilir. Bu nedenle Sprint 1 sonunda macro-F1 ve per-class F1'in raporda ana yorum metrikleri olarak kullanılmasına karar verilmiştir.

## Split Sonuçları

Lesion-aware 70/15/15 split üretildi:

| Split | Images | Unique lesion IDs | Image ratio |
| --- | ---: | ---: | ---: |
| Train | 6,981 | 5,229 | 69.71% |
| Validation | 1,532 | 1,120 | 15.30% |
| Test | 1,502 | 1,121 | 15.00% |

Split boyutları hedeflenen 70/15/15 oranına çok yakındır. Toplam image sayısı korunmuştur: 6,981 + 1,532 + 1,502 = 10,015.

Split class counts:

| Label | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| `akiec` | 222 | 53 | 52 |
| `bcc` | 361 | 82 | 71 |
| `bkl` | 772 | 160 | 167 |
| `df` | 71 | 24 | 20 |
| `mel` | 773 | 173 | 167 |
| `nv` | 4,683 | 1,018 | 1,004 |
| `vasc` | 99 | 22 | 21 |

Bu tablo, class imbalance'ın split'lerde de devam ettiğini gösterir. Test split'te `nv` 1,004 örnekle baskın kalırken `df` sadece 20, `vasc` ise 21 örneğe sahiptir. Sonraki sprintlerde per-class sonuçlar yorumlanırken bu support farkları özellikle belirtilmelidir.

## Leakage Kontrolü

Lesion-level leakage kontrolü temizdir:

| Pair | Shared lesion IDs |
| --- | ---: |
| Train / validation | 0 |
| Train / test | 0 |
| Validation / test | 0 |

Bu, Sprint 1'in en önemli sonucudur. Aynı lesion'a ait image'ların farklı split'lere dağılmadığı doğrulanmıştır. Böylece validation ve test sonuçları near-duplicate leakage ile yapay şekilde şişirilmez.

## `df` Split Sapması

`df` sınıfı 71/24/20 olarak ayrılmıştır. Bu yaklaşık olarak:

- Train: 61.74%
- Validation: 20.87%
- Test: 17.39%

Bu dağılım ideal 70/15/15 oranından sapar. Ancak bu bir veri hatası değil, lesion-aware grouping kaynaklı bilinçli bir tradeoff'tur. Küçük sınıflarda lesion grupları az olduğu için hem group isolation hem de tam class-ratio matching aynı anda garanti edilemeyebilir. Bu projede leakage prevention, image-level oranların birebir tutturulmasından daha öncelikli kabul edilmiştir.

Bu karar özellikle raporda net anlatılmalıdır: split oranındaki küçük/orta sapmalar, aynı lesion'ın train ve test'e düşmesini engellemek için kabul edilmiştir.

## Sonuçlar İyi mi, Beklenen mi?

Sprint 1 sonuçları beklenen ve sağlıklıdır. Dataset standart HAM10000 büyüklüğündedir, missing image yoktur, duplicate image ID yoktur, missing label yoktur ve lesion ID bilgisi kullanılabilir durumdadır. Split boyutları hedef oranlara yakındır ve en kritik kontrol olan cross-split lesion leakage sıfırdır.

Class imbalance ise ciddi ve beklenen bir HAM10000 özelliğidir. Bu durum modelleme aşamasında problem değildir, fakat evaluation protokolünü belirler. Sprint 2'de class weighting'in değerlendirilmesi ve macro-F1'in primary interpretation metric olarak seçilmesi doğrudan Sprint 1 bulgularından kaynaklanmıştır.

Bu yüzden Sprint 1'in ana çıktısı "temiz veri + leakage-free split + imbalance-aware evaluation plan" olarak özetlenebilir.

## Rapor İçin Ana Mesaj

Dataset bölümünde vurgulanması gereken ana mesaj: HAM10000 üzerinde modelleme yapmadan önce veri bütünlüğü doğrulanmış, 10,015 image'ın tamamı bulunmuş ve lesion-aware split ile train/validation/test arasında lesion leakage olmadığı gösterilmiştir.

Experimental setup veya evaluation protocol kısmında vurgulanması gereken ana mesaj: Dataset dağılımı ciddi şekilde imbalanced olduğu için accuracy tek başına yeterli değildir. Macro-F1 ve per-class F1, model davranışını daha doğru yorumlamak için ana analiz metrikleri olarak kullanılmalıdır.

Discussion kısmında vurgulanması gereken ana mesaj: Özellikle `df` ve `vasc` gibi küçük sınıflarda metric belirsizliği yüksektir. Bu sınıflardaki F1 değişimleri support düşük olduğu için dikkatli yorumlanmalıdır. Split oranlarındaki küçük sapmalar, leakage prevention tradeoff'u olarak kabul edilmiştir.

## Sprint 2'ye Bağlantı

Sprint 2 frozen feature extraction ve MLP baseline deneyleri bu Sprint 1 split'lerini doğrudan kullanmıştır. Bu bağlantı önemlidir çünkü Sprint 2 sonuçlarının güvenilirliği, Sprint 1'de temizlenen veri ve leakage-free split garantisine dayanır.

Sprint 2'de görülen `df` sınıfı düşük/oynak performansı da Sprint 1 bulgularıyla uyumludur: `df` hem dataset genelinde çok küçük bir sınıftır hem de test split'te sadece 20 örnekle temsil edilir. Bu nedenle Sprint 2 ve sonraki sprintlerde `df` sonuçları raporlanmalı, fakat yüksek belirsizlikle yorumlanmalıdır.

## Report-Ready Artifact List

Sprint 1 raporuna doğrudan bağlanabilecek küçük artifact'ler:

- `artifacts/report_assets/tables/class_distribution.csv`
- `artifacts/report_assets/tables/split_class_distribution.csv`
- `artifacts/report_assets/figures/class_distribution.png`
- `artifacts/report_assets/figures/split_class_distribution.png`
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

Raw images, processed metadata ve büyük generated artifacts git'e alınmamalıdır. Rapor için küçük CSV/PNG özetleri yeterlidir.

