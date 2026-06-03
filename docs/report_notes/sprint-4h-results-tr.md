# Sprint 4H Sonuç Notları: Targeted Fine-Tuning

Sprint 4H, Sprint 4F'teki agresif train-time augmentation düşüşünden sonra daha hedefli bir
fine-tuning hipotezini test etti. Deneyde ayrı bir `finetuned_targeted` feature source üretildi;
ResNet50 için `layer3 + layer4`, MobileNetV2 ve EfficientNetB0 için son feature blokları
fine-tune edildi. Loss tarafında class-balanced focal loss kullanıldı ve augmentation politikası
crop/affine içermeyecek şekilde sınırlandı.

## Artifact Kontrolü

Colab T4 koşusundan lokal projeye şu operasyonel artifactler alındı:

- 3 backbone checkpoint dosyası;
- 9 fine-tuned feature cache `.pt` dosyası;
- 11 MLP/fusion `model.pt` dosyası;
- 14 `metrics.json` dosyası;
- 4H'ye özgü report-ready CSV ve PNG çıktıları.

Büyük checkpoint, feature cache ve MLP `model.pt` dosyaları git dışında tutulur. Küçük CSV/PNG
report assetleri rapor üretimi için uygundur.

## Ana Sonuç

En iyi Sprint 4H cached-feature matrix sonucu:

| Koşu | Test macro-F1 | Accuracy | Weighted-F1 |
|---|---:|---:|---:|
| ResNet50 + MobileNetV2 + EfficientNetB0 concat | `0.643` | `0.768` | `0.779` |

Karşılaştırma:

| Deney | Test macro-F1 |
|---|---:|
| Sprint 4D weighted + `tta_rot4` | `0.733` |
| Sprint 4G ensemble | `0.707` |
| Canonical Sprint 4 three-backbone concat | `0.706` |
| Sprint 4F augmented concat | `0.645` |
| Sprint 4H targeted concat | `0.643` |
| Sprint 3 frozen best fusion | `0.595` |

Sprint 4H, canonical Sprint 4 ve Sprint 4D TTA sonucunu geçmedi. Bu nedenle yeni ana sonuç olarak
değil, daha hedefli fine-tuning stratejisinin beklenen genelleme artışını sağlamadığı kontrollü bir
negatif deney olarak raporlanmalıdır.

## Sınıf Bazlı Yorum

En iyi 4H matrix koşusunda az örnekli sınıflar için recall belirgin şekilde yükseldi:

| Sınıf | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| akiec | `0.462` | `0.692` | `0.554` | 52 |
| bcc | `0.550` | `0.775` | `0.643` | 71 |
| df | `0.542` | `0.650` | `0.591` | 20 |
| vasc | `0.621` | `0.857` | `0.720` | 21 |

Ancak bu recall artışı genel accuracy ve weighted-F1 tarafında düşüşle birlikte geldi. Özellikle
çoğunluk sınıfı `nv` için F1 `0.879` seviyesine inse de macro-F1 toplamda canonical Sprint 4'ün
altında kaldı.

## Teknik Yorum

Image-level fine-tuning head tarafında ResNet50 targeted policy test macro-F1 `0.647` üretti. Bu,
targeted fine-tuning'in ResNet50 üzerinde anlamlı bir sinyal taşıdığını gösteriyor. Fakat aynı
backbone'lardan çıkarılan feature cache'ler ile yapılan MLP/fusion matrix bu sinyali yeni bir best
sonuca dönüştüremedi. Bu durum, class-balanced focal loss ve daha derin ResNet50 unfreeze
politikasının az sınıf recall'ını artırırken cached-feature fusion temsillerinin genel dengesini
zayıflatmış olabileceğini düşündürür.

Final raporda Sprint 4H, literatüre yaklaşmak için denenmiş kontrollü bir training-policy extension
olarak konumlandırılabilir. Ana performans iddiası hâlâ Sprint 4D TTA sonucuna dayanmalıdır.

## Literatürle Bağlantı

Haque et al. (2026) ve Hu ve Yang (2023) gibi daha yüksek HAM10000 sonuçları raporlayan çalışmalar,
daha güçlü EfficientNet varyantları, attention modülleri, class-aware training veya explainability
eklentileriyle gelir. Sprint 4H bu literatür yönünü daha kontrollü bir şekilde denedi: ResNet50'de
daha fazla block açıldı, loss class-balanced focal loss'a çevrildi ve augmentation Sprint 4F'e göre
daha sınırlı tutuldu.

Sonucun canonical Sprint 4'ün altında kalması, literatürdeki yüksek skorların tek bir bileşenden
gelmediğini gösterir. Daha fazla layer açmak ve focal loss kullanmak az sınıfların recall'ını
artırabilir; fakat aynı anda majority class precision/weighted-F1 dengesini ve feature fusion
tamamlayıcılığını bozabilir. Sprint 4H bu trade-off'u görünür kıldığı için değerlidir: minority
recall artışı tek başına final macro-F1 veya weighted-F1 artışı garantilemez.
