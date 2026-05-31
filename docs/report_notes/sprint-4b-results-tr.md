# Sprint 4B Sonuç Notu: Class-Aware Fine-Tuning Extension

Bu not Sprint 4B için sonuç yorumu iskeletidir. Şu anda implementasyon hazırdır, fakat Colab GPU
üzerinde full class-aware fine-tuning ve screening run'ları henüz bu lokalde sonuçlanmış artifact
olarak bulunmamaktadır. Bu nedenle aşağıdaki metin sonuç iddiası değil, çalıştırma ve yorumlama
çerçevesidir.

## Amaç

Sprint 4B, canonical Sprint 4'ün yerine geçmez. Canonical Sprint 4 feature source'u
`finetuned` olarak kalır. Sprint 4B iki ayrı exploratory source kullanır:

- `finetuned_classaware`: canonical Sprint 4 unfreeze policy aynı kalır, loss class-balanced focal
  loss olur.
- `finetuned_deeper`: sadece ResNet50 için `layer3 + layer4 + fc` deeper partial unfreezing probe.

Ana soru, class-aware fine-tuning'in validation macro-F1 ve minority-class davranışını canonical
Sprint 4'e göre iyileştirip iyileştirmediğidir. Test metric model veya hyperparameter seçiminde
kullanılmamalıdır.

## Çalıştırılacak Komutlar

Class-aware backbone fine-tuning ve feature cache üretimi:

```bash
uv run python scripts/finetune_backbone.py \
  --config configs/experiments/sprint4b_classaware_backbones.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_classaware \
  --batch-size 32
```

Class-aware single-backbone MLP screening:

```bash
uv run python scripts/train_mlp.py \
  --config configs/experiments/sprint4b_classaware_feature_matrix.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_classaware
```

Deeper ResNet50 probe:

```bash
uv run python scripts/finetune_backbone.py \
  --config configs/experiments/sprint4b_deeper_screen.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_deeper \
  --backbone resnet50 \
  --batch-size 32

uv run python scripts/train_mlp.py \
  --config configs/experiments/sprint4b_deeper_screen.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_deeper
```

Screening asset refresh:

```bash
uv run python scripts/make_report_assets.py \
  --feature-source finetuned_classaware \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-root artifacts/features
```

Full class-aware matrix yalnızca validation stop/go geçerse çalıştırılmalıdır:

```bash
uv run python scripts/run_experiment_matrix.py \
  --config configs/experiments/sprint4b_classaware_feature_matrix.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_classaware
```

## Beklenen Artifact'ler

Screening sonrası beklenen küçük report-ready çıktılar:

- `artifacts/report_assets/tables/sprint4b_screening_results.csv`
- `artifacts/report_assets/tables/sprint4b_vs_canonical_single_backbone.csv`
- `artifacts/report_assets/tables/sprint4b_per_class_f1_gain.csv`
- `artifacts/report_assets/figures/sprint4b_val_macro_f1_screening.png`
- `artifacts/report_assets/figures/sprint4b_test_macro_f1_vs_canonical.png`
- `artifacts/report_assets/figures/sprint4b_per_class_f1_gain_heatmap.png`

Full class-aware matrix çalışırsa ek beklenen çıktılar:

- `artifacts/report_assets/tables/sprint4b_classaware_all_results.csv`
- `artifacts/report_assets/tables/sprint4b_classaware_vs_canonical_fusion_summary.csv`
- `artifacts/report_assets/tables/sprint4b_classaware_fusion_weight_summary.csv`
- `artifacts/report_assets/figures/sprint4b_classaware_fusion_comparison.png`
- `artifacts/report_assets/figures/sprint4b_classaware_concat_vs_weighted.png`
- `artifacts/report_assets/figures/sprint4b_classaware_learned_fusion_weights.png`
- `artifacts/report_assets/figures/sprint4b_best_confusion_matrix.png`

## Yorumlama Kuralları

- Expansion kararı validation macro-F1 ile verilir, test metric ile değil.
- En iyi class-aware single-backbone validation macro-F1, matched canonical Sprint 4 single'dan
  en az yaklaşık `0.015` iyi olursa full matrix düşünülebilir.
- Alternatif olarak üç backbone'dan en az ikisi matched canonical'a göre yaklaşık `0.010`
  validation macro-F1 gain gösterirse full matrix düşünülebilir.
- Accuracy veya weighted-F1 yaklaşık `0.03` veya daha fazla çökerse genişletme yapılmamalıdır.
- Minority recall artarken precision collapse varsa sonuç olumlu sayılmamalıdır.
- `df` ve `vasc` support düşük olduğu için per-class değişimler dikkatli yorumlanmalıdır.

## Sonuç Durumu

Henüz gerçek Sprint 4B sonucu yoktur. Colab run'ları tamamlandığında bu bölüm şu sorularla
doldurulmalıdır:

- Hangi feature cache'ler oluştu?
- Screening validation macro-F1 sonuçları canonical Sprint 4'e göre nasıl değişti?
- Full class-aware matrix çalıştı mı, çalışmadıysa validation-based gerekçe neydi?
- Macro-F1, accuracy, weighted-F1 ve per-class F1 tarafında tradeoff var mı?
- Büyük checkpoint/cache/run artifact'leri git dışında kaldı mı?
