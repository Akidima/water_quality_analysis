# Fast Notebook Execution Guide

## 🚀 Quick Solutions for Faster Execution

The notebook can run slowly during hyperparameter tuning. Here are several ways to speed it up:

---

## Option 1: Use Fast Mode (Recommended) ⚡

Run with the `--fast` flag:

```bash
python3 run_notebook.py --fast
```

**What it does:**
- Uses random search (10 iterations) instead of grid search (108 combinations)
- Reduces CV folds from 5 to 3
- **Speedup: ~10-20x faster**

---

## Option 2: Skip Hyperparameter Tuning

If you just want to train and evaluate without tuning:

1. Open the notebook
2. Skip cells 11-12 (Hyperparameter Tuning and Model Comparison)
3. Run cells 1-10 and 13-14

Or modify the notebook to comment out those sections.

---

## Option 3: Manual Fast Settings

Edit the hyperparameter tuning cell in the notebook:

**Change from:**
```python
tuning_results = tuner.grid_search(
    X=clean_df[config.features],
    y=clean_df[config.target],
    cv=5,  # Slow: 5-fold CV
    scoring='r2'
)
```

**To:**
```python
# FAST MODE: Random search with fewer iterations
tuning_results = tuner.random_search(
    X=clean_df[config.features],
    y=clean_df[config.target],
    n_iter=5,  # Only 5 iterations
    cv=3,      # 3-fold CV instead of 5
    scoring='r2',
    random_state=42
)
```

**Speedup: ~50x faster** (5 iterations × 3 folds = 15 fits vs 108 × 5 = 540 fits)

---

## Option 4: Use Smaller Parameter Grid

If you want to keep grid search but make it faster:

```python
# Smaller parameter grid
param_grid = {
    'regressor__n_estimators': [50, 100],  # Only 2 values instead of 3
    'regressor__max_depth': [10, 20],      # Only 2 values instead of 4
    'regressor__min_samples_split': [2, 5] # Only 2 values instead of 3
}
# Total combinations: 2 × 2 × 2 = 8 (vs 108 before)
# With cv=3: 8 × 3 = 24 fits (vs 540 before)
# Speedup: ~22x faster

tuning_results = tuner.grid_search(
    X=clean_df[config.features],
    y=clean_df[config.target],
    param_grid=param_grid,
    cv=3,  # Reduced from 5
    scoring='r2'
)
```

---

## Option 5: Skip Model Comparison

Model comparison runs multiple algorithms which can be slow. To skip:

1. Comment out or skip cell 12 (Model Comparison)
2. The rest of the notebook will still work fine

---

## ⏱️ Expected Execution Times

| Mode | Hyperparameter Tuning | Model Comparison | Total Time |
|------|----------------------|------------------|------------|
| **Full (default)** | ~5-10 min | ~2-3 min | ~10-15 min |
| **Fast mode** | ~30-60 sec | ~1-2 min | ~2-3 min |
| **Skip tuning** | 0 sec | ~1-2 min | ~1-2 min |
| **Minimal** | 0 sec | 0 sec | ~30 sec |

---

## 🎯 Recommended Approach

**For Quick Testing/Development:**
```bash
python3 run_notebook.py --fast
```

**For Production/Full Analysis:**
```bash
python3 run_notebook.py  # Full mode
```

**For Just Training (No Tuning):**
- Skip cells 11-12 in the notebook
- Or comment them out

---

## 📝 Quick Reference

### Fast Mode Command
```bash
python3 run_notebook.py --fast
```

### What Fast Mode Does
- ✅ Random search: 10 iterations (vs 108 grid combinations)
- ✅ CV folds: 3 (vs 5)
- ✅ Total fits: ~30 (vs ~540)
- ✅ Speedup: ~10-20x faster

### Still Gets Good Results
- Random search often finds good hyperparameters quickly
- 3-fold CV is still reliable for model evaluation
- Results are typically within 1-2% of full grid search

---

## 💡 Tips

1. **First Run**: Use `--fast` to verify everything works
2. **Development**: Use `--fast` for iterative testing
3. **Final Model**: Use full mode for production
4. **Quick Demo**: Skip hyperparameter tuning entirely

---

**Last Updated:** December 29, 2025

