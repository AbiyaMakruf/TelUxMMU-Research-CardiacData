# Fix Summary: PermissionError & Epoch Changes

## Issues Fixed

### 1. PermissionError in File Operations (Windows)
**Problem:** When saving temporary matplotlib files, Windows kept the files locked during the `os.rename()` operation, causing `PermissionError`.

**Affected Functions:**
- `train_model()` – line 560 (training_history.png)
- `train_model()` – line 572 (confusion_matrix_val.png)  
- `evaluate_and_log()` – line 693 (confusion_matrix_test.png)

**Root Cause:** Using `os.rename()` doesn't close the file handle on Windows before moving it, causing access conflicts.

**Solution Applied:**
```python
# OLD (broken on Windows):
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    # ... save to tmp.name ...
    os.rename(tmp.name, local_path)  # FAILS: file still locked

# NEW (works on Windows):
tmp_path = os.path.join(tempfile.gettempdir(), f"name_{run_name}.png")
# ... save to tmp_path ...
shutil.copy2(tmp_path, local_path)  # Copy (no lock issues)
os.remove(tmp_path)                  # Delete separately
```

### 2. Epoch Count Change
**Change:** `NUM_EPOCHS = 20` → `NUM_EPOCHS = 3`

**Reason:** Allows quick testing of the entire pipeline before running full training with DagsHub integration.

**File Modified:** `notebook-modeling.ipynb` (cell 9e08bc83)

---

## Files Changed

### `utils/modeling.py`
1. **Line 4:** Added `import shutil`
2. **Lines 555-562:** Fixed training history saving (`train_model()`)
3. **Lines 566-574:** Fixed validation confusion matrix saving (`train_model()`)
4. **Lines 685-694:** Fixed test confusion matrix saving (`evaluate_and_log()`)

**Total Changes:** 3 function fixes, consistent pattern across all artifact saving

### `notebook-modeling.ipynb`
1. **Cell 9e08bc83:** Changed NUM_EPOCHS from 20 to 3

---

## Commits

```
6edfa6f - Fix: PermissionError di evaluate_and_log()
81abd2a - Fix: PermissionError saat rename temp file & ubah NUM_EPOCHS ke 3
```

---

## Verification

✓ All syntax checks passed  
✓ No `os.rename()` patterns remaining  
✓ All `shutil.copy2()` patterns in place  
✓ `import shutil` present in file header  
✓ Code compiles without errors  

---

## How It Works Now

1. **Create temp file:** `tmp_path = os.path.join(tempfile.gettempdir(), f"name_{run_name}.png")`
2. **Save matplotlib figure to temp:** `plt.savefig(tmp_path)` → `plt.close()`
3. **Log to MLflow:** `mlflow.log_artifact(tmp_path, ...)`
4. **Copy to local:** `shutil.copy2(tmp_path, local_path)` (no lock issues)
5. **Clean up:** `os.remove(tmp_path)` (delete temp file)

---

## Next Steps

The notebook is now ready to test:
- With `NUM_EPOCHS=3`, the pipeline should complete without PermissionErrors
- Once verified working, increase `NUM_EPOCHS` back to 20 (or desired value)
- Then run full experiments with DagsHub integration

