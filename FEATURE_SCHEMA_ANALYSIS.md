# Feature Schema Analysis: RL Data Generation vs Brain Models

## Executive Summary

**CRITICAL FINDING**: There is a **complete mismatch** between feature schemas:
- `generate_rl_data.py` outputs **40 features** with SMC-focused indicators
- `rl_brain.py` expects **24 features** with different technical indicators
- `rl_brain_v2.py` expects **~40 features** (compatible with generate_rl_data.py)

**These are incompatible systems**: `rl_brain.py` cannot consume output from `generate_rl_data.py`.

---

## 1. GENERATE_RL_DATA.PY Feature Schema

**File**: `/root/bot/backtest/generate_rl_data.py`  
**Function**: `extract_features_for_instrument()` (lines 383-861)  
**Total Features**: 40  
**Location**: Feature assembly at lines 800-846

### Feature Order (Exact Index by Position)

#### [0-9] Structure Features (10 features)
```
0  struct_1d              - Daily structure direction (-1, 0, +1)
1  struct_4h              - 4H structure direction
2  struct_1h              - 1H structure direction
3  struct_15m             - 15m structure direction
4  struct_5m              - 5m structure direction
5  decay_1d               - Exponential decay since last break on daily (0-1)
6  decay_4h               - Exponential decay since last break on 4H (0-1)
7  decay_1h               - Exponential decay since last break on 1H (0-1)
8  decay_15m              - Exponential decay since last break on 15m (0-1)
9  decay_5m               - Exponential decay since last break on 5m (0-1)
```

#### [10-12] Bias Features (3 features)
```
10 bias_strong            - Strong bias from daily (0/1 flag)
11 premium_discount       - Discount/premium vs 4H (float)
12 daily_bias             - Alias for struct_1d (duplicate!)
```

#### [13-19] Component Confirmation Features (7 features)
```
13 h4_confirms            - 4H confirms daily bias (0/1 flag)
14 h4_poi                 - 4H POI (FVG/OB) aligned with bias (0/1 flag)
15 h1_confirms            - 1H confirms daily bias (0/1 flag)
16 h1_choch               - 1H CHoCH signal (0/1 flag)
17 has_entry_zone         - 15m entry zone (FVG/OB) exists (0/1 flag)
18 precision_trigger      - 5m precision trigger (0/1 flag)
19 volume_ok              - Volume check passes (0/1 flag)
```

#### [20-33] Market Context Features (14 features)
```
20 ema20_dist_5m          - (close - EMA20) / close on 5m
21 ema50_dist_5m          - (close - EMA50) / close on 5m
22 ema20_dist_1h          - (close - EMA20) / close on 1H
23 ema50_dist_1h          - (close - EMA50) / close on 1H
24 atr_5m_norm            - ATR(14) / close on 5m
25 atr_1h_norm            - ATR(14) / close on 1H
26 atr_daily_norm         - ATR(14) / close on daily
27 rsi_5m                 - RSI(14) / 100 on 5m (normalized to 0-1)
28 rsi_1h                 - RSI(14) / 100 on 1H (normalized to 0-1)
29 volume_ratio           - current_vol / ma_vol(20) on 5m, capped at 5.0
30 adx_1h                 - ADX(14) / 50 on 1H (normalized to 0-1)
31 alignment_score        - Weighted composite of SMC confirmations (0-1)
32 hour_sin               - sin(2π * hour / 24) for time of day
33 hour_cos               - cos(2π * hour / 24) for time of day
```

#### [34-39] Zone Activity Features (6 features)
```
34 fvg_bull_active        - Count of active bullish FVGs on 15m, normalized
35 fvg_bear_active        - Count of active bearish FVGs on 15m, normalized
36 ob_bull_active         - Count of active bullish order blocks on 15m, normalized
37 ob_bear_active         - Count of active bearish order blocks on 15m, normalized
38 liq_above_count        - Count of liquidity levels above on 1H, normalized
39 liq_below_count        - Count of liquidity levels below on 1H, normalized
```

### Data Cleaning in generate_rl_data.py

**At Generation Time**: NO aggressive cleaning
- Lines 337-790: Uses `np.isnan()`, `pd.notna()` checks only for **conditionals**
- Data types: All cast to `np.float32`
- NaN values are **preserved** in the output arrays

**Data Quality Issues NOT Caught**:
- Missing higher timeframe data → zeros filled manually
- Division by zero → checked but not always prevented
- Infinite values possible from ATR/RSI calculations

---

## 2. RL_BRAIN_V2.PY Feature Schema

**File**: `/root/bot/rl_brain_v2.py`  
**Function**: `prepare_features()` (lines 120-127)  
**Expected Feature Count**: Dynamic from parquet (typically ~40)

### Feature Discovery Method

```python
# Lines 120-127
def prepare_features(df: pd.DataFrame, task: str = "binary") -> tuple[np.ndarray, list[str]]:
    exclude = META_COLS | (ENTRY_QUALITY_EXCLUDE if task == "entry_quality" else set())
    feat_cols = [c for c in df.columns if c not in exclude]
    X = df[feat_cols].values.astype(np.float32)
    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)
    return X, feat_cols
```

### Excluded Metadata Columns (Line 49-50)
```python
META_COLS = {"timestamp", "symbol", "asset_class", "window",
             "label_action", "label_rr", "label_outcome", "label_profitable"}
```

### Entry Quality Task Exclusions (Line 55)
When `task="entry_quality"` (default, line 615):
```python
ENTRY_QUALITY_EXCLUDE = {"has_entry_zone", "alignment_score"}
```
**Reason**: "data leaks" - these correlate with the teacher's entry decisions

### Expected Features

**If consuming generate_rl_data.py output**:
- Total columns: 40 features + 1 timestamp = 41
- After excluding META_COLS (8): ~33 features
- After excluding ENTRY_QUALITY_EXCLUDE (2, for entry_quality task): ~31 features

**Actual features loaded**:
- All columns from parquet except metadata
- Feature names stored in `feat_names` variable (line 548)
- Feature order preserved from DataFrame column order

### Data Cleaning in rl_brain_v2.py

**At Load Time** (Line 126):
```python
X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)
```

**THIS MASKS UPSTREAM DATA QUALITY ISSUES**:
- NaN → 0.0 (silently)
- +inf → 5.0 (silently)
- -inf → -5.0 (silently)

**Impact**: Any NaN or inf values from generate_rl_data.py are **invisible** to training/evaluation.

---

## 3. RL_BRAIN.PY Feature Schema

**File**: `/root/bot/rl_brain.py`  
**Function**: `extract_features()` (lines 60-180)  
**BASE_OBS_DIM**: 24 (HARDCODED, line 41)  
**OBS_DIM**: 25 (24 + 1 coin-id feature, line 42)

### Feature Order (Exact Index by Position)

#### [0-15] Original Technical Features
```
0  alignment_score           - Alignment score [0, 1]
1  direction_sign            - -1 (short) or +1 (long)
2  atr_14_normalised         - ATR(14) / close
3  ema20_distance            - (close - EMA20) / close
4  ema50_distance            - (close - EMA50) / close
5  ema_cross                 - 1 if EMA20 > EMA50, else 0
6  volume_ratio              - current_vol / avg_vol(20), capped at 5.0
7  close_return_1            - 1-bar return
8  close_return_5            - 5-bar return
9  close_return_20           - 20-bar return
10 high_low_range_norm       - (H-L) / close (current bar)
11 rsi_14_normalised         - RSI(14) / 100 (normalized to 0-1)
12 setup_tier_encoded        - 0.5=AAA+, 1.0=AAA++ (from setup_tier param)
13 trade_style_encoded       - 0.0=scalp, 0.5=day, 1.0=swing (from trade_style param)
14 rr_ratio_normalised       - RR / 10, clamped to [0, 1]
15 daily_atr_pct_normalised  - daily ATR% * 10, clamped to [0, 1]
```

#### [16-23] AAA++ Additional Features
```
16 adx_normalized            - ADX / 50, clamped [0, 1]
17 session_score             - Session quality [0, 1] (parameter)
18 zone_quality              - Zone quality score [0, 1] (parameter)
19 volume_score              - 3-layer volume score [0, 1] (parameter)
20 momentum_score            - RSI+MACD confluence [0, 1] (parameter)
21 tf_agreement_score        - Multi-TF trend agreement [0, 1] (parameter)
22 spread_normalized         - 1.0 - spread (lower spread = higher score) [0, 1]
23 asset_class_id            - 0.0=crypto, 0.33=forex, 0.66=stocks, 1.0=commodities
```

### No Data Cleaning in rl_brain.py

At line 109:
```python
if n < 50:
    return np.zeros(BASE_OBS_DIM, dtype=np.float32)
```
Returns zeros if insufficient data, but **no `np.nan_to_num()` cleaning**.

---

## CRITICAL MISMATCHES

### ❌ 1. FEATURE COUNT MISMATCH

| Component | Feature Count | Type |
|-----------|--------------|------|
| `generate_rl_data.py` | 40 | Parquet features |
| `rl_brain.py` | 24 | Hardcoded array size |
| `rl_brain_v2.py` | ~40 | Dynamic from parquet |

**Result**: `rl_brain.py` cannot consume data from `generate_rl_data.py`.  
**Error**: Would fail with shape mismatch (expects 24, gets 40).

---

### ❌ 2. FEATURE SCHEMA MISMATCH

**generate_rl_data.py features NOT in rl_brain.py**:
- struct_1d, struct_4h, struct_1h, struct_15m, struct_5m
- decay_1d, decay_4h, decay_1h, decay_15m, decay_5m
- bias_strong, premium_discount
- h4_confirms, h4_poi, h1_confirms, h1_choch
- has_entry_zone, precision_trigger, volume_ok
- fvg_bull_active, fvg_bear_active, ob_bull_active, ob_bear_active
- liq_above_count, liq_below_count

**rl_brain.py features NOT in generate_rl_data.py**:
- direction_sign (parameter passed, not computed from data)
- setup_tier_encoded (parameter passed, not computed from data)
- trade_style_encoded (parameter passed, not computed from data)
- session_score (parameter passed, not computed from data)
- zone_quality (parameter passed, not computed from data)
- volume_score (parameter passed, not computed from data)
- momentum_score (parameter passed, not computed from data)
- tf_agreement_score (parameter passed, not computed from data)
- spread_normalized (parameter passed, not computed from data)
- close_return_1, close_return_5, close_return_20 (not in generate_rl_data.py)

**Conclusion**: These are **completely different feature sets** designed for different purposes.

---

### ✓ 3. COMPATIBLE: generate_rl_data.py ↔ rl_brain_v2.py

**Why it works**:
- `rl_brain_v2.py` uses dynamic feature discovery (line 123): `feat_cols = [c for c in df.columns if c not in exclude]`
- Features don't need to match a hardcoded schema
- Feature names preserved in `feat_names` list
- Compatible with XGBoost/NN training on arbitrary feature sets

**But there are issues**:

#### Issue 1: Data Leakage Exclusions
For `task="entry_quality"`:
```python
ENTRY_QUALITY_EXCLUDE = {"has_entry_zone", "alignment_score"}
```

These features ARE generated by `generate_rl_data.py` (indices 17, 31).

**Impact**: `rl_brain_v2` drops them to avoid teacher-knowledge leakage.
- **Expected**: ~40 features
- **Actual**: ~38 features for entry_quality task

#### Issue 2: Silent Data Cleaning (Line 126)
```python
X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)
```

**This masks data quality issues**:
- Any NaN in `generate_rl_data.py` output → silently becomes 0.0
- Any +inf → becomes 5.0
- Any -inf → becomes -5.0

**Danger**: If `generate_rl_data.py` has upstream bugs creating NaN/inf, training proceeds without error but with corrupted data.

---

## Data Cleaning & Data Quality Issues

### In generate_rl_data.py

**Generation Time**: No aggressive cleaning
- Line 337-790: `np.isnan()` and `pd.notna()` checks exist but only for **conditionals** (if statements)
- Missing data filled with zeros manually
- Float casting to float32 (no NaN handling)

**Potential Issues**:
1. **Missing higher timeframe data**: If df_1h/df_4h/df_1d are empty → manual zero-fill
2. **Division by zero**: Protected in some places (e.g., line 614), but not everywhere
3. **Invalid indicator values**: ATR, RSI calculations could produce edge cases
4. **Infinite values**: Possible from division by very small numbers

### In rl_brain_v2.py

**Load Time**: AGGRESSIVE cleaning masks everything
- Line 126: `np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)`
- Silently replaces all problematic values
- No logging or warnings

**Risk**: Data quality issues in `generate_rl_data.py` are **invisible** during training.

### In rl_brain.py

**No cleaning**: Returns zeros if `n < 50`, but otherwise assumes clean data.

---

## Summary Table

| Aspect | generate_rl_data.py | rl_brain_v2.py | rl_brain.py |
|--------|---|---|---|
| **Feature Count** | 40 | ~40 (dynamic) | 24 (hardcoded) |
| **Feature Discovery** | Fixed dict | Dynamic from DataFrame | Hardcoded array |
| **Data Cleaning at Load** | NO | YES (np.nan_to_num) | NO |
| **Compatible** | YES with v2 | YES with generate | NO with generate |
| **Data Quality Risk** | High (no cleaning) | Masked (cleaned silently) | Low (assumes clean) |

---

## Recommendations

1. **Fix rl_brain.py**: Either
   - Update it to use dynamic feature discovery like rl_brain_v2.py, OR
   - Document which 24 features it expects and ensure generate_rl_data.py matches

2. **Add data quality checks**: Before rl_brain_v2.py cleaning, log:
   - Count of NaN values per feature
   - Count of inf values per feature
   - Recommend fixing upstream rather than masking

3. **Remove data leakage exclusions**: If `has_entry_zone` and `alignment_score` are truly problematic, don't generate them in the first place.

4. **Standardize feature schemas**: Create a shared feature specification document (JSON schema) that both generate_rl_data.py and the brain models must conform to.
