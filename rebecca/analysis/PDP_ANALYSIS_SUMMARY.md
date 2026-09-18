# Partial Dependence Plot (PDP) Analysis Summary

**Status:** PDP generation in progress...  
**Generated:** Analyzing 5 input features (sku, price, order, duration, category)  
**Model:** Trained neural network predicting product quantity

---

## Quick Summary: What PDPs Tell Us

Partial Dependence Plots show **how each feature affects predicted quantity**, while keeping other features constant. This helps us understand:

1. **Which features matter most?** ← Feature importance ranking
2. **What's the direction of effect?** ← Positive or negative relationship
3. **Is the relationship linear?** ← Or are there non-linear patterns
4. **How sensitive are predictions?** ← Steep vs. flat slope

---

## Analysis Framework

### Feature 1: SKU (Product ID)
- **Range:** 0 to 74,998
- **Type:** Numerical product identifier
- **Expected:** Likely flat or noisy (product-specific variance)
- **Business meaning:** Different products, different demand

### Feature 2: PRICE (Product Price)
- **Range:** 0 to ~275 (scaled)
- **Type:** Continuous monetary value
- **Expected:** Negative relationship (higher price → lower quantity)
- **Business meaning:** Demand curve - price elasticity

### Feature 3: ORDER (Sequence Position)
- **Range:** 0 to many
- **Type:** Sequence/position in order
- **Expected:** Flat or weak trend
- **Business meaning:** Does order position affect sales?

### Feature 4: DURATION (Time Period)
- **Range:** 1 to ~32 (scaled)
- **Type:** Time-based feature
- **Expected:** STRONG positive relationship
- **Business meaning:** Seasonality/timing effect (feature importance: 0.97)

### Feature 5: CATEGORY (Product Category)
- **Range:** 0 to 32
- **Type:** Categorical identifier (encoded as numeric)
- **Expected:** Moderate trend or varied pattern
- **Business meaning:** Category-specific demand patterns

---

## Interpretation Quick Guide

### Slope (Steepness)

| Slope | Meaning | Example |
|-------|---------|---------|
| **Steep (+/-)** | Feature has large impact | Duration likely here |
| **Moderate** | Feature has measurable impact | Price likely here |
| **Flat (~0)** | Feature has little impact | SKU might be here |

### Direction

| Pattern | Meaning | Business Insight |
|---------|---------|-----------------|
| **Positive ↑** | Higher feature → Higher predictions | Duration: more time = more sales |
| **Negative ↓** | Higher feature → Lower predictions | Price: higher price = fewer sales |
| **Flat** | Feature doesn't matter | SKU identity alone may not predict |

### Non-linearity (Curve)

| Pattern | Meaning |
|---------|---------|
| **Straight line** | Linear relationship (consistent effect) |
| **Curved upward** | Effect accelerates at high values |
| **Curved downward** | Effect decelerates at high values |
| **S-shape** | Threshold or tipping point exists |
| **U-shape** | Optimal value in the middle |

---

## Expected Findings (Based on Feature Importance)

From our earlier analysis:
- **Duration** (importance 0.97): Should show VERY STEEP slope
- **Price** (importance 0.38): Should show MODERATE negative slope
- **Category** (importance 0.02): Should show WEAK/FLAT pattern
- **Order** (importance 0.01): Should show VERY WEAK/FLAT pattern
- **SKU** (importance 0.003): Should show FLAT/NOISY pattern

**Key insight:** PDP slopes should roughly match feature importance rankings!

---

## Validation Checklist

Once plots are generated, check:

- [ ] **Duration shows strong positive slope** - Matches importance (0.97)
- [ ] **Price shows negative slope** - Typical demand curve
- [ ] **SKU/Order/Category show weak slopes** - Match low importances
- [ ] **All PDPs make business sense** - No surprising patterns
- [ ] **No extreme outliers** - PDPs are smooth/continuous
- [ ] **Prediction ranges reasonable** - In valid quantity units

---

## How to Read the Individual Plots

Each PDP plot shows:

```
Y-axis: Predicted Quantity (units)
X-axis: Feature Value (scaled)

The curve shows:  
"As this feature changes, predicted quantity changes like this"

Shaded area: Confidence/uncertainty region
Red dots: Start and end points of feature range
```

---

## Files Generated

- `pdp_combined.png` - All 5 features in one view (fast version)
- `pdp_sku.png` - Detailed plot for SKU (if full version completes)
- `pdp_price.png` - Detailed plot for Price
- `pdp_order.png` - Detailed plot for Order
- `pdp_duration.png` - Detailed plot for Duration
- `pdp_category.png` - Detailed plot for Category

---

## Next Steps

1. **View the generated plots** → results/plots/pdp_*.png
2. **Compare slopes** → Match against feature importance
3. **Identify patterns** → Linear, curved, flat, noisy?
4. **Interpret for business** → What does each pattern mean?
5. **Validate** → Do results make business sense?
6. **Document findings** → Write up analysis

---

## Key Questions to Answer

After seeing the PDPs:

1. **Duration effect:** How strong is the time/seasonality impact?
2. **Price elasticity:** What's the demand curve shape?
3. **Category performance:** Do some categories predict higher/lower sales?
4. **Product identity:** Does SKU significantly affect predictions?
5. **Order position:** Does sequence in order matter?

---

## Technical Notes

### Scaling
- **Input features:** Scaled with StandardScaler (mean 0, std 1)
- **Predictions:** Unscaled to original quantity units (actual items sold)
- **PDP range:** -3 to +3 standard deviations (typical data range)

### Computation
- **Samples used:** ~20-30% of test data for speed
- **Feature range:** 25-100 points along feature spectrum
- **Averaging:** Mean prediction across all samples at each point

### Interpretation
- **No causality:** PDPs show association, not causation
- **Ceteris paribus:** Assumes other features stay at their values
- **Non-linear effects:** Only captures main effects, not interactions

---

## Quick Reference: Expected Patterns

```
DURATION (Expected: STEEP ↑)
   Q
   |     /
   |    /
   |   /
   |__/__________ Duration

PRICE (Expected: MODERATE ↓)
   Q
   |\
   | \
   |  \___
   |_______\ Price

SKU/ORDER/CATEGORY (Expected: FLAT)
   Q
   |__________
   |___________ Feature
```

---

## Generate Analysis

PDP scripts available:
- `code/partial_dependence.py` - Full detailed analysis (slower, more samples)
- `code/partial_dependence_fast.py` - Quick analysis (faster, reasonable quality)

Run either or both depending on your needs!

---

**Check back soon for detailed results and interpretation!**
