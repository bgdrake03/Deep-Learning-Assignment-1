# Partial Dependence Plot (PDP) Analysis Guide

## What are PDPs?

Partial Dependence Plots show how predicted values (quantity) change as you vary one feature while keeping all other features at their average values. This helps us understand the **marginal effect** of each feature on predictions.

**Key Insight:** PDPs answer the question: "If I change feature X while keeping everything else constant, how does the model's prediction change?"

---

## Feature Analysis Overview

### Dataset Features:
- **sku** (Product ID): Numerical identifier (0-74,998)
- **price** (Product Price): Continuous feature related to product cost
- **order** (Order Sequence): Sequence number in purchase order
- **duration** (Time Period): Duration in some time unit
- **category** (Product Category): Categorical identifier (0-32)
- **quantity** (Target): Units sold (what we're predicting)

---

## Expected PDP Patterns

### Feature: SKU (Product ID)

**What it represents:** Different products have different demand patterns

**Expected pattern:**
- Likely **relatively flat or noisy** - individual SKUs may have high variance
- Slight trends possible if certain product ranges sell better
- Should show: Random fluctuation suggesting product identity alone isn't strongly predictive

**Interpretation:**
- If **flat:** Product ID doesn't strongly predict quantity (good regularization)
- If **noisy:** High variance in product-specific demand
- If **trending:** Some SKUs systematically higher/lower demand

---

### Feature: Price

**What it represents:** Product pricing's effect on quantity sold

**Expected pattern:**
- **Negative relationship** (higher price → lower quantity) - typical demand curve
- Could be **non-linear:** steeper decline at high prices

**Interpretation:**
- **Negative slope:** Elasticity - customers buy less at higher prices
- **Steepness:** How sensitive demand is to price changes
- **Non-linearity:** If curve bends, price sensitivity varies by price range

**Business insight:** "For every unit increase in price, quantity drops by X units"

---

### Feature: Order (Sequence Position)

**What it represents:** Which item in the order sequence is this?

**Expected pattern:**
- Could be **negative** (earlier items sell more? or later items?)
- Or **relatively flat** if position doesn't matter

**Interpretation:**
- **Positive trend:** Items appearing later in orders sell more
- **Negative trend:** Items appearing earlier in orders sell more
- **Flat:** Order position doesn't matter for quantity

**Business insight:** "Sequential position in orders affects demand by X%"

---

### Feature: Duration (Time Period)

**What it represents:** Time period when the sale occurs

**Expected pattern:**
- Likely **strongly positive** - time/seasonality often drives sales
- Could be **non-linear:** increasing then plateauing, or seasonal oscillation

**Interpretation:**
- **Strong positive slope:** More sales in longer/later time periods
- **Non-linearity:** If duration effect is non-linear, there may be seasonal patterns
- **This is likely the MOST IMPORTANT feature** (remember: importance analysis showed duration at 0.97)

**Business insight:** "Products sold during longer/later periods have much higher demand"

---

### Feature: Category

**What it represents:** Product category identifier

**Expected pattern:**
- Likely **moderate trend** - some categories sell more than others
- Could show several different patterns across category range

**Interpretation:**
- **Positive trend:** Higher category IDs → more sales
- **Negative trend:** Higher category IDs → fewer sales
- **Non-linear:** Some categories are "sweet spots" for demand

**Business insight:** "Category differences account for X% of quantity variance"

---

## How to Interpret the Plots

### Slope (Steepness)
- **Steep slope:** Feature has large effect on predictions
- **Flat slope:** Feature has small effect
- **Slope direction:** + means increasing, - means decreasing relationship

### Non-linearity (Curvature)
- **Straight line:** Linear relationship
- **Curved:** Non-linear - effect varies depending on feature value
- **S-shape:** Threshold effects (e.g., "price only matters above $10")
- **U-shape:** Optimal value in the middle (e.g., "medium duration best")

### Range
- **Wide range (e.g., 15-35):** Feature causes big prediction changes
- **Narrow range (e.g., 22-23):** Feature causes small prediction changes

---

## Key Patterns to Look For

### Pattern 1: Strong Linear Relationship ➡️
```
Prediction
    ^
    |     /
    |    /
    |   /
    |  /
    | /
    |/____> Feature
```
**Meaning:** Clear, consistent effect - feature is important
**Example:** Duration likely shows this

---

### Pattern 2: Flat/Weak Relationship ➡️
```
Prediction
    ^
    |________________
    |
    |
    |________________> Feature
```
**Meaning:** Feature has little effect - weak importance
**Example:** SKU might show this

---

### Pattern 3: Non-linear Relationship ➡️
```
Prediction
    ^
    |    /\
    |   /  \
    |  /    \
    | /      \___
    |/____________> Feature
```
**Meaning:** Effect varies by value - complex relationship
**Example:** Price or category might show this

---

## Expected Results Summary

| Feature | Expected Importance | Expected Relationship |
|---------|-------------------|----------------------|
| SKU | Low | Flat/Noisy |
| Price | Moderate-High | Negative (demand curve) |
| Order | Low-Moderate | Flat or trending |
| Duration | **Very High** | Positive/Strong |
| Category | Moderate | Positive or varied |

---

## Validation Against Feature Importance

Our earlier analysis showed feature importance (permutation):
- **duration**: 0.97 (DOMINANT)
- **price**: 0.38 (moderate)
- **category**: 0.02 (weak)
- **order**: 0.01 (very weak)
- **sku**: 0.003 (negligible)

**PDPs should align:** Features with high importance should show steeper slopes and wider prediction ranges.

---

## Handling Edge Cases

### Scaled Features
All input features are scaled (StandardScaler: mean 0, std 1)
- **-3 to +3** is typical range
- **Original units:** Check `y_mean` and `y_std` for unscaling

### Categorical Variables
SKU and Category are numerically encoded (not one-hot)
- Treat as continuous in PDPs
- Patterns show trends across category space, not individual categories

### Target Scaling
Predictions are unscaled from standardized units
- **Model outputs:** Standardized (mean 0, std 1)
- **Plots show:** Original quantity units (actual items sold)

---

## Questions PDPs Answer

1. **Which features matter most?** → Steepest slopes
2. **What's the demand curve for price?** → See price's PDP
3. **Is there seasonality/timing effect?** → See duration's PDP
4. **Are there non-linear patterns?** → See curved PDPs
5. **What's the prediction range?** → Y-axis of each plot

---

## Business Takeaways

After seeing the PDPs, you'll understand:
- ✅ Which factors drive product sales
- ✅ How pricing affects demand (elasticity)
- ✅ Whether timing/duration matters
- ✅ Category and SKU effects
- ✅ Non-linear patterns and thresholds

**Actionable Insights:**
- If duration is strong → Focus on timing/seasonality
- If price is negative → Understand price elasticity
- If flat → Feature may not be predictive → could remove

---

## Files Generated

- `partial_dependence_plots.png` - All 5 features in one figure
- `pdp_sku.png` - Individual detailed plot for SKU
- `pdp_price.png` - Individual detailed plot for Price
- `pdp_order.png` - Individual detailed plot for Order
- `pdp_duration.png` - Individual detailed plot for Duration
- `pdp_category.png` - Individual detailed plot for Category

Each individual plot includes:
- Main PDP curve with confidence region
- Endpoints marked
- Trend direction (increasing/decreasing)
- Non-linearity measure
- Prediction statistics (min, max, range)
