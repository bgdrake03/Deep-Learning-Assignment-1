# Feature Interpretation Guide - Partial Dependence Analysis

## Overview

This guide helps you interpret each feature's Partial Dependence Plot (PDP) in the context of predicting product quantity sold.

---

## FEATURE 1: SKU (Stock Keeping Unit)

### What it is
Unique product identifier (0-74,998). Each SKU represents a different product.

### What to expect in PDP
- **Likely FLAT or NOISY** line
- Why? Random variation between products
- Could show slight trend if some product ranges sell better

### How to interpret
- **Flat line** → Product identity alone doesn't strongly predict quantity ✓
- **Noisy/scattered** → High variance in individual product sales ✓
- **Clear trend** → Some SKUs systematically sell more (unexpected)

### Business meaning
- If FLAT: Focus on OTHER factors (price, time, category) not product itself
- If NOISY: Each product is unique; hard to predict from SKU alone
- If TRENDING: Could indicate product line effects (early SKUs sell diff than late)

### Question to answer
"Does knowing the SKU tell us much about how many will sell?"
- **Answer from importance: NO (0.003)** → PDP should be flat

---

## FEATURE 2: PRICE (Product Price)

### What it is
Product pricing in scaled units (original range: $0 to ~$275).

### What to expect in PDP
- **Negative slope ↓** (classic demand curve)
- Steeper = more price sensitive
- May be non-linear (stronger at extremes)

### How to interpret

**NEGATIVE slope (↓):**
```
Pred
  |    /
  |   /  ← Higher price = Lower quantity
  |  /
  |/__________ Price
```
✓ Normal - higher prices reduce demand (economics 101)

**Slope interpretation:**
- **Steep decline:** Price very elastic (customers sensitive to price)
- **Gentle decline:** Price less elastic (customers relatively insensitive)
- **Flat:** Price doesn't affect quantity (would be unusual)

**Non-linearity:**
- **S-curve:** Price only matters above/below threshold
- **Curved:** Elasticity varies by price range
- **Linear:** Constant price sensitivity

### Business meaning
- **Understand elasticity:** "Every $1 price increase reduces quantity by X%"
- **Pricing strategy:** Identify optimal price point (max revenue)
- **Sensitivity:** How much do customers react to price changes?

### Question to answer
"How sensitive is demand to price changes?"
- **Expected from importance: Moderately important (0.38)** → PDP should show moderate negative slope

---

## FEATURE 3: ORDER (Order Sequence Position)

### What it is
Position in purchase order sequence (1st item, 2nd item, 3rd item, etc.).

### What to expect in PDP
- **Likely FLAT or slight trend**
- Why? Position in order may not directly affect quantity
- Could be positive or negative depending on customer behavior

### How to interpret

**FLAT line:**
```
Pred
  |__________________
  |__________________Order
```
✓ Order position doesn't affect quantity (items ordered first don't sell more)

**Positive slope ↑:**
```
Pred
  |        /
  |      /
  |    /
  |  /
  |/____________ Order
```
✗ Later items in order sell MORE (could indicate increasing needs)

**Negative slope ↓:**
```
Pred
  |\
  | \
  |  \
  |   \___
  |________\ Order
```
✗ Earlier items in order sell MORE (could indicate priority/necessity)

### Business meaning
- **Customer behavior:** Do customers buy MORE of items ordered first/last?
- **Order dynamics:** Is there a pattern to which items sell more?
- **Bundling:** Are later items sold as add-ons?

### Question to answer
"Does position in the purchase order affect quantity sold?"
- **Expected from importance: NO (0.01)** → PDP should be very flat

---

## FEATURE 4: DURATION (Time Period)

### What it is
Duration or time period (scaled, likely months or seasons).

### What to expect in PDP
- **STRONG POSITIVE slope ↑ ↑ ↑**
- Steepest of all features
- May be non-linear (acceleration or plateauing)

### How to interpret

**STRONG positive slope (↑):**
```
Pred
  |              /
  |           /
  |        /
  |    /
  |  /
  |/________________ Duration
```
✓ STRONG TIME/SEASONALITY EFFECT
(This should be the MOST IMPORTANT feature!)

**Why this matters:**
- Time-based trends dominate predictions
- Products sell MUCH MORE in certain periods
- Could be: seasonal demand, growth over time, end-of-period urgency

**Non-linear patterns:**
- **Accelerating:** Early months flat, then suddenly increased demand
- **Plateauing:** Strong early growth, then levels off
- **Exponential:** Compounding effect over time
- **Seasonal:** Oscillating (not visible in single PDP, but non-linearity shows it)

### Business meaning
- **Seasonality:** "Q4 sales 300% higher than Q1"
- **Growth trajectory:** "Sales accelerating over time"
- **Urgency:** "End-of-period buying drives quantity"
- **Peak seasons:** Identify when demand peaks

### Question to answer
"How much does timing/period affect quantity sold?"
- **Expected from importance: VERY MUCH (0.97)** → PDP should show STEEP positive slope

---

## FEATURE 5: CATEGORY (Product Category)

### What it is
Product category identifier (0-32). Categories group related products.

### What to expect in PDP
- **Moderate trend or varied pattern**
- Could be positive, negative, or mixed
- May show non-linearity (some categories better than others)

### How to interpret

**Positive trend ↑:**
```
Pred
  |            /
  |         /
  |      /
  |   /
  |/_____________ Category
```
✓ Higher category IDs sell more (e.g., electronics sell more than basics)

**Negative trend ↓:**
```
Pred
  |\
  | \
  |  \
  |   \___
  |________\ Category
```
✓ Lower category IDs sell more (e.g., essentials more popular)

**Non-linear (U-shape):**
```
Pred
  |  \        /
  |   \    /
  |    \  /
  |_____\/_________ Category
```
✓ Categories at extremes sell more (specialized niches)

**Non-linear (N-shape):**
```
Pred
  |    /\
  |   /  \
  |  /    \
  |/        \______ Category
```
✓ Some specific categories are "sweet spots"

### Business meaning
- **Category performance:** Which categories drive sales?
- **Strategic focus:** Where to invest marketing?
- **Product assortment:** Which categories to expand?
- **Segmentation:** Different customer preferences by category

### Question to answer
"Do some product categories sell significantly more than others?"
- **Expected from importance: MODERATELY (0.02)** → PDP should show weak/moderate trend

---

## Comparison Matrix

| Feature | Expected Importance | Expected Slope | Expected Shape | Visual |
|---------|-------------------|-----------------|-----------------|--------|
| **SKU** | Very Low (0.003) | Flat | Noisy | ═══════ |
| **Price** | Moderate (0.38) | Negative | Linear or Curved | ╲╲╲═══ |
| **Order** | Very Low (0.01) | Flat/Slight | Flat | ╲═══════ |
| **Duration** | VERY HIGH (0.97) | POSITIVE | May Curve | ╱╱╱╱╱ |
| **Category** | Low (0.02) | Slight | Varied | ╱╲══╱ |

---

## Common Interpretation Mistakes

❌ **Mistake 1:** "PDP is flat, so feature is useless"
✓ **Correct:** Flat might mean feature is already well-predicted by others (interactions)

❌ **Mistake 2:** "Steep slope = causation"
✓ **Correct:** Correlation not causation. PDPs show association, not cause-effect

❌ **Mistake 3:** "My PDP doesn't match expectations"
✓ **Correct:** May indicate:
- Feature interactions
- Non-linear relationships
- Data quality issues
- Model-specific learned patterns

❌ **Mistake 4:** "PDP shows actual customer behavior"
✓ **Correct:** Shows model's LEARNED behavior (may not reflect reality)

---

## How to Use This Guide

1. **Generate PDPs** → Run `python code/partial_dependence_fast.py`
2. **For each PDP:**
   - Find it in the chart
   - Look up feature above
   - Compare actual slope/shape to "What to expect"
   - Read "How to interpret"
3. **Match expectations:** Do your results align with feature importance?
4. **Document findings:** Note any surprises or interesting patterns
5. **Answer business questions:** Extract actionable insights

---

## Quick Checklist

After seeing all PDPs, verify:

- [ ] Duration shows STRONG positive slope? (should be strongest)
- [ ] Price shows negative slope? (should be moderate strength)
- [ ] SKU/Order/Category show weaker slopes? (should be weak)
- [ ] All slopes match the feature importance rankings?
- [ ] No extreme outliers or impossible values?
- [ ] Prediction ranges make business sense?
- [ ] Any surprising patterns identified and understood?

---

## Example Analysis

**"What if duration PDP was FLAT?"**

This would mean: "Timing/seasonality doesn't affect quantity sold"

But we expect importance of 0.97 (very high!)

Possible explanations:
1. Non-linear pattern not visible in simple slope
2. Seasonal effect is categorical, not continuous
3. Model learned different time representation
4. Data quality issue in duration feature
5. Feature scaling issue affecting PDP

Action: Investigate further, check raw data, verify model

---

**"What if price PDP was POSITIVE?"**

This would mean: "Higher prices = Higher quantities" (unusual!)

Possible explanations:
1. Premium products attract bulk orders
2. High-price items are "luxury" with dedicated customers
3. Prices increase AFTER high demand (correlation, not cause)
4. Interaction with category (luxury categories)

Action: Good finding! Indicates market segmentation

---

## Next: Document Your Findings

Template for your analysis:
```
Feature: [Name]
Importance: [Value]
Expected: [What we predicted]
Actual: [What PDP shows]
Trend: [Positive/Negative/Flat]
Strength: [Steep/Moderate/Flat]
Non-linearity: [Yes/No]
Interpretation: [What it means]
Business Insight: [Action/Decision]
```

Happy interpreting! 📊
