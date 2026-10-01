# Where did the fuel-economy gains go? (regression on EPA test cars, 2013–2025)

Engines and transmissions improved a lot in this period, but cars also got heavier and more powerful. This project
separates the two with a log-log regression on 6,341 EPA test vehicles. It measures how much less fuel a car of the
**same** weight, power, body and drive used each model year, and how much of that gain bigger cars took back.

Build log: https://mrrishit909.github.io/projects/fuel-economy/

## Data

US EPA, "Data on Cars Used for Testing Fuel Economy", model-year files 2012–2026:
https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy. Each row is one lab test of
one test vehicle, with its test weight, rated horsepower, engine, transmission and measured fuel economy.
`download.py` fetches the 15 files (not committed).

## Steps

| Step | File | What it does |
|---|---|---|
| 1 | `download.py` | the 15 model-year files |
| 2 | `analysis.py` | cleaning funnel; model; clustered errors; decomposition; grouped cross-validation; robustness; city-vs-highway physics |
| 3 | `charts.py` | year effects, decomposition, size and power trends, physics |
| 4 | `check.py` | funnel recomputes; NumPy refit matches; decomposition adds up; robustness signs hold |

## Cleaning, and three traps in the data

66,976 test results → 54,496 on EPA certification gasoline → 46,404 city or highway tests → 41,409 measured (not
analytically derived) → 41,207 not police vehicles → **18,247 after counting each test once** → 14,907 first tested
2013–2025 → 6,937 vehicles with both a city and a highway result → 6,426 not hybrids → **6,341** usable.

1. **Carry-over tests.** A model that isn't redesigned keeps its certification, so the same test is listed again
   next year: 99.9% of repeated test numbers have identical results. Counting them in every year would make old
   technology look current. Each test is kept once, in its first model year. The 2012 file is used only to recognise
   tests carried over from before 2013.
2. **Hidden hybrids.** The files have no hybrid flag. Hybrids are found three ways:
   - by name;
   - by a plug-in's charge-depleting tests;
   - by physics: conventional cars use about a third less fuel per mile on the highway than in the city, but hybrids
     don't, so a city/highway fuel ratio under 1.2 marks a hybrid.

   Checking the largest residuals and the most powerful cars found more (the Volvo XC90 T8, LaFerrari, McLaren P1 and
   Audi "e-tron" plug-ins), which are now caught.
   48-volt mild hybrids stay in as an engine technology.
3. **Placeholder horsepower.** Three Audis are listed at 999 hp (an A8 has 335). They are dropped.

The test fuel stays the same throughout: CO₂ × MPG is about 8,875 g per gallon every year, so no year effect comes from
a fuel change. The 2026 file is a partial January release: 265 vehicles against 2025's 360, lighter and less powerful, with
the CVT share jumping from 6% to 20% (`results/partial_2026.json`). So the comparison ends at 2025.

## Model

`log(gallons per 100 miles) ~ log(weight) + log(horsepower) + car/truck + drive + model year`

Consumption combines the two tests as 0.55 × city + 0.45 × highway in gallons per 100 miles, which is how EPA weights
them. Averaging MPG would be wrong. Logs make the coefficients elasticities. The model explains 82% of the variation,
or 87% when engine and transmission are added.

**Why manufacturer-clustered standard errors:** one maker's cars share engines and engineering, so they aren't 6,341
independent observations. Clustered by 31 manufacturer groups, the standard errors are 3.5–4.2× the heteroskedasticity-robust
ones (weight 0.056 vs 0.016). 31 clusters is near the usual minimum of about 30, so the intervals are approximate.

**Why grouped cross-validation:** the same model is tested many times, so random folds can put a car's twin in the
training set. Holding out whole models gives R² 0.815 against 0.820 for random folds (median error 6.6% vs 6.5%).
The leak is small here because the model is simple; a flexible model would gain more from memorising.

## Results

| Holding the others fixed | Effect on fuel used per mile |
|---|---:|
| 10% heavier | **+3.0%** (elasticity 0.31) |
| 10% more horsepower | **+3.7%** (elasticity 0.38) |
| Truck (vs car) | +7.5% |
| All-wheel/4-wheel or rear drive (vs front) | +2.6% / +3.1%, not significant |

**Where the gains went, 2013 to 2025** (tested vehicles; these parts add up exactly in log terms):

| Part | Change in fuel used per mile |
|---|---:|
| Technology: same weight, power, body and drive | **−14.4%** |
| Heavier (average test weight 4,212 → 4,581 lb) | +2.6% |
| More powerful (average 286 → 330 hp) | +5.8% |
| Body and drive mix (trucks 23% → 44% of tested vehicles) | +1.3% |
| **Net change** | **−5.8%** |

Bigger, more powerful vehicles took back **61%** of the technology gain. On raw averages, the 2025 change isn't even
statistically clear (95% CI −15% to +4%). Holding size and power fixed, it is (−17% to −12%).

**What the technology was:** adding displacement, gears, transmission type and cylinders to the model shrinks the 2025
effect from −14.4% to −6.4%. These measured engine and transmission changes explain **57%** of the gain:
- each extra gear: −2.5%;
- downsizing (smaller engines at the same power, usually turbocharged): 10% less displacement, about 1% less fuel;
- CVTs: −21%.

CVTs come with other efficiency choices, mostly from Japanese makers, so that −21% is not the gearbox alone.

**Robustness**

| Version | n | Weight elasticity | Power elasticity | 2025 vs 2013 |
|---|---:|---:|---:|---:|
| Main | 6,341 | 0.31 | 0.38 | −14.4% |
| Hybrids kept, with a hybrid dummy | 6,840 | 0.30 | 0.39 | −13.7% |
| City test only | 6,341 | 0.32 | 0.41 | −15.7% |
| Highway test only | 6,341 | 0.28 | 0.32 | −11.9% |
| Each model counted once per year | 1,848 | 0.32 | 0.40 | −15.9% |

**City vs highway physics (a separate model):** with EPA's road-load coefficients added, the air-drag term's elasticity
is **0.21 on the highway vs 0.11 in the city**, about twice as large, as physics predicts (drag grows with speed squared).
Weight matters about equally on both tests (0.34 vs 0.31). The rolling/friction term is not significant once weight is
in the model. The largest VIF is 3.0.

## Not done

- **Not sales-weighted:** these are the vehicles EPA tests, not what people bought. EPA's Automotive Trends Report is the
  sales-weighted view.
- Lab test values (unadjusted), not window-sticker MPG.
- 48-volt mild hybrids are not separated out, and turbocharging is only visible through displacement.
- Associations, not causes: the year effect is everything that changed at fixed weight, power, body and drive.

## Run it

```
python3 -m venv venv && ./venv/bin/pip install pandas numpy statsmodels scikit-learn matplotlib scipy openpyxl
./venv/bin/python download.py && ./venv/bin/python analysis.py && ./venv/bin/python charts.py && ./venv/bin/python check.py
```
