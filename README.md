# Data Detective: Finding the Secret Behind Future Revenue Losses

A B2B company (800 customers, 30 months) sees revenue falling by **14.7%** – but nobody knows why, or who will leave next.
This project solves it with a "time machine", process mining and machine learning. **All data is simulated.**

## Approach
1. **Time machine** – versioned snapshots of the customer table, like Databricks **Delta Lake Time Travel** (`VERSION AS OF`). Comparing month 12 with month 30 shows the South region's late-delivery rate jumping from **7% to 28%** and complaints from **16 to 69**.
2. **Process mining** (Celonis-style) – 75,700 order-to-cash events: process variants, rework loops and lead times. After a logistics change, South orders show **26% rework/complaint cases** (vs. 4% before) and lead times rising from 3.9 to 5.9 days.
3. **Machine learning with backtest** – logistic-regression churn model trained "as of" month 18, tested out-of-time on months 25–30: **AUC 0.76**. Main drivers: late-delivery rate and complaints.
4. **Revenue at risk** – South holds **24% of customers but 41% of the revenue at risk**.

**The secret:** late deliveries → complaints → customers leave 1–3 months later. A fixable root cause, not just a forecast.

## Run
```bash
pip install -r requirements.txt
python data_detective.py
```
Full output: [`detective_output.txt`](detective_output.txt)

*Author: Dhrukumar Valand*
