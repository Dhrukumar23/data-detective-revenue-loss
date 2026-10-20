"""
Data Detective: Finding the Secret Behind Future Revenue Losses
---------------------------------------------------------------
Portfolio project by Dhrukumar Valand (simulated data only)

Story: A B2B company sees stable revenue today, but management wants to know
whether revenue losses are coming – and why.

Approach (inspired by Databricks + Celonis):
 1. "Time machine": keep versioned snapshots of the customer table
    (like Delta Lake Time Travel: SELECT ... VERSION AS OF / TIMESTAMP AS OF)
    and compare customer health between versions.
 2. Process mining (Celonis-style): analyse the order-to-cash event log –
    process variants, throughput times, rework loops – to find WHERE the
    process breaks.
 3. Machine learning with a backtest: train a churn model on the data
    "as of" month 18, check it against what really happened in months 19–24,
    then score today's customers and estimate revenue at risk.

Run: python data_detective.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

rng = np.random.default_rng(11)
N_CUST, MONTHS = 800, 30          # months 1..30, "today" = month 30
CHANGE_MONTH = 15                  # hidden cause: new logistics partner for region "South"

cust = pd.DataFrame({
    "customer_id": [f"C{1000+i}" for i in range(N_CUST)],
    "region": rng.choice(["North", "South", "East", "West"], N_CUST, p=[.3, .25, .25, .2]),
    "segment": rng.choice(["Key Account", "Mid", "Small"], N_CUST, p=[.15, .45, .4]),
})
base_value = {"Key Account": 9000, "Mid": 2500, "Small": 800}
cust["avg_order"] = [rng.normal(base_value[s], base_value[s] * .2) for s in cust.segment]

# ---------------- order-to-cash event log ----------------
events, orders = [], []
churn_month = {}
late_hist = {c: 0 for c in cust.customer_id}
oid = 0
for m in range(1, MONTHS + 1):
    for _, c in cust.iterrows():
        if c.customer_id in churn_month:
            continue
        if rng.random() > 0.55:                          # orders roughly every other month
            continue
        oid += 1
        o = f"O{oid:06d}"
        p_late = 0.06
        if c.region == "South" and m >= CHANGE_MONTH:
            p_late = 0.38                                # the hidden problem
        late = rng.random() < p_late
        rework = late and rng.random() < .5              # delivery date changed / re-shipped
        complaint = late and rng.random() < .45
        lead = rng.gamma(3, 1.2) + (6 if late else 0)
        value = max(100, rng.normal(c.avg_order, c.avg_order * .15))
        orders.append((o, c.customer_id, m, value, late, complaint, lead))
        steps = ["Order created", "Order confirmed", "Goods shipped"]
        if rework:
            steps += ["Delivery date changed", "Goods shipped"]
        steps += ["Goods delivered"]
        if complaint:
            steps += ["Complaint received"]
        steps += ["Invoice sent", "Payment received"]
        for s in steps:
            events.append((o, s))
        late_hist[c.customer_id] = late_hist[c.customer_id] * 0.8 + (1 if late else 0) + (1 if complaint else 0)
        # churn risk grows with recent late deliveries / complaints – with a delay
        p_churn = 0.003 + 0.05 * max(0, late_hist[c.customer_id] - 1)
        if rng.random() < p_churn:
            churn_month[c.customer_id] = m + int(rng.integers(1, 4))   # leaves 1–3 months later

orders = pd.DataFrame(orders, columns=["order_id", "customer_id", "month", "value", "late",
                                       "complaint", "lead_days"]).merge(cust, on="customer_id")
orders = orders[~orders.apply(lambda r: r.customer_id in churn_month and r.month >= churn_month[r.customer_id], axis=1)]
log = pd.DataFrame(events, columns=["order_id", "activity"])
log = log[log.order_id.isin(orders.order_id)]


def snapshot(as_of):
    """Customer table as it looked at the end of month `as_of` (Delta Time Travel)."""
    h = orders[orders.month <= as_of]
    last6 = h[h.month > as_of - 6]
    prev6 = h[(h.month > as_of - 12) & (h.month <= as_of - 6)]
    s = cust.set_index("customer_id")[["region", "segment"]].copy()
    s["rev_last6"] = last6.groupby("customer_id").value.sum()
    s["rev_prev6"] = prev6.groupby("customer_id").value.sum()
    s["orders_last6"] = last6.groupby("customer_id").size()
    s["late_rate_last6"] = last6.groupby("customer_id").late.mean()
    s["complaints_last6"] = last6.groupby("customer_id").complaint.sum()
    s["avg_lead_last6"] = last6.groupby("customer_id").lead_days.mean()
    s["months_since_order"] = as_of - h.groupby("customer_id").month.max()
    s = s.fillna({"rev_last6": 0, "rev_prev6": 0, "orders_last6": 0, "late_rate_last6": 0,
                  "complaints_last6": 0, "avg_lead_last6": 0, "months_since_order": 12})
    s["rev_trend"] = (s.rev_last6 - s.rev_prev6) / (s.rev_prev6 + 1)
    return s


FEATURES = ["rev_trend", "orders_last6", "late_rate_last6", "complaints_last6",
            "avg_lead_last6", "months_since_order"]

if __name__ == "__main__":
    total = orders.groupby("month").value.sum()
    print("=== 0. What management sees ===")
    print(f"Revenue months 1–6:  EUR {total.loc[1:6].sum():,.0f}")
    print(f"Revenue months 25–30: EUR {total.loc[25:30].sum():,.0f}  "
          f"({total.loc[25:30].sum() / total.loc[1:6].sum() - 1:+.1%})")

    print("\n=== 1. Time machine: compare snapshot versions (month 12 vs. month 30) ===")
    v12, v30 = snapshot(12), snapshot(30)
    cmp = pd.DataFrame({
        "late_rate_m12": v12.groupby("region").late_rate_last6.mean(),
        "late_rate_m30": v30.groupby("region").late_rate_last6.mean(),
        "complaints_m12": v12.groupby("region").complaints_last6.sum(),
        "complaints_m30": v30.groupby("region").complaints_last6.sum(),
    }).round(3)
    print(cmp.to_string())

    print("\n=== 2. Process mining: order-to-cash event log ===")
    variants = log.groupby("order_id").activity.apply(lambda a: " > ".join(a))
    vc = variants.value_counts()
    print(f"{len(log):,} events, {variants.size:,} cases, {vc.size} process variants")
    happy = vc.index[0]
    print(f"Happy path share: {vc.iloc[0] / vc.sum():.1%}")
    o2 = orders.set_index("order_id")
    o2["variant_ok"] = variants.reindex(o2.index) == happy
    o2["period"] = np.where(o2.month >= CHANGE_MONTH, "after", "before")
    pm = o2.groupby(["region", "period"]).agg(
        rework_or_complaint=("variant_ok", lambda x: 1 - x.mean()),
        avg_lead_days=("lead_days", "mean")).round(2).unstack()
    print(pm.to_string())

    print("\n=== 3. Churn model with backtest (trained 'as of' month 18) ===")
    train = snapshot(18)
    train = train[train.months_since_order < 12]                      # active customers
    churned_19_24 = {c for c, m in churn_month.items() if 19 <= m <= 24}
    y = train.index.isin(churned_19_24).astype(int)
    X = train[FEATURES]
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    # out-of-time test: features as of month 24, churn in months 25–30
    test = snapshot(24)
    test = test[test.months_since_order < 12]
    yt = test.index.isin({c for c, m in churn_month.items() if 25 <= m <= 30}).astype(int)
    pt = model.predict_proba(test[FEATURES])[:, 1]
    print(f"Out-of-time AUC: {roc_auc_score(yt, pt):.2f} | "
          f"precision@0.5: {precision_score(yt, pt > .5):.2f} | recall@0.5: {recall_score(yt, pt > .5):.2f}")
    coef = pd.Series(model.coef_[0], index=FEATURES).sort_values(ascending=False)
    print("Strongest drivers (positive = more churn):")
    print(coef.round(2).to_string())

    print("\n=== 4. Today (month 30): revenue at risk in the next 6 months ===")
    now = snapshot(30)
    now = now[now.months_since_order < 12].copy()
    # balanced model ranks customers; an unweighted model gives realistic probabilities for EUR estimates
    calib = LogisticRegression(max_iter=1000).fit(X, y)
    now["p_churn"] = calib.predict_proba(now[FEATURES])[:, 1]
    now["high_risk"] = model.predict_proba(now[FEATURES])[:, 1] > .5
    now["revenue_at_risk"] = now.p_churn * now.rev_last6
    risk = now.groupby("region").agg(customers=("p_churn", "size"),
                                     high_risk=("high_risk", "sum"),
                                     revenue_at_risk=("revenue_at_risk", "sum")).sort_values(
        "revenue_at_risk", ascending=False)
    risk["share"] = risk.revenue_at_risk / risk.revenue_at_risk.sum()
    risk["share_of_customers"] = risk.customers / risk.customers.sum()
    print(risk.round({"revenue_at_risk": 0, "share": 3, "share_of_customers": 3}).to_string())
    print(f"\nTotal revenue at risk: EUR {now.revenue_at_risk.sum():,.0f} "
          f"(~{now.revenue_at_risk.sum() / now.rev_last6.sum():.0%} of last-6-month revenue)")
    print("Secret uncovered: since month 15 the South region's late-delivery rate jumped "
          "(new logistics partner) -> rework + complaints -> customers leave 1–3 months later.")
