# Golden Dataset — Agentic RAG QA

20 verified questions across Tesla, GM, Ford (2022–2025) 10-K filings.
All answers are grounded in specific sections of the filings.

**Structure:** 5 Tesla + 5 GM + 5 Ford + 5 Cross-company

---

## Tesla (5 questions)

### T1 — Simple
**Question:** What were Tesla's total revenues in 2024?

**Expected answer:** Tesla recognized total revenues of $97.69 billion in 2024, representing an increase of $917 million compared to the prior year.

**Source:** Tesla 2024 10-K — ITEM 7. Management's Discussion and Analysis

---

### T2 — Simple
**Question:** Who is Tesla's Chief Executive Officer?

**Expected answer:** Elon Musk is Tesla's Chief Executive Officer, also referred to as Technoking of Tesla.

**Source:** Tesla 2024 10-K — ITEM 1A. Risk Factors

---

### T3 — Medium
**Question:** How did Tesla's automotive gross margin change from 2022 to 2024?

**Expected answer:** Tesla's automotive gross margin declined from 28.5% in 2022 to 19.4% in 2023 and further to 18.4% in 2024.

**Source:** Tesla 2024 10-K — ITEM 7. Cost of Revenues and Gross Margin

---

### T4 — Medium
**Question:** How did Tesla's R&D spending change from 2023 to 2024?

**Expected answer:** Tesla's R&D expenses increased by $571 million (14%) in 2024 compared to 2023, rising from $3.969 billion to $4.540 billion.

**Source:** Tesla 2024 10-K — ITEM 7. Research and Development

---

### T5 — Hard
**Question:** What were Tesla's main risk factors related to its dependence on key personnel in 2024?

**Expected answer:** Tesla identified a heavy dependence on Elon Musk as a key risk. Musk is involved in multiple ventures outside Tesla, and his departure or reduced involvement could materially harm the company. No key employee is bound by a fixed-term employment agreement.

**Source:** Tesla 2024 10-K — ITEM 1A. Risk Factors

---

## GM (5 questions)

### G1 — Simple
**Question:** What was GM Financial's total revenue in 2024?

**Expected answer:** GM Financial's total revenue was $15.875 billion in 2024, up from $14.225 billion in 2023.

**Source:** GM 2024 10-K — ITEM 7. GM Financial Revenue

---

### G2 — Simple
**Question:** What was GM's net income attributable to stockholders in 2024?

**Expected answer:** GM's net income attributable to stockholders was $6.008 billion in 2024.

**Source:** GM 2024 10-K — Consolidated Statements of Income

---

### G3 — Medium
**Question:** How did GM's net income change from 2022 to 2024?

**Expected answer:** GM's net income attributable to stockholders was $9.934 billion in 2022, $10.127 billion in 2023, and $6.008 billion in 2024 — a significant decline in 2024.

**Source:** GM 2024 10-K — Consolidated Statements of Income

---

### G4 — Medium
**Question:** What were GM's main cybersecurity risks in 2024?

**Expected answer:** GM highlighted cybersecurity threats as a significant operational risk, including risks to its connected vehicles, manufacturing systems, and customer data. The company noted that a successful cyberattack could disrupt operations and harm its reputation.

**Source:** GM 2024 10-K — ITEM 1C. Cybersecurity

---

### G5 — Hard
**Question:** What were GM's key risks related to its electric vehicle strategy in 2024?

**Expected answer:** GM identified risks including the pace of EV adoption being slower than anticipated, intense competition in the EV market, high capital requirements for EV manufacturing, and dependence on battery supply chains. The company also noted uncertainty around government EV incentives and charging infrastructure availability.

**Source:** GM 2024 10-K — ITEM 1A. Risk Factors

---

## Ford (5 questions)

### F1 — Simple
**Question:** What was Ford's net income attributable to Ford Motor Company in 2024?

**Expected answer:** Ford's net income attributable to Ford Motor Company was $5.879 billion in 2024, compared to $4.347 billion in 2023.

**Source:** Ford 2024 10-K — ITEM 7. Management's Discussion and Analysis

---

### F2 — Simple
**Question:** What was Ford's total revenue in 2024?

**Expected answer:** Ford's total revenues were $184.992 billion in 2024.

**Source:** Ford 2024 10-K — Consolidated Statements of Income

---

### F3 — Medium
**Question:** How did Ford's net income change from 2023 to 2024?

**Expected answer:** Ford's net income attributable to Ford Motor Company increased by $1.532 billion in 2024 compared to 2023, rising from $4.347 billion to $5.879 billion, driven by lower special items and higher Ford Pro segment performance.

**Source:** Ford 2024 10-K — ITEM 7. Management's Discussion and Analysis

---

### F4 — Medium
**Question:** What restructuring risks did Ford highlight in its 2024 10-K?

**Expected answer:** Ford highlighted risks that restructuring actions — such as plant closures, shift reductions, and programme cancellations — may not deliver anticipated cost savings, could harm its reputation, and may cause the company to incur significant charges. Supplier financial distress was also cited as a production constraint risk.

**Source:** Ford 2024 10-K — ITEM 1A. Risk Factors

---

### F5 — Hard
**Question:** What were Ford's total revenues across 2022, 2023, and 2024, and what was the trend?

**Expected answer:** Ford's total revenues were $158.057 billion in 2022, $176.191 billion in 2023, and $184.992 billion in 2024 — a consistent upward trend across all three years.

**Source:** Ford 2024 10-K — Consolidated Statements of Income

---

## Cross-Company (5 questions)

### X1 — Simple
**Question:** Which company had higher total revenues in 2024 — Tesla or Ford?

**Expected answer:** Ford had significantly higher total revenues than Tesla in 2024. Ford's total revenues were $184.992 billion while Tesla's total revenues were $97.69 billion — Ford was nearly twice the size of Tesla by revenue.

**Source:** Tesla 2024 10-K — ITEM 7; Ford 2024 10-K — Consolidated Statements of Income

---

### X2 — Medium
**Question:** How did Tesla's and Ford's net income compare in 2024?

**Expected answer:** Tesla's net income attributable to common stockholders was $7.091 billion in 2024, while Ford's net income attributable to Ford Motor Company was $5.879 billion. Tesla was more profitable on a net income basis despite Ford having much higher revenues.

**Source:** Tesla 2024 10-K — Consolidated Statements of Operations; Ford 2024 10-K — ITEM 7

---

### X3 — Medium
**Question:** What cybersecurity risks did Tesla and GM have in common in 2024?

**Expected answer:** Both Tesla and GM identified cybersecurity threats as material risks in 2024. Both companies highlighted risks of operational disruption, intellectual property theft, and reputational damage from cyberattacks. Tesla specifically referenced risks under Item 106(a) of Regulation S-K, while GM noted risks to its connected vehicle systems and customer data.

**Source:** Tesla 2024 10-K — ITEM 1C. Cybersecurity; GM 2024 10-K — ITEM 1C. Cybersecurity

---

### X4 — Hard
**Question:** How did Tesla's automotive gross margin trend from 2022 to 2024 compare to Ford's net income trend over the same period?

**Expected answer:** Tesla's automotive gross margin declined sharply from 28.5% in 2022 to 18.4% in 2024, indicating significant margin compression. Ford's net income attributable to Ford Motor Company went from a loss of $1.981 billion in 2022 to $4.347 billion in 2023 and $5.879 billion in 2024 — a dramatic recovery. The contrast shows Tesla facing profitability pressure while Ford reversed a net loss into consistent profit over the same period.

**Source:** Tesla 2024 10-K — ITEM 7; Ford 2024 10-K — ITEM 7

---

### X5 — Hard
**Question:** Which of the three companies — Tesla, GM, or Ford — highlighted the most significant risks related to electric vehicle adoption in their 2024 10-K filings?

**Expected answer:** All three companies highlighted EV-related risks, but the emphasis differed. Tesla, as a pure-play EV company, focused on production scaling, battery supply, and margin pressure from price competition. GM highlighted slower-than-expected EV adoption, capital intensity, and charging infrastructure gaps as it transitions its legacy business. Ford flagged EV programme costs and restructuring risks tied to its EV investments. GM's risk disclosure was arguably the most extensive given its ongoing transition from an ICE-dominant business.

**Source:** Tesla 2024 10-K — ITEM 1A; GM 2024 10-K — ITEM 1A; Ford 2024 10-K — ITEM 1A

---

## How to Use

Run each question through `query.py` and compare the agent's answer to the expected answer:

```bash
python3 query.py "What were Tesla's total revenues in 2024?"
```

Check:
1. **Factual accuracy** — does the answer match the expected answer?
2. **Citations** — does every fact have an inline citation?
3. **Tool calls** — did the agent retrieve from the right company and year?
4. **Tool call count** — single-company questions should use 1-2 calls; cross-company questions 2-4 calls
