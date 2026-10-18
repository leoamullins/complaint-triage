---
title: Complaint Triage
emoji: 📨
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
short_description: Routes consumer finance complaints by product
---

# Complaint Triage

Classifies US consumer finance complaints into 10 product groups with calibrated confidence. Complaints below 0.80 confidence are marked for a person to review; the rest can be routed automatically.

Fine-tuned RoBERTa-base on CFPB complaint narratives: test macro-F1 0.753, expected calibration error 0.018. At the 0.80 threshold, 25% of complaints go to a person and the auto-routed rest are 93.7% accurate.

## Use it

Interactive docs: `/docs`

```bash
curl -X POST https://leoamullins-complaint-triage.hf.space/classify \
  -H "Content-Type: application/json" \
  -d '{"texts": ["They keep calling me about a debt I already paid off years ago."]}'
```

```json
{"results": [{"product": "Debt collection", "confidence": 0.955, "route": "auto"}]}
```

`POST /classify` takes 1 to 64 texts. `GET /health` returns the class labels.

Code, data pipeline and evaluation: https://github.com/leoamullins/complaint-triage
