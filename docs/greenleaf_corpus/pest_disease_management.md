---
title: Greenhouse Pest and Disease Management Context
source_name: Compiled from CEA horticulture references and Canadian greenhouse industry reporting
source_url: https://thegrower.org/news/green-lighting-canadian-agricultural-growth
fetch_date: 2026-05-27
category: pest_disease
region: Canada / BC
---

# Greenhouse Pest and Disease Management Context

GreenLeaf's sensor alerts are overwhelmingly dominated by High Pest Pressure events (about 99.5% of all alerts in the dataset). This document provides context for interpreting pest and disease management in BC greenhouses.

## Why pest pressure dominates
- In controlled environments, the warm, humid, stable conditions that are ideal for crops are also ideal for many pests and pathogens.
- Pest pressure tends to build continuously and can spike quickly, which is why it generates the majority of sensor alerts in a monitoring system tuned to flag rising risk.

## Major disease challenges in Canadian greenhouses
- Tomato brown rugose fruit virus (ToBRFV), the "rugose" virus, is a significant ongoing challenge for greenhouse growers in British Columbia, Ontario, and Quebec.
- It is difficult to detect and manage because it is stable and easily transmitted by mechanical means (tools, hands, contact).
- It causes misshapen fruit and discoloration, with documented deficits in yield, fruit size, and flavor even in tolerant varieties.
- It has also disrupted international trade, with import restrictions on tomato and pepper seed, transplants, and fruit from affected countries.

## Integrated and reduced pest strategies
- Integrated Pest Management (IPM) combines multiple control approaches (biological controls, monitoring, targeted treatment) in a more balanced, targeted way than routine calendar spraying.
- Reduced-pest strategies test whether lower chemical intervention can still protect crop outcomes, which matters for both cost and market/environmental positioning.
- GreenLeaf's experimental treatments include both Integrated Pest and Reduced Pest strategies, allowing direct comparison against routine baseline management.

## Precision response to pest alerts
- The value of a pest alert depends on response speed. Acting quickly when pest pressure rises can prevent a manageable problem from becoming a yield-threatening outbreak.
- This connects directly to GreenLeaf's action_delay_days and post_action_stress_delta metrics: faster responses to pest-pressure alerts are associated with better stress recovery.

## Relevance to GreenLeaf
Because nearly all of GreenLeaf's alerts are pest-pressure alerts, the operation's "precision response" story is largely a pest-response story. The dashboard's response-time findings (same-day response producing substantially more stress reduction than delayed response) are therefore primarily about how quickly the team responds to rising pest pressure. The Integrated Pest and Reduced Pest treatment plots offer a way to evaluate whether smarter pest strategies improve outcomes versus routine spraying.
