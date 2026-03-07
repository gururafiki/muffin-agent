A professional, role-based public-equity investment process in a hedge-fund / prop-style setting can be organized as a 15-step, end‑to‑end loop from idea sourcing through post‑mortem attribution.  The same skeleton works for long‑only, multi‑manager pods, and quant/prop firms with role weights and a few steps shifting (detailed in the variations section).[1][2][3][4]

***

## E2E Workflow Overview

### Linear lifecycle (15 core steps)

| # | Step | Primary Owner(s) | Key Decision / Gate |
|---|------|------------------|---------------------|
| 1 | Idea Sourcing & Screening | Analyst | Does the idea clear minimum quantitative, quality, and liquidity screens to warrant work? [5][6] |
| 2 | Market Regime & Top‑Down Context | PM, Risk | Is the current macro/liquidity regime supportive or at least not hostile to the idea’s factor, sector, and style profile? [7][8] |
| 3 | Sector / Industry & Thematic View | Analyst, PM | Is the industry structure, cycle position, and thematic backdrop attractive enough to seek alpha in this space now? [9][10] |
| 4 | Business, Moat, Mgmt & ESG Triage | Analyst | Does the company pass “must‑have” qualitative checks on business model, moat, management, governance, and material ESG risks? [6][11] |
| 5 | Financial Quality & Fundamental Deep Dive | Analyst | Do historical financials show durability (growth, margins, FCF, balance sheet strength) consistent with the targeted style (quality/value/growth)? [6][12] |
| 6 | Forecasting & Scenario Modeling | Analyst, PM | Are forward projections and scenarios (bull/base/bear) internally coherent and grounded enough to support a thesis? [1][13] |
| 7 | Valuation & Relative Value | Analyst, PM | Is the expected return vs. intrinsic value and peers compelling, with a clear mispricing vs. the current market? [6][14] |
| 8 | Risk & Downside / Stress Testing | Risk, PM | Is downside (idiosyncratic + factor + liquidity) acceptable vs. portfolio risk budget, with pre‑defined loss and risk triggers? [15][4] |
| 9 | Thesis Synthesis & Investment Case | Analyst, PM | Is there a crisp bull/base/bear thesis, catalysts roadmap, and conviction score that justify capital and risk usage? [6][13] |
|10| Portfolio Fit & Position Sizing | PM, Risk | Does size, gross/net, and factor exposure fit portfolio objectives and constraints, given correlations and tracking error? [16][17] |
|11| Pre‑Trade Approval & Compliance Checks | PM, Risk, Compliance | Does the trade meet mandate, limits, ESG/negative screens, and regulatory/issuer restrictions; any hard vetoes? [11][18] |
|12| Execution & Implementation | Trader | Is the order executed with appropriate urgency, venue selection, and market‑impact control relative to alpha decay? [19][20] |
|13| Ongoing Monitoring & Thesis Drift | Analyst, PM, Risk | Are fundamentals, price action, sentiment, ESG, and risk signals still aligned with the original thesis and risk budget? [7][20] |
|14| Exit / Trim / Add Decision | PM | Is the position upgraded, reduced, exited, or rotated based on thesis break, valuation, better opportunities, or risk triggers? [21][22] |
|15| Post‑Mortem & Performance Attribution | Risk, PM, Analyst | What portion of P&L came from stock‑picking vs. factors, sizing, timing, and macro; what process changes are required? [23][24] |

Key handoffs and escalations:
- Analyst → PM at Steps 3–4 (green‑light for full work) and 9 (IC‑style decision).[13][1]
- PM ↔ Risk at Steps 2, 8, 10, 11, 13 for vetoes on risk, exposure, or mandate breaches.[4][16]
- PM → Trader at Step 12, with feedback loops from execution quality back to sizing and liquidity assumptions.[19][20]

***

## Detailed Steps and Responsibilities

Below, each step is specified in an agent‑executable format: **What**, **Who Owns**, **Inputs/Outputs**, **Success Criteria/Gates**.

### 1. Idea Sourcing & Screening

- **What**: Systematically generate candidates via quant screens (value, quality, momentum, size, volatility), 13F/ownership moves, insider activity, and manager letters, then apply hard filters for liquidity, market‑cap, and investability universe (region, sectors, ESG exclusions).[5][6][25]
- **Who Owns**: Analyst owns; PM is approver for adding a name to the “active research list.”[1]
- **Inputs/Outputs**: Inputs: screening universes, factor definitions, watchlists, external idea sources; Output: ranked list of candidates tagged with key metrics and flags (e.g., value+quality, high short interest).[6][5]
- **Success Criteria/Gates**: Name only progresses if it meets minimum liquidity, free‑float, and factor/quality thresholds, and is within sector and mandate scope; otherwise it is logged and discarded or watch‑listed.[4][6]

### 2. Market Regime & Top‑Down Context

- **What**: Classify the current macro/market regime (e.g., high growth/low inflation, stagflation, tightening liquidity) using indicators like GDP, PMIs, inflation, policy rates, curves, volatility indices, and liquidity proxies, and map how regime historically impacts the idea’s style and sector.[7][8][26]
- **Who Owns**: PM and Risk co‑own; macro/quant team (if present) provides regime classification; Analyst uses this as context, not as a stop‑go on stock‑level alpha.[27][7]
- **Inputs/Outputs**: Inputs: macro datasets, regime models, factor performance by regime; Output: labeled regime, factor tailwind/headwind assessment, and recommended beta/gross/net ranges.[28][7]
- **Success Criteria/Gates**: Idea can still advance in adverse regimes but may require lower sizing, lower net exposure, tighter stop‑loss bands, or explicit justification of why idiosyncratic alpha dominates regime drag.[16][7]

### 3. Sector / Industry & Thematic View

- **What**: Analyze industry structure (5‑forces, concentration, barriers), cycle position, regulatory backdrop, and major themes (e.g., energy transition, AI, healthcare policy) to determine if the “pond” is attractive and how competition and dispersion look for long and short ideas.[9][10][29]
- **Who Owns**: Sector Analyst owns; PM approves sector risk budget and focuses (e.g., overweight/underweight sectors or themes).[2][30]
- **Inputs/Outputs**: Inputs: industry reports, sector returns vs. market, dispersion, M&A and regulatory trends; Output: sector scorecard (tailwinds/headwinds, cycle stage), peer list, and thematic mappings.[31][9]
- **Success Criteria/Gates**: Only proceed if there is sufficient dispersion and prospective alpha (e.g., clear winners/losers, event opportunities) and the sector does not breach portfolio concentration or thematic risk limits.[16][4]

### 4. Business, Moat, Management & ESG Triage

- **What**: Rapid qualitative triage of the company’s business model, competitive advantage, management quality, capital allocation, governance, and material ESG issues before heavy modeling effort.[11][12][6]
- **Who Owns**: Analyst owns; PM and ESG/Stewardship specialist (if present) act as approvers for borderline governance/ESG names.[18][11]
- **Inputs/Outputs**: Inputs: 10‑K/annual report, investor presentations, transcripts, ownership/insider data, ESG ratings and controversies; Outputs: one‑pager with business description, moat assessment, management/ownership notes, and ESG red/amber/green flags.[32][5][6]
- **Success Criteria/Gates**: Must pass Tier‑1 “must‑haves” (understandable business, non‑toxic governance, no extreme leverage, no uninvestable ESG controversies under mandate) or be rejected or kept as a short‑only candidate.[33][6][11]

### 5. Financial Quality & Fundamental Deep Dive

- **What**: Analyze historical financials for growth, margin structure, cash conversion, leverage, capital intensity, and capital allocation to classify quality and cyclicality, identifying key drivers, red flags, and fundamental sensitivities.[12][6]
- **Who Owns**: Analyst owns; PM reviews for alignment with style (e.g., quality, value, turnaround), Risk validates leverage and solvency metrics vs. firm standards.[1][4]
- **Inputs/Outputs**: Inputs: at least 5–10 years of income, balance sheet, cash flow statements, segment/geographic data; Outputs: cleaned time series, key ratios (ROIC, FCF yield, leverage, coverage, Z‑score), and initial driver map (pricing, volume, mix, cost, FX).[6][12]
- **Success Criteria/Gates**: Company should either show robust, explainable economics (for long) or clear structural weaknesses (for short); if fundamentals are too noisy or opaque to model reliably, idea is downgraded or dropped.[20][12][6]

### 6. Forecasting & Scenario Modeling

- **What**: Build a forward model (typically full three‑statement or at least revenue/EBIT/FCF + balance‑sheet items) with explicit assumptions for volumes, pricing, margins, capex, working capital, and capital allocation, then construct bull/base/bear scenarios and probabilities.[13][1]
- **Who Owns**: Analyst owns modeling; PM approves key assumptions (long‑term growth, normalized margins, terminal economics) and scenario probabilities.[13][1]
- **Inputs/Outputs**: Inputs: historical model from Step 5, management guidance, industry KPIs, macro/sector linkages; Outputs: forward financials under three scenarios, sensitivity tables (e.g., ±x% volume/margin), and key “what‑if” levers.[5][13]
- **Success Criteria/Gates**: Model must be internally consistent (balance sheet ties, cash uses funded), assumptions must be grounded in external and historical evidence, and scenarios must show asymmetric payoff, not just linear tweaks on current consensus.[1][13]

### 7. Valuation & Relative Value

- **What**: Apply multiple valuation methods appropriate for the business (DCF, multiples, sum‑of‑parts, residual income) and benchmark against peers and history, incorporating current market narratives and factor exposures.[6][1]
- **Who Owns**: Analyst owns valuation work; PM challenges inputs and selects the decision‑relevant metrics (e.g., EV/EBITDA vs. P/E vs. EV/FCF) and target multiples/discount rates.[14][6]
- **Inputs/Outputs**: Inputs: model outputs, peer group metrics, historical multiples, factor and style diagnostics; Outputs: valuation range by method, implied upside/downside, and relative value stack vs. peers and alternatives.[14][6]
- **Success Criteria/Gates**: Proceed only if there is a clear dislocation: e.g., target trades at a discount/premium relative to its quality and growth profile vs. peers and intrinsic value, with a plausible mechanism and catalysts for re‑pricing.[20][16][6]

### 8. Risk & Downside / Stress Testing

- **What**: Quantify idiosyncratic and systematic risks via factor models, beta/volatility, drawdown analysis, liquidity, crowding, and stress scenarios (macro shocks, regulatory events, ESG incidents), defining explicit stop‑loss, max drawdown, and risk triggers ex‑ante.[15][7][4]
- **Who Owns**: Risk owns portfolio‑level and factor analysis; PM owns trade‑off between alpha and risk budget; Analyst supplies idiosyncratic risk map and event list.[4][16]
- **Inputs/Outputs**: Inputs: position‑level factor loadings, trading/liquidity stats, borrow/short data if applicable, macro stress templates; Outputs: VaR/expected shortfall, factor contribution to risk, concentration metrics, downside scenarios with P&L impacts.[34][4]
- **Success Criteria/Gates**: Trade is blocked or resized if it breaches risk limits (ex‑ante drawdown, single‑name, sector, factor, liquidity, or ESG constraints) or if downside in credible bear scenarios is disproportionate to thesis quality and sizing rules.[21][15][4]

### 9. Thesis Synthesis & Investment Case

- **What**: Convert analysis into a concise investment case: 1–2 line thesis, key drivers, edge vs. consensus, bull/base/bear with probabilities, expected return distribution, catalysts timeline, and pre‑defined “thesis break” conditions.[6][13]
- **Who Owns**: Analyst drafts; PM owns final thesis, conviction score, and go/no‑go decision (often via an IC‑style or pod‑level discussion).[13][1]
- **Inputs/Outputs**: Inputs: all previous steps, consensus estimates, sentiment/positioning data; Outputs: written memo/IC deck, explicit conviction rating (e.g., 1–5) and time horizon, with checklists completed (fundamental, risk, ESG).[35][11][6]
- **Success Criteria/Gates**: Idea is approved only if edge vs. consensus is clear, catalysts and verification points are identified, downside is tolerable at proposed sizing, and the thesis can be monitored via specific KPIs and events.[6][13]

### 10. Portfolio Fit & Position Sizing

- **What**: Translate conviction and risk characteristics into size, entry plan, and hedge structure, accounting for portfolio diversification, factor exposures, correlation, liquidity, and targeted tracking error or Sharpe; enforce “survival constraints” on single‑name and theme concentration.[17][14][16]
- **Who Owns**: PM owns sizing and portfolio fit; Risk has veto on exposures and concentration; Analyst recommends size band based on conviction and risk profile.[16][4]
- **Inputs/Outputs**: Inputs: portfolio analytics (factor and sector exposures, correlation matrix, risk budget, gross/net), conviction scores, liquidity/volatility stats; Outputs: target and max position size, gross/net impact, hedges (index, sector, factor), and scaling plan (build/trim/add rules).[14][4]
- **Success Criteria/Gates**: Final size respects hard risk constraints, conviction hierarchy (larger weights only for highest‑quality ideas), and liquidity (e.g., days‑to‑liquidate), avoiding accidental concentration and “false equality” across names.[17][4][16]

### 11. Pre‑Trade Approval & Compliance Checks

- **What**: Run the order through mandate, regulatory, and internal policy checks, including restricted lists, position/issuer/sector/ESG limits, short‑selling/borrow availability, and any client‑specific constraints.[11][18]
- **Who Owns**: PM initiates; Risk verifies risk/limit usage; Compliance has hard veto for legal, regulatory, mandate, or restricted‑list violations.[32][11]
- **Inputs/Outputs**: Inputs: proposed orders with size and instrument details, current holdings and limits, compliance rules engine, restricted lists, borrow/locate data; Outputs: approved, modified, or rejected orders with rationale and any conditions (e.g., partial size, time‑in‑force).[18][20]
- **Success Criteria/Gates**: No order proceeds to the trader without automated and manual checks clearing; exceptions require documented overrides with sign‑off from designated approvers (e.g., PM + CCO).[11][18]

### 12. Execution & Implementation

- **What**: Execute orders using appropriate algorithms and venues (VWAP, POV, liquidity‑seeking) balancing market impact, information leakage, and alpha decay, while monitoring fills vs. benchmarks and adapting tactics to intraday conditions.[19][20]
- **Who Owns**: Trader owns execution; PM sets urgency and tolerance for slippage; Risk oversees aggregate trading risk and limits.[2][19]
- **Inputs/Outputs**: Inputs: approved orders, current book, market data, venue/algorithm choices, borrow/collateral info; Outputs: fills, execution reports vs. benchmarks (VWAP, arrival price), and updated positions and cash/financing usage.[19][20]
- **Success Criteria/Gates**: Execution achieves acceptable slippage vs. pre‑trade estimates and respects risk and exposure constraints; large deviations or liquidity issues trigger escalation to PM/Risk and potential order revision.[16][19]

### 13. Ongoing Monitoring & Thesis Drift

- **What**: Continuously monitor fundamentals, price/volume, factor behavior, sentiment, ESG incidents, and risk metrics; update models after earnings and events, and re‑underwrite the thesis when key KPIs or regime conditions change.[7][18][20]
- **Who Owns**: Analyst owns name‑level monitoring; PM owns overall portfolio monitoring; Risk oversees exposures, factor drifts, and limit utilization.[4][16]
- **Inputs/Outputs**: Inputs: earnings releases, transcripts, news feeds, alternative data, broker research, ESG alerts, risk reports; Outputs: periodic update notes, revised forecasts and conviction scores, and status tags (e.g., “on track,” “watch,” “review,” “thesis break candidate”).[7][18][13]
- **Success Criteria/Gates**: Positions are reviewed on a fixed cadence and around scheduled catalysts; significant negative developments (fundamental, ESG, or regime) or risk breaches automatically trigger re‑assessment and potential de‑risking per pre‑defined rules.[21][18][7]

### 14. Exit / Trim / Add Decision

- **What**: Make explicit “upgrade/hold/trim/exit” decisions based on valuation reaching target, thesis realization or break, better capital uses (“upgrade” ideas), risk signals, liquidity conditions, and pre‑set stops/targets and time‑based reviews.[22][20][21]
- **Who Owns**: PM owns decision; Analyst recommends with updated thesis and valuation; Risk enforces stop‑loss and drawdown policies where codified.[21][4]
- **Inputs/Outputs**: Inputs: updated valuation and thesis, P&L path, risk metrics, opportunity set vs. existing and new ideas; Outputs: closing or resizing orders plus short rationale notes explaining whether the move is due to thesis completion, break, upgrade to a better idea, or risk control.[22][21]
- **Success Criteria/Gates**: Exits and trims align with inverse of buy criteria (thesis broken, weaker capital allocation vs. alternatives, valuation stretched, risk intolerable), not purely short‑term price moves, and are documented to reduce behavioral biases.[35][22][21]

### 15. Post‑Mortem & Performance Attribution

- **What**: After material exits or periodically, conduct structured post‑mortems and performance attribution separating security selection, factor exposures, timing, sizing, and execution, and assess decision quality vs. outcomes.[23][24][36]
- **Who Owns**: Risk owns quantitative attribution; PM and Analyst jointly own qualitative post‑mortems; firm leadership (CIO/Head of Risk) reviews patterns and process changes.[37][23]
- **Inputs/Outputs**: Inputs: trade and position histories, factor returns, benchmarks, original memos, and monitoring notes; Outputs: attribution reports (e.g., stock‑picking vs. beta/factors), behavioral/decision analysis, and updated checklists, sizing rules, and process improvements.[24][23][35]
- **Success Criteria/Gates**: Findings are fed back into screening, modeling, risk limits, and sizing frameworks (e.g., refining “must‑have” checks or stop rules) rather than remaining descriptive; repeated process failures trigger explicit remediation (training, rule changes, or role adjustments).[36][24][37]

***

## Role Accountability Matrix

Indicative ownership / veto weights by step and core roles (percentages are approximate “share of decision control” for that step).

| Step | Analyst | PM | Risk | Trader | Compliance |
|------|--------|----|------|--------|------------|
| 1. Idea Sourcing & Screening | 80% own (design and run screens, initial triage) [5][6] | 15% veto (can refuse coverage) [1] | 5% veto (liquidity/universe rules) [4] | 0% | 0% |
| 2. Market Regime & Top‑Down | 20% use (context only) [7] | 40% own (positioning vs. regime) [7][29] | 40% veto (beta/gross/net, factor risk) [4][16] | 0% | 0% |
| 3. Sector / Industry View | 70% own (sector analysis, peer set) [9][10] | 25% veto (sector risk budget) [2] | 5% consult (sector limits) [4] | 0% | 0% |
| 4. Business, Moat, Mgmt & ESG | 70% own (qualitative checks) [6][12] | 10% veto (doesn’t fit style) [1] | 0% | 0% | 20% veto (ESG/mandate exclusions) [11][18] |
| 5. Financial Deep Dive | 80% own (financial modeling, quality assessment) [6][12] | 15% veto (style fit, economics) [1] | 5% veto (leverage/solvency) [4] | 0% | 0% |
| 6. Forecasting & Scenarios | 70% own (model build, scenarios) [1][13] | 25% veto (key assumptions, probabilities) [13] | 5% consult (macro/FX/credit stresses) [7] | 0% | 0% |
| 7. Valuation & Relative Value | 70% own (methods, comps, ranges) [6][14] | 25% veto (decision metric, target) [1] | 5% consult (valuation vs. factor risk) [14] | 0% | 0% |
| 8. Risk & Downside | 30% own (idiosyncratic risk map) [15] | 30% own (accept/reject risk–reward) [4] | 40% veto (limits, stress, VaR) [4][16] | 0% | 0% |
| 9. Thesis & Investment Case | 60% own (memo, thesis articulation) [6][13] | 35% veto (go/no‑go, conviction) [1] | 5% consult (risk caveats) [4] | 0% | 0% |
|10. Portfolio Fit & Sizing | 25% consult (conviction input) [17] | 45% own (final size, hedges) [16][14] | 30% veto (exposures, concentration) [4][16] | 0% | 0% |
|11. Pre‑Trade & Compliance | 10% consult (clarify thesis/urgency) [11] | 30% own (submit orders) [18] | 20% veto (risk/limit issues) [4] | 0% | 40% veto (mandate/regulation) [11][18] |
|12. Execution | 5% consult (urgency, price sensitivity) [19] | 15% consult (high‑level urgency) [19] | 10% veto (trading limits) [4] | 70% own (route, algo, execution) [19][20] | 0% |
|13. Monitoring & Drift | 50% own (fundamentals, news, ESG) [7][18] | 25% own (portfolio‑level view) [4] | 25% veto (breach‑driven actions) [4][16] | 0% | 0% |
|14. Exit / Trim / Add | 30% consult (updated thesis/valuation) [22][21] | 55% own (final decision, rotation) [21] | 15% veto (forced de‑risking, stops) [4][15] | 0% | 0% |
|15. Post‑Mortem & Attribution | 30% own (qualitative review) [36] | 30% own (decisions & lessons) [24] | 40% own (quant attribution, diagnostics) [23][37] | 0% | 0% |

***

## Discovered Variations Across Firm Types

### Responsibility and process shifts

| Area | Long‑Only Fundamental | Hedge (L/S, Multi‑Manager, Pods) | Quant / Prop / Market‑Making |
|------|-----------------------|-----------------------------------|------------------------------|
| Use of shorts | Rare or for limited hedging; bulk of work on longs and benchmarks. [38][39] | Symmetric long/short pipelines; explicit short thesis, borrow, and crowding checks; market‑neutral or low‑beta targets common. [20][34][2] | Shorts primarily arise from model‑driven signals, relative value pairs, or inventory hedging; no qualitative “short thesis” memo for every leg. [19][40] |
| Portfolio construction | Benchmarked, higher net beta, lower tracking error; PM emphasizes relative performance and risk vs. index. [38][16] | Gross/net, factor, and exposure tightly risk‑budgeted; multi‑manager platforms dynamically re‑allocate capital across pods based on Sharpe and drawdown rules. [41][3][42] | Book optimized around risk models, capacity, and turnover; sizing often automated from signal strength, risk, and cost models. [19][40][26] |
| Market regime use | Macro mostly for tilt/overlays and sector tilts; stock‑picking remains primary driver. [29][43] | Regime feeds into gross/net, factor tilts, and sector/strategy risk budgets (e.g., long/short equity vs. macro vs. arbitrage). [31][29][7] | Regimes often embedded in the models themselves (regime‑switching, volatility, liquidity) and can auto‑scale risk or flip signals. [7][27][28] |
| Role of Risk | Emphasis on tracking error vs. benchmark and client guidelines; risk more “advisory” in some shops. [16] | Central risk has strong veto; drawdown and VaR triggers can cut or close pod capital (e.g., 5% / 7.5% pod stops at platforms like Millennium). [41][44][42] | Risk team merged with quant research; model validation, stress testing, and kill‑switches for strategies are core responsibilities. [40][26] |
| Execution | Often via external brokers/OMS with focus on minimizing cost vs. benchmark; less intraday trading. [39] | Dedicated execution/trading with internal algos, broker selection, and financing/stock‑loan optimization. [2][30][4] | Execution and research deeply intertwined; heavy emphasis on market microstructure, low‑latency, and automation (e.g., Jane Street). [19][40][45] |
| ESG & stewardship | ESG integration, engagement, and voting central to process; stewardship teams important. [11][18] | ESG more often used as risk screen and controversy monitor; engagement variable by mandate. [11][33] | ESG usually only when trading client capital under ESG‑restricted mandates; prop books often have minimal ESG overlays. [11][33] |
| Monitoring cadence | Quarterly earnings cycles, regular strategy reviews; lower turnover portfolios. [46][39] | Higher frequency P&L, factor, and exposure monitoring; daily risk calls common, higher turnover in some strategies. [31][4] | Monitoring is near‑real‑time with automated alerts for breakdowns in model behavior, slippage, and risk; human review focuses on anomalies. [19][40][26] |
| Post‑mortem depth | Emphasis on performance vs. benchmark, client communication, and sell discipline reviews. [22][23] | Strategy‑ and pod‑level attribution informs capital allocation and PM turnover; platform “Darwinism” prunes underperforming teams. [41][3][31] | Systematic, data‑driven evaluation of models and signals; experiments tracked as in R&D, with rigorous A/B testing of model changes. [40][26][24] |

***

## Research Appendix

- **Key sources and frameworks used**  
  - Fundamental analysis and equity‑research checklists from recent practitioner guides (e.g., Shyft/Lockstep and Winvesta checklists for screening, competitive analysis, management and financial health).[25][5][6]
  - ESG integration and stewardship processes from PRI technical guides and ESG due‑diligence frameworks highlighting policy, governance, integration into analysis, and monitoring/reporting.[47][18][32][11]
  - Long/short equity and hedge‑fund industry primers detailing risk management, gross/net and factor control, and strategy behavior across regimes.[10][29][31][20][4]
  - Firm‑specific descriptions of investment and trading approaches at Citadel, Millennium, and Jane Street, informing role boundaries, centralized risk, and quant/trading integration.[3][30][40][41][42][45][48][2][19]
  - Position sizing, sell discipline, and post‑mortem/decision‑attribution frameworks from Resonanz, Janus Henderson, Averaging Up, CFAs’ decision‑attribution work, and institutional post‑mortem case studies.[23][24][36][37][17][35][14][21]

- **Critical gaps and follow‑ups for an AI agent implementation**  
  - Many details on exact numeric limits (e.g., pod‑level VaR caps, firm‑specific drawdown rules, concentration thresholds) are proprietary; an AI agent should parameterize these as configuration inputs rather than hard‑coding.[41][42][4]
  - Quant and HFT/market‑making shops provide only high‑level public descriptions of model and risk‑control workflows; replicating their processes requires additional internal documentation on model approval, sandboxing, and rollback procedures.[26][40][19]
  - Client‑ and mandate‑specific ESG, regulatory, and reporting requirements vary widely; the agent should treat these as policy layers (mandate schema) over the generic workflow, populated from firm‑internal rule sets.[18][32][11]
  - Execution‑level details (algo selection logic, broker/routing preferences, crossing rules) are only partially documented in public sources and would need augmentation from internal best‑execution and trading‑desk manuals.[39][20][19]

Sources
[1] Equity Research: Equity Excellence: A Guide to Equity Research for Hedge Fund Careers - FasterCapital https://www.fastercapital.com/content/Equity-Research--Equity-Excellence--A-Guide-to-Equity-Research-for-Hedge-Fund-Careers.html
[2] Citadel International Equities https://www.citadel.com/what-we-do/equities/citadel-international-equities/
[3] Millennium Management | TrendSpider Learning Center https://trendspider.com/learning-center/millennium-management/
[4] [PDF] Edge with hedge: primer for equity long/short funds | Aurum https://www.aurum.com/wp-content/uploads/250930-Equity-long-short-funds-primer.pdf
[5] Invest smarter: A five-step checklist for company analysis - Shyft https://www.shyft.co.za/en-ZA/steps-for-better-fundamental-analysis
[6] Building your fundamental analysis checklist for stock picking https://www.winvesta.in/blog/investors/building-your-fundamental-analysis-checklist-for-stock-picking
[7] Global Macro Hedge Fund Strategies and Forecasting Models https://chenjiazizhong.com/2025/03/18/global-macro-hedge-fund-strategies-and-forecasting-models/
[8] [PDF] Incorporating Market Regimes into Large-Scale Stock Portfolios https://mpra.ub.uni-muenchen.de/121552/1/MPRA_paper_121552.pdf
[9] H2 2025 hedge fund outlook - Aberdeen Investments https://www.aberdeeninvestments.com/en-us/institutional/insights-and-research/h2-2025-hedge-fund-outlook
[10] Top Hedge Fund Industry Trends for 2025 | Portfolio for the Future https://caia.org/blog/2025/01/23/top-hedge-fund-industry-trends-2025
[11] ESG integration in listed equity: A technical guide https://www.unpri.org/listed-equity/esg-integration-in-listed-equity-a-technical-guide/11273.article
[12] Equity Research: 5 Important Fundamental Analysis Components https://www.equentis.com/blog/equity-research-fundamental-analysis/
[13] How to find and develop good ideas (discretionary l/s equity) https://www.wallstreetoasis.com/forum/hedge-fund/the-investment-process-how-to-find-and-develop-good-ideas-discretionary-ls-equity
[14] [PDF] OPTIMIZING RISK-ADJUSTED CONVICTION - Janus Henderson https://cdn.janushenderson.com/webdocs/Optimizing_Risk_Adjusted_Conviction_Implied_Alpha.pdf
[15] Risk Management Strategies In Long Short Equity Hedge Funds - FasterCapital https://fastercapital.com/topics/risk-management-strategies-in-long-short-equity-hedge-funds.html
[16] A More Appealing Environment for Equity Long/Short Strategies https://www.cambridgeassociates.com/insight/a-more-appealing-environment-for-equity-long-short-strategies/
[17] Position Sizing – The Final Act of Conviction - Averaging Up https://averagingup.com/position-sizing-the-final-act-of-conviction/
[18] [PDF] ESG INTEGRATION IN LISTED EQUITY: A TECHNICAL GUIDE https://www.saoicmai.in/elibrary/practical-guide-to-ESG-integration-for-equity-investing.pdf
[19] What We Do https://www.janestreet.com/what-we-do/overview/
[20] Understanding Long-Short Equity Hedge Funds Strategies https://www.tejwin.com/en/insight/long-short-equity/
[21] Position Sizing & Sell Discipline: A Modern Allocator's Framework https://resonanzcapital.com/insights/position-sizing-sell-discipline-a-modern-allocators-framework
[22] Sell Discipline and Overcoming Behavioral Biases | EP159 https://www.mawer.com/the-art-of-boring/podcast/from-buy-to-bye-sell-discipline-and-overcoming-behavioral-biases-ep159/
[23] Ex-Post Performance Attribution Analysis https://www.northstarrisk.com/expostperformanceattributionanalysis
[24] Decision Attribution: Portfolio Manager Skill vs. Past Performance https://blogs.cfainstitute.org/investor/2023/01/31/decision-attribution-portfolio-manager-skill-vs-past-performance/
[25] Iii) Historical Performance https://lockstep.beehiiv.com/p/checklist-for-fundamental-analysis-of-stocks
[26] [PDF] Systematic Macro Hedge Funds: Trending into the New Regime https://insightcommunity.mercer.com/api/v1/uploads/c0c9a252573b447794fb37aa0640cd95.pdf?public=true
[27] [PDF] Decoding Market Regimes Machine Learning Insights into US Asset ... https://www.ssga.com/library-content/assets/pdf/global/pc/2025/decoding-market-regimes-with-machine-learning.pdf
[28] Modeling Regime Structure and Informational Drivers of Stock ... https://arxiv.org/html/2504.18958v1
[29] 2025 Hedge fund investor barometer - Amundi Research Center https://research-center.amundi.com/article/2025-hedge-fund-investor-barometer
[30] Equities Businesses - Citadel https://www.citadel.com/what-we-do/equities/
[31] Aurum Hedge Fund Industry Deep Dive Q1 2025 review https://www.aurum.com/wp-content/uploads/Aurum-Hedge-Fund-Industry-Deep-Dive-Q1-2025.pdf
[32] ESG Due Diligence Checklist - ESG Investing Questionnaire - Neotas https://www.neotas.com/esg-due-diligence-checklist/
[33] A Practical Guide to ESG Integration for Equity Investing: Navigating the Intersection of Sustainability and Financial Performance - Accountend https://accountend.com/a-practical-guide-to-esg-integration-for-equity-investing-navigating-the-intersection-of-sustainability-and-financial-performance/
[34] Theory and evidence from long/short equity hedge funds https://www.sciencedirect.com/science/article/abs/pii/S0927539811000211
[35] [PDF] Conducting an Effective Post-Mortem Exercise for Better Investment ... https://www.td.com/content/dam/tdgis/document/ca/en/pdf/insights/thought-leadership/benefits-of-post-mortem-exercise-en.pdf
[36] Why Every Investor Needs a Post-Mortem Analysis (And How to Do It Right) https://www.worldlyinvest.com/p/post-mortem-analysis
[37] [PDF] Performance Analysis and Attribution with Alternative Investments https://uncipc.org/wp-content/uploads/2022/02/IPC-Performance-Attribution-Analysis-2022-01-23.pdf
[38] [PDF] Long-Short Equity - Meketa Investment Group https://meketa.com/wp-content/uploads/2019/12/Long-Short-Equity-FINAL.pdf
[39] What do long-only funds look for in a trading system? - LSEG https://www.lseg.com/en/insights/data-analytics/what-do-long-only-funds-look-for-in-a-trading-system
[40] Quantitative Research :: Jane Street https://www.janestreet.com/quantitative-research/
[41] Millennium Management's Multi-Strategy Trading Architecture https://navnoorbawa.substack.com/p/millennium-managements-multi-strategy
[42] Millennium's Pod System: How Platform Design Beats Star Portfolio ... https://www.confluencegp.com/articles-and-news/millennium-s-pod-system-how-platform-design-beats-star-portfolio-managers
[43] [PDF] 2024 Revealed: ten convictions for a blinding future - Candriam https://www.candriam.com/siteassets/campagne/outlooks/outlook-2024/202401/2024_01_outlook_2024_gb.pdf
[44] Millennium Management: Overview, History, and Investments https://www.fool.com/investing/how-to-invest/famous-investors/millennium-management/
[45] Jane Street: Trading, Research, And A Deep Dive https://vault.nimc.gov.ng/blog/jane-street-trading-research-and-a-deep-dive-1764814072
[46] Stocks for the Long Runs—Investing with Discipline https://www.linkedin.com/pulse/stocks-long-runsinvesting-discipline-nutshell-asset-management-kybze
[47] [PDF] Global ESG due diligence+ study 2024 https://assets.kpmg.com/content/dam/kpmg/xx/pdf/2024/06/esg-due-diligence-study-2024.pdf
[48] Citadel: Relentless Optimization at Global Scale - Quartr https://quartr.com/insights/company-research/citadel-relentless-optimization-at-global-scale
[49] I put together a 5 step checklist on how to fundamental analysis for a ... https://www.reddit.com/r/ValueInvesting/comments/1c66ow0/i_put_together_a_5_step_checklist_on_how_to/
[50] Citadel's Value Strategy https://www.citadelfund.com/value-investing/citadels-value-strategy/
[51] Millennium Management Review - SmartAsset.com https://smartasset.com/financial-advisor/millennium-management-review
[52] Citadel: Overview, History, and Investments | The Motley Fool https://www.fool.com/investing/how-to-invest/famous-investors/citadel/
[53] Final Round Jane Street Quant Research https://www.reddit.com/r/quantfinance/comments/1mmfs88/final_round_jane_street_quant_research/
[54] Jane Street: Trading, Research, And Career Insights https://littlecrumblybits.com/blog/jane-street-trading-research-and-1761750958613
[55] Hedge Fund: The Investment Life Cycle https://www.wallstreetoasis.com/forum/hedge-fund/hedge-fund-the-investment-life-cycle
[56] Back to basics: The process of exiting a private equity investment https://www.taxadvisermagazine.com/article/back-basics-process-exiting-private-equity-investment
[57] How I Generate Investment Ideas - CFA Institute Enterprising Investor https://blogs.cfainstitute.org/investor/2019/04/08/how-i-generate-investment-ideas/
[58] What Is an Exit Strategy in Private Equity? - EQT Group https://eqtgroup.com/thinq/Education/what-is-an-exit-strategy
[59] Vanguard's portfolio construction framework https://www.vanguard.co.uk/professional/vanguard-365/investment-knowledge/portfolio-construction/portfolio-construction-framework
