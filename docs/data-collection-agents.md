# Data Collection Agents

Grouping of OpenBB MCP tools into data collection agents based on:
- Whether tools require a `symbol` (ticker-specific) vs no symbol (market-wide/macro)
- Parameter patterns and data domain coherence
- Purpose in stock valuation workflow

## Architecture Context

```
Criterion Evaluation Agent
  → identifies what data is needed for a valuation criterion
  → routes to one or more Data Collection Agents below
```

**Ticker-specific agents** are called once per stock being analyzed.
**Market-wide agents** are called once per analysis run, providing shared context.

---

## Tier 1: Ticker-Specific Agents (called per stock)

### 1. Equity Fundamentals (exists)

Financial statements, ratios, metrics, growth, segments, management, ESG.
All require `symbol`.

| Tool | Required Params | Notes |
|------|----------------|-------|
| `equity_fundamental_balance` | symbol | Balance sheet |
| `equity_fundamental_balance_growth` | symbol | Balance sheet growth |
| `equity_fundamental_cash` | symbol | Cash flow statement |
| `equity_fundamental_cash_growth` | symbol | Cash flow growth |
| `equity_fundamental_income` | symbol | Income statement |
| `equity_fundamental_income_growth` | symbol | Income growth |
| `equity_fundamental_reported_financials` | symbol | As-reported financials |
| `equity_fundamental_metrics` | symbol [MULTI] | Key metrics (PE, EV/EBITDA, etc.) |
| `equity_fundamental_ratios` | symbol [MULTI] | Extensive financial ratios |
| `equity_fundamental_dividends` | symbol [MULTI] | Dividend history |
| `equity_fundamental_historical_eps` | symbol [MULTI] | Historical EPS |
| `equity_fundamental_employee_count` | symbol [MULTI] | Employee count history |
| `equity_fundamental_revenue_per_geography` | symbol | Revenue by geography |
| `equity_fundamental_revenue_per_segment` | symbol | Revenue by segment |
| `equity_fundamental_management` | symbol | Executive team |
| `equity_fundamental_management_compensation` | symbol [MULTI] | Exec compensation |
| `equity_fundamental_management_discussion_analysis` | symbol | MD&A section |
| `equity_fundamental_transcript` | symbol | Earnings call transcripts |
| `equity_fundamental_trailing_dividend_yield` | symbol | Trailing div yield |
| `equity_fundamental_historical_splits` | symbol | Stock split history |
| `equity_fundamental_filings` | symbol (opt) | SEC filings |
| `equity_fundamental_esg_score` | symbol [MULTI] | ESG scores |
| `equity_fundamental_search_attributes` | query | Search Intrinio data tags |
| `equity_fundamental_latest_attributes` | symbol, tag | Latest Intrinio data tag |
| `equity_fundamental_historical_attributes` | symbol, tag | Historical Intrinio data tag |

**~25 tools.** Status: **Implemented.**

---

### 2. Equity Price

Historical and current price data, market cap. All require `symbol`.

| Tool | Required Params | Notes |
|------|----------------|-------|
| `equity_price_quote` | symbol [MULTI] | Current quote (price, volume, etc.) |
| `equity_price_historical` | symbol [MULTI] | OHLCV history with interval support |
| `equity_price_nbbo` | symbol | National best bid/offer (spread) |
| `equity_price_performance` | symbol [MULTI] | Returns across timeframes |
| `equity_historical_market_cap` | symbol [MULTI] | Market cap history |

**5 tools.** All share `start_date`/`end_date` patterns. Priority: **High.**

---

### 3. Equity Estimates

Analyst forecasts, consensus, and forward-looking projections. Ticker-specific.

| Tool | Required Params | Notes |
|------|----------------|-------|
| `equity_estimates_price_target` | symbol (opt) [MULTI] | Analyst price targets |
| `equity_estimates_historical` | symbol [MULTI] | Historical estimate revisions |
| `equity_estimates_consensus` | symbol (opt) [MULTI] | Consensus target & recommendation |
| `equity_estimates_forward_sales` | symbol (opt) [MULTI] | Forward sales estimates |
| `equity_estimates_forward_ebitda` | symbol (opt) [MULTI] | Forward EBITDA estimates |
| `equity_estimates_forward_eps` | symbol (opt) [MULTI] | Forward EPS estimates |
| `equity_estimates_forward_pe` | symbol (opt) [MULTI] | Forward PE estimates |
| `equity_estimates_analyst_search` | analyst/firm name | Search analysts by name (no symbol) |

**8 tools.** `analyst_search` is a lookup helper, rest are ticker-specific. Priority: **High.**

---

### 4. Equity Ownership & Short Interest

Who holds the stock and who's betting against it. All require `symbol`.

| Tool | Required Params | Notes |
|------|----------------|-------|
| `equity_ownership_major_holders` | symbol | Top holders by quarter |
| `equity_ownership_institutional` | symbol [MULTI] | Institutional ownership stats |
| `equity_ownership_insider_trading` | symbol | Insider buys/sells |
| `equity_ownership_share_statistics` | symbol [MULTI] | Float, shares outstanding |
| `equity_ownership_form_13f` | symbol | 13F filings (CIK or ticker) |
| `equity_ownership_government_trades` | symbol (opt) [MULTI] | Congressional trades |
| `equity_shorts_fails_to_deliver` | symbol | FTD data |
| `equity_shorts_short_volume` | symbol | Short volume |
| `equity_shorts_short_interest` | symbol | Short interest & days to cover |

**9 tools.** Coherent group answering "who's buying/selling this stock?" Priority: **High.**

---

### 5. Company News

Ticker-specific news and sentiment.

| Tool | Required Params | Notes |
|------|----------------|-------|
| `news_company` | symbol (opt) [MULTI] | Company news with rich filtering (source, sentiment, relevance, language) |
| `news_world` | — | Global news (can filter by topic/term) |

**2 tools.** Both in one agent — the agent decides which to call based on context.
`news_company` has extensive filtering: `sentiment`, `business_relevance_*`, `topics`, `channels`, etc. Priority: **High.**

---

### 6. Options

Options data for a specific underlying.

| Tool | Required Params | Notes |
|------|----------------|-------|
| `derivatives_options_chains` | symbol | Full options chain with Greeks |
| `derivatives_options_surface` | data (array) | Post-processes chains output for vol surface |

**2 tools.** `options_surface` takes raw data as input (output of `chains`), not a symbol directly. Priority: **Medium.**

---

## Tier 2: Market-Wide / Context Agents (called once per analysis)

### 7. Economy & Macro

Macroeconomic indicators. Parameterized by `country`, `date range`, or series IDs — no equity `symbol`.

| Tool | Required Params | Notes |
|------|----------------|-------|
| `economy_gdp_forecast` | — | GDP forecasts by country |
| `economy_gdp_nominal` | — | Nominal GDP by country |
| `economy_gdp_real` | — | Real GDP by country |
| `economy_cpi` | — | Consumer Price Index |
| `economy_unemployment` | — | Unemployment by country |
| `economy_interest_rates` | — | Interest rates by country/duration |
| `economy_money_measures` | — | M1/M2 money supply |
| `economy_pce` | — | Personal Consumption Expenditures |
| `economy_composite_leading_indicator` | — | CLI by country |
| `economy_calendar` | — | Economic events calendar |
| `economy_risk_premium` | — | Market risk premium by country |
| `economy_country_profile` | country | Country statistics overview |
| `economy_available_indicators` | — | List available indicators |
| `economy_indicators` | — | Get indicators by country/symbol |
| `economy_share_price_index` | — | OECD share price index |
| `economy_house_price_index` | — | OECD house price index |
| `economy_retail_prices` | — | Retail prices for common items |
| `economy_balance_of_payments` | — | Balance of payments reports |
| `economy_export_destinations` | country | Top export destinations |
| `economy_direction_of_trade` | — | IMF trade statistics |
| `economy_central_bank_holdings` | — | Central bank balance sheet |
| `economy_primary_dealer_positioning` | — | Primary dealer stats |
| `economy_primary_dealer_fails` | — | Fails to deliver/receive |
| `economy_fomc_documents` | — | FOMC meeting documents |
| `economy_fred_search` | — | Search FRED series |
| `economy_fred_series` | symbol (series ID) | Get FRED data by series ID |
| `economy_fred_release_table` | release_id | FRED release data |
| `economy_fred_regional` | symbol (series ID) | Regional FRED data |
| `economy_survey_bls_series` | symbol (series ID) | BLS time series |
| `economy_survey_bls_search` | — | Search BLS surveys |
| `economy_survey_sloos` | — | Senior Loan Officer survey |
| `economy_survey_university_of_michigan` | — | Consumer sentiment |
| `economy_survey_economic_conditions_chicago` | — | Chicago economic conditions |
| `economy_survey_manufacturing_outlook_texas` | — | Texas manufacturing |
| `economy_survey_manufacturing_outlook_ny` | — | Empire State manufacturing |
| `economy_survey_nonfarm_payrolls` | — | Nonfarm payrolls |
| `economy_shipping_port_info` | — | Port metadata |
| `economy_shipping_port_volume` | — | Port trade volumes |
| `economy_shipping_chokepoint_info` | — | Chokepoint metadata |
| `economy_shipping_chokepoint_volume` | — | Chokepoint transit volumes |

**~38 tools.** Large but coherent — all macro/economic with `country`/`date` params.
Note: FRED tools use `symbol` for series IDs (e.g., "GDP", "UNRATE"), not equity tickers. Priority: **High.**

---

### 8. Fixed Income & Rates

Rates, yields, spreads, bonds. No equity symbol. Critical for discount rate / WACC calculations.

| Tool | Required Params | Notes |
|------|----------------|-------|
| `fixedincome_rate_sofr` | — | Secured Overnight Financing Rate |
| `fixedincome_rate_effr` | — | Effective Federal Funds Rate |
| `fixedincome_rate_effr_forecast` | — | Fed Funds projections |
| `fixedincome_rate_ameribor` | — | AMERIBOR |
| `fixedincome_rate_sonia` | — | Sterling overnight rate |
| `fixedincome_rate_estr` | — | Euro short-term rate |
| `fixedincome_rate_ecb` | — | ECB interest rates |
| `fixedincome_rate_iorb` | — | Interest on reserve balances |
| `fixedincome_rate_dpcredit` | — | Discount window rate |
| `fixedincome_rate_overnight_bank_funding` | — | Overnight bank funding rate |
| `fixedincome_spreads_tcm` | — | Treasury constant maturity spreads |
| `fixedincome_spreads_tcm_effr` | — | TCM minus EFFR |
| `fixedincome_spreads_treasury_effr` | — | T-bill minus EFFR |
| `fixedincome_government_yield_curve` | — | Yield curves by country |
| `fixedincome_government_treasury_rates` | — | Treasury rates |
| `fixedincome_government_treasury_auctions` | — | Treasury auctions |
| `fixedincome_government_treasury_prices` | — | Treasury prices |
| `fixedincome_government_tips_yields` | — | TIPS yields |
| `fixedincome_corporate_hqm` | — | High quality market bond yields |
| `fixedincome_corporate_spot_rates` | — | Corporate spot rates |
| `fixedincome_corporate_commercial_paper` | — | Commercial paper rates |
| `fixedincome_corporate_bond_prices` | — | Corporate bond prices (by issuer/ISIN) |
| `fixedincome_bond_indices` | — | Bond indices |
| `fixedincome_mortgage_indices` | — | Mortgage indices |

**~22 tools.** All parameterized by `date range`, `maturity`, `country`. Priority: **High.**

---

### 9. ETF & Index

Sector context, relative valuation, benchmark comparison. Uses index/ETF symbols (not equity tickers).

| Tool | Required Params | Notes |
|------|----------------|-------|
| `index_price_historical` | symbol (index) [MULTI] | Index levels history |
| `index_constituents` | symbol (index) | Index components |
| `index_sectors` | symbol (index) | Sector weights |
| `index_sp500_multiples` | — | S&P 500 PE / Shiller PE |
| `index_snapshots` | — | All indices current levels |
| `index_available` | — | List all indices |
| `index_search` | — | Search indices |
| `etf_info` | symbol (ETF) [MULTI] | ETF overview |
| `etf_sectors` | symbol (ETF) [MULTI] | ETF sector weights |
| `etf_countries` | symbol (ETF) [MULTI] | ETF country weights |
| `etf_holdings` | symbol (ETF) | ETF holdings |
| `etf_historical` | symbol (ETF) [MULTI] | ETF price history |
| `etf_price_performance` | symbol (ETF) [MULTI] | ETF returns |
| `etf_nport_disclosure` | symbol (ETF) | SEC NPORT filings |
| `etf_equity_exposure` | symbol (stock) [MULTI] | Which ETFs hold this stock? |
| `etf_search` | — | Search ETFs |
| `etf_discovery_gainers` | — | Top ETF gainers |
| `etf_discovery_losers` | — | Top ETF losers |
| `etf_discovery_active` | — | Most active ETFs |

**~18 tools.** Note: `etf_equity_exposure` takes a stock ticker (reverse lookup). Priority: **Medium.**

---

### 10. Discovery & Screening

Market-wide scans, calendars, peer comparisons. Mostly no symbol required.

| Tool | Required Params | Symbol? | Notes |
|------|----------------|---------|-------|
| `equity_screener` | — | No | Multi-filter screener (cap, sector, valuation, etc.) |
| `equity_discovery_gainers` | — | No | Top gainers |
| `equity_discovery_losers` | — | No | Top losers |
| `equity_discovery_active` | — | No | Most active by volume |
| `equity_discovery_undervalued_large_caps` | — | No | Pre-built screen |
| `equity_discovery_undervalued_growth` | — | No | Pre-built screen |
| `equity_discovery_aggressive_small_caps` | — | No | Pre-built screen |
| `equity_discovery_growth_tech` | — | No | Pre-built screen |
| `equity_discovery_top_retail` | — | No | Retail investor activity |
| `equity_discovery_filings` | — | No | Recent SEC filings |
| `equity_discovery_latest_financial_reports` | — | No | Latest quarterly/annual reports |
| `equity_calendar_ipo` | symbol (opt) | Optional | IPO calendar |
| `equity_calendar_dividend` | — | No | Dividend calendar |
| `equity_calendar_splits` | — | No | Splits calendar |
| `equity_calendar_events` | — | No | Company events calendar |
| `equity_calendar_earnings` | — | No | Earnings calendar |
| `equity_compare_groups` | — | No | Sector/industry valuation & performance |
| `equity_market_snapshots` | — | No | All stocks current data |
| `equity_search` | — | No | Ticker/company lookup |
| `equity_profile` | symbol [MULTI] | **Yes (REQ)** | Company info (sector, industry, description) |
| `equity_compare_peers` | symbol | **Yes (REQ)** | Find peer companies |
| `equity_compare_company_facts` | symbol (opt) [MULTI] | Optional | Compare reported facts |
| `equity_darkpool_otc` | symbol (opt) | Optional | Dark pool volume |

**~23 tools.** Mixed bag. `equity_profile` and `equity_compare_peers` require a symbol and could alternatively live in a ticker-specific agent. Priority: **Medium.**

---

## Tier 3: Specialized (Deprioritize)

### 11. Currency & Commodities

FX rates, commodity prices, energy outlook. Relevant for companies with commodity/FX exposure.

| Tool | Required Params | Notes |
|------|----------------|-------|
| `currency_price_historical` | symbol (FX pair) [MULTI] | FX history (e.g., "EURUSD") |
| `currency_search` | — | Find currency pairs |
| `currency_reference_rates` | — | Official reference rates |
| `currency_snapshots` | — | Current FX snapshot |
| `commodity_price_spot` | — | Commodity spot prices (WTI, Brent, natgas, etc.) |
| `commodity_petroleum_status_report` | — | EIA weekly petroleum report |
| `commodity_short_term_energy_outlook` | — | EIA 18-month energy projections |
| `crypto_price_historical` | symbol [MULTI] | Crypto price history |
| `crypto_search` | — | Search crypto pairs |

**~9 tools.** Priority: **Low.**

---

### 12. Regulatory & Filings

SEC lookups, CFTC reports, congressional bills. Reference/compliance data.

| Tool | Required Params | Notes |
|------|----------------|-------|
| `regulators_sec_filing_headers` | url (opt) | Filing headers by URL |
| `regulators_sec_htm_file` | url (opt) | Raw HTML from SEC |
| `regulators_sec_cik_map` | symbol | Ticker → CIK |
| `regulators_sec_institutions_search` | — | Search institutions by name |
| `regulators_sec_schema_files` | — | SEC XML schema directory |
| `regulators_sec_symbol_map` | query | CIK → ticker |
| `regulators_sec_rss_litigation` | — | SEC litigation RSS |
| `regulators_sec_sic_search` | — | SIC code lookup |
| `regulators_cftc_cot_search` | — | Search COT reports |
| `regulators_cftc_cot` | — | Commitment of Traders data |
| `uscongress_bills` | — | Congressional bills |
| `uscongress_bill_text_urls` | bill_url | Bill document URLs |
| `uscongress_bill_info` | — | Bill metadata |
| `uscongress_bill_text` | — | Bill full text |

**~14 tools.** Overlaps with `equity_fundamental_filings` and `equity_discovery_filings`. Priority: **Low.**

---

### 13. Fama-French

Academic factor data for quantitative analysis.

| Tool | Required Params | Notes |
|------|----------------|-------|
| `famafrench_factors` | — | 3/5 factor model data |
| `famafrench_us_portfolio_returns` | — | US portfolio returns by size/value/etc. |
| `famafrench_regional_portfolio_returns` | — | Regional portfolio returns |
| `famafrench_country_portfolio_returns` | — | Country portfolio returns |
| `famafrench_international_index_returns` | — | International index returns |
| `famafrench_breakpoints` | — | Size/value breakpoints |

**6 tools.** No equity symbol. Parameterized by region, factor, frequency. Priority: **Low.**

---

## Derivatives — Futures (Unassigned)

These don't fit cleanly into equity analysis but could be relevant for commodities or macro context.

| Tool | Required Params | Notes |
|------|----------------|-------|
| `derivatives_futures_historical` | symbol (futures contract) [MULTI] | Futures price history |
| `derivatives_futures_curve` | symbol (futures contract) | Term structure |
| `derivatives_futures_instruments` | — | Reference data for available contracts |
| `derivatives_futures_info` | symbol (opt) [MULTI] | Current trading statistics |
| `derivatives_options_unusual` | symbol (opt) | Unusual options activity (market-wide if no symbol) |
| `derivatives_options_snapshots` | — | Options market snapshot |

Could fold into **Currency & Commodities** (agent 11) or **Options** (agent 6) depending on use case.

---

## Mapping: Valuation Criteria → Data Agents

| Valuation Criterion | Primary Agents | Supporting Agents |
|---------------------|---------------|-------------------|
| Intrinsic value (DCF) | Fundamentals, Estimates | Fixed Income (discount rate) |
| Relative valuation (multiples) | Fundamentals, Estimates | Discovery (peers/sector), ETF & Index |
| Technical / momentum | Price | — |
| Analyst sentiment | Estimates | — |
| Insider / institutional conviction | Ownership | — |
| News sentiment | News | — |
| Macro environment | Economy & Macro | Fixed Income |
| Options-implied sentiment | Options | — |
| Commodity / FX exposure | Currency & Commodities | Economy & Macro |
| ESG / governance | Fundamentals (ESG tools) | — |

---

## Implementation Priority

1. **Equity Price** (5 tools) — small, high value
2. **Equity Estimates** (8 tools) — critical for forward-looking valuation
3. **Equity Ownership** (9 tools) — conviction signals
4. **Company News** (2 tools) — sentiment analysis
5. **Economy & Macro** (38 tools) — macro context
6. **Fixed Income** (22 tools) — discount rates
7. **Options** (2 tools) — implied sentiment
8. **ETF & Index** (18 tools) — relative valuation context
9. **Discovery & Screening** (23 tools) — stock selection
10. **Currency & Commodities** (9 tools) — exposure analysis
11. **Regulatory** (14 tools) — compliance
12. **Fama-French** (6 tools) — quant analysis

---

## Utility Tools (Not Assigned to Any Agent)

These are OpenBB MCP meta-tools for managing tool activation:

- `available_categories` — list tool categories
- `available_tools` — list tools in a category
- `activate_tools` — activate a tool
- `deactivate_tools` — deactivate a tool
- `list_prompts` — list available prompts
- `execute_prompt` — run a prompt
