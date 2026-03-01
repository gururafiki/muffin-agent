# OpenBB MCP Server Setup

Muffin Agent uses OpenBB as its financial data backbone via the [Model Context Protocol](https://modelcontextprotocol.io/). The MCP server exposes OpenBB's data providers as tools that agents can call.

> We recommend running OpenBB in a **separate virtual environment** from the main muffin-agent, since OpenBB has heavy dependencies.

## Installation

```bash
cd extras/openbb

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Configure Data Providers

OpenBB aggregates data from many providers. Some work out of the box, others require a free or paid API key. Configure your keys in `~/.openbb_platform/user_settings.json` (created automatically on first run):

```json
{
  "credentials": {
    "bls_api_key": null,
    "cftc_app_token": null,
    "congress_gov_api_key": null,
    "econdb_api_key": null,
    "eia_api_key": null,
    "fmp_api_key": null,
    "fred_api_key": null,
    "nasdaq_api_key": null,
    "polygon_api_key": null,
    "benzinga_api_key": null,
    "intrinio_api_key": null,
    "alpha_vantage_api_key": null,
    "biztoc_api_key": null,
    "tradier_api_key": null,
    "tradier_account_type": "sandbox",
    "tradingeconomics_api_key": null,
    "tiingo_token": null
  },
  "preferences": {},
  "defaults": {
    "commands": {}
  }
}
```

Replace `null` with `"your-key"` for each provider you want to use.

### Providers that work without an API key

| Provider | Data |
|----------|------|
| yfinance | Price quotes, historical OHLCV, some fundamentals |
| SEC | SEC filings, reported financials |
| ECB | European Central Bank economic data |
| Federal Reserve | US economic indicators |
| OECD | Economic and development statistics |
| IMF | International Monetary Fund data |
| US Government | Government open data |
| US EIA | Energy information |
| CBOE | Options exchange data |
| FINRA | Regulatory trading data |
| FinViz | Stock screening and analysis |
| Fama-French | Academic factor research data |
| Seeking Alpha | Investment research |
| Deribit | Crypto derivatives |
| TMX | Canadian market data |

### Providers with free API keys

We recommend signing up for all of these — they cover the majority of financial data Muffin Agent needs.

| Provider | Key in `user_settings.json` | Free tier limits | Signup |
|----------|----------------------------|-----------------|--------|
| **FMP** | `fmp_api_key` | 250 req/day, ~5yr history, 150+ endpoints | [financialmodelingprep.com](https://site.financialmodelingprep.com/developer) |
| **FRED** | `fred_api_key` | Free, no hard limits | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| **Polygon** | `polygon_api_key` | 5 req/min, EOD data | [polygon.io](https://polygon.io/) |
| **Alpha Vantage** | `alpha_vantage_api_key` | 25 req/day | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| **Tiingo** | `tiingo_token` | 50 symbols/hr, 30yr price history | [tiingo.com](https://www.tiingo.com/) |
| **Nasdaq** | `nasdaq_api_key` | 50k req/day, 40+ free datasets | [data.nasdaq.com](https://data.nasdaq.com/) |
| **BLS** | `bls_api_key` | 500 req/day (25/day without key) | [bls.gov](https://data.bls.gov/registrationEngine/) |
| **Congress.gov** | `congress_gov_api_key` | 5,000 req/hr | [api.congress.gov](https://api.congress.gov/sign-up/) |
| **CFTC** | `cftc_app_token` | 1,000 req/hr (Socrata token) | [data.socrata.com](https://evergreen.data.socrata.com/signup) |
| **EconDB** | `econdb_api_key` | Free registration | [econdb.com](https://www.econdb.com/) |
| **BizToc** | `biztoc_api_key` | Free tier via RapidAPI | [rapidapi.com/biztoc](https://rapidapi.com/thma/api/biztoc) |

### Paid-only providers

| Provider | Key in `user_settings.json` | Notes |
|----------|----------------------------|-------|
| Benzinga | `benzinga_api_key` | Financial news and events |
| Intrinio | `intrinio_api_key` | Financial data and research |
| Tradier | `tradier_api_key` | Brokerage and options (sandbox available) |
| Trading Economics | `tradingeconomics_api_key` | Macroeconomic indicators |

For the full list of providers and their capabilities, see the [OpenBB Providers documentation](https://docs.openbb.co/odp/python/extensions/providers).

## Start the Server

```bash
# Activate the OpenBB venv
cd extras/openbb
source .venv/bin/activate

# Start on port 8001 (must match muffin-agent config)
openbb-mcp --port 8001
```

The server must be running at `http://127.0.0.1:8001/mcp` before using Muffin Agent.

## Verify with MCP Inspector (optional)

The MCP inspector lets you browse available tools and test them interactively:

```bash
npx @modelcontextprotocol/inspector \
  .venv/bin/openbb-mcp \
  --transport stdio
```

## Troubleshooting

**Port already in use:**
```bash
lsof -i :8001  # Find what's using the port
# Either kill the process or use a different port:
openbb-mcp --port 8002
# (update src/muffin_agent/config.py if you change the port)
```

**Missing provider data:** If a tool returns errors, the underlying provider may need an API key. Check the tables above and add the key to `~/.openbb_platform/user_settings.json`.

**Check available tools:** Use the MCP inspector (above) to see which tools are loaded and test them directly.

For full MCP server configuration options, see the [OpenBB MCP docs](https://docs.openbb.co/odp/python/extensions/interface/openbb-mcp).
