# 📄 NDA Analyzer

AI-powered Non-Disclosure Agreement analysis tool using DeepSeek models via Fireworks AI.

## Features

- **Document Processing**: Upload PDF and DOCX files
- **AI Analysis**: Extract key clauses, assess risks, and generate summaries
- **Semantic Search**: Find specific information within documents
- **Risk Scoring**: Get an overall risk assessment (0-100)
- **Export**: Download extracted clauses as CSV

## Quick Start

### 1. Get API Key

Sign up at [Fireworks AI](https://fireworks.ai/) and get your API key.

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your FIREWORKS_API_KEY
```

### 3. Run with Docker (Recommended)

```bash
docker-compose up --build
```

Access the app at: http://localhost:8501

### 4. Run Locally (Alternative)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Configuration

### Models Available

| Model | Best For | Notes |
|-------|----------|-------|
| DeepSeek V3.2 | General analysis | Recommended, latest version |
| DeepSeek V3.1 | Complex documents | Hybrid thinking modes |
| DeepSeek R1 | Deep reasoning | Best for complex legal analysis |

### Streamlit Secrets (for deployment)

Create `.streamlit/secrets.toml`:

```toml
[fireworks]
api_key = "your_api_key_here"
```

## Project Structure

```
nda_analyzer/
├── app.py                 # Main Streamlit application
├── backend/
│   ├── __init__.py
│   ├── analyzer.py        # LLM analysis logic
│   ├── embedder.py        # Vector embeddings & search
│   ├── loader.py          # Document parsing
│   └── utils.py           # Shared utilities
├── prompts/
│   ├── clause_extractor.txt
│   ├── risk_assessor.txt
│   └── summarizer.txt
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## API Reference

### Fireworks AI Endpoints

The app uses the OpenAI-compatible chat completions endpoint:

```
POST https://api.fireworks.ai/inference/v1/chat/completions
```

### Model IDs

- `accounts/fireworks/models/deepseek-v3p2` - DeepSeek V3.2
- `accounts/fireworks/models/deepseek-v3p1` - DeepSeek V3.1
- `accounts/fireworks/models/deepseek-r1-0528` - DeepSeek R1 (latest)

## Troubleshooting

### "FIREWORKS_API_KEY not set"

Make sure you have either:
- Set the environment variable: `export FIREWORKS_API_KEY=your_key`
- Created a `.env` file with `FIREWORKS_API_KEY=your_key`
- Added it to Streamlit secrets (for cloud deployment)

### "API call failed: 429"

You're hitting rate limits. The app has built-in retry logic, but if it persists:
- Wait a few minutes
- Consider upgrading your Fireworks plan

### "No clauses extracted"

The LLM sometimes struggles with poorly formatted documents. Try:
- Ensuring the PDF has selectable text (not scanned images)
- Using a cleaner document format
- Adjusting the temperature slider

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See LICENSE file for details.

## Changelog

### v2.0.0 (January 2025)
- Updated to DeepSeek V3.2 and R1-0528 models
- Migrated from `/completions` to `/chat/completions` API
- Added retry logic with exponential backoff
- Fixed CSV parser to handle 6-field format
- Improved chunking with overlap
- Added file validation
- Optimized Docker image with multi-stage build
- Added semantic search UI
- Better error handling throughout

### v1.0.0
- Initial release
