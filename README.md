# Streamlit Data Chatbot

A Streamlit-based web application for data analysis through natural language queries using PandasAI.

## Project Structure

```
data-chatbot-dashboard/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── components/
│   ├── __init__.py
│   ├── file_handler.py        # File upload components
│   ├── chat_interface.py      # Chat interface components
│   └── visualization.py       # Chart rendering components
├── services/
│   ├── __init__.py
│   └── pandas_agent.py        # PandasAI agent wrapper
└── utils/
    ├── __init__.py
    └── session_manager.py      # Session state management
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Features

- Upload CSV and Excel files
- Natural language data queries
- Interactive visualizations
- Chat-based interface
- Session state management

## Development Status

This project is currently under development. Features are being implemented incrementally according to the specification tasks.