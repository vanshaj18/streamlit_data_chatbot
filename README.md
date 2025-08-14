# Data Chatbot Dashboard

A powerful, intelligent web application for data analysis through natural language queries. Built with Streamlit and powered by PandasAI, this dashboard allows users to upload CSV/Excel files and analyze their data using conversational AI.

## ✨ Features

- **Natural Language Queries**: Ask questions about your data in plain English
- **Multiple File Formats**: Support for CSV and Excel files (.xlsx, .xls)
- **Interactive Visualizations**: Automatic chart generation with Plotly and Matplotlib
- **Smart Caching**: Optimized performance with chart and computation caching
- **Memory Management**: Efficient handling of large datasets with automatic optimization
- **Session Persistence**: Maintain chat history and data throughout your session
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Error Handling**: Graceful error handling with helpful suggestions

## 🚀 Quick Start

### Prerequisites

- Python >= 3.10 < 3.12 (PandasAI beta 3.0 needs python < 3.12)
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd data-analysis-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   
   Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 📊 Usage

### 1. Upload Your Data
- Click the file uploader in the "Data Upload" section
- Select a CSV or Excel file (max 50MB)
- Wait for the data to process and preview

### 2. Start Analyzing
Use the chat interface to ask questions like:
- "Show me the first 10 rows of my data"
- "What is the average sales by region?"
- "Create a bar chart of product categories"
- "Find outliers in the price column"

### 3. Explore Insights
- Get instant text responses and visualizations
- Build on previous questions for deeper analysis
- Use the example queries for inspiration

## 💡 Example Queries

### Data Exploration
- "What are the column names in my dataset?"
- "Show me basic statistics for all numeric columns"
- "How many rows and columns does my data have?"

### Analysis
- "What is the correlation between [column1] and [column2]?"
- "Group the data by [column] and show counts"
- "Find the top 10 values in [column]"

### Visualizations
- "Create a histogram of [column]"
- "Plot [column1] vs [column2] as a scatter plot"
- "Show me a heatmap of correlations"

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit for the web interface
- **AI Engine**: PandasAI for natural language processing
- **Visualization**: Plotly and Matplotlib for charts
- **Data Processing**: Pandas for data manipulation

### Performance Optimizations
- **Data Loading**: Optimized CSV/Excel parsing with encoding detection
- **Memory Management**: Automatic data type optimization and cleanup
- **Caching**: Smart caching of visualizations and computations
- **Responsive Design**: Adaptive layouts for different screen sizes

### File Support
- **CSV Files**: All encodings (UTF-8, Latin-1, CP1252, ISO-8859-1)
- **Excel Files**: .xlsx and .xls formats
- **Size Limits**: Up to 50MB files, 1M rows, 1K columns

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Required for PandasAI functionality
- `STREAMLIT_SERVER_PORT`: Optional, defaults to 8501
- `STREAMLIT_SERVER_ADDRESS`: Optional, defaults to localhost

### Memory Settings
The application automatically manages memory with configurable limits:
- Maximum chat history: 100 messages
- Chart cache size: 50 charts
- Automatic cleanup every 50 operations

## 📁 Project Structure

```
data-analysis-dashboard/
├── app.py                      # Main application entry point
├── components/                 # UI components
│   ├── chat_interface.py      # Chat interface logic
│   ├── file_handler.py        # File upload and processing
│   ├── visualization.py       # Chart rendering with caching
│   └── example_queries.py     # Example queries and help
├── services/                   # Business logic
│   └── pandas_agent.py        # PandasAI integration
├── utils/                      # Utilities
│   ├── session_manager.py     # Session state management
│   └── error_handler.py       # Error handling
├── docs/                       # Documentation
│   └── user_guide.md          # Comprehensive user guide
├── tests/                      # Test files
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run integration tests:
```bash
python tests/run_integration_tests.py
```

## 🔒 Privacy & Security

- **No Data Storage**: Your data is only kept in memory during your session
- **Session Isolation**: Each user session is completely separate
- **Automatic Cleanup**: Data is cleared when you close the browser
- **Local Processing**: Analysis happens locally, not sent to external services

## 🐛 Troubleshooting

### Common Issues

**File Upload Problems**
- Ensure file is under 50MB
- Check file format (CSV or Excel only)
- Try saving Excel files as CSV

**Query Not Working**
- Check column names match exactly (case-sensitive)
- Try simpler queries first
- Use the data preview to see available columns

**Performance Issues**
- Large datasets may take longer to process
- Try filtering data first for faster analysis
- Use the memory optimization button in the sidebar

**API Key Issues**
- Ensure your Gemini API key is set in the `.env` file
- Check that the API key is valid and has sufficient quota

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [PandasAI](https://pandas-ai.readthedocs.io/) for natural language data analysis
- [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/) for visualizations
- [Pandas](https://pandas.pydata.org/) for data manipulation

## 📞 Support

If you encounter any issues or have questions:
1. Check the [User Guide](docs/user_guide.md) for detailed instructions
2. Look at the troubleshooting section above
3. Open an issue on GitHub
4. Use the debug mode in the application for technical details

---

**Built with ❤️ using Streamlit and PandasAI**