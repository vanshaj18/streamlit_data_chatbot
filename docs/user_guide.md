# Data Chatbot Dashboard - User Guide

## Overview

The Data Chatbot Dashboard is an intelligent web application that allows you to analyze your data using natural language queries. Simply upload your CSV or Excel files and start asking questions about your data in plain English!

## Getting Started

### 1. Upload Your Data

1. Click on the file uploader in the "Data Upload" section
2. Select a CSV (.csv) or Excel (.xlsx, .xls) file from your computer
3. Wait for the file to process and see the data preview
4. Your data is now ready for analysis!

**Supported File Formats:**
- CSV files (with various encodings)
- Excel files (.xlsx, .xls)
- Maximum file size < 50MB
- Maximum rows: 1,000,000
- Maximum columns < 1,000

### 2. Ask Questions About Your Data

Once your data is uploaded, use the chat interface to ask questions in natural language. The AI will analyze your data and provide insights, answers, and visualizations.

## Example Queries

Here are some example questions you can ask about your data:

### Basic Data Exploration
- "Show me the first 10 rows of my data"
- "What are the column names in my dataset?"
- "How many rows and columns does my data have?"
- "What are the data types of each column?"
- "Show me basic statistics for all numeric columns"

### Data Analysis
- "What is the average value of [column_name]?"
- "Find the maximum and minimum values in [column_name]"
- "How many unique values are in [column_name]?"
- "Show me the correlation between [column1] and [column2]"
- "What percentage of values in [column_name] are missing?"

<!-- ### Filtering and Grouping
- "Show me all rows where [column_name] is greater than 100"
- "Filter the data to show only [condition]"
- "Group the data by [column_name] and show the count"
- "What is the average [column1] for each [column2]?"
- "Show me the top 10 values in [column_name]" -->

### Visualizations
- "Create a bar chart of [column_name]"
- "Show me a histogram of [column_name]"
- "Plot [column1] vs [column2] as a scatter plot"
- "Create a line chart showing [column_name] over time"
- "Make a pie chart of [column_name] distribution"
- "Show me a heatmap of correlations between numeric columns"

<!-- ### Advanced Analysis
- "Identify outliers in [column_name]"
- "Show me the trend in [column_name] over [time_column]"
- "Compare [column1] across different [column2] categories"
- "Find patterns in my data"
- "What insights can you provide about this dataset?" -->

## Sample Datasets

To help you get started, here are some types of data that work well with the dashboard:

### Sales Data
Example columns: Date, Product, Sales_Amount, Region, Customer_ID
Example questions:
- "What are the total sales by region?"
- "Show me sales trends over time"
- "Which products are the top performers?"

### Financial Data
Example columns: Date, Account, Amount, Category, Description
Example questions:
- "What is my spending by category?"
- "Show me income vs expenses over time"
- "Find unusual transactions"

### Survey Data
Example columns: Respondent_ID, Age, Gender, Rating, Comments
Example questions:
- "What is the average rating by age group?"
- "Show me the distribution of responses"
- "Compare ratings between different demographics"

### Inventory Data
Example columns: Product_ID, Product_Name, Quantity, Price, Category
Example questions:
- "Which products are low in stock?"
- "What is the total inventory value?"
- "Show me inventory by category"

## Tips for Better Results

### 1. Be Specific
- Instead of "show me data", try "show me the top 10 customers by sales amount"
- Use exact column names when possible
- Specify the type of visualization you want

### 2. Start Simple
- Begin with basic questions to understand your data structure
- Build up to more complex analysis
- Use the data preview to familiarize yourself with column names

### 3. Use Natural Language
- You don't need to use technical terms
- Ask questions as you would to a human analyst
- The AI understands context from previous questions

### 4. Iterate and Refine
- If a query doesn't work as expected, try rephrasing it
- Build on previous questions for deeper analysis
- Use follow-up questions to drill down into insights

## Features

### Chat Interface
- **Persistent History**: Your conversation is saved during your session
- **Multiple Response Types**: Get text answers, data tables, and visualizations
- **Error Handling**: Helpful error messages and suggestions when queries fail

### Data Management
- **Session Persistence**: Your uploaded data stays available during your session
- **Memory Optimization**: Efficient handling of large datasets
- **File Validation**: Automatic validation of file formats and sizes

### Visualizations
- **Interactive Charts**: Plotly-powered interactive visualizations
- **Static Charts**: Matplotlib charts for publication-ready graphics
- **Responsive Design**: Charts adapt to different screen sizes
- **Chart Caching**: Improved performance for repeated visualizations

### Performance Features
- **Optimized Loading**: Efficient processing of large files
- **Memory Management**: Automatic cleanup to prevent memory issues
- **Caching**: Smart caching of charts and computations

## Troubleshooting

### Common Issues

**File Upload Problems:**
- Ensure your file is under 50MB
- Check that the file format is CSV or Excel
- Try saving your Excel file as CSV if upload fails

**Query Not Working:**
- Check column names in the data preview
- Try simpler queries first
- Use exact column names (case-sensitive)

**Slow Performance:**
- Large datasets may take longer to process
- Try filtering data first for faster analysis
- Consider using a smaller sample of your data

**Memory Issues:**
- The system automatically manages memory
- Very large datasets may require simplification
- Clear chat history if needed using the sidebar button

**Visualization Issues:**
- Plots, charts are prepared by LLM not any backend code.
- It sometime runs into cold start issues
- False table name issues
- Complex charts

**PandasAI issue:**
- PandasAI is prone to LLM code generation errors


### Getting Help

If you encounter issues:
1. Check the error message for specific guidance
2. Try rephrasing your question
3. Start with simpler queries and build up complexity
4. Use the debug mode in the sidebar for technical details

## Privacy and Security

- **No Data Storage**: Your data is only kept in memory during your session
- **Session Isolation**: Each user session is completely separate
- **Automatic Cleanup**: Data is automatically cleared when you close the browser
- **Local Processing**: Analysis happens on the server, not sent to external services

## Best Practices

### Data Preparation
- Clean your data before uploading for best results
- Ensure column names are descriptive and consistent
- Remove or handle missing values appropriately

### Query Strategy
- Start with exploratory questions to understand your data
- Use specific column names and values
- Build complex analysis step by step

### Performance Optimization
- Filter large datasets before complex analysis
- Use appropriate visualization types for your data
- Clear chat history periodically for better performance

## Advanced Features

### Debug Mode
Enable debug mode in the sidebar to see:
- Session state information
- Memory usage statistics
- Technical details about processing

### Memory Management
The system automatically:
- Optimizes data types for memory efficiency
- Manages chat history length
- Caches visualizations for better performance
- Cleans up unused resources

### Chart Customization
While you can't directly customize charts, you can:
- Request specific chart types in your queries
- Ask for different data groupings or filters
- Request multiple views of the same data

---

*Built with ❤️ using Streamlit and PandasAI*