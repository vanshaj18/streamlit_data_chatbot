"""
Example queries component for user guidance.
"""

import streamlit as st
from typing import List, Dict, Any


# Example queries organized by category
EXAMPLE_QUERIES = {
    "Data Exploration": [
        "Show me the first 10 rows of my data",
        "What are the column names in my dataset?",
        "How many rows and columns does my data have?",
        "Show me basic statistics for all numeric columns",
        "What are the data types of each column?"
    ],
    "Data Analysis": [
        "What is the average value of [column_name]?",
        "Find the maximum and minimum values in [column_name]",
        "How many unique values are in [column_name]?",
        "Show me the correlation between numeric columns",
        "What percentage of values are missing in each column?"
    ],
    "Filtering & Grouping": [
        "Show me all rows where [column_name] is greater than 100",
        "Group the data by [column_name] and show the count",
        "What is the average [column1] for each [column2]?",
        "Show me the top 10 values in [column_name]",
        "Filter the data to show only [specific_condition]"
    ],
    "Visualizations": [
        "Create a bar chart of [column_name]",
        "Show me a histogram of [column_name]",
        "Plot [column1] vs [column2] as a scatter plot",
        "Create a line chart showing trends over time",
        "Make a pie chart of [column_name] distribution",
        "Show me a heatmap of correlations"
    ],
    "Advanced Analysis": [
        "Identify outliers in [column_name]",
        "Show me trends in the data over time",
        "Compare values across different categories",
        "Find patterns and insights in my data",
        "What are the key findings from this dataset?"
    ]
}

# Context-specific examples based on common data types
CONTEXT_EXAMPLES = {
    "sales": {
        "description": "Sales & Revenue Data",
        "queries": [
            "What are the total sales by region?",
            "Show me sales trends over the last 12 months",
            "Which products are the top performers?",
            "Create a chart showing monthly revenue",
            "What is the average order value?"
        ]
    },
    "financial": {
        "description": "Financial Data",
        "queries": [
            "What is my spending by category?",
            "Show me income vs expenses over time",
            "Find transactions above $1000",
            "Create a budget analysis chart",
            "What are my largest expense categories?"
        ]
    },
    "survey": {
        "description": "Survey & Feedback Data",
        "queries": [
            "What is the average rating by demographic?",
            "Show me the distribution of responses",
            "Compare satisfaction across different groups",
            "Create a chart of response frequencies",
            "What are the most common feedback themes?"
        ]
    },
    "inventory": {
        "description": "Inventory & Stock Data",
        "queries": [
            "Which products are low in stock?",
            "What is the total inventory value?",
            "Show me inventory levels by category",
            "Find products that need reordering",
            "Create a stock level visualization"
        ]
    },
    "customer": {
        "description": "Customer Data",
        "queries": [
            "Who are my top customers by value?",
            "Show me customer acquisition trends",
            "What is the customer lifetime value?",
            "Create a customer segmentation analysis",
            "Which customers haven't purchased recently?"
        ]
    }
}


def render_example_queries(show_context_examples: bool = True) -> None:
    """
    Render example queries to help users get started.
    
    Args:
        show_context_examples: Whether to show context-specific examples
    """
    st.markdown("### 💡 Example Queries")
    st.markdown("Not sure what to ask? Here are some examples to get you started:")
    
    # Create tabs for different categories
    tabs = st.tabs(list(EXAMPLE_QUERIES.keys()))
    
    for i, (category, queries) in enumerate(EXAMPLE_QUERIES.items()):
        with tabs[i]:
            st.markdown(f"**{category}**")
            
            for query in queries:
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"• {query}")
                
                with col2:
                    # Create a unique key for each button
                    button_key = f"use_{category}_{queries.index(query)}"
                    if st.button("Use", key=button_key, help="Click to use this query"):
                        # Set the query in session state for the chat interface to pick up
                        st.session_state['suggested_query'] = query
                        st.rerun()
    
    # Show context-specific examples if enabled
    if show_context_examples:
        st.markdown("---")
        st.markdown("### 📊 Examples by Data Type")
        st.markdown("Choose examples based on your type of data:")
        
        # Create columns for context examples
        cols = st.columns(2)
        
        for i, (context_key, context_data) in enumerate(CONTEXT_EXAMPLES.items()):
            col_index = i % 2
            
            with cols[col_index]:
                with st.expander(f"📈 {context_data['description']}"):
                    for query in context_data['queries']:
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.markdown(f"• {query}")
                        
                        with col2:
                            button_key = f"use_context_{context_key}_{context_data['queries'].index(query)}"
                            if st.button("Use", key=button_key, help="Click to use this query"):
                                st.session_state['suggested_query'] = query
                                st.rerun()


def render_quick_start_guide() -> None:
    """Render a quick start guide for new users."""
    st.markdown("### 🚀 Quick Start Guide")
    
    with st.expander("How to use this dashboard", expanded=False):
        st.markdown("""
        **Step 1: Upload Your Data**
        - Click the file uploader and select a CSV or Excel file
        - Wait for the data to load and preview
        
        **Step 2: Ask Questions**
        - Use the chat interface to ask questions in natural language
        - Start with simple questions like "Show me the first 10 rows"
        - Use exact column names from your data
        
        **Step 3: Explore and Analyze**
        - Build on previous questions for deeper insights
        - Request specific visualizations
        - Try different types of analysis
        
        **Tips for Better Results:**
        - Be specific in your questions
        - Use column names exactly as they appear in your data
        - Start simple and build complexity gradually
        - Ask for specific chart types when you want visualizations
        """)


def render_query_suggestions(dataframe_columns: List[str] = None) -> None:
    """
    Render query suggestions based on the uploaded data.
    
    Args:
        dataframe_columns: List of column names from the uploaded DataFrame
    """
    if not dataframe_columns:
        return
    
    st.markdown("### 🎯 Suggestions for Your Data")
    st.markdown("Based on your uploaded data, here are some relevant questions:")
    
    # Generate suggestions based on column names
    suggestions = _generate_column_based_suggestions(dataframe_columns)
    
    for suggestion in suggestions:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"• {suggestion}")
        
        with col2:
            button_key = f"use_suggestion_{suggestions.index(suggestion)}"
            if st.button("Use", key=button_key, help="Click to use this query"):
                st.session_state['suggested_query'] = suggestion
                st.rerun()


def _generate_column_based_suggestions(columns: List[str]) -> List[str]:
    """
    Generate query suggestions based on column names.
    
    Args:
        columns: List of column names
        
    Returns:
        List of suggested queries
    """
    suggestions = []
    
    # Basic exploration
    suggestions.append("Show me basic statistics for all columns")
    suggestions.append("What are the data types of each column?")
    
    # Column-specific suggestions
    for col in columns[:5]:  # Limit to first 5 columns to avoid clutter
        # Clean column name for display
        clean_col = col.replace('_', ' ').title()
        
        # Suggest based on common column name patterns
        if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
            suggestions.append(f"Show me trends in data over {col}")
        elif any(keyword in col.lower() for keyword in ['amount', 'price', 'cost', 'value', 'revenue', 'sales']):
            suggestions.append(f"What is the average {clean_col}?")
            suggestions.append(f"Show me the distribution of {clean_col}")
        elif any(keyword in col.lower() for keyword in ['category', 'type', 'status', 'region', 'department']):
            suggestions.append(f"Group data by {clean_col} and show counts")
            suggestions.append(f"Create a bar chart of {clean_col}")
        elif any(keyword in col.lower() for keyword in ['id', 'name', 'customer', 'user']):
            suggestions.append(f"How many unique values are in {clean_col}?")
        else:
            suggestions.append(f"Show me the top 10 values in {clean_col}")
    
    # Visualization suggestions
    if len(columns) >= 2:
        suggestions.append(f"Create a scatter plot of {columns[0]} vs {columns[1]}")
        suggestions.append("Show me correlations between numeric columns")
    
    return suggestions[:8]  # Limit to 8 suggestions


def get_suggested_query() -> str:
    """
    Get and clear any suggested query from session state.
    
    Returns:
        The suggested query string, or empty string if none
    """
    if 'suggested_query' in st.session_state:
        query = st.session_state['suggested_query']
        del st.session_state['suggested_query']
        return query
    return ""


def render_help_section() -> None:
    """Render a help section with tips and troubleshooting."""
    st.markdown("### ❓ Need Help?")
    
    with st.expander("Common Issues & Solutions"):
        st.markdown("""
        **Query not working?**
        - Check that column names match exactly (case-sensitive)
        - Try simpler queries first
        - Use the data preview to see available columns
        
        **File upload issues?**
        - Ensure file is under 50MB
        - Check file format (CSV or Excel only)
        - Try saving Excel files as CSV
        
        **Slow performance?**
        - Large datasets take longer to process
        - Try filtering data first
        - Use simpler visualizations
        
        **Memory issues?**
        - The system automatically manages memory
        - Clear chat history if needed
        - Consider using a smaller dataset
        """)
    
    with st.expander("Best Practices"):
        st.markdown("""
        **For better results:**
        - Start with data exploration queries
        - Use specific column names and values
        - Build complex analysis step by step
        - Be specific about the type of chart you want
        
        **Data preparation tips:**
        - Clean your data before uploading
        - Use descriptive column names
        - Handle missing values appropriately
        - Keep file sizes reasonable (<50MB)
        """)