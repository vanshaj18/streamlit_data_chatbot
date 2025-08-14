"""
x - Main Application
A Streamlit-based web application for data analysis through natural language queries.
"""

import streamlit as st
import traceback
from typing import Optional
from utils.error_handler import handle_query_error, ErrorCategory, safe_execute

from utils.session_manager import (
    initialize_session, 
    get_session_summary,
    has_dataframe,
    get_dataframe,
    get_agent,
    set_agent
)
from components.file_handler import render_file_upload
from components.chat_interface import (
    render_chat_interface, 
    get_chat_stats,
    clear_chat,
    display_error_message,
    display_text_response
)
from services.pandas_agent import PandasAgent
from dotenv import load_dotenv

load_dotenv()


def initialize_agent_if_needed() -> Optional[PandasAgent]:
    """
    Initialize PandasAI agent if data is loaded but agent doesn't exist.
    
    Returns:
        Optional[PandasAgent]: The agent instance or None if not ready
    """
    def _initialize_agent():
        # Check if we have data but no agent
        if has_dataframe() and get_agent() is None:
            df = get_dataframe()
            if df is not None:
                # Create and initialize agent
                agent = PandasAgent()
                print(agent.is_initialized())
                success = agent.initialize_agent(df)
                
                if success:
                    set_agent(agent)
                    return agent
                else:
                    raise RuntimeError("PandasAI agent could not be initialized. Please ensure you have set your GEMINI_API_KEY environment variable.")
        
        return get_agent()
    
    return safe_execute(
        _initialize_agent,
        ErrorCategory.API_ERROR,
        context={"operation": "agent_initialization"},
        fallback_value=None
    )


def process_user_query(query: str) -> None:
    """
    Process a user query through the PandasAI agent.
    
    Args:
        query: The user's natural language query
    """
    def _process_query():
        # Ensure agent is initialized
        agent = initialize_agent_if_needed()

        print("agent: ", agent)
        
        if agent is None:
            error_info = handle_query_error(
                """Unable to initialize data analysis agent. 
                Please ensure your data is uploaded and API KEY is set.""",
                query,
                show_ui=False
            )
            display_error_message(error_info.user_message)
            return
        
        # Check if agent is properly initialized
        if not agent.is_initialized():
            error_info = handle_query_error(
                "Data analysis agent is not properly configured. Please check your API key and try again.",
                query,
                show_ui=False
            )
            display_error_message(error_info.user_message)
            return
        
        # Show processing indicator
        with st.spinner("Analyzing the query..."):
            # Process the query through the agent
            response = agent.process_query(query)
            
            # Handle the response based on type
            if response.response_type == "error":
                display_error_message(response.error_message)
            elif response.response_type == "text":
                display_text_response(response.content)
            elif response.response_type == "dataframe":
                display_text_response("Here's the result of your query:")
                st.dataframe(response.content)
            elif response.response_type == "plot":
                # Use the chat interface to display chart responses
                from components.chat_interface import display_chart_response
                display_chart_response("Here's the visualization for your query:", response.content)
            else:
                display_text_response(f"Query processed successfully. Response type: {response.response_type}")
    
    # Use safe_execute to handle any unexpected errors
    safe_execute(
        _process_query,
        ErrorCategory.QUERY_PROCESSING,
        context={"query": query, "operation": "query_processing"},
        fallback_value=None
    )


def render_sidebar():
    """Render the sidebar with controls and statistics."""
    with st.sidebar:
        st.header("🔧 Dashboard Controls")
        
        # Session summary
        session_summary = get_session_summary()
        
        # Data status
        st.subheader("📊 Data Status")
        if session_summary['has_dataframe']:
            st.success("✅ Data loaded")
            if session_summary['file_info']:
                file_info = session_summary['file_info']
                st.write(f"**File:** {file_info['filename']}")
                st.write(f"**Rows:** {file_info['row_count']:,}")
                st.write(f"**Columns:** {file_info['column_count']}")
                size_mb = file_info['file_size'] / (1024 * 1024) if file_info['file_size'] else 0
                st.write(f"**Size:** {size_mb:.1f} MB")
        else:
            st.warning("⚠️ No data loaded")
        
        # Chat statistics
        st.subheader("💬 Chat Statistics")
        stats = get_chat_stats()
        
        if stats["total_messages"] > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", stats["total_messages"])
                st.metric("Questions", stats["user_messages"])
            with col2:
                st.metric("Responses", stats["agent_messages"])
                st.metric("Errors", stats["error_messages"])
            
            # Session timing
            if stats["session_start"] and stats["last_message"]:
                duration = stats["last_message"] - stats["session_start"]
                st.write(f"**Session duration:** {duration}")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History", type="secondary"):
                clear_chat()
        else:
            st.info("No conversation yet")
        
        # Agent status
        st.subheader("🤖 Agent Status")
        if session_summary['has_agent']:
            st.success("✅ Agent ready")
        else:
            if session_summary['has_dataframe']:
                st.info("🔄 Agent will initialize on first query")
            else:
                st.warning("⚠️ Upload data to enable agent")
        
        # Performance and Memory section
        st.subheader("⚡ Performance")
        if session_summary.get('memory_usage'):
            memory_info = session_summary['memory_usage']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Data Memory", f"{memory_info['dataframe_memory_mb']:.1f} MB")
            with col2:
                st.metric("Chat Memory", f"{memory_info['chat_memory_mb']:.1f} MB")
            
            # Memory optimization button
            if st.button("🧹 Optimize Memory", help="Clean up memory usage"):
                from utils.session_manager import optimize_session_memory
                result = optimize_session_memory()
                if result['optimization_successful']:
                    st.success("Memory optimized successfully!")
                else:
                    st.error("Memory optimization failed")
                st.rerun()
        
        # Chart cache info
        try:
            from components.visualization import get_cache_stats
            cache_stats = get_cache_stats()
            if cache_stats['cache_size'] > 0:
                st.write(f"**Chart Cache:** {cache_stats['cache_size']}/{cache_stats['max_size']} charts")
        except ImportError:
            pass
        
        # Debug mode toggle
        st.subheader("🔧 Debug Options")
        debug_mode = st.checkbox("Enable debug mode", value=False)
        st.session_state['debug_mode'] = debug_mode
        
        if debug_mode:
            with st.expander("Session Debug Info"):
                st.json(session_summary)
            
            with st.expander("Performance Stats"):
                try:
                    from components.visualization import get_cache_stats
                    cache_stats = get_cache_stats()
                    st.json(cache_stats)
                except ImportError:
                    st.write("Visualization cache not available")


def render_main_content():
    """Render the main content area with file upload and chat interface."""
    # Main header
    st.title("Lalika - AI Analytics Bot")
    st.markdown("""
    Welcome to your intelligent data analysis companion!
    """)
    
    # Check if this is the first visit
    if not has_dataframe() and not get_chat_stats()["total_messages"]:
        st.info("""
        👋 **Getting Started:**
        1. Upload a CSV or Excel file using the file uploader below
        2. Once your data is loaded, use the chat interface to ask questions
        3. Get instant insights and visualizations based on your queries
        """)
        
        # Show example queries for new users
        from components.example_queries import render_example_queries, render_quick_start_guide
        
        # Quick start guide
        render_quick_start_guide()
        
        # Example queries
        render_example_queries(show_context_examples=True)
    
    # Create layout columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📁 Data Upload")
        render_file_upload()
        
        # Show data-specific suggestions if data is loaded
        if has_dataframe():
            from components.example_queries import render_query_suggestions
            df = get_dataframe()
            if df is not None:
                render_query_suggestions(list(df.columns))
    
    with col2:
        st.header("💬 Chat Interface")
        render_chat_interface()
        
        # Show help section
        if has_dataframe():
            from components.example_queries import render_help_section
            render_help_section()


def handle_query_processing():
    """Handle any pending query processing from the chat interface."""
    # Check if there's a new query to process
    # This will be triggered by the chat interface when a user submits a query
    if 'pending_query' in st.session_state and st.session_state.pending_query:
        query = st.session_state.pending_query
        print("user query: ", query)

        st.session_state.pending_query = None  # Clear the pending query
        
        # Process the query
        process_user_query(query)


def main():
    """Main application entry point."""
    def _main_app():
        # Initialize session state
        initialize_session()
        
        # Configure Streamlit page
        st.set_page_config(
            page_title="Lalika - AI Analytics Bot",
            page_icon="☄❤️",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/vanshaj18/streamlit_data_chatbot',
                'Report a bug': 'https://github.com/vanshaj18/streamlit_data_chatbot/issues',
                'About': 
                """"
                    A multi-modal intelligent data analysis tool powered by PandasAI. Upload your data and ask questions in natural language!
                """
            }
        )
        
        # Render sidebar
        render_sidebar()
        
        # Render main content
        render_main_content()
        
        # Handle any pending query processing
        handle_query_processing()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "Built with ❤️ using Streamlit and PandasAI"
            "</div>", 
            unsafe_allow_html=True
        )
    
    # Use safe_execute for the entire application
    safe_execute(
        _main_app,
        ErrorCategory.SYSTEM_ERROR,
        context={"operation": "main_application"},
        fallback_value=None
    )


if __name__ == "__main__":
    main()