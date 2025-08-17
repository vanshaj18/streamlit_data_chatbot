"""
Data preprocessing utilities for reliable chart generation.

This module provides data cleaning and formatting functions specifically
designed to prepare data for plotting operations, handling common issues
that can cause chart generation failures.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_MAX_CATEGORIES = 20  # Maximum categories for categorical plots
DEFAULT_SAMPLE_SIZE = 10000  # Default sample size for large datasets
MIN_DATA_POINTS = 2  # Minimum data points required for plotting


class PlottingDataPreprocessor:
    """
    Data preprocessor specifically designed for chart generation.
    
    Handles missing values, data type formatting, sampling, and validation
    to ensure reliable chart generation across different visualization libraries.
    """
    
    def __init__(self, max_categories: int = DEFAULT_MAX_CATEGORIES,
                 max_sample_size: int = DEFAULT_SAMPLE_SIZE):
        """
        Initialize the plotting data preprocessor.
        
        Args:
            max_categories: Maximum number of categories to display in categorical plots
            max_sample_size: Maximum number of data points to use for large datasets
        """
        self.max_categories = max_categories
        self.max_sample_size = max_sample_size
        self.logger = logging.getLogger(__name__)
    
    def clean_data_for_plotting(self, data: pd.DataFrame, 
                               chart_type: Optional[str] = None,
                               columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Clean and prepare data for chart generation.
        
        Args:
            data: Input DataFrame to clean
            chart_type: Type of chart being generated (bar, histogram, pie, scatter, etc.)
            columns: Specific columns to focus on for cleaning
            
        Returns:
            pd.DataFrame: Cleaned DataFrame ready for plotting
        """
        try:
            if data is None or data.empty:
                raise ValueError("Input data is None or empty")
            
            # Create a copy to avoid modifying original data
            cleaned_data = data.copy()
            
            # Focus on specific columns if provided
            if columns:
                available_columns = [col for col in columns if col in cleaned_data.columns]
                if not available_columns:
                    raise ValueError(f"None of the specified columns {columns} exist in the data")
                cleaned_data = cleaned_data[available_columns]
            
            # Apply chart-type specific cleaning
            if chart_type:
                cleaned_data = self._apply_chart_specific_cleaning(cleaned_data, chart_type)
            
            # General cleaning steps
            cleaned_data = self._handle_missing_values(cleaned_data, chart_type)
            cleaned_data = self._format_data_types(cleaned_data)
            cleaned_data = self._handle_large_datasets(cleaned_data)
            cleaned_data = self._validate_data_for_plotting(cleaned_data)
            
            self.logger.info(f"Data cleaned for plotting: {cleaned_data.shape} -> ready for {chart_type or 'general'} chart")
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Error cleaning data for plotting: {str(e)}")
            raise ValueError(f"Failed to clean data for plotting: {str(e)}")
    
    def handle_missing_values(self, data: pd.DataFrame, 
                             strategy: str = "auto",
                             chart_type: Optional[str] = None) -> pd.DataFrame:
        """
        Handle missing values in data based on chart type and strategy.
        
        Args:
            data: Input DataFrame
            strategy: Strategy for handling missing values ("auto", "drop", "fill", "interpolate")
            chart_type: Type of chart being generated
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        try:
            if data.empty:
                return data
            
            cleaned_data = data.copy()
            
            if strategy == "auto":
                # Automatic strategy based on chart type and data characteristics
                cleaned_data = self._auto_handle_missing_values(cleaned_data, chart_type)
            elif strategy == "drop":
                # Drop rows with any missing values, but keep at least some data
                cleaned_data = cleaned_data.dropna(how='any')
                # If all data was removed, try dropping only rows where all values are missing
                if cleaned_data.empty:
                    cleaned_data = data.dropna(how='all')
            elif strategy == "fill":
                # Fill missing values with appropriate defaults
                cleaned_data = self._fill_missing_values(cleaned_data)
            elif strategy == "interpolate":
                # Interpolate missing values for numeric columns
                cleaned_data = self._interpolate_missing_values(cleaned_data)
            else:
                self.logger.error(f"Unknown missing value strategy: {strategy}")
                raise ValueError(f"Unknown missing value strategy: {strategy}")
            
            # Ensure we still have data after cleaning
            if cleaned_data.empty:
                self.logger.warning("All data was removed during missing value handling")
                return data.dropna(how='all')  # Keep rows with at least some data
            
            missing_before = data.isnull().sum().sum()
            missing_after = cleaned_data.isnull().sum().sum()
            self.logger.info(f"Missing values handled: {missing_before} -> {missing_after}")
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            # Re-raise the exception to maintain proper error handling
            raise
    
    def format_categorical_data(self, data: pd.DataFrame, 
                               columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Format categorical data for proper chart display.
        
        Args:
            data: Input DataFrame
            columns: Specific columns to format (if None, auto-detect categorical columns)
            
        Returns:
            pd.DataFrame: DataFrame with properly formatted categorical data
        """
        try:
            formatted_data = data.copy()
            
            # Auto-detect categorical columns if not specified
            if columns is None:
                columns = self._detect_categorical_columns(formatted_data)
            
            for col in columns:
                if col not in formatted_data.columns:
                    continue
                
                # Convert to string and handle special values
                formatted_data[col] = formatted_data[col].astype(str)
                
                # Replace common problematic values
                formatted_data[col] = formatted_data[col].replace({
                    'nan': 'Unknown',
                    'None': 'Unknown',
                    'null': 'Unknown',
                    '': 'Empty'
                })
                
                # Limit number of categories for better visualization
                if formatted_data[col].nunique() > self.max_categories:
                    formatted_data = self._limit_categories(formatted_data, col)
            
            self.logger.info(f"Categorical data formatted for {len(columns)} columns")
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error formatting categorical data: {str(e)}")
            return data
    
    def validate_numeric_columns(self, data: pd.DataFrame, 
                                columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Validate and clean numeric columns for plotting.
        
        Args:
            data: Input DataFrame
            columns: Specific numeric columns to validate (if None, auto-detect)
            
        Returns:
            pd.DataFrame: DataFrame with validated numeric columns
        """
        try:
            validated_data = data.copy()
            
            # Auto-detect numeric columns if not specified
            if columns is None:
                columns = list(validated_data.select_dtypes(include=[np.number]).columns)
            
            for col in columns:
                if col not in validated_data.columns:
                    continue
                
                # Convert to numeric, handling errors
                validated_data[col] = pd.to_numeric(validated_data[col], errors='coerce')
                
                # Handle infinite values
                validated_data[col] = validated_data[col].replace([np.inf, -np.inf], np.nan)
                
                # Remove extreme outliers that might break plotting
                if validated_data[col].notna().sum() > 0:
                    validated_data = self._handle_extreme_outliers(validated_data, col)
            
            self.logger.info(f"Numeric columns validated: {len(columns)} columns")
            return validated_data
            
        except Exception as e:
            self.logger.error(f"Error validating numeric columns: {str(e)}")
            return data
    
    def sample_large_dataset(self, data: pd.DataFrame, 
                            max_points: Optional[int] = None,
                            strategy: str = "random") -> pd.DataFrame:
        """
        Sample large datasets for better chart performance.
        
        Args:
            data: Input DataFrame
            max_points: Maximum number of points to keep (if None, use default)
            strategy: Sampling strategy ("random", "systematic", "stratified")
            
        Returns:
            pd.DataFrame: Sampled DataFrame
        """
        try:
            if max_points is None:
                max_points = self.max_sample_size
            
            if len(data) <= max_points:
                return data
            
            if strategy == "random":
                sampled_data = data.sample(n=max_points, random_state=42)
            elif strategy == "systematic":
                step = len(data) // max_points
                sampled_data = data.iloc[::step][:max_points]
            elif strategy == "stratified":
                # Try stratified sampling if there's a categorical column
                categorical_cols = self._detect_categorical_columns(data)
                if categorical_cols:
                    sampled_data = self._stratified_sample(data, categorical_cols[0], max_points)
                else:
                    # Fall back to random sampling
                    sampled_data = data.sample(n=max_points, random_state=42)
            else:
                self.logger.error(f"Unknown sampling strategy: {strategy}")
                raise ValueError(f"Unknown sampling strategy: {strategy}")
            
            self.logger.info(f"Dataset sampled: {len(data)} -> {len(sampled_data)} points")
            return sampled_data
            
        except Exception as e:
            self.logger.error(f"Error sampling dataset: {str(e)}")
            # Re-raise the exception to maintain proper error handling
            raise
    
    def aggregate_for_performance(self, data: pd.DataFrame, 
                                 group_by: str,
                                 agg_column: str,
                                 agg_method: str = "mean") -> pd.DataFrame:
        """
        Aggregate data for better chart performance and clarity.
        
        Args:
            data: Input DataFrame
            group_by: Column to group by
            agg_column: Column to aggregate
            agg_method: Aggregation method ("mean", "sum", "count", "median")
            
        Returns:
            pd.DataFrame: Aggregated DataFrame
        """
        try:
            if group_by not in data.columns:
                raise ValueError(f"Columns {group_by} not found in data")
            
            # Perform aggregation
            if agg_method == "count":
                aggregated = data.groupby(group_by).size().reset_index(name=agg_column)
            else:
                if agg_column not in data.columns:
                    raise ValueError(f"Columns {group_by} or {agg_column} not found in data")
                    
                agg_func = {
                    "mean": "mean",
                    "sum": "sum", 
                    "median": "median",
                    "min": "min",
                    "max": "max"
                }.get(agg_method, "mean")
                
                aggregated = data.groupby(group_by)[agg_column].agg(agg_func).reset_index()
            
            # Sort by aggregated values for better visualization
            aggregated = aggregated.sort_values(agg_column, ascending=False)
            
            self.logger.info(f"Data aggregated: {len(data)} -> {len(aggregated)} groups")
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error aggregating data: {str(e)}")
            raise ValueError(f"Error aggregating data: {str(e)}")
    
    def _apply_chart_specific_cleaning(self, data: pd.DataFrame, chart_type: str) -> pd.DataFrame:
        """Apply cleaning specific to chart type."""
        chart_type_lower = chart_type.lower()
        
        if chart_type_lower in ["pie", "donut"]:
            # For pie charts, ensure we have positive values and limit categories
            return self._clean_for_pie_chart(data)
        elif chart_type_lower in ["histogram", "hist"]:
            # For histograms, ensure numeric data and handle outliers
            return self._clean_for_histogram(data)
        elif chart_type_lower in ["scatter", "scatterplot"]:
            # For scatter plots, ensure we have two numeric columns
            return self._clean_for_scatter_plot(data)
        elif chart_type_lower in ["bar", "column"]:
            # For bar charts, handle categorical data and aggregation
            return self._clean_for_bar_chart(data)
        else:
            return data
    
    def _handle_missing_values(self, data: pd.DataFrame, chart_type: Optional[str]) -> pd.DataFrame:
        """Handle missing values based on chart type."""
        return self.handle_missing_values(data, strategy="auto", chart_type=chart_type)
    
    def _auto_handle_missing_values(self, data: pd.DataFrame, chart_type: Optional[str]) -> pd.DataFrame:
        """Automatically determine the best strategy for handling missing values."""
        missing_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns))
        
        if missing_percentage > 0.5:
            # Too many missing values, drop rows with missing values
            return data.dropna()
        elif missing_percentage > 0.1:
            # Moderate missing values, fill with appropriate defaults
            return self._fill_missing_values(data)
        else:
            # Few missing values, interpolate for numeric, drop for others
            return self._interpolate_missing_values(data)
    
    def _fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate defaults."""
        filled_data = data.copy()
        
        for col in filled_data.columns:
            if filled_data[col].dtype in ['object', 'category']:
                # Fill categorical with 'Unknown'
                filled_data[col] = filled_data[col].fillna('Unknown')
            elif filled_data[col].dtype in ['int64', 'float64']:
                # Fill numeric with median
                median_val = filled_data[col].median()
                if pd.isna(median_val):
                    # If median is NaN (all values are NaN), fill with 0
                    filled_data[col] = filled_data[col].fillna(0)
                else:
                    filled_data[col] = filled_data[col].fillna(median_val)
            elif filled_data[col].dtype == 'datetime64[ns]':
                # Fill datetime with forward fill
                filled_data[col] = filled_data[col].ffill()
        
        return filled_data
    
    def _interpolate_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing values for numeric columns."""
        interpolated_data = data.copy()
        
        numeric_columns = interpolated_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            interpolated_data[col] = interpolated_data[col].interpolate()
        
        # Drop rows with missing values in non-numeric columns
        non_numeric_columns = interpolated_data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_columns) > 0:
            interpolated_data = interpolated_data.dropna(subset=non_numeric_columns)
        
        return interpolated_data
    
    def _format_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Format data types for optimal plotting."""
        formatted_data = data.copy()
        
        # Skip formatting if data is empty
        if len(formatted_data) == 0:
            return formatted_data
        
        # Convert object columns that should be categorical
        for col in formatted_data.select_dtypes(include=['object']).columns:
            if len(formatted_data) > 0 and formatted_data[col].nunique() / len(formatted_data) < 0.5:
                formatted_data[col] = formatted_data[col].astype('category')
        
        # Ensure numeric columns are properly typed
        for col in formatted_data.columns:
            if formatted_data[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    numeric_series = pd.to_numeric(formatted_data[col], errors='coerce')
                    # Only replace if conversion was successful for some values
                    if not numeric_series.isna().all():
                        formatted_data[col] = numeric_series
                except Exception:
                    pass  # Keep original data type if conversion fails
        
        return formatted_data
    
    def _handle_large_datasets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle large datasets by sampling if necessary."""
        if len(data) > self.max_sample_size:
            return self.sample_large_dataset(data)
        return data
    
    def _validate_data_for_plotting(self, data: pd.DataFrame) -> pd.DataFrame:
        """Final validation to ensure data is suitable for plotting."""
        if data.empty:
            raise ValueError("Insufficient data points for plotting (data is empty after preprocessing)")
        
        if len(data) < MIN_DATA_POINTS:
            raise ValueError(f"Insufficient data points for plotting (minimum {MIN_DATA_POINTS})")
        
        # Remove columns that are all NaN
        data = data.dropna(axis=1, how='all')
        
        if data.empty:
            raise ValueError("Insufficient data points for plotting (no valid columns remaining)")
        
        return data
    
    def _detect_categorical_columns(self, data: pd.DataFrame) -> List[str]:
        """Detect columns that should be treated as categorical."""
        categorical_columns = []
        
        for col in data.columns:
            if data[col].dtype in ['object', 'category']:
                categorical_columns.append(col)
            elif data[col].dtype in ['int64', 'float64']:
                # Check if numeric column has few unique values (might be categorical)
                if data[col].nunique() <= 10 and data[col].nunique() / len(data) < 0.1:
                    categorical_columns.append(col)
        
        return categorical_columns
    
    def _limit_categories(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Limit the number of categories in a column for better visualization."""
        limited_data = data.copy()
        
        # Get top categories by frequency
        top_categories = limited_data[column].value_counts().head(self.max_categories - 1).index
        
        # Replace less frequent categories with 'Other'
        limited_data[column] = limited_data[column].apply(
            lambda x: x if x in top_categories else 'Other'
        )
        
        return limited_data
    
    def _handle_extreme_outliers(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove extreme outliers that might break plotting."""
        cleaned_data = data.copy()
        
        Q1 = cleaned_data[column].quantile(0.25)
        Q3 = cleaned_data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Use a more conservative outlier detection (3 * IQR instead of 1.5)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Only remove extreme outliers, keep moderate ones
        outlier_mask = (cleaned_data[column] < lower_bound) | (cleaned_data[column] > upper_bound)
        outliers_removed = outlier_mask.sum()
        
        if outliers_removed > 0:
            cleaned_data = cleaned_data[~outlier_mask]
            self.logger.info(f"Removed {outliers_removed} extreme outliers from {column}")
        
        return cleaned_data
    
    def _stratified_sample(self, data: pd.DataFrame, column: str, max_points: int) -> pd.DataFrame:
        """Perform stratified sampling based on a categorical column."""
        try:
            # Calculate sample size per category
            category_counts = data[column].value_counts()
            total_categories = len(category_counts)
            
            if total_categories == 0:
                return data.sample(n=min(max_points, len(data)), random_state=42)
            
            samples_per_category = max(1, max_points // total_categories)
            
            sampled_dfs = []
            for category in category_counts.index:
                category_data = data[data[column] == category]
                sample_size = min(samples_per_category, len(category_data))
                sampled_dfs.append(category_data.sample(n=sample_size, random_state=42))
            
            return pd.concat(sampled_dfs, ignore_index=True)
            
        except Exception as e:
            self.logger.warning(f"Stratified sampling failed: {str(e)}, falling back to random sampling")
            return data.sample(n=min(max_points, len(data)), random_state=42)
    
    def _clean_for_pie_chart(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data specifically for pie charts."""
        cleaned_data = data.copy()
        
        # For pie charts, we need at least one numeric column with positive values
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            # Find the first numeric column with some positive values
            for col in numeric_columns:
                # Convert to absolute values to handle negative numbers
                abs_values = cleaned_data[col].abs()
                # Only filter if we have some positive values to work with
                if abs_values.gt(0).any():
                    cleaned_data[col] = abs_values
                    # Only remove rows where this specific column is zero/negative
                    cleaned_data = cleaned_data[cleaned_data[col] > 0]
                    break
        
        return cleaned_data
    
    def _clean_for_histogram(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data specifically for histograms."""
        cleaned_data = data.copy()
        
        # Focus on numeric columns only
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            # Keep only the first numeric column for histogram
            cleaned_data = cleaned_data[numeric_columns[:1]]
            
            # Remove outliers more aggressively for histograms
            for col in numeric_columns[:1]:
                cleaned_data = self._handle_extreme_outliers(cleaned_data, col)
        
        return cleaned_data
    
    def _clean_for_scatter_plot(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data specifically for scatter plots."""
        cleaned_data = data.copy()
        
        # Ensure we have at least two numeric columns
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < 2:
            raise ValueError("Scatter plot requires at least two numeric columns")
        
        # Keep only numeric columns
        cleaned_data = cleaned_data[numeric_columns]
        
        # Remove rows where both x and y are missing
        cleaned_data = cleaned_data.dropna(how='all')
        
        return cleaned_data
    
    def _clean_for_bar_chart(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data specifically for bar charts."""
        cleaned_data = data.copy()
        
        # Limit categories for better visualization
        categorical_columns = self._detect_categorical_columns(cleaned_data)
        for col in categorical_columns:
            if cleaned_data[col].nunique() > self.max_categories:
                cleaned_data = self._limit_categories(cleaned_data, col)
        
        return cleaned_data


# Convenience functions for direct use
def clean_data_for_plotting(data: pd.DataFrame, 
                           chart_type: Optional[str] = None,
                           columns: Optional[List[str]] = None,
                           max_categories: int = DEFAULT_MAX_CATEGORIES,
                           max_sample_size: int = DEFAULT_SAMPLE_SIZE) -> pd.DataFrame:
    """
    Convenience function to clean data for plotting.
    
    Args:
        data: Input DataFrame to clean
        chart_type: Type of chart being generated
        columns: Specific columns to focus on
        max_categories: Maximum number of categories for categorical plots
        max_sample_size: Maximum number of data points for large datasets
        
    Returns:
        pd.DataFrame: Cleaned DataFrame ready for plotting
    """
    preprocessor = PlottingDataPreprocessor(max_categories, max_sample_size)
    return preprocessor.clean_data_for_plotting(data, chart_type, columns)


def handle_missing_values_for_plotting(data: pd.DataFrame, 
                                      strategy: str = "auto",
                                      chart_type: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to handle missing values for plotting.
    
    Args:
        data: Input DataFrame
        strategy: Strategy for handling missing values
        chart_type: Type of chart being generated
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    preprocessor = PlottingDataPreprocessor()
    return preprocessor.handle_missing_values(data, strategy, chart_type)


def format_categorical_data_for_plotting(data: pd.DataFrame, 
                                        columns: Optional[List[str]] = None,
                                        max_categories: int = DEFAULT_MAX_CATEGORIES) -> pd.DataFrame:
    """
    Convenience function to format categorical data for plotting.
    
    Args:
        data: Input DataFrame
        columns: Specific columns to format
        max_categories: Maximum number of categories to display
        
    Returns:
        pd.DataFrame: DataFrame with properly formatted categorical data
    """
    preprocessor = PlottingDataPreprocessor(max_categories=max_categories)
    return preprocessor.format_categorical_data(data, columns)


def validate_numeric_columns_for_plotting(data: pd.DataFrame, 
                                         columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convenience function to validate numeric columns for plotting.
    
    Args:
        data: Input DataFrame
        columns: Specific numeric columns to validate
        
    Returns:
        pd.DataFrame: DataFrame with validated numeric columns
    """
    preprocessor = PlottingDataPreprocessor()
    return preprocessor.validate_numeric_columns(data, columns)


def sample_large_dataset_for_plotting(data: pd.DataFrame, 
                                     max_points: Optional[int] = None,
                                     strategy: str = "random") -> pd.DataFrame:
    """
    Convenience function to sample large datasets for plotting.
    
    Args:
        data: Input DataFrame
        max_points: Maximum number of points to keep
        strategy: Sampling strategy
        
    Returns:
        pd.DataFrame: Sampled DataFrame
    """
    preprocessor = PlottingDataPreprocessor()
    return preprocessor.sample_large_dataset(data, max_points, strategy)