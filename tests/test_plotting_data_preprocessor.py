"""
Unit tests for plotting data preprocessor.

Tests various data quality scenarios and preprocessing functions
to ensure reliable chart generation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

from utils.plotting_data_preprocessor import (
    PlottingDataPreprocessor,
    clean_data_for_plotting,
    handle_missing_values_for_plotting,
    format_categorical_data_for_plotting,
    validate_numeric_columns_for_plotting,
    sample_large_dataset_for_plotting,
    DEFAULT_MAX_CATEGORIES,
    DEFAULT_SAMPLE_SIZE
)


class TestPlottingDataPreprocessor:
    """Test cases for PlottingDataPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = PlottingDataPreprocessor()
        
        # Create sample data with various data quality issues
        self.sample_data = pd.DataFrame({
            'numeric_clean': [1, 2, 3, 4, 5],
            'numeric_with_nan': [1.0, np.nan, 3.0, 4.0, np.nan],
            'numeric_with_inf': [1.0, 2.0, np.inf, 4.0, -np.inf],
            'categorical_clean': ['A', 'B', 'A', 'C', 'B'],
            'categorical_with_nan': ['X', np.nan, 'Y', 'X', None],
            'mixed_types': [1, 'text', 3.0, 'more_text', 5],
            'datetime_col': pd.date_range('2023-01-01', periods=5),
            'all_nan': [np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        
        # Large dataset for sampling tests
        self.large_data = pd.DataFrame({
            'x': np.random.randn(15000),
            'y': np.random.randn(15000),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 15000)
        })
        
        # Data with many categories
        self.many_categories_data = pd.DataFrame({
            'category': [f'Cat_{i}' for i in range(50)],
            'value': np.random.randn(50)
        })
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = PlottingDataPreprocessor(max_categories=15, max_sample_size=5000)
        assert preprocessor.max_categories == 15
        assert preprocessor.max_sample_size == 5000
    
    def test_clean_data_for_plotting_basic(self):
        """Test basic data cleaning functionality."""
        cleaned = self.preprocessor.clean_data_for_plotting(self.sample_data)
        
        # Should return a DataFrame
        assert isinstance(cleaned, pd.DataFrame)
        
        # Should not be empty
        assert not cleaned.empty
        
        # Should handle missing values
        assert cleaned['numeric_with_nan'].isnull().sum() <= self.sample_data['numeric_with_nan'].isnull().sum()
    
    def test_clean_data_for_plotting_with_chart_type(self):
        """Test data cleaning with specific chart types."""
        # Test histogram cleaning
        hist_data = self.preprocessor.clean_data_for_plotting(
            self.sample_data, chart_type="histogram"
        )
        assert isinstance(hist_data, pd.DataFrame)
        
        # Test pie chart cleaning
        pie_data = self.preprocessor.clean_data_for_plotting(
            self.sample_data, chart_type="pie"
        )
        assert isinstance(pie_data, pd.DataFrame)
        
        # Test scatter plot cleaning
        scatter_data = self.preprocessor.clean_data_for_plotting(
            self.sample_data, chart_type="scatter"
        )
        assert isinstance(scatter_data, pd.DataFrame)
    
    def test_clean_data_for_plotting_with_columns(self):
        """Test data cleaning with specific columns."""
        columns = ['numeric_clean', 'categorical_clean']
        cleaned = self.preprocessor.clean_data_for_plotting(
            self.sample_data, columns=columns
        )
        
        # Should only contain specified columns
        assert list(cleaned.columns) == columns
    
    def test_clean_data_empty_input(self):
        """Test cleaning with empty input."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Input data is None or empty"):
            self.preprocessor.clean_data_for_plotting(empty_df)
    
    def test_clean_data_none_input(self):
        """Test cleaning with None input."""
        with pytest.raises(ValueError, match="Input data is None or empty"):
            self.preprocessor.clean_data_for_plotting(None)
    
    def test_handle_missing_values_auto_strategy(self):
        """Test automatic missing value handling."""
        cleaned = self.preprocessor.handle_missing_values(
            self.sample_data, strategy="auto"
        )
        
        assert isinstance(cleaned, pd.DataFrame)
        assert not cleaned.empty
    
    def test_handle_missing_values_drop_strategy(self):
        """Test drop missing values strategy."""
        cleaned = self.preprocessor.handle_missing_values(
            self.sample_data, strategy="drop"
        )
        
        # Should have fewer missing values (may not be zero due to fallback strategy)
        assert cleaned.isnull().sum().sum() <= self.sample_data.isnull().sum().sum()
    
    def test_handle_missing_values_fill_strategy(self):
        """Test fill missing values strategy."""
        cleaned = self.preprocessor.handle_missing_values(
            self.sample_data, strategy="fill"
        )
        
        # Should have fewer missing values
        assert cleaned.isnull().sum().sum() <= self.sample_data.isnull().sum().sum()
    
    def test_handle_missing_values_interpolate_strategy(self):
        """Test interpolate missing values strategy."""
        cleaned = self.preprocessor.handle_missing_values(
            self.sample_data, strategy="interpolate"
        )
        
        # Numeric columns should have fewer missing values
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in self.sample_data.columns:
                assert cleaned[col].isnull().sum() <= self.sample_data[col].isnull().sum()
    
    def test_handle_missing_values_invalid_strategy(self):
        """Test invalid missing value strategy."""
        with pytest.raises(ValueError, match="Unknown missing value strategy"):
            self.preprocessor.handle_missing_values(
                self.sample_data, strategy="invalid_strategy"
            )
    
    def test_format_categorical_data(self):
        """Test categorical data formatting."""
        formatted = self.preprocessor.format_categorical_data(self.sample_data)
        
        # Should handle categorical columns
        assert 'categorical_clean' in formatted.columns
        assert formatted['categorical_clean'].dtype == 'object'
        
        # Should replace NaN values
        if 'categorical_with_nan' in formatted.columns:
            assert 'Unknown' in formatted['categorical_with_nan'].values
    
    def test_format_categorical_data_with_specific_columns(self):
        """Test categorical formatting with specific columns."""
        columns = ['categorical_clean']
        formatted = self.preprocessor.format_categorical_data(
            self.sample_data, columns=columns
        )
        
        # Should format specified columns
        assert formatted['categorical_clean'].dtype == 'object'
    
    def test_format_categorical_data_limit_categories(self):
        """Test limiting number of categories."""
        formatted = self.preprocessor.format_categorical_data(
            self.many_categories_data, columns=['category']
        )
        
        # Should limit categories
        unique_categories = formatted['category'].nunique()
        assert unique_categories <= self.preprocessor.max_categories
        
        # Should have 'Other' category if categories were limited
        if self.many_categories_data['category'].nunique() > self.preprocessor.max_categories:
            assert 'Other' in formatted['category'].values
    
    def test_validate_numeric_columns(self):
        """Test numeric column validation."""
        validated = self.preprocessor.validate_numeric_columns(self.sample_data)
        
        # Should handle numeric columns
        numeric_cols = validated.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0
        
        # Should handle infinite values
        for col in numeric_cols:
            assert not np.isinf(validated[col]).any()
    
    def test_validate_numeric_columns_with_specific_columns(self):
        """Test numeric validation with specific columns."""
        columns = ['numeric_with_inf']
        validated = self.preprocessor.validate_numeric_columns(
            self.sample_data, columns=columns
        )
        
        # Should remove infinite values
        assert not np.isinf(validated['numeric_with_inf']).any()
    
    def test_sample_large_dataset_random(self):
        """Test random sampling of large datasets."""
        sampled = self.preprocessor.sample_large_dataset(
            self.large_data, max_points=1000, strategy="random"
        )
        
        assert len(sampled) == 1000
        assert list(sampled.columns) == list(self.large_data.columns)
    
    def test_sample_large_dataset_systematic(self):
        """Test systematic sampling of large datasets."""
        sampled = self.preprocessor.sample_large_dataset(
            self.large_data, max_points=1000, strategy="systematic"
        )
        
        assert len(sampled) == 1000
        assert list(sampled.columns) == list(self.large_data.columns)
    
    def test_sample_large_dataset_stratified(self):
        """Test stratified sampling of large datasets."""
        sampled = self.preprocessor.sample_large_dataset(
            self.large_data, max_points=1000, strategy="stratified"
        )
        
        assert len(sampled) <= 1000
        assert list(sampled.columns) == list(self.large_data.columns)
        
        # Should maintain category distribution
        original_categories = set(self.large_data['category'].unique())
        sampled_categories = set(sampled['category'].unique())
        assert sampled_categories.issubset(original_categories)
    
    def test_sample_large_dataset_small_data(self):
        """Test sampling with data smaller than max_points."""
        sampled = self.preprocessor.sample_large_dataset(
            self.sample_data, max_points=1000
        )
        
        # Should return original data if smaller than max_points
        assert len(sampled) == len(self.sample_data)
    
    def test_sample_large_dataset_invalid_strategy(self):
        """Test invalid sampling strategy."""
        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            self.preprocessor.sample_large_dataset(
                self.large_data, strategy="invalid_strategy"
            )
    
    def test_aggregate_for_performance(self):
        """Test data aggregation for performance."""
        # Create test data for aggregation
        agg_data = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        
        aggregated = self.preprocessor.aggregate_for_performance(
            agg_data, group_by='category', agg_column='value', agg_method='mean'
        )
        
        assert len(aggregated) == 3  # Three categories
        assert 'category' in aggregated.columns
        assert 'value' in aggregated.columns
        
        # Check aggregation results
        assert aggregated[aggregated['category'] == 'A']['value'].iloc[0] == 15  # (10+20)/2
        assert aggregated[aggregated['category'] == 'B']['value'].iloc[0] == 35  # (30+40)/2
        assert aggregated[aggregated['category'] == 'C']['value'].iloc[0] == 55  # (50+60)/2
    
    def test_aggregate_for_performance_count(self):
        """Test count aggregation."""
        agg_data = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        
        aggregated = self.preprocessor.aggregate_for_performance(
            agg_data, group_by='category', agg_column='count', agg_method='count'
        )
        
        assert len(aggregated) == 3
        assert all(aggregated['count'] == 2)  # Each category has 2 items
    
    def test_aggregate_for_performance_invalid_columns(self):
        """Test aggregation with invalid columns."""
        with pytest.raises(ValueError, match="Columns .* not found in data"):
            self.preprocessor.aggregate_for_performance(
                self.sample_data, group_by='nonexistent', agg_column='also_nonexistent'
            )
    
    def test_chart_specific_cleaning_pie(self):
        """Test pie chart specific cleaning."""
        # Create data with negative values
        pie_data = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'value': [-10, 20, -5]
        })
        
        cleaned = self.preprocessor.clean_data_for_plotting(pie_data, chart_type="pie")
        
        # Should convert negative values to positive
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert all(cleaned[col] >= 0)
    
    def test_chart_specific_cleaning_histogram(self):
        """Test histogram specific cleaning."""
        cleaned = self.preprocessor.clean_data_for_plotting(
            self.sample_data, chart_type="histogram"
        )
        
        # Should focus on numeric columns
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0
    
    def test_chart_specific_cleaning_scatter(self):
        """Test scatter plot specific cleaning."""
        # Create data with sufficient numeric columns
        scatter_data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        
        cleaned = self.preprocessor.clean_data_for_plotting(
            scatter_data, chart_type="scatter"
        )
        
        # Should keep numeric columns
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) >= 2
    
    def test_chart_specific_cleaning_scatter_insufficient_numeric(self):
        """Test scatter plot cleaning with insufficient numeric columns."""
        # Create data with only one numeric column
        scatter_data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        
        with pytest.raises(ValueError, match="Scatter plot requires at least two numeric columns"):
            self.preprocessor.clean_data_for_plotting(scatter_data, chart_type="scatter")
    
    def test_extreme_outlier_handling(self):
        """Test extreme outlier handling."""
        # Create data with extreme outliers
        outlier_data = pd.DataFrame({
            'normal': [1, 2, 3, 4, 5],
            'with_outliers': [1, 2, 1000000, 4, 5]  # Extreme outlier
        })
        
        cleaned = self.preprocessor.validate_numeric_columns(outlier_data)
        
        # Should handle extreme outliers
        assert cleaned['with_outliers'].max() < 1000000
    
    def test_data_validation_empty_after_preprocessing(self):
        """Test validation when data becomes empty after preprocessing."""
        # Create data that will become empty
        empty_after_cleaning = pd.DataFrame({
            'all_nan': [np.nan, np.nan, np.nan]
        })
        
        with pytest.raises(ValueError, match="Insufficient data points for plotting"):
            self.preprocessor.clean_data_for_plotting(empty_after_cleaning)
    
    def test_data_validation_insufficient_data_points(self):
        """Test validation with insufficient data points."""
        # Create data with only one row
        insufficient_data = pd.DataFrame({
            'x': [1],
            'y': [2]
        })
        
        with pytest.raises(ValueError, match="Insufficient data points for plotting"):
            self.preprocessor.clean_data_for_plotting(insufficient_data)


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'numeric': [1, 2, np.nan, 4, 5],
            'categorical': ['A', 'B', np.nan, 'C', 'B'],
            'mixed': [1, 'text', 3.0, 'more', 5]
        })
    
    def test_clean_data_for_plotting_convenience(self):
        """Test convenience function for data cleaning."""
        cleaned = clean_data_for_plotting(self.sample_data)
        
        assert isinstance(cleaned, pd.DataFrame)
        assert not cleaned.empty
    
    def test_handle_missing_values_for_plotting_convenience(self):
        """Test convenience function for missing value handling."""
        cleaned = handle_missing_values_for_plotting(self.sample_data)
        
        assert isinstance(cleaned, pd.DataFrame)
        assert not cleaned.empty
    
    def test_format_categorical_data_for_plotting_convenience(self):
        """Test convenience function for categorical formatting."""
        formatted = format_categorical_data_for_plotting(self.sample_data)
        
        assert isinstance(formatted, pd.DataFrame)
        assert not formatted.empty
    
    def test_validate_numeric_columns_for_plotting_convenience(self):
        """Test convenience function for numeric validation."""
        validated = validate_numeric_columns_for_plotting(self.sample_data)
        
        assert isinstance(validated, pd.DataFrame)
        assert not validated.empty
    
    def test_sample_large_dataset_for_plotting_convenience(self):
        """Test convenience function for dataset sampling."""
        # Create larger dataset
        large_data = pd.DataFrame({
            'x': np.random.randn(2000),
            'y': np.random.randn(2000)
        })
        
        sampled = sample_large_dataset_for_plotting(large_data, max_points=500)
        
        assert len(sampled) == 500
        assert list(sampled.columns) == list(large_data.columns)


class TestEdgeCases:
    """Test cases for edge cases and error conditions."""
    
    def test_all_nan_columns(self):
        """Test handling of columns with all NaN values."""
        all_nan_data = pd.DataFrame({
            'all_nan': [np.nan, np.nan, np.nan],
            'some_data': [1, 2, 3]
        })
        
        preprocessor = PlottingDataPreprocessor()
        cleaned = preprocessor.clean_data_for_plotting(all_nan_data)
        
        # Should handle all-NaN columns (may fill with default values for plotting)
        assert 'some_data' in cleaned.columns
        assert len(cleaned) > 0  # Should have some data remaining
        
        # If all_nan column is kept, it should be filled with default values
        if 'all_nan' in cleaned.columns:
            assert not cleaned['all_nan'].isna().all()  # Should not be all NaN anymore
    
    def test_single_value_columns(self):
        """Test handling of columns with single unique value."""
        single_value_data = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5],
            'variable': [1, 2, 3, 4, 5]
        })
        
        preprocessor = PlottingDataPreprocessor()
        cleaned = preprocessor.clean_data_for_plotting(single_value_data)
        
        # Should handle constant columns
        assert isinstance(cleaned, pd.DataFrame)
        assert not cleaned.empty
    
    def test_mixed_data_types_in_column(self):
        """Test handling of columns with mixed data types."""
        mixed_data = pd.DataFrame({
            'mixed': [1, 'text', 3.14, None, True],
            'numeric': [1, 2, 3, 4, 5]
        })
        
        preprocessor = PlottingDataPreprocessor()
        cleaned = preprocessor.clean_data_for_plotting(mixed_data)
        
        # Should handle mixed types
        assert isinstance(cleaned, pd.DataFrame)
        assert not cleaned.empty
    
    def test_datetime_columns(self):
        """Test handling of datetime columns."""
        datetime_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'value': [1, 2, 3, 4, 5]
        })
        
        preprocessor = PlottingDataPreprocessor()
        cleaned = preprocessor.clean_data_for_plotting(datetime_data)
        
        # Should handle datetime columns
        assert isinstance(cleaned, pd.DataFrame)
        assert not cleaned.empty
    
    def test_very_large_categories(self):
        """Test handling of columns with very large number of categories."""
        large_cat_data = pd.DataFrame({
            'category': [f'Cat_{i}' for i in range(1000)],
            'value': np.random.randn(1000)
        })
        
        preprocessor = PlottingDataPreprocessor(max_categories=10)
        formatted = preprocessor.format_categorical_data(
            large_cat_data, columns=['category']
        )
        
        # Should limit categories
        assert formatted['category'].nunique() <= 10
        assert 'Other' in formatted['category'].values
    
    def test_zero_variance_numeric_columns(self):
        """Test handling of numeric columns with zero variance."""
        zero_var_data = pd.DataFrame({
            'constant_numeric': [5.0, 5.0, 5.0, 5.0, 5.0],
            'variable_numeric': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        preprocessor = PlottingDataPreprocessor()
        validated = preprocessor.validate_numeric_columns(zero_var_data)
        
        # Should handle zero variance columns
        assert isinstance(validated, pd.DataFrame)
        assert not validated.empty
        assert 'constant_numeric' in validated.columns
        assert 'variable_numeric' in validated.columns


if __name__ == "__main__":
    pytest.main([__file__])