"""Tests for utils.aggregate module."""

import pandas as pd
import os
import tempfile
import shutil

from utils.aggregate import aggregate_results, collect_results


class TestCollectResults:
    """Tests for collect_results function."""

    def setup_method(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_collect_single_region(self):
        """Test collecting results from single region."""
        # Create test CSV files for monkeyF V1 (arrays 0-7)
        df1 = pd.DataFrame({"Layer": ["layer1", "layer2"], "Score": [0.5, 0.6]})
        df1.to_csv(os.path.join(self.test_dir, "monkeyF_arr_0.csv"), index=False)

        df2 = pd.DataFrame({"Layer": ["layer1", "layer2"], "Score": [0.55, 0.65]})
        df2.to_csv(os.path.join(self.test_dir, "monkeyF_arr_1.csv"), index=False)

        results = collect_results(self.test_dir, monkey="monkeyF")

        assert "V1" in results
        assert len(results["V1"]) == 2
        assert isinstance(results["V1"][0], pd.DataFrame)

    def test_collect_multiple_regions(self):
        """Test collecting results from multiple regions."""
        # V1: arrays 0-7
        df_v1 = pd.DataFrame({"Layer": ["layer1"], "Score": [0.5]})
        df_v1.to_csv(os.path.join(self.test_dir, "monkeyF_arr_0.csv"), index=False)

        # IT: arrays 8-12
        df_it = pd.DataFrame({"Layer": ["layer1"], "Score": [0.6]})
        df_it.to_csv(os.path.join(self.test_dir, "monkeyF_arr_8.csv"), index=False)

        # V4: arrays 13-15
        df_v4 = pd.DataFrame({"Layer": ["layer1"], "Score": [0.7]})
        df_v4.to_csv(os.path.join(self.test_dir, "monkeyF_arr_13.csv"), index=False)

        results = collect_results(self.test_dir, monkey="monkeyF")

        assert "V1" in results
        assert "IT" in results
        assert "V4" in results
        assert len(results["V1"]) == 1
        assert len(results["IT"]) == 1
        assert len(results["V4"]) == 1

    def test_collect_filters_by_monkey(self):
        """Test that only specified monkey's results are collected."""
        # MonkeyF file
        df_f = pd.DataFrame({"Layer": ["layer1"], "Score": [0.5]})
        df_f.to_csv(os.path.join(self.test_dir, "monkeyF_arr_0.csv"), index=False)

        # MonkeyN file
        df_n = pd.DataFrame({"Layer": ["layer1"], "Score": [0.6]})
        df_n.to_csv(os.path.join(self.test_dir, "monkeyN_arr_0.csv"), index=False)

        results_f = collect_results(self.test_dir, monkey="monkeyF")
        results_n = collect_results(self.test_dir, monkey="monkeyN")

        # MonkeyF should only have 1 file
        assert "V1" in results_f
        assert len(results_f["V1"]) == 1

        # MonkeyN should only have 1 file
        assert "V1" in results_n
        assert len(results_n["V1"]) == 1

    def test_collect_ignores_aggregate_files(self):
        """Test that aggregate files are ignored."""
        # Regular file
        df1 = pd.DataFrame({"Layer": ["layer1"], "Score": [0.5]})
        df1.to_csv(os.path.join(self.test_dir, "monkeyF_arr_0.csv"), index=False)

        # Aggregate file (should be ignored)
        df_agg = pd.DataFrame({"model": ["model1"], "score": [0.8]})
        df_agg.to_csv(os.path.join(self.test_dir, "agg_results_V1.csv"), index=False)

        results = collect_results(self.test_dir, monkey="monkeyF")

        # Should only collect 1 file (not the aggregate)
        assert "V1" in results
        assert len(results["V1"]) == 1

    def test_collect_empty_directory(self):
        """Test collecting from empty directory."""
        results = collect_results(self.test_dir, monkey="monkeyF")
        assert results == {}

    def test_collect_no_matching_monkey(self):
        """Test collecting when no files match the monkey."""
        df = pd.DataFrame({"Layer": ["layer1"], "Score": [0.5]})
        df.to_csv(os.path.join(self.test_dir, "monkeyF_arr_0.csv"), index=False)

        results = collect_results(self.test_dir, monkey="monkeyN")
        assert results == {}

    def test_collect_monkeyN_regions(self):
        """Test collecting results for monkeyN with correct region mapping."""
        # V1: arrays 0-7
        df_v1 = pd.DataFrame({"Layer": ["layer1"], "Score": [0.5]})
        df_v1.to_csv(os.path.join(self.test_dir, "monkeyN_arr_0.csv"), index=False)

        # V4: arrays 8-11
        df_v4 = pd.DataFrame({"Layer": ["layer1"], "Score": [0.6]})
        df_v4.to_csv(os.path.join(self.test_dir, "monkeyN_arr_8.csv"), index=False)

        # IT: arrays 12-15
        df_it = pd.DataFrame({"Layer": ["layer1"], "Score": [0.7]})
        df_it.to_csv(os.path.join(self.test_dir, "monkeyN_arr_12.csv"), index=False)

        results = collect_results(self.test_dir, monkey="monkeyN")

        assert "V1" in results
        assert "V4" in results
        assert "IT" in results


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def setup_method(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_aggregate_single_model_single_region(self):
        """Test aggregating results for single model and region."""
        # Create model directory
        model_dir = os.path.join(self.test_dir, "model1")
        os.makedirs(model_dir)

        # Create test data - V1 region (arrays 0-7 for monkeyF)
        df = pd.DataFrame(
            {"Layer": ["layer1", "layer2", "layer1", "layer2"], "Score": [0.5, 0.6, 0.52, 0.58]}
        )
        df.to_csv(os.path.join(model_dir, "monkeyF_arr_0.csv"), index=False)

        aggregate_results(self.test_dir, monkey="monkeyF")

        # Check that aggregate file was created
        agg_file = os.path.join(self.test_dir, "agg_results_V1.csv")
        assert os.path.exists(agg_file)

        # Check contents
        agg_df = pd.read_csv(agg_file)
        assert len(agg_df) == 1
        assert "model" in agg_df.columns
        assert "best_layer" in agg_df.columns
        assert "best_layer_score" in agg_df.columns
        assert "best_layer_std" in agg_df.columns

    def test_aggregate_multiple_models(self):
        """Test aggregating results for multiple models."""
        # Create model directories
        model1_dir = os.path.join(self.test_dir, "model1")
        model2_dir = os.path.join(self.test_dir, "model2")
        os.makedirs(model1_dir)
        os.makedirs(model2_dir)

        # Create test data for both models
        df1 = pd.DataFrame({"Layer": ["layer1", "layer2"], "Score": [0.5, 0.6]})
        df1.to_csv(os.path.join(model1_dir, "monkeyF_arr_0.csv"), index=False)

        df2 = pd.DataFrame({"Layer": ["layer1", "layer2"], "Score": [0.7, 0.8]})
        df2.to_csv(os.path.join(model2_dir, "monkeyF_arr_0.csv"), index=False)

        aggregate_results(self.test_dir, monkey="monkeyF")

        # Check aggregate file
        agg_file = os.path.join(self.test_dir, "agg_results_V1.csv")
        agg_df = pd.read_csv(agg_file)

        assert len(agg_df) == 2
        assert set(agg_df["model"]) == {"model1", "model2"}

    def test_aggregate_best_layer_selection(self):
        """Test that best layer is correctly selected."""
        model_dir = os.path.join(self.test_dir, "model1")
        os.makedirs(model_dir)

        # Create data where layer2 is better than layer1
        df = pd.DataFrame(
            {
                "Layer": ["layer1", "layer1", "layer2", "layer2"],
                "Score": [0.3, 0.35, 0.8, 0.85],  # layer2 has higher mean
            }
        )
        df.to_csv(os.path.join(model_dir, "monkeyF_arr_0.csv"), index=False)

        aggregate_results(self.test_dir, monkey="monkeyF")

        agg_df = pd.read_csv(os.path.join(self.test_dir, "agg_results_V1.csv"))
        assert agg_df.iloc[0]["best_layer"] == "layer2"
        assert agg_df.iloc[0]["best_layer_score"] > 0.8

    def test_aggregate_multiple_regions(self):
        """Test aggregating results across multiple regions."""
        model_dir = os.path.join(self.test_dir, "model1")
        os.makedirs(model_dir)

        # V1 data (array 0)
        df_v1 = pd.DataFrame({"Layer": ["layer1"], "Score": [0.5]})
        df_v1.to_csv(os.path.join(model_dir, "monkeyF_arr_0.csv"), index=False)

        # IT data (array 8)
        df_it = pd.DataFrame({"Layer": ["layer1"], "Score": [0.6]})
        df_it.to_csv(os.path.join(model_dir, "monkeyF_arr_8.csv"), index=False)

        # V4 data (array 13)
        df_v4 = pd.DataFrame({"Layer": ["layer1"], "Score": [0.7]})
        df_v4.to_csv(os.path.join(model_dir, "monkeyF_arr_13.csv"), index=False)

        aggregate_results(self.test_dir, monkey="monkeyF")

        # Check that all three aggregate files were created
        assert os.path.exists(os.path.join(self.test_dir, "agg_results_V1.csv"))
        assert os.path.exists(os.path.join(self.test_dir, "agg_results_IT.csv"))
        assert os.path.exists(os.path.join(self.test_dir, "agg_results_V4.csv"))

    def test_aggregate_empty_results_dir(self):
        """Test aggregating from empty results directory."""
        # Should not crash
        aggregate_results(self.test_dir, monkey="monkeyF")

        # No aggregate files should be created
        csv_files = [f for f in os.listdir(self.test_dir) if f.endswith(".csv")]
        assert len(csv_files) == 0

    def test_aggregate_model_without_region(self):
        """Test that models without certain regions are skipped for those regions."""
        model_dir = os.path.join(self.test_dir, "model1")
        os.makedirs(model_dir)

        # Only V1 data
        df = pd.DataFrame({"Layer": ["layer1"], "Score": [0.5]})
        df.to_csv(os.path.join(model_dir, "monkeyF_arr_0.csv"), index=False)

        aggregate_results(self.test_dir, monkey="monkeyF")

        # Only V1 aggregate should exist
        assert os.path.exists(os.path.join(self.test_dir, "agg_results_V1.csv"))
        assert not os.path.exists(os.path.join(self.test_dir, "agg_results_IT.csv"))
        assert not os.path.exists(os.path.join(self.test_dir, "agg_results_V4.csv"))

    def test_aggregate_std_calculation(self):
        """Test that standard deviation is correctly calculated."""
        model_dir = os.path.join(self.test_dir, "model1")
        os.makedirs(model_dir)

        # Data with known std
        df = pd.DataFrame(
            {
                "Layer": ["layer1", "layer1", "layer1"],
                "Score": [0.4, 0.5, 0.6],  # std should be ~0.1
            }
        )
        df.to_csv(os.path.join(model_dir, "monkeyF_arr_0.csv"), index=False)

        aggregate_results(self.test_dir, monkey="monkeyF")

        agg_df = pd.read_csv(os.path.join(self.test_dir, "agg_results_V1.csv"))
        assert abs(agg_df.iloc[0]["best_layer_std"] - 0.1) < 0.01
