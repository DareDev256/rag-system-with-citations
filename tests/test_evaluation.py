"""Tests for run_evaluation() — the evaluation pipeline orchestrator.

Mocks perform_search, synthesize_answer, calculate_citation_coverage,
pandas DataFrame I/O, and os.makedirs. No real LLM, FAISS, or file writes.
"""
from unittest.mock import patch, MagicMock, call
import pytest


# Shared fixtures -------------------------------------------------------

def _make_search_results(doc_ids=("doc_001", "doc_002")):
    return [{"doc_id": d, "snippet": f"Snippet for {d}"} for d in doc_ids]


def _make_synth_result(answer="Answer [doc_001].", citations=None, confidence=0.8):
    if citations is None:
        citations = [{"doc_id": "doc_001", "snippet": "Snippet for doc_001"}]
    return {"answer": answer, "citations_used": citations, "confidence": confidence}


# ── run_evaluation ─────────────────────────────────────────────


class TestRunEvaluation:
    """Tests for the main evaluation pipeline."""

    @patch("src.eval.evaluate.os.makedirs")
    @patch("src.eval.evaluate.pd.DataFrame")
    @patch("src.eval.evaluate.calculate_citation_coverage", return_value=0.75)
    @patch("src.eval.evaluate.synthesize_answer")
    @patch("src.eval.evaluate.perform_search")
    def test_happy_path_calls_pipeline_for_each_item(
        self, mock_search, mock_synth, mock_coverage, mock_df_cls, mock_makedirs
    ):
        from src.eval.evaluate import run_evaluation, EVAL_DATA

        mock_search.return_value = _make_search_results()
        mock_synth.return_value = _make_synth_result()
        mock_df = MagicMock()
        mock_df_cls.return_value = mock_df

        run_evaluation()

        assert mock_search.call_count == len(EVAL_DATA)
        assert mock_synth.call_count == len(EVAL_DATA)
        assert mock_coverage.call_count == len(EVAL_DATA)

    @patch("src.eval.evaluate.os.makedirs")
    @patch("src.eval.evaluate.pd.DataFrame")
    @patch("src.eval.evaluate.calculate_citation_coverage", return_value=1.0)
    @patch("src.eval.evaluate.synthesize_answer")
    @patch("src.eval.evaluate.perform_search")
    def test_perform_search_called_with_k_3(
        self, mock_search, mock_synth, mock_coverage, mock_df_cls, mock_makedirs
    ):
        from src.eval.evaluate import run_evaluation

        mock_search.return_value = _make_search_results()
        mock_synth.return_value = _make_synth_result()
        mock_df_cls.return_value = MagicMock()

        run_evaluation()

        # Every call should pass k=3
        for c in mock_search.call_args_list:
            assert c == call(c[0][0], k=3)

    @patch("src.eval.evaluate.os.makedirs")
    @patch("src.eval.evaluate.pd.DataFrame")
    @patch("src.eval.evaluate.calculate_citation_coverage", return_value=0.5)
    @patch("src.eval.evaluate.synthesize_answer")
    @patch("src.eval.evaluate.perform_search")
    def test_synthesize_receives_search_results(
        self, mock_search, mock_synth, mock_coverage, mock_df_cls, mock_makedirs
    ):
        from src.eval.evaluate import run_evaluation

        search_res = _make_search_results()
        mock_search.return_value = search_res
        mock_synth.return_value = _make_synth_result()
        mock_df_cls.return_value = MagicMock()

        run_evaluation()

        for c in mock_synth.call_args_list:
            assert c[0][1] is search_res  # second positional arg = search results

    @patch("src.eval.evaluate.os.makedirs")
    @patch("src.eval.evaluate.pd.DataFrame")
    @patch("src.eval.evaluate.calculate_citation_coverage", return_value=0.9)
    @patch("src.eval.evaluate.synthesize_answer")
    @patch("src.eval.evaluate.perform_search")
    def test_creates_reports_directory(
        self, mock_search, mock_synth, mock_coverage, mock_df_cls, mock_makedirs
    ):
        from src.eval.evaluate import run_evaluation

        mock_search.return_value = _make_search_results()
        mock_synth.return_value = _make_synth_result()
        mock_df_cls.return_value = MagicMock()

        run_evaluation()

        mock_makedirs.assert_called_once_with("reports", exist_ok=True)

    @patch("src.eval.evaluate.os.makedirs")
    @patch("src.eval.evaluate.pd.DataFrame")
    @patch("src.eval.evaluate.calculate_citation_coverage", return_value=0.6)
    @patch("src.eval.evaluate.synthesize_answer")
    @patch("src.eval.evaluate.perform_search")
    def test_saves_csv_to_reports_dir(
        self, mock_search, mock_synth, mock_coverage, mock_df_cls, mock_makedirs
    ):
        from src.eval.evaluate import run_evaluation

        mock_search.return_value = _make_search_results()
        mock_synth.return_value = _make_synth_result()
        mock_df = MagicMock()
        mock_df_cls.return_value = mock_df

        run_evaluation()

        mock_df.to_csv.assert_called_once_with("reports/eval_results.csv", index=False)

    @patch("src.eval.evaluate.os.makedirs")
    @patch("src.eval.evaluate.pd.DataFrame")
    @patch("src.eval.evaluate.calculate_citation_coverage", return_value=0.5)
    @patch("src.eval.evaluate.synthesize_answer")
    @patch("src.eval.evaluate.perform_search")
    def test_prints_markdown_table(
        self, mock_search, mock_synth, mock_coverage, mock_df_cls, mock_makedirs, capsys
    ):
        from src.eval.evaluate import run_evaluation

        mock_search.return_value = _make_search_results()
        mock_synth.return_value = _make_synth_result()
        mock_df = MagicMock()
        mock_df.to_markdown.return_value = "| query | latency_ms |"
        mock_df_cls.return_value = mock_df

        run_evaluation()

        output = capsys.readouterr().out
        assert "Evaluation Results:" in output
        assert "| query | latency_ms |" in output
        assert "reports/eval_results.csv" in output

    @patch("src.eval.evaluate.os.makedirs")
    @patch("src.eval.evaluate.pd.DataFrame")
    @patch("src.eval.evaluate.calculate_citation_coverage", return_value=1.0)
    @patch("src.eval.evaluate.synthesize_answer")
    @patch("src.eval.evaluate.perform_search")
    def test_keyword_match_true_when_all_present(
        self, mock_search, mock_synth, mock_coverage, mock_df_cls, mock_makedirs
    ):
        """First EVAL_DATA item expects 'retrieval', 'generation', 'LLM' — answer has all."""
        from src.eval.evaluate import run_evaluation

        mock_search.return_value = _make_search_results()
        mock_synth.return_value = _make_synth_result(
            answer="Retrieval augmented Generation uses an LLM."
        )
        mock_df_cls.return_value = MagicMock()

        run_evaluation()

        # Inspect the list of dicts passed to DataFrame constructor
        results = mock_df_cls.call_args[0][0]
        first_result = results[0]
        assert first_result["keyword_match"] is True

    @patch("src.eval.evaluate.os.makedirs")
    @patch("src.eval.evaluate.pd.DataFrame")
    @patch("src.eval.evaluate.calculate_citation_coverage", return_value=0.0)
    @patch("src.eval.evaluate.synthesize_answer")
    @patch("src.eval.evaluate.perform_search")
    def test_keyword_match_false_when_missing(
        self, mock_search, mock_synth, mock_coverage, mock_df_cls, mock_makedirs
    ):
        """Answer missing expected keywords → keyword_match is False."""
        from src.eval.evaluate import run_evaluation

        mock_search.return_value = _make_search_results()
        mock_synth.return_value = _make_synth_result(
            answer="I have no relevant information."
        )
        mock_df_cls.return_value = MagicMock()

        run_evaluation()

        results = mock_df_cls.call_args[0][0]
        # All items should be False since answer has none of the expected keywords
        assert all(r["keyword_match"] is False for r in results)

    @patch("src.eval.evaluate.os.makedirs")
    @patch("src.eval.evaluate.pd.DataFrame")
    @patch("src.eval.evaluate.calculate_citation_coverage", return_value=0.0)
    @patch("src.eval.evaluate.synthesize_answer")
    @patch("src.eval.evaluate.perform_search")
    def test_keyword_match_is_case_insensitive(
        self, mock_search, mock_synth, mock_coverage, mock_df_cls, mock_makedirs
    ):
        """'retrieval' matches 'RETRIEVAL' because both are lowered."""
        from src.eval.evaluate import run_evaluation

        mock_search.return_value = _make_search_results()
        mock_synth.return_value = _make_synth_result(
            answer="RETRIEVAL augmented GENERATION with an LLM model. "
                   "NEIL ARMSTRONG and BUZZ ALDRIN walked on the moon in 1969. "
                   "The EAGLE has landed."
        )
        mock_df_cls.return_value = MagicMock()

        run_evaluation()

        results = mock_df_cls.call_args[0][0]
        # All 3 eval items should match when answer has all keywords (case-insensitive)
        assert all(r["keyword_match"] is True for r in results)

    @patch("src.eval.evaluate.os.makedirs")
    @patch("src.eval.evaluate.pd.DataFrame")
    @patch("src.eval.evaluate.calculate_citation_coverage", return_value=0.0)
    @patch("src.eval.evaluate.synthesize_answer")
    @patch("src.eval.evaluate.perform_search")
    def test_result_records_contain_expected_keys(
        self, mock_search, mock_synth, mock_coverage, mock_df_cls, mock_makedirs
    ):
        from src.eval.evaluate import run_evaluation

        mock_search.return_value = _make_search_results()
        mock_synth.return_value = _make_synth_result()
        mock_df_cls.return_value = MagicMock()

        run_evaluation()

        results = mock_df_cls.call_args[0][0]
        expected_keys = {"query", "latency_ms", "citation_coverage", "keyword_match", "answer_length"}
        for r in results:
            assert set(r.keys()) == expected_keys

    @patch("src.eval.evaluate.os.makedirs")
    @patch("src.eval.evaluate.pd.DataFrame")
    @patch("src.eval.evaluate.calculate_citation_coverage", return_value=0.5)
    @patch("src.eval.evaluate.synthesize_answer")
    @patch("src.eval.evaluate.perform_search")
    def test_latency_is_positive_float(
        self, mock_search, mock_synth, mock_coverage, mock_df_cls, mock_makedirs
    ):
        from src.eval.evaluate import run_evaluation

        mock_search.return_value = _make_search_results()
        mock_synth.return_value = _make_synth_result()
        mock_df_cls.return_value = MagicMock()

        run_evaluation()

        results = mock_df_cls.call_args[0][0]
        for r in results:
            assert isinstance(r["latency_ms"], float)
            assert r["latency_ms"] >= 0

    @patch("src.eval.evaluate.os.makedirs")
    @patch("src.eval.evaluate.pd.DataFrame")
    @patch("src.eval.evaluate.calculate_citation_coverage", return_value=0.42)
    @patch("src.eval.evaluate.synthesize_answer")
    @patch("src.eval.evaluate.perform_search")
    def test_coverage_value_forwarded_from_metric(
        self, mock_search, mock_synth, mock_coverage, mock_df_cls, mock_makedirs
    ):
        """The exact value returned by calculate_citation_coverage appears in results."""
        from src.eval.evaluate import run_evaluation

        mock_search.return_value = _make_search_results()
        mock_synth.return_value = _make_synth_result()
        mock_df_cls.return_value = MagicMock()

        run_evaluation()

        results = mock_df_cls.call_args[0][0]
        assert all(r["citation_coverage"] == 0.42 for r in results)

    @patch("src.eval.evaluate.os.makedirs")
    @patch("src.eval.evaluate.pd.DataFrame")
    @patch("src.eval.evaluate.calculate_citation_coverage", return_value=0.0)
    @patch("src.eval.evaluate.synthesize_answer")
    @patch("src.eval.evaluate.perform_search")
    def test_answer_length_matches_actual_length(
        self, mock_search, mock_synth, mock_coverage, mock_df_cls, mock_makedirs
    ):
        from src.eval.evaluate import run_evaluation

        answer = "Exactly forty-two characters of test text!"
        mock_search.return_value = _make_search_results()
        mock_synth.return_value = _make_synth_result(answer=answer)
        mock_df_cls.return_value = MagicMock()

        run_evaluation()

        results = mock_df_cls.call_args[0][0]
        assert all(r["answer_length"] == len(answer) for r in results)
