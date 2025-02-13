try:
    import streamlit as st
except ImportError:
    print("(!) The code inside `autogoal_streamlit` requires `streamlit`.")
    print("(!) Fix it by running `pip install autogoal[streamlit]`.")
    raise


from autogoal.search import Logger
from autogoal_streamlit.cli import typer_app
import pandas as pd
import altair as alt


class StreamlitLogger(Logger):
    def __init__(self):
        self.evaluations = 0
        self.current = 0
        self.status = st.info("Waiting for evaluation start.")
        self.progress = st.progress(0)
        self.error_log = st.empty()
        self.best_fn = 0
        self.current_pipeline = st.code("")
        self.best_pipeline = None

        self.data = pd.DataFrame(
            {"iteration": [0, 0], "fitness": [0, 0], "category": ["current", "best"]}
        )

        chart = (
            alt.Chart(self.data)
            .mark_line()
            .encode(x="iteration", y="fitness", color="category")
        )

        # Display the updated chart in the Streamlit app
        self.chart = st.altair_chart(chart, use_container_width=True)

    def begin(self, evaluations, pop_size):
        self.status.info(f"Starting evaluation for {evaluations} iterations.")
        self.progress.progress(0)
        self.evaluations = evaluations

    def update_best(self, new_best, new_fn, previous_best, previous_fn):
        self.best_fn = new_fn
        self.best_pipeline = repr(new_best)

    def sample_solution(self, solution):
        self.current += 1
        self.status.info(
            f"""
            [Best={self.best_fn:0.3}] 🕐 Iteration {self.current}/{self.evaluations}.
            """
        )
        self.progress.progress(self.current / self.evaluations)
        self.current_pipeline.code(repr(solution))

    def eval_solution(self, solution, fitness):
        iteration = len(self.data) - 1

        new_data = pd.DataFrame(
            {
                "iteration": [iteration, iteration],
                "fitness": [float(max(fitness, 0)), float(max(self.best_fn, 0))],
                "category": ["current", "best"],
            }
        )

        self.data = pd.concat([self.data, new_data], ignore_index=True)

        chart = (
            alt.Chart(self.data)
            .mark_line()
            .encode(x="iteration", y="fitness", color="category")
        )

        self.chart.altair_chart(chart, use_container_width=True)

    def end(self, best, best_fn):
        self.status.success(
            f"""
            **Evaluation completed:** 👍 Best solution={best_fn:0.3}
            """
        )
        self.progress.progress(1.0)
        self.current_pipeline.code(self.best_pipeline)
