import argparse
import json
import os
import sqlite3
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
MAIN_BRANCH = "develop"


def calculate_threshold_value(
    avg, threshold_range={"min": 10, "min_time": 100, "max": 100, "max_time": 1000}
):
    clamp = lambda x, min_val, max_val: max(min_val, min(x, max_val))
    lerp = lambda x, y, t: x + (y - x) * t
    return lerp(
        threshold_range["min"],
        threshold_range["max"],
        1
        - (
            (
                clamp(
                    avg,
                    threshold_range["min_time"],
                    threshold_range["max_time"],
                )
                - threshold_range["min_time"]
            )
            / (threshold_range["max_time"] - threshold_range["min_time"])
        ),
    )


class BenchmarkHistoryTracker:
    def __init__(self, db_path="benchmark_history.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                commit_hash TEXT,
                branch TEXT,
                platform TEXT,
                architecture TEXT
            )
        """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                benchmark_name TEXT,
                real_time REAL,
                cpu_time REAL,
                iterations INTEGER,
                time_unit TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(id)
            )
        """
        )

        self.conn.commit()

    def get_all_branches(self, platform=None, architecture=None):
        """Retrieve all unique branch names from the database"""
        cursor = self.conn.cursor()
        query = "SELECT DISTINCT branch FROM benchmark_runs WHERE 1=1"
        params = []
        if platform:
            query += " AND platform = ?"
            params.append(platform)
        if architecture:
            query += " AND architecture = ?"
            params.append(architecture)
        cursor.execute(query, params)
        return [row[0] for row in cursor.fetchall()]

    def get_all_branch_combinations(self):
        """Retrieve all unique (branch, platform, architecture) combinations from the database"""
        cursor = self.conn.cursor()
        query = "SELECT DISTINCT branch, platform, architecture FROM benchmark_runs"
        cursor.execute(query)
        return [row for row in cursor.fetchall()]

    def get_database_entries(self, branch=None, platform=None, architecture=None):
        """Retrieve all commit_hash and branch entries from the database"""
        cursor = self.conn.cursor()
        query = "SELECT commit_hash, branch FROM benchmark_runs WHERE 1=1"
        params = []
        if branch:
            query += " AND branch = ?"
            params.append(branch)
        if platform:
            query += " AND platform = ?"
            params.append(platform)
        if architecture:
            query += " AND architecture = ?"
            params.append(architecture)
        cursor.execute(query, params)
        return cursor.fetchall()

    def remove_benchmark_run(
        self, commit_hash=None, branch="main", platform=None, architecture=None
    ):
        """Remove a benchmark run from the history and the benchmark_results that reference this run"""
        cursor = self.conn.cursor()
        run_ids = []
        query = "SELECT DISTINCT id FROM benchmark_runs WHERE branch = ?"
        params = [branch]
        if platform:
            query += " AND platform = ?"
            params.append(platform)
        if architecture:
            query += " AND architecture = ?"
            params.append(architecture)
        if commit_hash:
            query += " AND commit_hash = ?"
            params.append(commit_hash)
        cursor.execute(query, params)
        run_ids = [row[0] for row in cursor.fetchall()]

        if run_ids:
            cursor.executemany(
                "DELETE FROM benchmark_results WHERE run_id = ?",
                [(rid,) for rid in run_ids],
            )
            cursor.executemany(
                "DELETE FROM benchmark_runs WHERE id = ?",
                [(rid,) for rid in run_ids],
            )
            self.conn.commit()
            cursor.execute("VACUUM")
        self.conn.commit()

    def add_benchmark_run(
        self,
        json_file,
        commit_hash=None,
        branch="main",
        platform=None,
        architecture=None,
    ):
        """Add a new benchmark run to the history"""
        with open(json_file) as f:
            data = json.load(f)

        timestamp = datetime.fromisoformat(
            data.get("context", {}).get("date", None)
        ).isoformat()

        # Create run entry with explicit timestamp
        cursor = self.conn.cursor()
        # make sure there is no existing entry with the same commit_hash, branch, platform, architecture
        cursor.execute(
            """
            SELECT id FROM benchmark_runs 
            WHERE commit_hash = ? AND branch = ? AND platform = ? AND architecture = ?
            """,
            (commit_hash, branch, platform, architecture),
        )
        existing = cursor.fetchone()
        if existing:
            print(
                f"Run with commit {commit_hash}, branch {branch}, platform {platform}, architecture {architecture} already exists as id {existing[0]}, skipping."
            )
            return existing[0]
        cursor.execute(
            """
            INSERT INTO benchmark_runs (timestamp, commit_hash, branch, platform, architecture)
            VALUES (?, ?, ?, ?, ?)
            """,
            (timestamp, commit_hash, branch, platform, architecture),
        )

        run_id = cursor.lastrowid

        # Add individual benchmark results
        for benchmark in data["benchmarks"]:
            cursor.execute(
                """
                INSERT INTO benchmark_results 
                (run_id, benchmark_name, real_time, cpu_time, iterations, time_unit)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    benchmark["name"],
                    float(benchmark.get("real_time", 0)),
                    float(benchmark.get("cpu_time", 0)),
                    benchmark.get("iterations", 0),
                    benchmark.get("time_unit", "μs"),
                ),
            )

        self.conn.commit()
        print(
            f"Added run {run_id} with {len(data['benchmarks'])} benchmarks at {timestamp}"
        )
        return run_id

    def get_benchmark_history(
        self,
        benchmark_name,
        platform=None,
        architecture=None,
        branch=None,
        max_entries=None,
        timestamp_max=None,
    ):
        """Get historical data for a specific benchmark. If max_entries is set, limit to that many most recent entries and if timestamp_max is set, limit to the values before that timestamp."""
        query = """
            SELECT 
                br.timestamp, 
                br.commit_hash, 
                res.real_time, 
                res.cpu_time,
                br.id as run_id
            FROM benchmark_results res
            JOIN benchmark_runs br ON res.run_id = br.id
            WHERE res.benchmark_name = ?
        """
        params = [benchmark_name]

        if platform:
            query += " AND br.platform = ?"
            params.append(platform)
        if architecture:
            query += " AND br.architecture = ?"
            params.append(architecture)
        if branch:
            query += " AND br.branch = ?"
            params.append(branch)
        if timestamp_max:
            query += " AND br.timestamp <= ?"
            params.append(timestamp_max)

        # Order by timestamp descending to get most recent entries first
        query += " ORDER BY br.timestamp DESC"

        if max_entries is not None and max_entries > 0:
            query += f" LIMIT {int(max_entries)}"

        df = pd.read_sql_query(query, self.conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["real_time"] = pd.to_numeric(df["real_time"], errors="coerce")
            df["cpu_time"] = pd.to_numeric(df["cpu_time"], errors="coerce")
            # Sort back in ascending order for time series display
            df = df.sort_values("timestamp")

        return df

    def get_all_benchmark_names(self, branch=None, platform=None, architecture=None):
        """Retrieve all unique benchmark names from the database"""
        cursor = self.conn.cursor()
        query = "SELECT DISTINCT benchmark_name FROM benchmark_results WHERE 1=1"
        params = []
        if platform or architecture or branch:
            query += " AND run_id IN (SELECT id FROM benchmark_runs WHERE 1=1"
            if platform:
                query += " AND platform = ?"
                params.append(platform)
            if architecture:
                query += " AND architecture = ?"
                params.append(architecture)
            if branch:
                query += " AND branch = ?"
                params.append(branch)
            query += ")"
        query += " ORDER BY benchmark_name"
        cursor.execute(query, params)
        return [row[0] for row in cursor.fetchall()]

    def get_benchmark_names_by_pattern(
        self,
        pattern=None,
        exclude_pattern=None,
        branch=None,
        platform=None,
        architecture=None,
    ):
        """Get benchmark names matching a pattern"""
        import re

        all_names = self.get_all_benchmark_names(
            branch=branch, platform=platform, architecture=architecture
        )

        if pattern:
            regex = re.compile(pattern)
            all_names = [name for name in all_names if regex.search(name)]

        if exclude_pattern:
            exclude_regex = re.compile(exclude_pattern)
            all_names = [name for name in all_names if not exclude_regex.search(name)]

        return all_names

    def extract_benchmark_category(self, benchmark_name):
        """Extract category from benchmark name (up to first / and removing template args)"""
        # First, remove template arguments (anything between < and >)
        name_without_templates = re.sub(r"<[^>]*>", "", benchmark_name)

        # Then extract up to the first forward slash
        if "/" in name_without_templates:
            category = name_without_templates.split("/")[0]
        else:
            # If no slash, use the whole name as category
            category = name_without_templates

        # Clean up any trailing/leading whitespace
        return category.strip()

    def group_benchmarks_by_category(
        self, branch=None, platform=None, architecture=None
    ):
        """Group all benchmarks by their category"""
        benchmark_names = self.get_all_benchmark_names(
            branch=branch, platform=platform, architecture=architecture
        )
        categories = defaultdict(list)

        for bench_name in benchmark_names:
            category = self.extract_benchmark_category(bench_name)
            categories[category].append(bench_name)

        # Sort categories and benchmarks within each category
        sorted_categories = {}
        for cat in sorted(categories.keys()):
            sorted_categories[cat] = sorted(categories[cat])

        return sorted_categories

    def plot_benchmark_evolution(
        self,
        benchmark_names=None,
        metric="cpu_time",
        threshold_range={"min": 10, "min_time": 100, "max": 100, "max_time": 1000},
        branch=None,
        platform=None,
        architecture=None,
        with_regression_detection=True,
    ):
        """Create evolution plot for specified benchmarks"""
        if benchmark_names is None:
            benchmark_names = self.get_all_benchmark_names(
                branch=branch, platform=platform, architecture=architecture
            )

        if isinstance(benchmark_names, str):
            benchmark_names = [benchmark_names]

        fig = go.Figure()

        # Color palette for multiple lines
        colors = [
            "#60a5fa",
            "#fb923c",
            "#4ade80",
            "#a78bfa",
            "#f472b6",
            "#fbbf24",
            "#34d399",
            "#c084fc",
        ]

        for i, bench_name in enumerate(benchmark_names):
            df = self.get_benchmark_history(
                bench_name, platform=platform, architecture=architecture, branch=branch
            )

            if not df.empty and metric in df.columns:
                valid_data = df[df[metric].notna()]

                if not valid_data.empty:
                    color = colors[i % len(colors)]
                    fig.add_trace(
                        go.Scatter(
                            x=valid_data["timestamp"].tolist(),
                            y=valid_data[metric].tolist(),
                            text=valid_data["commit_hash"].tolist(),
                            mode="lines+markers",
                            name=bench_name,
                            line=dict(color=color, width=4),
                            marker=dict(
                                size=6, color=color, line=dict(color="white", width=1)
                            ),
                            hovertemplate=(
                                f"<b>{bench_name}</b><br>"
                                + "Date: %{x|%Y-%m-%d %H:%M}<br>"
                                + f"{metric}: %{{y:.2f}} μs<br>"
                                + "Commit: %{text}<br>"
                                + "<extra></extra>"
                            ),
                        )
                    )

                    if branch is not None and branch != MAIN_BRANCH:
                        df_historic = self.get_benchmark_history(
                            bench_name,
                            platform=platform,
                            architecture=architecture,
                            branch=MAIN_BRANCH,
                            max_entries=10,
                            timestamp_max=df["timestamp"].min().isoformat(),
                        )
                        if df_historic.empty or metric not in df_historic.columns:
                            continue
                        historic_data = df_historic[df_historic[metric].notna()]
                        color = colors[i + 1 % len(colors)]

                        fig.add_trace(
                            go.Scatter(
                                x=historic_data["timestamp"].tolist(),
                                y=historic_data[metric].tolist(),
                                text=historic_data["commit_hash"].tolist(),
                                mode="lines+markers",
                                name=bench_name + " (historical)",
                                line=dict(color=color, width=2),
                                marker=dict(
                                    size=3,
                                    color=color,
                                    line=dict(color="white", width=1),
                                ),
                                hovertemplate=(
                                    f"<b>{bench_name} (historical)</b><br>"
                                    + "Date: %{x|%Y-%m-%d %H:%M}<br>"
                                    + f"{metric}: %{{y:.2f}} μs<br>"
                                    + "Commit: %{text}<br>"
                                    + "<extra></extra>"
                                ),
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=[
                                    historic_data["timestamp"].tail(1).tolist()[0],
                                    valid_data["timestamp"].head(1).tolist()[0],
                                ],
                                y=[
                                    historic_data[metric].tail(1).tolist()[0],
                                    valid_data[metric].head(1).tolist()[0],
                                ],
                                mode="lines",
                                line=dict(color=color, width=2, dash="dash"),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )
                        if with_regression_detection:
                            valid_data = pd.concat([historic_data, valid_data])
                if with_regression_detection:
                    self.plot_regression_detection_df(
                        bench_name,
                        threshold_range,
                        metric,
                        fig,
                        dataframe=valid_data,
                    )

        fig.update_layout(
            title=None,  # Remove title as we have it in the wrapper
            xaxis_title=None,
            yaxis_title=f"{metric} (μs)" if "time" in metric else metric,
            hovermode="x unified",
            height=500,
            plot_bgcolor="#0f1723",
            paper_bgcolor="#0f1723",
            font=dict(color="#94a3b8", size=11),
            xaxis=dict(
                type="date",
                tickformat="%Y-%m-%d\n%H:%M",
                gridcolor="#1e293b",
                linecolor="#2d3748",
                zerolinecolor="#2d3748",
                tickfont=dict(color="#64748b"),
                title_font=dict(color="#94a3b8", size=12),
            ),
            yaxis=dict(
                gridcolor="#1e293b",
                linecolor="#2d3748",
                zerolinecolor="#2d3748",
                tickfont=dict(color="#64748b"),
                title_font=dict(color="#94a3b8", size=12),
            ),
            hoverlabel=dict(
                bgcolor="#1a2332",
                font_size=12,
                font_family="monospace",
                font_color="#e2e8f0",
                bordercolor="#3b82f6",
            ),
            margin=dict(t=20, b=40, l=50, r=20),
            legend=dict(
                bgcolor="rgba(26, 35, 50, 0.8)",
                bordercolor="#2d3748",
                borderwidth=1,
                font=dict(color="#94a3b8", size=10),
            ),
        )

        return fig

    def plot_regression_detection_df(
        self,
        benchmark_name,
        threshold_range={"min": 10, "min_time": 100, "max": 100, "max_time": 1000},
        metric="cpu_time",
        figure=None,
        dataframe=None,
    ):
        """Plot with regression detection highlighting from a provided dataframe"""
        if dataframe is None:
            return go.Figure().add_annotation(text="No data available", showarrow=False)

        df = dataframe[dataframe[metric].notna()].copy()

        if len(df) < 2:
            return go.Figure().add_annotation(
                text="Insufficient data for regression analysis", showarrow=False
            )

        df["rolling_avg"] = (
            df[metric].rolling(window=min(5, len(df)), min_periods=1).mean()
        )
        df["pct_change"] = df[metric].pct_change() * 100

        fig = figure or go.Figure()

        if not figure:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"].tolist(),
                    y=df[metric].tolist(),
                    mode="lines+markers",
                    name="Real Time",
                    line=dict(color="#4ade80", width=4),
                    hovertemplate="Date: %{x|%Y-%m-%d %H:%M}<br>Time: %{y:.2f} μs<extra></extra>",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"].tolist(),
                y=df["rolling_avg"].tolist(),
                mode="lines",
                name="Rolling Average",
                line=dict(color="gray", dash="dash", width=2),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d %H:%M}<br>Avg: %{y:.2f} μs<extra></extra>"
                    if not figure
                    else "Avg: %{y:.2f} μs<extra></extra>"
                ),
            )
        )

        avg = df[metric].mean()
        threshold = calculate_threshold_value(avg, threshold_range)

        regressions = df[df["pct_change"] > threshold]
        if not regressions.empty:
            fig.add_trace(
                go.Scatter(
                    x=regressions["timestamp"].tolist(),
                    y=regressions[metric].tolist(),
                    mode="markers",
                    name=f"Regression Warning",
                    marker=dict(color="red", size=14, symbol="circle-dot"),
                    hovertemplate=(
                        f">{threshold}%<br>Date: %{{x|%Y-%m-%d %H:%M}}<br>Time: %{{y:.2f}} μs<extra></extra>"
                        if not figure
                        else f">{threshold}%"
                    ),
                )
            )

        if not figure:
            fig.update_layout(
                title=f"Performance: {benchmark_name}",
                xaxis_title="Date",
                yaxis_title="Time (μs)",
                hovermode="x unified",
                height=500,
            )

        return fig

    def plot_regression_detection(
        self,
        benchmark_name,
        threshold_range={"min": 10, "min_time": 100, "max": 100, "max_time": 1000},
        metric="cpu_time",
        figure=None,
        platform=None,
        architecture=None,
        branch=None,
    ):
        """Plot with regression detection highlighting"""
        df = self.get_benchmark_history(
            benchmark_name, platform=platform, architecture=architecture, branch=branch
        )

        if df.empty:
            return go.Figure().add_annotation(text="No data available", showarrow=False)

        df = df[df[metric].notna()].copy()

        if len(df) < 2:
            return go.Figure().add_annotation(
                text="Insufficient data for regression analysis", showarrow=False
            )

        df["rolling_avg"] = (
            df[metric].rolling(window=min(5, len(df)), min_periods=1).mean()
        )
        df["pct_change"] = df[metric].pct_change() * 100

        fig = figure or go.Figure()

        if not figure:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"].tolist(),
                    y=df[metric].tolist(),
                    mode="lines+markers",
                    name="Real Time",
                    line=dict(color="#4ade80", width=4),
                    hovertemplate="Date: %{x|%Y-%m-%d %H:%M}<br>Time: %{y:.2f} μs<extra></extra>",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"].tolist(),
                y=df["rolling_avg"].tolist(),
                mode="lines",
                name="Rolling Average",
                line=dict(color="gray", dash="dash", width=2),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d %H:%M}<br>Avg: %{y:.2f} μs<extra></extra>"
                    if not figure
                    else "Avg: %{y:.2f} μs<extra></extra>"
                ),
            )
        )

        avg = df[metric].mean()
        threshold = calculate_threshold_value(avg, threshold_range)

        regressions = df[df["pct_change"] > threshold]
        if not regressions.empty:
            fig.add_trace(
                go.Scatter(
                    x=regressions["timestamp"].tolist(),
                    y=regressions[metric].tolist(),
                    mode="markers",
                    name=f"Regression Warning",
                    marker=dict(color="red", size=14, symbol="circle-dot"),
                    hovertemplate=(
                        f">{threshold}%<br>Date: %{{x|%Y-%m-%d %H:%M}}<br>Time: %{{y:.2f}} μs<extra></extra>"
                        if not figure
                        else f">{threshold}%"
                    ),
                )
            )

        if not figure:
            fig.update_layout(
                title=f"Performance: {benchmark_name}",
                xaxis_title="Date",
                yaxis_title="Time (μs)",
                hovermode="x unified",
                height=500,
                plot_bgcolor="#0f1723",
                paper_bgcolor="#0f1723",
                font=dict(color="#94a3b8", size=11),
                xaxis=dict(
                    type="date",
                    tickformat="%Y-%m-%d\n%H:%M",
                    gridcolor="#1e293b",
                    linecolor="#2d3748",
                    zerolinecolor="#2d3748",
                    tickfont=dict(color="#64748b"),
                    title_font=dict(color="#94a3b8", size=12),
                ),
                yaxis=dict(
                    gridcolor="#1e293b",
                    linecolor="#2d3748",
                    zerolinecolor="#2d3748",
                    tickfont=dict(color="#64748b"),
                    title_font=dict(color="#94a3b8", size=12),
                ),
                hoverlabel=dict(
                    bgcolor="#1a2332",
                    font_size=12,
                    font_family="monospace",
                    font_color="#e2e8f0",
                    bordercolor="#3b82f6",
                ),
                margin=dict(t=20, b=40, l=50, r=20),
                legend=dict(
                    bgcolor="rgba(26, 35, 50, 0.8)",
                    bordercolor="#2d3748",
                    borderwidth=1,
                    font=dict(color="#94a3b8", size=10),
                ),
            )

        return fig

    def plot_all_benchmarks_evolution(
        self,
        metric="cpu_time",
        separate_plots=False,
        branch=None,
        platform=None,
        architecture=None,
    ):
        """Plot evolution for all benchmarks automatically"""
        benchmark_names = self.get_all_benchmark_names(
            branch=branch, platform=platform, architecture=architecture
        )

        if not benchmark_names:
            print("No benchmarks found in database")
            return None

        if separate_plots:
            figures = {}
            for bench_name in benchmark_names:
                fig = self.plot_benchmark_evolution(
                    [bench_name],
                    metric,
                    platform=platform,
                    architecture=architecture,
                    branch=branch,
                )
                figures[bench_name] = fig
            return figures
        else:
            return self.plot_benchmark_evolution(
                benchmark_names,
                metric,
                platform=platform,
                architecture=architecture,
                branch=branch,
            )

    def detect_all_regressions(
        self,
        threshold_range={"min": 10, "min_time": 100, "max": 100, "max_time": 1000},
        metric="cpu_time",
        branch=None,
        platform=None,
        architecture=None,
    ):
        """Run regression detection on all benchmarks"""
        benchmark_names = self.get_all_benchmark_names(
            branch=branch, platform=platform, architecture=architecture
        )
        regression_results = {}

        for bench_name in benchmark_names:
            df = self.get_benchmark_history(
                bench_name, platform=platform, architecture=architecture, branch=branch
            )
            if len(df) > 1:
                df["pct_change"] = df[metric].pct_change() * 100

                avg = df[metric].mean()
                threshold = calculate_threshold_value(avg, threshold_range)
                regressions = df[df["pct_change"] > threshold]

                if not regressions.empty:
                    regression_results[bench_name] = {
                        "count": len(regressions),
                        "max_regression": regressions["pct_change"].max(),
                        "dates": regressions["timestamp"].tolist(),
                        "fig": self.plot_regression_detection(
                            bench_name,
                            threshold_range,
                            metric,
                            platform=platform,
                            architecture=architecture,
                            branch=branch,
                        ),
                    }

        return regression_results

    def create_combined_dashboard(
        self,
        output_file="dashboard.html",
        metric="cpu_time",
        threshold_range={"min": 10, "min_time": 100, "max": 100, "max_time": 1000},
        branch=None,
        platform=None,
        architecture=None,
    ):
        """Create dashboard with pagination by benchmark category"""
        return self.create_paginated_dashboard(
            output_file,
            metric=metric,
            threshold_range=threshold_range,
            branch=branch,
            platform=platform,
            architecture=architecture,
        )

    def create_paginated_dashboard(
        self,
        output_file="dashboard.html",
        metric="cpu_time",
        threshold_range={"min": 10, "min_time": 100, "max": 100, "max_time": 1000},
        branch=None,
        platform=None,
        architecture=None,
    ):
        """Create a paginated dashboard organized by benchmark categories"""

        categories = self.group_benchmarks_by_category(
            branch=branch, platform=platform, architecture=architecture
        )

        if not categories:
            print("No benchmarks found in database")
            return None

        print(
            f"Creating paginated dashboard for {sum(len(v) for v in categories.values())} benchmarks in {len(categories)} categories..."
        )

        # guarantee the dir exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Calculate summary statistics
        total_benchmarks = sum(len(benchmarks) for benchmarks in categories.values())
        category_stats = {}

        css_styles = ""
        with open(os.path.join(os.path.dirname(__file__), "plot.css"), "r") as f:
            css_styles = f.read()

        html_header = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benchmark Performance Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                {css_styles}
            </style>
        </head>"""

        js_scripts = ""
        with open(os.path.join(os.path.dirname(__file__), "plot.js"), "r") as f:
            js_scripts = f.read()

        html_js = f"""
            <script>
                const GLOBAL_TOTAL_BENCHMARKS = {total_benchmarks};
                const GLOBAL_TOTAL_CATEGORIES = {len(categories)};
                {js_scripts}
            </script>
        """

        file_ext = os.path.splitext(output_file)[1]
        category_file_base_name = (
            os.path.splitext(os.path.basename(output_file))[0] + "_"
        )
        category_file_base = os.path.join(
            os.path.dirname(output_file), category_file_base_name
        )

        def make_tabs_container_html(active_index=None):
            html = f"""
                <div class="tabs-container">
                    <div class="tabs">
                        <button class="tab overview-tab{ ' active' if active_index is None else '' }" onclick="location.href='{os.path.basename(output_file)}'">
                            Overview
                            <span class="tab-badge">{total_benchmarks}</span>
                        </button>
            """
            for i, (category, benchmarks) in enumerate(categories.items()):
                html += f"""
                        <button class="tab { ' active' if active_index == i else '' }" onclick="location.href='{category_file_base_name}{category}{file_ext}'">
                            {category}
                            <span class="tab-badge">{len(benchmarks)}</span>
                        </button>
                """

            html += """
                    </div>
                </div>
            """
            return html

        # Generate HTML with tabbed interface
        html_body_start = f"""
        {html_header}
        <body>            
            {make_tabs_container_html()}
        """

        # Create overview tab content
        overall_stats = {"regression": 0, "improvement": 0, "neutral": 0}

        # First pass: calculate all statistics
        for category, benchmarks in categories.items():
            category_stats[category] = {"regression": 0, "improvement": 0, "neutral": 0}

            for bench_name in benchmarks:
                df = self.get_benchmark_history(
                    bench_name,
                    platform=platform,
                    architecture=architecture,
                    branch=branch,
                )
                if not df.empty:
                    latest = df.iloc[-1][metric]
                    avg = df[metric].mean()
                    pct_change = ((latest - avg) / avg * 100) if avg > 0 else 0

                    threshold = calculate_threshold_value(avg, threshold_range)

                    if pct_change > threshold:
                        category_stats[category]["regression"] += 1
                        overall_stats["regression"] += 1
                    elif pct_change < -threshold:
                        category_stats[category]["improvement"] += 1
                        overall_stats["improvement"] += 1
                    else:
                        category_stats[category]["neutral"] += 1
                        overall_stats["neutral"] += 1

        # Create overview tab
        html_string = f"""
        {html_body_start}
            <div class="container">
                <div id="overview" class="tab-content active">
                    <div class="category-summary">
                        <div class="category-title">Overall Summary</div>
                        <div class="category-stats">
                            <div class="stat-card">
                                <div class="stat-value" style="color: #667eea;">{total_benchmarks}</div>
                                <div class="stat-label">Total Benchmarks</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" style="color: #667eea;">{len(categories)}</div>
                                <div class="stat-label">Categories</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" style="color: #dc3545;">{overall_stats["regression"]}</div>
                                <div class="stat-label">Regressions</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" style="color: #28a745;">{overall_stats["improvement"]}</div>
                                <div class="stat-label">Improvements</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" style="color: #6c757d;">{overall_stats["neutral"]}</div>
                                <div class="stat-label">Stable</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="category-summary">
                        <div class="category-title">Categories Overview</div>
                        <table style="width: 100%; border-collapse: collapse;">
                            <thead>
                                <tr style="border-bottom: 2px solid #e0e0e0;">
                                    <th style="text-align: left; padding: 0.5rem;">Category</th>
                                    <th style="text-align: center; padding: 0.5rem;">Total</th>
                                    <th style="text-align: center; padding: 0.5rem;">Regressions</th>
                                    <th style="text-align: center; padding: 0.5rem;">Improvements</th>
                                    <th style="text-align: center; padding: 0.5rem;">Stable</th>
                                </tr>
                            </thead>
                            <tbody>
        """

        # Add category rows to overview
        for i, (category, benchmarks) in enumerate(categories.items()):
            stats = category_stats[category]
            html_string += f"""
                                <tr style="border-bottom: 1px solid #f0f0f0; cursor: pointer;" onclick="location.href='{category_file_base_name}{category}{file_ext}'">
                                    <td style="padding: 0.5rem; font-weight: 500;">{category}</td>
                                    <td style="text-align: center; padding: 0.5rem;">{len(benchmarks)}</td>
                                    <td style="text-align: center; padding: 0.5rem; color: #dc3545;">{stats['regression']}</td>
                                    <td style="text-align: center; padding: 0.5rem; color: #28a745;">{stats['improvement']}</td>
                                    <td style="text-align: center; padding: 0.5rem; color: #6c757d;">{stats['neutral']}</td>
                                </tr>
            """

        html_string += """
                            </tbody>
                        </table>
                    </div>
                </div>
        """

        # Create content for each category tab
        plot_id = 0
        for cat_idx, (category, benchmarks) in enumerate(categories.items()):
            safe_id = f"cat_{cat_idx}"
            stats = category_stats[category]

            cat_html_string = f"""
            {html_header}
            <body>            
                {make_tabs_container_html(cat_idx)}
            <div class="container">
                <div id="{safe_id}" class="tab-content active">
                    <div class="controls">
                        <input type="text" 
                               class="search-box" 
                               id="searchBox_{safe_id}" 
                               placeholder="🔍 Search benchmarks in {category}..."
                               onkeyup="filterBenchmarks('{safe_id}')">
                        <div class="filter-buttons">
                            <button class="filter-btn active" onclick="filterByStatus('{safe_id}', 'all', this)" style="color: #667eea;">{len(benchmarks)} Benchmarks</button>
                            <button class="filter-btn" onclick="filterByStatus('{safe_id}', 'regression', this)" style="color: #dc3545;">{stats['regression']} Regressions</button>
                            <button class="filter-btn" onclick="filterByStatus('{safe_id}', 'improvement', this)" style="color: #28a745;">{stats['improvement']} Improvements</button>
                            <button class="filter-btn" onclick="filterByStatus('{safe_id}', 'neutral', this)" style="color: #6c757d;">{stats['neutral']} Stable</button>
                        </div>
                    </div>                    
                    <div id="benchmarks_{safe_id}">
            """

            # Add benchmark sections for this category
            for bench_name in benchmarks:
                df = self.get_benchmark_history(
                    bench_name,
                    platform=platform,
                    architecture=architecture,
                    branch=branch,
                )

                if not df.empty:
                    # Calculate statistics
                    latest = df.iloc[-1][metric]
                    best = df[metric].min()
                    avg = df[metric].mean()
                    pct_change = ((latest - avg) / avg * 100) if avg > 0 else 0

                    threshold = calculate_threshold_value(avg, threshold_range)

                    # Determine status
                    if pct_change > threshold:
                        status_class = "stat-regression"
                        status_text = f"↑ {pct_change:.1f}%"
                        status_type = "regression"
                    elif pct_change < -threshold:
                        status_class = "stat-improvement"
                        status_text = f"↓ {abs(pct_change):.1f}%"
                        status_type = "improvement"
                    else:
                        status_class = "stat-neutral"
                        status_text = f"→ {abs(pct_change):.1f}%"
                        status_type = "neutral"

                    # Create plots
                    fig_evolution = self.plot_benchmark_evolution(
                        [bench_name],
                        platform=platform,
                        architecture=architecture,
                        branch=branch,
                        metric=metric,
                        threshold_range=threshold_range,
                        with_regression_detection=True,
                    )
                    # fig_regression = self.plot_regression_detection(bench_name)

                    # Optimize plot layouts
                    fig_evolution.update_layout(
                        height=300,
                        margin=dict(t=30, b=30, l=40, r=40),
                        showlegend=False,
                        title=None,
                        autosize=True,
                    )

                    safe_bench_name = bench_name.replace("<", "&lt;").replace(
                        ">", "&gt;"
                    )

                    # Add section
                    cat_html_string += f"""
                        <div class="benchmark-section" data-name="{bench_name.lower()}" data-status="{status_type}" data-category="{safe_id}">
                            <div class="benchmark-header">
                                <div class="benchmark-title">{safe_bench_name}</div>
                                <div class="benchmark-stats">
                                    <span class="stat-badge {status_class}">{status_text}</span>
                                    <span class="stat-badge stat-neutral">Latest: {latest:.2f} μs</span>
                                    <span class="stat-badge stat-neutral">Best: {best:.2f} μs</span>
                                    <span class="stat-badge stat-neutral">Avg: {avg:.2f} μs</span>
                                </div>
                            </div>
                            
                            <div class="plots-container">
                                <div class="plot-wrapper">
                                    <div class="plot-responsive">
                                        {fig_evolution.to_html(include_plotlyjs=False, div_id=f"plot_evo_{plot_id}", full_html=False, config=dict(responsive=True, displayModeBar=False))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    """
                    plot_id += 1

            cat_html_string += f"""
                    </div>
                </div>
            </div>
                {html_js}
            </body>
        </html>
        """
            with open(
                f"{category_file_base}{category}{file_ext}", "w", encoding="utf-8"
            ) as f:
                f.write(cat_html_string)

        # Add JavaScript
        html_string += f"""
            </div>
            {html_js}
        </body>
        </html>
        """

        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_string)

        print(f"Paginated dashboard created successfully: {output_file}")
        print(f"  Categories: {len(categories)}")
        for category, benchmarks in categories.items():
            stats = category_stats[category]
            print(
                f"  - {category}: {len(benchmarks)} benchmarks (R:{stats['regression']}, I:{stats['improvement']}, S:{stats['neutral']})"
            )

        return output_file

    def create_scrollable_dashboard(
        self,
        output_file="dashboard.html",
        metric="cpu_time",
        threshold_range={"min": 10, "min_time": 100, "max": 100, "max_time": 1000},
        branch=None,
        platform=None,
        architecture=None,
    ):
        """Fallback to original scrollable dashboard if pagination is not desired"""
        return self.create_paginated_dashboard(
            output_file,
            metric,
            threshold_range=threshold_range,
            branch=branch,
            platform=platform,
            architecture=architecture,
        )


# Example usage
if __name__ == "__main__":
    DATABASE_PATH = "database.db"
    OUTPUT_DIR = ""
    DATA_DIR = ""

    if os.path.exists(os.path.join(PROJECT_ROOT_DIR, "settings.json")):
        with open(os.path.join(PROJECT_ROOT_DIR, "settings.json"), "r") as f:
            settings = json.load(f)
            DATABASE_PATH = settings.get("database", "")
            OUTPUT_DIR = settings.get("output", "")
            DATA_DIR = settings.get("data", "")
            MAIN_BRANCH = settings.get("main-branch", "develop")

    parser = argparse.ArgumentParser(description="Benchmark History Tracker")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--add",
        nargs=5,
        metavar=("FILE", "COMMIT_HASH", "BRANCH", "PLATFORM", "ARCHITECTURE"),
        help="Add a new benchmark run: FILE COMMIT_HASH BRANCH PLATFORM ARCHITECTURE",
    )
    group.add_argument(
        "--add-dir",
        nargs=1,
        metavar=("DIRECTORY"),
        help="Add a new benchmark run from all JSON files in DIRECTORY, the directory name is used as a branch name, and the filenames are the short git sha1",
    )
    group.add_argument(
        "--remove",
        nargs=4,
        metavar=("COMMIT_HASH", "BRANCH", "PLATFORM", "ARCHITECTURE"),
        help="Remove a benchmark run: COMMIT_HASH BRANCH PLATFORM ARCHITECTURE",
    )
    group.add_argument(
        "--remove-branch",
        nargs=3,
        metavar=("BRANCH", "PLATFORM", "ARCHITECTURE"),
        help="Remove a benchmark run: BRANCH PLATFORM ARCHITECTURE",
    )
    group.add_argument(
        "--list", nargs="?", const=None, help="List all database entries"
    )
    group.add_argument(
        "--plot",
        metavar=("OUTPUT_FILE", "BRANCH", "PLATFORM", "ARCHITECTURE"),
        nargs="?",
        const=(os.path.join(OUTPUT_DIR, "dashboard.html"), None, None, None),
        help="Plot benchmark results and save to OUTPUT_FILE (default: dashboard.html)",
    )
    group.add_argument(
        "--plot-ci",
        action="store_true",
        help="When set will use the CI flow to plot and output",
    )
    args = parser.parse_args()

    tracker = BenchmarkHistoryTracker(db_path=DATABASE_PATH)

    if args.add_dir:
        # get all files recursively in the directory
        all_files = []
        for root, dirs, files in os.walk(args.add_dir[0]):
            for file in files:
                all_files.append(os.path.join(root, file))
        for filename in all_files:
            if filename.endswith(".json"):
                # first part of the path is the platform, then architecture, then branch is the remainder.
                # the filename is the commit hash
                file_rel_path = os.path.relpath(filename, args.add_dir[0])
                platform = file_rel_path.split(os.sep)[0]
                architecture = file_rel_path.split(os.sep)[1]
                branch = "/".join(file_rel_path.split(os.sep)[2:-1])
                commit_hash = os.path.splitext(os.path.basename(filename))[0]
                print(
                    f"Adding benchmark run from file: {filename} (commit: '{commit_hash}', branch: '{branch}', platform: '{platform}', architecture: '{architecture}')"
                )
                tracker.add_benchmark_run(
                    json_file=filename,
                    commit_hash=commit_hash,
                    branch=branch,
                    platform=platform,
                    architecture=architecture,
                )

    if args.add:
        tracker.add_benchmark_run(
            json_file=args.add[0],
            commit_hash=args.add[1],
            branch=args.add[2],
            platform=args.add[3],
            architecture=args.add[4],
        )

    if args.remove:
        tracker.remove_benchmark_run(*args.remove)

    if args.remove_branch:
        tracker.remove_benchmark_run(branch=args.remove_branch)

    if args.list:
        print("Listing database entries:")
        print(tracker.get_database_entries(branch=args.list))

    def plot_branch(output_file, branch, platform, architecture):
        tracker.create_combined_dashboard(
            output_file=output_file,
            branch=branch,
            metric="cpu_time",
            platform=platform,
            architecture=architecture,
        )

    if args.plot_ci:
        all_branches = tracker.get_all_branch_combinations()
        for [branch, platform, architecture] in all_branches:
            output_file = os.path.join(
                OUTPUT_DIR, platform, architecture, branch, "index.html"
            )
            plot_branch(
                output_file=output_file,
                branch=branch,
                platform=platform,
                architecture=architecture,
            )

    if args.plot:
        plot_branch(
            output_file=args.plot[0],
            branch=args.plot[1],
            platform=args.plot[2],
            architecture=args.plot[3],
        )
