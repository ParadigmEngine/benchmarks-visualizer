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
                branch TEXT
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

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ci_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                last_plot_id INTEGER,
                FOREIGN KEY (last_plot_id) REFERENCES benchmark_runs(id)
            )
            """
        )
        self.conn.commit()

    def set_last_plot(self):
        """Set the last ID that was plotted by the ci"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(id) FROM benchmark_runs")
        last_run_id = cursor.fetchone()[0]
        if last_run_id:
            cursor.execute(
                "UPDATE ci_cache SET last_plot_id = ?",
                (last_run_id,),
            )
        self.conn.commit()

    def get_last_plot(self):
        """Retrieve the last plotted ID from the ci_cache"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT last_plot_id FROM ci_cache")
        temp = cursor.fetchone()
        id = temp[0] if temp else -1
        return id

    def get_all_runs_since_last_plot(self):
        """Retrieve all benchmark runs since the last plotted ID"""
        last_plot_id = self.get_last_plot()
        cursor = self.conn.cursor()
        if last_plot_id:
            cursor.execute(
                "SELECT * FROM benchmark_runs WHERE id > ?",
                (last_plot_id,),
            )
        return cursor.fetchall()

    def get_all_branches_since_last_plot(self):
        """Retrieve all unique branch names from benchmark_runs since the last plotted ID"""
        last_plot_id = self.get_last_plot()
        cursor = self.conn.cursor()
        if last_plot_id:
            cursor.execute(
                "SELECT DISTINCT branch FROM benchmark_runs WHERE id > ?",
                (last_plot_id,),
            )
        return [row[0] for row in cursor.fetchall()]

    def get_all_branches(self):
        """Retrieve all unique branch names from the database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT branch FROM benchmark_runs")
        return [row[0] for row in cursor.fetchall()]

    def get_database_entries(self, branch=None):
        """Retrieve all commit_hash and branch entries from the database"""
        cursor = self.conn.cursor()
        if branch:
            cursor.execute(
                "SELECT commit_hash, branch FROM benchmark_runs WHERE branch = ?",
                (branch,),
            )
        else:
            cursor.execute("SELECT commit_hash, branch FROM benchmark_runs")
        return cursor.fetchall()

    def remove_benchmark_run(self, commit_hash=None, branch="main"):
        """Remove a benchmark run from the history and the benchmark_results that reference this run"""
        cursor = self.conn.cursor()
        run_ids = []
        if commit_hash:
            cursor.execute(
                "SELECT DISTINCT id FROM benchmark_runs WHERE commit_hash = ? AND branch = ?",
                (commit_hash, branch),
            )
            run_ids = [row[0] for row in cursor.fetchall()]

        else:
            # select the run_id from benchmark_runs
            cursor.execute(
                "SELECT DISTINCT id FROM benchmark_runs WHERE branch = ?",
                branch,
            )
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

    def add_benchmark_run(self, json_file, commit_hash=None, branch="main"):
        """Add a new benchmark run to the history"""
        with open(json_file) as f:
            data = json.load(f)

        timestamp = datetime.fromisoformat(
            data.get("context", {}).get("date", None)
        ).isoformat()

        # Create run entry with explicit timestamp
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO benchmark_runs (timestamp, commit_hash, branch)
            VALUES (?, ?, ?)
        """,
            (timestamp, commit_hash, branch),
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

    def get_benchmark_history(self, benchmark_name):
        """Get historical data for a specific benchmark"""
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
            ORDER BY br.timestamp ASC
        """

        df = pd.read_sql_query(query, self.conn, params=[benchmark_name])

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["real_time"] = pd.to_numeric(df["real_time"], errors="coerce")
            df["cpu_time"] = pd.to_numeric(df["cpu_time"], errors="coerce")

        return df

    def get_all_benchmark_names(self, branch=None):
        """Retrieve all unique benchmark names from the database"""
        cursor = self.conn.cursor()
        if branch:
            cursor.execute(
                """
                SELECT DISTINCT res.benchmark_name 
                FROM benchmark_results res
                JOIN benchmark_runs br ON res.run_id = br.id
                WHERE br.branch = ?
                ORDER BY res.benchmark_name
                """,
                (branch,),
            )
        else:
            cursor.execute(
                "SELECT DISTINCT benchmark_name FROM benchmark_results ORDER BY benchmark_name"
            )
        return [row[0] for row in cursor.fetchall()]

    def get_benchmark_names_by_pattern(
        self, pattern=None, exclude_pattern=None, branch=None
    ):
        """Get benchmark names matching a pattern"""
        import re

        all_names = self.get_all_benchmark_names(branch=branch)

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

    def group_benchmarks_by_category(self, branch=None):
        """Group all benchmarks by their category"""
        benchmark_names = self.get_all_benchmark_names(branch=branch)
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
    ):
        """Create evolution plot for specified benchmarks"""
        if benchmark_names is None:
            benchmark_names = self.get_all_benchmark_names(branch=branch)

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
            df = self.get_benchmark_history(bench_name)

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

            self.plot_regression_detection(bench_name, threshold_range, metric, fig)

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

    def plot_regression_detection(
        self,
        benchmark_name,
        threshold_range={"min": 10, "min_time": 100, "max": 100, "max_time": 1000},
        metric="cpu_time",
        figure=None,
    ):
        """Plot with regression detection highlighting"""
        df = self.get_benchmark_history(benchmark_name)

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
        self, metric="cpu_time", separate_plots=False, branch=None
    ):
        """Plot evolution for all benchmarks automatically"""
        benchmark_names = self.get_all_benchmark_names(branch=branch)

        if not benchmark_names:
            print("No benchmarks found in database")
            return None

        if separate_plots:
            figures = {}
            for bench_name in benchmark_names:
                fig = self.plot_benchmark_evolution([bench_name], metric)
                figures[bench_name] = fig
            return figures
        else:
            return self.plot_benchmark_evolution(benchmark_names, metric)

    def detect_all_regressions(
        self,
        threshold_range={"min": 10, "min_time": 100, "max": 100, "max_time": 1000},
        metric="cpu_time",
        branch=None,
    ):
        """Run regression detection on all benchmarks"""
        benchmark_names = self.get_all_benchmark_names(branch=branch)
        regression_results = {}

        for bench_name in benchmark_names:
            df = self.get_benchmark_history(bench_name)
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
                            bench_name, threshold_range
                        ),
                    }

        return regression_results

    def create_combined_dashboard(
        self,
        output_file="dashboard.html",
        metric="cpu_time",
        threshold_range={"min": 10, "min_time": 100, "max": 100, "max_time": 1000},
        branch=None,
    ):
        """Create dashboard with pagination by benchmark category"""
        return self.create_paginated_dashboard(
            output_file, metric=metric, threshold_range=threshold_range, branch=branch
        )

    def create_paginated_dashboard(
        self,
        output_file="dashboard.html",
        metric="cpu_time",
        threshold_range={"min": 10, "min_time": 100, "max": 100, "max_time": 1000},
        branch=None,
    ):
        """Create a paginated dashboard organized by benchmark categories"""

        categories = self.group_benchmarks_by_category(branch=branch)

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

        def make_tabs_container_html(active_index=None):
            html = f"""
                <div class="tabs-container">
                    <div class="tabs">
                        <button class="tab overview-tab{ ' active' if active_index is None else '' }" onclick="location.href='{output_file}'">
                            Overview
                            <span class="tab-badge">{total_benchmarks}</span>
                        </button>
            """
            for i, (category, benchmarks) in enumerate(categories.items()):
                html += f"""
                        <button class="tab { ' active' if active_index == i else '' }" onclick="location.href='{output_file.split('.')[0]}_{category}.html'">
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
                df = self.get_benchmark_history(bench_name)
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
                                <tr style="border-bottom: 1px solid #f0f0f0; cursor: pointer;" onclick="location.href='{output_file.split('.')[0]}_{category}.html'">
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
                df = self.get_benchmark_history(bench_name)

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
                    fig_evolution = self.plot_benchmark_evolution([bench_name])
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
                f"{output_file.split('.')[0]}_{category}.html", "w", encoding="utf-8"
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
    ):
        """Fallback to original scrollable dashboard if pagination is not desired"""
        return self.create_paginated_dashboard(
            output_file, metric, threshold_range=threshold_range, branch=branch
        )


# Example usage
if __name__ == "__main__":
    import sys

    DATABASE_PATH = "database.db"
    OUTPUT_DIR = ""
    DATA_DIR = ""

    if os.path.exists(os.path.join(PROJECT_ROOT_DIR, "settings.json")):
        with open(os.path.join(PROJECT_ROOT_DIR, "settings.json"), "r") as f:
            settings = json.load(f)
            DATABASE_PATH = settings.get("database", "")
            OUTPUT_DIR = settings.get("output", "")
            DATA_DIR = settings.get("data", "")

    parser = argparse.ArgumentParser(description="Benchmark History Tracker")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--add",
        nargs=3,
        metavar=("FILE", "COMMIT_HASH", "BRANCH"),
        help="Add a new benchmark run: FILE COMMIT_HASH BRANCH",
    )
    group.add_argument(
        "--add-dir",
        nargs=1,
        metavar=("DIRECTORY"),
        help="Add a new benchmark run from all JSON files in DIRECTORY, the directory name is used as a branch name, and the filenames are the short git sha1",
    )
    group.add_argument(
        "--remove",
        nargs=2,
        metavar=("COMMIT_HASH", "BRANCH"),
        help="Remove a benchmark run: COMMIT_HASH BRANCH",
    )
    group.add_argument(
        "--remove-branch",
        nargs=1,
        metavar=("BRANCH"),
        help="Remove a benchmark run: BRANCH",
    )
    group.add_argument(
        "--list", nargs="?", const=None, help="List all database entries"
    )
    group.add_argument(
        "--plot",
        metavar=("OUTPUT_FILE", "BRANCH"),
        nargs="?",
        const=(os.path.join(OUTPUT_DIR, "dashboard.html"), None),
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
        all_files = os.listdir(args.add_dir[0])
        dirname = os.path.basename(args.add_dir[0])
        for filename in all_files:
            if filename.endswith(".json") and len(filename) == 7 + 5:
                commit_hash = filename.split(".")[0]
                tracker.add_benchmark_run(
                    os.path.join(args.add_dir[0], filename),
                    commit_hash=commit_hash,
                    branch=dirname,
                )

    if args.add:
        tracker.add_benchmark_run(*args.add)

    if args.remove:
        tracker.remove_benchmark_run(*args.remove)

    if args.remove_branch:
        tracker.remove_benchmark_run(branch=args.remove_branch)

    if args.list:
        print("Listing database entries:")
        print(tracker.get_database_entries(branch=args.list))

    def plot_branch(output_file, branch):
        tracker.plot_all_benchmarks_evolution(
            metric="cpu_time", separate_plots=True, branch=branch
        )
        tracker.detect_all_regressions(metric="cpu_time", branch=branch)
        tracker.create_combined_dashboard(
            output_file=output_file, branch=branch, metric="cpu_time"
        )

    if args.plot_ci:
        all_branches = tracker.get_all_branches_since_last_plot()
        for branch in all_branches:
            output_file = os.path.join(OUTPUT_DIR, branch, "index.html")
            plot_branch(output_file=output_file, branch=branch)

        tracker.set_last_plot()

    if args.plot:
        plot_branch(output_file=args.plot[0], branch=args.plot[1])
