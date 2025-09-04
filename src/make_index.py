import json
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))


def load_settings():
    settings_path = os.path.join(PROJECT_ROOT_DIR, "settings.json")
    with open(settings_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    settings = load_settings()

    output_path = os.path.join(PROJECT_ROOT_DIR, settings["output"])
    os.makedirs(output_path, exist_ok=True)
    index_path = os.path.join(output_path, "index.html")

    all_indexes = []
    # get all directories in output_path, see if the have an `index.html` and add them to all_indexes
    for root, dirs, files in os.walk(output_path):
        if "index.html" in files and root != output_path:
            rel_path = os.path.relpath(root, output_path)
            components = rel_path.split(os.sep)
            if len(components) < 3:
                print(f"Skipping invalid directory structure: {rel_path}")
                continue
            platform = components[0]
            architecture = components[1]
            branch = "/".join(components[2:])
            all_indexes.append(
                {"platform": platform, "architecture": architecture, "branch": branch}
            )
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0
">
    <title>Benchmark Index</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        h1 {
            text-align: center;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
        a {
            color: #007BFF;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>Benchmark Index</h1>
    <table>
        <tr>
            <th>Platform</th>
            <th>Architecture</th>
            <th>Branch</th>
            <th>Link</th>
        </tr>
"""
    for entry in sorted(
        all_indexes, key=lambda x: (x["platform"], x["architecture"], x["branch"])
    ):
        link = (
            f"{entry['platform']}/{entry['architecture']}/{entry['branch']}/index.html"
        )
        index_html += f"""        <tr>
            <td>{entry['platform']}</td>
            <td>{entry['architecture']}</td>
            <td>{entry['branch']}</td>
            <td><a href="{link}">View Benchmarks</a></td>
        </tr>
"""
    index_html += """    </table>
</body>
</html>
"""
    with open(index_path, "w") as f:
        f.write(index_html)
    print(f"Index written to {index_path}")
