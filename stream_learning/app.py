from pathlib import Path
import re
import textwrap

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


APP_FOLDER = Path(__file__).resolve().parent
RESULTS_FOLDER = APP_FOLDER / "stream_learner_results"


def get_accuracy(text):
    match = re.search(r"Accuracy:\s*([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def get_snapshots(text):
    pattern = r"^\s*Date:\s*(\d{4}-\d{2}-\d{2})(?:\s+\d{2}:\d{2}:\d{2})?\s*$"

    matches = list(
        re.finditer(
            pattern,
            text,
            re.MULTILINE
        )
    )

    snapshots = []

    for i, match in enumerate(matches):
        date = match.group(1)

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        snapshot_text = text[start:end].strip()

        snapshot_text = re.sub(
            r"^=+\s*",
            "",
            snapshot_text
        ).strip()

        snapshots.append({
            "date": date,
            "text": snapshot_text
        })

    return snapshots


def detect_tree_format(text):
    if "Split test:" in text:
        return "SOHOT"

    if (
        re.search(r"^\s*Leaf\s*$", text, re.MULTILINE)
        and "Class distribution:" in text
    ):
        return "SOHOT"

    if re.search(r"^\s*if\s+\[att", text, re.MULTILINE):
        return "HAT"

    if re.search(r"^\s*Leaf\s+\[class:", text, re.MULTILINE):
        return "HAT"

    return "UNKNOWN"


def parse_tree(text):
    tree_format = detect_tree_format(text)

    if tree_format == "HAT":
        return parse_hat_tree(text)

    if tree_format == "SOHOT":
        return parse_sohot_tree(text)

    return [], []


def parse_hat_tree(text):
    nodes = []
    edges = []
    stack = []
    counter = 0

    for raw_line in text.splitlines():
        line = raw_line.expandtabs(2)
        stripped = line.strip()

        if not stripped or stripped.startswith("Accuracy:"):
            continue

        if stripped.startswith("if "):
            node_type = "condition"
            label = stripped[3:].rstrip(":").strip()
        elif stripped.startswith("Leaf"):
            node_type = "leaf"
            label = stripped
        else:
            continue

        indent = len(line) - len(line.lstrip(" "))
        node_id = f"node_{counter}"
        counter += 1

        while stack and stack[-1]["indent"] >= indent:
            stack.pop()

        nodes.append({
            "id": node_id,
            "label": label,
            "type": node_type,
            "level": len(stack) + 1,
        })

        parent_id = stack[-1]["id"] if stack else "root"
        edges.append((parent_id, node_id, ""))

        if node_type == "condition":
            stack.append({"indent": indent, "id": node_id})

    return nodes, edges


def parse_sohot_tree(text):
    nodes = []
    edges = []
    lines = text.splitlines()
    node_stack = []
    branch_at_indent = {}
    counter = 0
    i = 0

    while i < len(lines):
        raw_line = lines[i].expandtabs(2)
        stripped = raw_line.strip()

        if not stripped or stripped.startswith("Accuracy:"):
            i += 1
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if stripped in ("LEFT:", "RIGHT:"):
            branch_at_indent[indent] = stripped.rstrip(":")
            i += 1
            continue

        if stripped.startswith("Split test:"):
            split_label = stripped.replace("Split test:", "", 1).strip()
            feature_importance = []
            j = i + 1

            if j < len(lines) and lines[j].strip() == "Feature importance:":
                fi_indent = len(lines[j]) - len(lines[j].lstrip(" "))
                j += 1

                while j < len(lines):
                    next_line = lines[j].expandtabs(2)
                    next_stripped = next_line.strip()

                    if not next_stripped:
                        j += 1
                        continue

                    next_indent = len(next_line) - len(next_line.lstrip(" "))

                    if next_indent <= fi_indent:
                        break

                    if next_stripped not in ("LEFT:", "RIGHT:"):
                        feature_importance.append(next_stripped)

                    j += 1

            useful_fi = [value for value in feature_importance if value != "n/a"]
            label = split_label

            if useful_fi:
                label += "\nFeature importance:\n" + "\n".join(useful_fi)

            node_id = f"node_{counter}"
            counter += 1

            while node_stack and node_stack[-1]["indent"] >= indent:
                node_stack.pop()

            nodes.append({
                "id": node_id,
                "label": label,
                "type": "condition",
                "level": len(node_stack) + 1,
            })

            if node_stack:
                parent_id = node_stack[-1]["id"]
                branch = branch_at_indent.get(indent - 2, "")
            else:
                parent_id = "root"
                branch = ""

            edges.append((parent_id, node_id, branch))
            node_stack.append({"id": node_id, "indent": indent})
            i += 1
            continue

        if stripped == "Leaf":
            leaf_lines = []
            j = i + 1

            while j < len(lines):
                next_line = lines[j].expandtabs(2)
                next_stripped = next_line.strip()

                if not next_stripped:
                    j += 1
                    continue

                next_indent = len(next_line) - len(next_line.lstrip(" "))

                if next_indent <= indent:
                    break

                leaf_lines.append(next_stripped)
                j += 1

            label_lines = ["Leaf"]

            for leaf_line in leaf_lines:
                if leaf_line.startswith("Class distribution:"):
                    distribution = leaf_line.replace("Class distribution:", "", 1).strip()
                    label_lines.append("Class distribution:")
                    label_lines.append(distribution)
                elif leaf_line.startswith("Predicted class:"):
                    label_lines.append(leaf_line)
                elif leaf_line.startswith("P(x->leaf):"):
                    label_lines.append(leaf_line)

            node_id = f"node_{counter}"
            counter += 1

            while node_stack and node_stack[-1]["indent"] >= indent:
                node_stack.pop()

            parent_id = node_stack[-1]["id"] if node_stack else "root"
            branch = branch_at_indent.get(indent - 2, "")

            nodes.append({
                "id": node_id,
                "label": "\n".join(label_lines),
                "type": "leaf",
                "level": len(node_stack) + 1,
            })

            edges.append((parent_id, node_id, branch))
            i = j
            continue

        i += 1

    return nodes, edges


def wrap_text(text, width=45):
    wrapped_lines = []

    for line in text.splitlines():
        if not line:
            wrapped_lines.append("")
            continue

        parts = textwrap.wrap(line, width=width, break_long_words=False)
        wrapped_lines.extend(parts if parts else [line])

    return "\n".join(wrapped_lines)


def create_tree(nodes, edges):
    net = Network(
        height="700px",
        width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )

    net.add_node(
        "root",
        label="Decision Tree",
        shape="box",
        color="#D9EAF7",
        level=0,
    )

    for node in nodes:
        color = "#D8F3DC" if node["type"] == "leaf" else "#FFF3BF"

        net.add_node(
            node["id"],
            label=wrap_text(node["label"]),
            title=node["label"],
            shape="box",
            color=color,
            level=node["level"],
        )

    for parent, child, edge_label in edges:
        net.add_edge(
            parent,
            child,
            label=edge_label,
            arrows="to",
        )

    net.set_options("""
    {
        "layout": {
            "hierarchical": {
                "enabled": true,
                "direction": "UD",
                "sortMethod": "directed",
                "levelSeparation": 160,
                "nodeSpacing": 300,
                "treeSpacing": 350
            }
        },
        "physics": {
            "enabled": false
        },
        "interaction": {
            "dragView": true,
            "zoomView": true,
            "navigationButtons": true,
            "keyboard": true
        },
        "nodes": {
            "margin": 12,
            "font": {
                "size": 13
            }
        },
        "edges": {
            "font": {
                "size": 12,
                "align": "middle"
            },
            "smooth": {
                "enabled": true,
                "type": "cubicBezier",
                "forceDirection": "vertical"
            }
        }
    }
    """)

    return net.generate_html()


st.set_page_config(
    page_title="Stream Learner Tree Viewer",
    layout="wide",
)

st.title("Stream Learner Tree Viewer")

if not RESULTS_FOLDER.exists():
    st.error(f'Folder "{RESULTS_FOLDER}" was not found.')
    st.stop()

files = sorted(RESULTS_FOLDER.glob("*.txt"))

if not files:
    st.warning(f'No .txt files found in "{RESULTS_FOLDER}".')
    st.stop()

file_names = [file.name for file in files]
default_file = "HAT_confidence_0.1.txt"
default_index = file_names.index(default_file) if default_file in file_names else 0

selected_file = st.selectbox(
    "Select result file",
    file_names,
    index=default_index
)

file_path = RESULTS_FOLDER / selected_file
full_text = file_path.read_text(encoding="utf-8")

snapshots = get_snapshots(full_text)

if snapshots:
    positions = list(range(len(snapshots)))
    slider_key = f"time_slider_{selected_file}"
    max_position = len(snapshots) - 1

    # Initialize slider at the latest tree
    if slider_key not in st.session_state:
        st.session_state[slider_key] = max_position

    # Safety if the selected file changes
    st.session_state[slider_key] = min(
        st.session_state[slider_key],
        max_position
    )

    def move_snapshot(delta, key, max_pos):
        new_position = st.session_state[key] + delta
        st.session_state[key] = max(
            0,
            min(new_position, max_pos)
        )

    back_col, slider_col, forward_col = st.columns(
        [1, 8, 1],
        vertical_alignment="bottom"
    )

    with back_col:
        st.button(
            "◀",
            key=f"back_{selected_file}",
            on_click=move_snapshot,
            args=(-1, slider_key, max_position),
            disabled=st.session_state[slider_key] == 0,
            use_container_width=True
        )

    with slider_col:
        selected_position = st.select_slider(
            "Over time",
            options=positions,
            format_func=lambda pos: snapshots[pos]["date"],
            key=slider_key
        )

    with forward_col:
        st.button(
            "▶",
            key=f"forward_{selected_file}",
            on_click=move_snapshot,
            args=(1, slider_key, max_position),
            disabled=st.session_state[slider_key] == max_position,
            use_container_width=True
        )

    selected_snapshot = snapshots[selected_position]
    text = selected_snapshot["text"]

    st.caption(
        f"Date: {selected_snapshot['date']}"
    )

else:
    text = full_text
    st.warning("No dated snapshots found.")


# Everything below uses ONLY the selected snapshot

tree_format = detect_tree_format(text)

accuracy = get_accuracy(text)

nodes, edges = parse_tree(text)

st.caption(
    f"Detected tree type: {tree_format}"
)

if accuracy is not None:
    st.metric(
        "Accuracy",
        f"{accuracy:.2f}%"
    )

st.write(
    f"Tree contains **{len(nodes)} nodes**."
)

if nodes:
    tree_html = create_tree(nodes, edges)

    components.html(
        tree_html,
        height=750,
        scrolling=True
    )
else:
    st.warning(
        "No tree structure could be found."
    )

with st.expander("Show original tree output"):
    st.code(
        text,
        language="text"
    )