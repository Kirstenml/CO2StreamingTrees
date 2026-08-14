const RESULTS_FOLDER = "stream_learner_results";
const MANIFEST_FILE = `${RESULTS_FOLDER}/files.json`;
const fileSelect = document.getElementById("file-select");
const timeControls = document.getElementById("time-controls");
const timeSlider = document.getElementById("time-slider");
const backButton = document.getElementById("back-button");
const forwardButton = document.getElementById("forward-button");
const dateOutput = document.getElementById("date-output");
const statusOutput = document.getElementById("status");
const treeTypeOutput = document.getElementById("tree-type");
const accuracyOutput = document.getElementById("accuracy");
const nodeCountOutput = document.getElementById("node-count");
const sourceOutput = document.getElementById("source-output");
const treeContainer = document.getElementById("tree");

let snapshots = [];
let network = null;

function setStatus(message, type = "") {
  statusOutput.textContent = message;
  statusOutput.className = `status ${type}`.trim();
}

function getAccuracy(text) {
  const match = text.match(/Accuracy:\s*([0-9]+(?:\.[0-9]+)?)/);
  return match ? Number.parseFloat(match[1]) : null;
}

function getSnapshots(text) {
  const pattern = /^[ \t]*Date:[ \t]*(\d{4}-\d{2}-\d{2})(?:[ \t]+\d{2}:\d{2}:\d{2})?[ \t]*$/gm;
  const matches = [...text.matchAll(pattern)];

  return matches.map((match, index) => {
    const start = match.index + match[0].length;
    const end = index + 1 < matches.length ? matches[index + 1].index : text.length;
    const snapshotText = text
      .slice(start, end)
      .replace(/^[ \t]*=+[ \t]*$/gm, "")
      .trim();

    return {
      date: match[1],
      text: snapshotText,
    };
  });
}

function detectTreeFormat(text) {
  if (text.includes("Split test:")) {
    return "SOHOT";
  }

  if (/^\s*Leaf\s*$/m.test(text) && text.includes("Class distribution:")) {
    return "SOHOT";
  }

  if (/^\s*if\s+\[att/m.test(text)) {
    return "HAT";
  }

  if (/^\s*Leaf\s+\[class:/m.test(text)) {
    return "HAT";
  }

  return "UNKNOWN";
}

function parseTree(text) {
  const treeFormat = detectTreeFormat(text);

  if (treeFormat === "HAT") {
    return parseHatTree(text);
  }

  if (treeFormat === "SOHOT") {
    return parseSohotTree(text);
  }

  return { nodes: [], edges: [] };
}

function leadingSpaces(line) {
  return line.length - line.replace(/^ +/, "").length;
}

function parseHatTree(text) {
  const nodes = [];
  const edges = [];
  const stack = [];
  let counter = 0;

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.replace(/\t/g, "  ");
    const stripped = line.trim();

    if (!stripped || stripped.startsWith("Accuracy:")) {
      continue;
    }

    let type;
    let label;

    if (stripped.startsWith("if ")) {
      type = "condition";
      label = stripped.slice(3).replace(/:\s*$/, "").trim();
    } else if (stripped.startsWith("Leaf")) {
      type = "leaf";
      label = stripped;
    } else {
      continue;
    }

    const indent = leadingSpaces(line);
    const id = `node_${counter++}`;

    while (stack.length && stack[stack.length - 1].indent >= indent) {
      stack.pop();
    }

    nodes.push({
      id,
      label,
      type,
      level: stack.length + 1,
    });

    const parentId = stack.length ? stack[stack.length - 1].id : "root";
    edges.push({
      from: parentId,
      to: id,
      label: "",
    });

    if (type === "condition") {
      stack.push({ indent, id });
    }
  }

  return { nodes, edges };
}

function parseSohotTree(text) {
  const nodes = [];
  const edges = [];
  const lines = text.split(/\r?\n/);
  const nodeStack = [];
  const branchAtIndent = new Map();
  let counter = 0;
  let i = 0;

  while (i < lines.length) {
    const rawLine = lines[i].replace(/\t/g, "  ");
    const stripped = rawLine.trim();

    if (!stripped || stripped.startsWith("Accuracy:")) {
      i += 1;
      continue;
    }

    const indent = leadingSpaces(rawLine);

    if (stripped === "LEFT:" || stripped === "RIGHT:") {
      branchAtIndent.set(indent, stripped.slice(0, -1));
      i += 1;
      continue;
    }

    if (stripped.startsWith("Split test:")) {
      const splitLabel = stripped.replace("Split test:", "").trim();
      const featureImportance = [];
      let j = i + 1;

      if (j < lines.length && lines[j].trim() === "Feature importance:") {
        const fiLine = lines[j].replace(/\t/g, "  ");
        const fiIndent = leadingSpaces(fiLine);
        j += 1;

        while (j < lines.length) {
          const nextLine = lines[j].replace(/\t/g, "  ");
          const nextStripped = nextLine.trim();

          if (!nextStripped) {
            j += 1;
            continue;
          }

          const nextIndent = leadingSpaces(nextLine);

          if (nextIndent <= fiIndent) {
            break;
          }

          if (nextStripped !== "LEFT:" && nextStripped !== "RIGHT:") {
            featureImportance.push(nextStripped);
          }

          j += 1;
        }
      }

      const usefulImportance = featureImportance.filter(value => value !== "n/a");
      let label = splitLabel;

      if (usefulImportance.length) {
        label += `\nFeature importance:\n${usefulImportance.join("\n")}`;
      }

      const id = `node_${counter++}`;

      while (nodeStack.length && nodeStack[nodeStack.length - 1].indent >= indent) {
        nodeStack.pop();
      }

      nodes.push({
        id,
        label,
        type: "condition",
        level: nodeStack.length + 1,
      });

      let parentId = "root";
      let branch = "";

      if (nodeStack.length) {
        parentId = nodeStack[nodeStack.length - 1].id;
        branch = branchAtIndent.get(indent - 2) || "";
      }

      edges.push({
        from: parentId,
        to: id,
        label: branch,
      });

      nodeStack.push({ id, indent });
      i += 1;
      continue;
    }

    if (stripped === "Leaf") {
      const leafLines = [];
      let j = i + 1;

      while (j < lines.length) {
        const nextLine = lines[j].replace(/\t/g, "  ");
        const nextStripped = nextLine.trim();

        if (!nextStripped) {
          j += 1;
          continue;
        }

        const nextIndent = leadingSpaces(nextLine);

        if (nextIndent <= indent) {
          break;
        }

        leafLines.push(nextStripped);
        j += 1;
      }

      const labelLines = ["Leaf"];

      for (const leafLine of leafLines) {
        if (leafLine.startsWith("Class distribution:")) {
          const distribution = leafLine.replace("Class distribution:", "").trim();
          labelLines.push("Class distribution:");
          labelLines.push(distribution);
        } else if (leafLine.startsWith("Predicted class:")) {
          labelLines.push(leafLine);
        } else if (leafLine.startsWith("P(x->leaf):")) {
          labelLines.push(leafLine);
        }
      }

      const id = `node_${counter++}`;

      while (nodeStack.length && nodeStack[nodeStack.length - 1].indent >= indent) {
        nodeStack.pop();
      }

      const parentId = nodeStack.length ? nodeStack[nodeStack.length - 1].id : "root";
      const branch = branchAtIndent.get(indent - 2) || "";

      nodes.push({
        id,
        label: labelLines.join("\n"),
        type: "leaf",
        level: nodeStack.length + 1,
      });

      edges.push({
        from: parentId,
        to: id,
        label: branch,
      });

      i = j;
      continue;
    }

    i += 1;
  }

  return { nodes, edges };
}

function wrapText(text, width = 45) {
  return text
    .split("\n")
    .flatMap(line => {
      if (!line) {
        return [""];
      }

      const words = line.split(/\s+/);
      const result = [];
      let current = "";

      for (const word of words) {
        const candidate = current ? `${current} ${word}` : word;

        if (candidate.length <= width || !current) {
          current = candidate;
        } else {
          result.push(current);
          current = word;
        }
      }

      if (current) {
        result.push(current);
      }

      return result.length ? result : [line];
    })
    .join("\n");
}

function renderTree(nodes, edges) {
  if (network) {
    network.destroy();
    network = null;
  }

  treeContainer.innerHTML = "";

  if (!nodes.length) {
    treeContainer.innerHTML = '<div class="empty-tree">No tree structure could be found.</div>';
    return;
  }

  const networkNodes = [
    {
      id: "root",
      label: "Decision Tree",
      shape: "box",
      color: "#D9EAF7",
      level: 0,
      margin: 12,
    },
    ...nodes.map(node => ({
      id: node.id,
      label: wrapText(node.label),
      title: node.label.replace(/\n/g, "<br>"),
      shape: "box",
      color: node.type === "leaf" ? "#D8F3DC" : "#FFF3BF",
      level: node.level,
      margin: 12,
    })),
  ];

  const networkEdges = edges.map(edge => ({
    from: edge.from,
    to: edge.to,
    label: edge.label,
    arrows: "to",
  }));

  const data = {
    nodes: new vis.DataSet(networkNodes),
    edges: new vis.DataSet(networkEdges),
  };

  const options = {
    layout: {
      hierarchical: {
        enabled: true,
        direction: "UD",
        sortMethod: "directed",
        levelSeparation: 160,
        nodeSpacing: 300,
        treeSpacing: 350,
      },
    },
    physics: {
      enabled: false,
    },
    interaction: {
      dragView: true,
      zoomView: true,
      navigationButtons: true,
      keyboard: true,
      hover: true,
    },
    nodes: {
      margin: 12,
      font: {
        size: 13,
        face: "Arial",
      },
    },
    edges: {
      font: {
        size: 12,
        align: "middle",
      },
      smooth: {
        enabled: true,
        type: "cubicBezier",
        forceDirection: "vertical",
      },
    },
  };

  network = new vis.Network(treeContainer, data, options);

  network.once("afterDrawing", () => {
    network.fit({
      animation: {
        duration: 250,
        easingFunction: "easeInOutQuad",
      },
    });
  });
}

function updateSnapshot(index) {
  if (!snapshots.length) {
    return;
  }

  const position = Math.max(0, Math.min(index, snapshots.length - 1));
  timeSlider.value = String(position);

  const snapshot = snapshots[position];
  const treeFormat = detectTreeFormat(snapshot.text);
  const accuracy = getAccuracy(snapshot.text);
  const { nodes, edges } = parseTree(snapshot.text);

  dateOutput.textContent = snapshot.date;
  treeTypeOutput.textContent = treeFormat;
  accuracyOutput.textContent = accuracy === null ? "–" : `${accuracy.toFixed(2)}%`;
  nodeCountOutput.textContent = String(nodes.length);
  sourceOutput.textContent = snapshot.text;

  backButton.disabled = position === 0;
  forwardButton.disabled = position === snapshots.length - 1;

  renderTree(nodes, edges);
}

function configureSnapshots(text) {
  snapshots = getSnapshots(text);

  if (!snapshots.length) {
    timeControls.hidden = true;

    const treeFormat = detectTreeFormat(text);
    const accuracy = getAccuracy(text);
    const { nodes, edges } = parseTree(text);

    treeTypeOutput.textContent = treeFormat;
    accuracyOutput.textContent = accuracy === null ? "–" : `${accuracy.toFixed(2)}%`;
    nodeCountOutput.textContent = String(nodes.length);
    sourceOutput.textContent = text;
    renderTree(nodes, edges);
    setStatus("No dated snapshots found; showing the file as one tree.", "warning");
    return;
  }

  timeControls.hidden = false;
  timeSlider.min = "0";
  timeSlider.max = String(snapshots.length - 1);
  timeSlider.step = "1";
  timeSlider.disabled = snapshots.length === 1;

  updateSnapshot(snapshots.length - 1);
  setStatus(`${snapshots.length} snapshot${snapshots.length === 1 ? "" : "s"} loaded.`);
}

async function loadResultFile(filename) {
    if (!filename) {
        return;
    }

    setStatus(
        `Loading ${filename}...`
    );

    fileSelect.disabled = true;

    try {
        const fileUrl = new URL(
            `${RESULTS_FOLDER}/${encodeURIComponent(filename)}`,
            new URL("./", window.location.href)
        );

        const response = await fetch(
            fileUrl,
            {
                cache: "no-store"
            }
        );

        if (!response.ok) {
            throw new Error(
                `HTTP ${response.status}`
            );
        }

        const text = await response.text();

        configureSnapshots(text);

    } catch (error) {
        snapshots = [];

        timeControls.hidden = true;
        treeContainer.innerHTML = "";
        sourceOutput.textContent = "";

        treeTypeOutput.textContent = "–";
        accuracyOutput.textContent = "–";
        nodeCountOutput.textContent = "–";

        setStatus(
            `Could not load "${filename}": ${error.message}`,
            "error"
        );

    } finally {
        fileSelect.disabled = false;
    }
}

// function getGitHubRepository() {
//   const hostname = window.location.hostname;
//
//   if (!hostname.endsWith(".github.io")) {
//     return null;
//   }
//
//   const owner = hostname.split(".")[0];
//   const pathParts = window.location.pathname.split("/").filter(Boolean);
//   const repo = pathParts.length ? pathParts[0] : `${owner}.github.io`;
//
//   return { owner, repo };
// }
//
// async function discoverFromGitHub() {
//   const repository = getGitHubRepository();
//
//   if (!repository) {
//     return [];
//   }
//
//   const { owner, repo } = repository;
//   const apiUrl =
//     `https://api.github.com/repos/${encodeURIComponent(owner)}/` +
//     `${encodeURIComponent(repo)}/contents/${RESULTS_FOLDER}`;
//
//   const response = await fetch(apiUrl, {
//     headers: {
//       Accept: "application/vnd.github+json",
//     },
//   });
//
//   if (!response.ok) {
//     throw new Error(`GitHub API returned HTTP ${response.status}`);
//   }
//
//   const content = await response.json();
//
//   if (!Array.isArray(content)) {
//     return [];
//   }
//
//   return content
//     .filter(item => item.type === "file" && item.name.toLowerCase().endsWith(".txt"))
//     .map(item => item.name)
//     .sort((a, b) => a.localeCompare(b));
// }

async function discoverFromManifest() {
    const manifestUrl = new URL(
        MANIFEST_FILE,
        new URL("./", window.location.href)
    );

    const response = await fetch(
        manifestUrl,
        {
            cache: "no-store"
        }
    );

    if (!response.ok) {
        throw new Error(
            `files.json returned HTTP ${response.status}`
        );
    }

    const manifest = await response.json();

    const files = Array.isArray(manifest)
        ? manifest
        : manifest.files;

    if (!Array.isArray(files)) {
        throw new Error(
            "files.json must contain a 'files' array."
        );
    }

    return files
        .filter(
            filename =>
                typeof filename === "string" &&
                filename.toLowerCase().endsWith(".txt")
        )
        .sort(
            (a, b) => a.localeCompare(b)
        );
}


async function discoverFromDirectoryListing() {
    const directoryUrl = new URL(
        `${RESULTS_FOLDER}/`,
        new URL("./", window.location.href)
    );

    const response = await fetch(
        directoryUrl,
        {
            cache: "no-store"
        }
    );

    if (!response.ok) {
        throw new Error(
            `Directory returned HTTP ${response.status}`
        );
    }

    const html = await response.text();

    const documentHtml = new DOMParser().parseFromString(
        html,
        "text/html"
    );

    const files = [];

    for (const link of documentHtml.querySelectorAll("a")) {
        const href = link.getAttribute("href");

        if (!href) {
            continue;
        }

        try {
            const url = new URL(
                href,
                directoryUrl
            );

            const parts = url.pathname
                .split("/")
                .filter(Boolean);

            const filename = decodeURIComponent(
                parts[parts.length - 1] || ""
            );

            if (
                filename
                    .toLowerCase()
                    .endsWith(".txt")
            ) {
                files.push(filename);
            }

        } catch (error) {
            console.warn(
                "Could not parse directory entry:",
                href
            );
        }
    }

    return [...new Set(files)]
        .sort(
            (a, b) => a.localeCompare(b)
        );
}


async function discoverResultFiles() {
    // First choice:
    // files.json generated by GitHub Actions.
    try {
        const files = await discoverFromManifest();

        if (files.length > 0) {
            console.log(
                "Files loaded from files.json:",
                files
            );

            return files;
        }

    } catch (error) {
        console.log(
            "No files.json available. Trying directory listing.",
            error
        );
    }


    // Local-development fallback.
    try {
        const files =
            await discoverFromDirectoryListing();

        if (files.length > 0) {
            console.log(
                "Files discovered locally:",
                files
            );

            return files;
        }

    } catch (error) {
        console.error(
            "Directory discovery failed:",
            error
        );
    }


    throw new Error(
        `No .txt files found in ${RESULTS_FOLDER}/`
    );
}

function populateFileSelect(files) {
    fileSelect.innerHTML = "";

    if (!files.length) {
        fileSelect.disabled = true;

        fileSelect.innerHTML =
            "<option>No result files found</option>";

        setStatus(
            "No .txt files were found in stream_learner_results/.",
            "error"
        );

        return;
    }

    for (const filename of files) {
        const option = document.createElement("option");

        option.value = filename;
        option.textContent = filename;

        fileSelect.appendChild(option);
    }

    const preferred = "HAT_confidence_0.1.txt";

    fileSelect.value = files.includes(preferred)
        ? preferred
        : files[0];

    fileSelect.disabled = false;

    loadResultFile(fileSelect.value);
}

fileSelect.addEventListener("change", () => {
  loadResultFile(fileSelect.value);
});

timeSlider.addEventListener("input", () => {
  updateSnapshot(Number.parseInt(timeSlider.value, 10));
});

backButton.addEventListener("click", () => {
  updateSnapshot(Number.parseInt(timeSlider.value, 10) - 1);
});

forwardButton.addEventListener("click", () => {
  updateSnapshot(Number.parseInt(timeSlider.value, 10) + 1);
});

window.addEventListener("resize", () => {
  if (network) {
    network.redraw();
  }
});

async function init() {
    if (typeof vis === "undefined") {
        setStatus(
            "vis-network could not be loaded.",
            "error"
        );

        return;
    }

    setStatus(
        "Loading result files..."
    );

    try {
        const files = await discoverResultFiles();

        populateFileSelect(files);
    } catch (error) {
        console.error(
            "Could not load files.json:",
            error
        );

        fileSelect.disabled = true;

        fileSelect.innerHTML =
            "<option>No result files found</option>";

        setStatus(
            `Could not load result file list: ${error.message}`,
            "error"
        );
    }
}


init();
