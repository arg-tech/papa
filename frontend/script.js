document.getElementById("submitButton").addEventListener("click", async () => {
  const selectedFile = document.getElementById("fileInput").files[0];

  if (!selectedFile) {
    alert("Please select a file first.");
    return;
  }

  const reader = new FileReader();
  const imcSwitch = document.getElementById("imcSwitch");
  const advancedSwitch = document.getElementById("advancedSwitch");

  reader.onload = async (e) => {
    const rawJSON = JSON.parse(e.target.result);

    // log to see if the json is being parsed correctly
    console.log("Parsed JSON:", rawJSON);

    const response = await fetch("/api/validate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        xaif: rawJSON,
        hasIntermapConnections: imcSwitch.checked,
      }),
    });

    const data = await response.json();
    console.log("Server response:", data);

    if (data.errors.length === 0) {
      let displayMessage = "No errors to display";
      document.getElementById("responseBox").innerText = displayMessage;
      return;
    }

    let html;
    if (data.errors.length < 3) {
      html = `<table class="table table-hover">`;
    } else {
      html = `<table class='table table-striped table-hover'>`;
    }

    function applyAdvancedToggle() {
      const show = advancedSwitch.checked;
      document.querySelectorAll(".advanced-col").forEach((cell) => {
        cell.classList.toggle("d-none", !show);
      });
    }

    advancedSwitch.addEventListener("change", applyAdvancedToggle);
    html += `<thead>
      <tr>
        <th>#</th>
        <th>Error Description</th>
        <th>Node Text</th>
        <th class="advanced-col text-nowrap d-none">Node ID</th>
        <th class="advanced-col text-nowrap d-none">Edge ID</th>
      </tr>
    </thead>
    <tbody class="table-group-divider">`;
    for (const [index, error] of data.errors.entries()) {
      html += `<tr>
        <td>${index + 1}</td>
        <td>${error.description}</td>
        <td>${error.nodes[0].text}</td>`;

      // Add nodes to table
      html +=
        '<td class="advanced-col d-none">' +
        (error.nodes?.map((node) => node.nodeID).join(", ") || "N/A") +
        "</td>";

      // Add edges to table
      html +=
        '<td class="advanced-col d-none">' +
        (error.edges?.map((edge) => edge.edgeID).join(", ") || "N/A") +
        "</td>";

      html += `</tr>`;
    }
    html += "</tbody></table>";
    console.log(html);
    document.getElementById("responseBox").innerHTML = html;
    applyAdvancedToggle();
  };

  reader.readAsText(selectedFile);
});
