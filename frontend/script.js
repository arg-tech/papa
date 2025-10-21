document.getElementById("submitButton").addEventListener("click", async () => {
  const selectedFile = document.getElementById("fileInput").files[0];

  if (!selectedFile) {
    alert("Please select a file first.");
    return;
  }

  const reader = new FileReader();
  const imcSwitch = document.getElementById("imcSwitch");

  reader.onload = async (e) => {
    const rawJSON = JSON.parse(e.target.result);

    // log to see if the json is being parsed correctly
    console.log("Parsed JSON:", rawJSON);

    const response = await fetch("/api/all_analytics", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        xaif: rawJSON,
      }),
    });

    const data = await response.json();
    console.log("Server response:", data);

    // creating a downloadable file for the user
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");

    a.href = url;
    a.download = "updated.json"; // name of the file
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    // if (data.errors.length === 0) {
    //   let displayMessage = "No errors to display";
    //   document.getElementById("responseBox").innerText = displayMessage;
    //   return;
    // }

    // let html;
    // if (data.errors.length < 3) {
    //   html = `<table class="table table-hover">`;
    // } else {
    //   html = `<table class='table table-striped table-hover'>`;
    // }

    // html += `<thead>
    //   <tr>
    //     <th>#</th>
    //     <th>Error Description</th>
    //     <th>Node Text</th>
    //     <th class="advanced-col text-nowrap d-none">Node ID</th>
    //     <th class="advanced-col text-nowrap d-none">Edge ID</th>
    //   </tr>
    // </thead>
    // <tbody class="table-group-divider">`;
    // for (const [index, error] of data.errors.entries()) {
    //   html += `<tr>
    //     <td>${index + 1}</td>
    //     <td>${error.description}</td>
    //     <td>${error.nodes[0].text}</td>`;

    //   // Add nodes to table
    //   html +=
    //     '<td class="advanced-col d-none">' +
    //     (error.nodes?.map((node) => node.nodeID).join(", ") || "N/A") +
    //     "</td>";

    //   // Add edges to table
    //   html +=
    //     '<td class="advanced-col d-none">' +
    //     (error.edges?.map((edge) => edge.edgeID).join(", ") || "N/A") +
    //     "</td>";

    //   html += `</tr>`;
    // }
    // html += "</tbody></table>";
    // console.log(html);
    // document.getElementById("responseBox").innerHTML = html;
  };

  reader.readAsText(selectedFile);
});
