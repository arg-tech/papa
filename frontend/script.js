document.getElementById("submitButton").addEventListener("click", async () => {
  const selectedFile = document.getElementById("fileInput").files[0];

  if (!selectedFile) {
    alert("Please select a file first.");
    return;
  }

  const reader = new FileReader();
  const nodeLevel = document.getElementById("node_level");
  const speaker = document.getElementById("speaker");
  const forecast = document.getElementById("forecast");

  reader.onload = async (e) => {
    const rawJSON = JSON.parse(e.target.result);

    // log to see if the json is being parsed correctly
    console.log("Parsed JSON:", rawJSON);
    console.log(nodeLevel.checked);
    console.log(speaker.checked);
    console.log(forecast.checked);

    const response = await fetch("/api/all_analytics", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        xaif: rawJSON,
        node_level: nodeLevel.checked,
        speaker: speaker.checked,
        forecast: forecast.checked,
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

    // displaying the json file on frontend if there is a response
    if (response.status === 200) {
      document.getElementById("json").style = "display: flex";
      document.querySelector('#json').data = data.analytics;
      document.getElementById("responseBox").style = "display: none";
    }
    // displaying an error message in response box if papa fails
    else {
      document.getElementById('json').style = "display: none";
      document.getElementById("responseBox").style = "display: flex";
      document.getElementById("responseBox").innerHTML = "There was an error in the analytics code. Open the downloaded file to see more details"
    }
  };

  reader.readAsText(selectedFile);
});
