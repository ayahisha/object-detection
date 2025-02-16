async function fetchDetection() {
    try {
        let response = await fetch("http://127.0.0.1:5000/detect");
        let data = await response.json();
        document.getElementById("result").innerHTML = `Detected Object: <span>${data.detected_object}</span>`;
    } catch (error) {
        console.error("Error:", error);
        alert("Error detecting object.");
    }
}
