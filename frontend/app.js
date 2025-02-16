async function fetchHistoricalData(location) {
    try {
        const response = await fetch(`http://127.0.0.1:5000/get_historical_data?location=${location}`);
        const data = await response.json();
        if (data.image_url) {
            const historicalScene = document.getElementById('historical-scene');
            historicalScene.setAttribute('src', `http://127.0.0.1:5000${data.image_url}`);
        } else {
            alert("No historical data available for this location.");
        }
    } catch (error) {
        console.error("Error fetching historical data:", error);
    }
}

// Example: Fetch data for San Francisco
fetchHistoricalData("San Francisco");
