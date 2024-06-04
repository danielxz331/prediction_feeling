document.getElementById('analyzeButton').addEventListener('click', function() {
    const reviewText = document.getElementById('reviewText').value;

    if (reviewText.trim() === "") {
        alert("Please enter a movie review text.");
        return;
    }

    fetch('http://localhost:8888/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: reviewText }),
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        resultDiv.textContent = `Sentiment: ${data.sentiment}`;
        resultDiv.style.color = data.sentiment === 'positive' ? 'green' : 'red';
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
