document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('newsInput');
    const charCount = document.getElementById('charCount');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.getElementById('btnLoader');
    
    const resultContainer = document.getElementById('resultContainer');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const confidencePct = document.getElementById('confidencePct');
    const confidenceFill = document.getElementById('confidenceFill');
    const resultMessage = document.getElementById('resultMessage');

    // Update character count
    input.addEventListener('input', () => {
        const length = input.value.length;
        charCount.textContent = `${length} character${length !== 1 ? 's' : ''}`;
    });

    analyzeBtn.addEventListener('click', async () => {
        const text = input.value.trim();
        if (!text) {
            // Simple shake animation
            input.parentElement.animate([
                { transform: 'translateX(0)' },
                { transform: 'translateX(-10px)' },
                { transform: 'translateX(10px)' },
                { transform: 'translateX(-10px)' },
                { transform: 'translateX(10px)' },
                { transform: 'translateX(0)' }
            ], { duration: 500 });
            return;
        }

        // Loading state
        setLoading(true);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!response.ok) throw new Error('Network response was not ok');

            const data = await response.json();
            showResult(data);

        } catch (error) {
            console.error('Error:', error);
            alert('Failed to analyze the article. Ensure the server is running.');
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        analyzeBtn.disabled = isLoading;
        btnText.style.display = isLoading ? 'none' : 'block';
        btnLoader.style.display = isLoading ? 'block' : 'none';
        
        if (isLoading) {
            resultContainer.classList.add('hidden');
            resultContainer.style.position = 'absolute';
            // Reset width to trigger animation later
            confidenceFill.style.width = '0%';
        }
    }

    function showResult(data) {
        resultContainer.style.position = 'relative';
        resultContainer.classList.remove('hidden', 'result-real', 'result-fake');
        
        const isReal = data.raw_prediction === 1;
        const confidenceValue = (data.confidence * 100).toFixed(1);

        if (isReal) {
            resultContainer.classList.add('result-real');
            resultIcon.textContent = '✅';
            resultTitle.textContent = 'REAL News';
            resultMessage.textContent = 'Based on our analysis, this article contains characteristics typical of factual news reporting.';
        } else {
            resultContainer.classList.add('result-fake');
            resultIcon.textContent = '❌';
            resultTitle.textContent = 'FAKE News';
            resultMessage.textContent = 'Our model has flagged this article. It contains linguistic patterns commonly associated with deceptive or fabricated news.';
        }

        confidencePct.textContent = `${confidenceValue}%`;
        
        // Trigger reflow for animation
        void resultContainer.offsetWidth;
        
        // Animate progress bar
        setTimeout(() => {
            confidenceFill.style.width = `${confidenceValue}%`;
        }, 100);
    }
});
