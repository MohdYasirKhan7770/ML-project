document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('newsInput');
    const charCount = document.getElementById('charCount');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.getElementById('btnLoader');
    const healthBadge = document.getElementById('healthBadge');

    const resultContainer = document.getElementById('resultContainer');
    const resultTitle = document.getElementById('resultTitle');
    const verdictText = document.getElementById('verdictText');
    const confidencePct = document.getElementById('confidencePct');
    const confidenceFill = document.getElementById('confidenceFill');
    const agentMessage = document.getElementById('agentMessage');
    const agentSummary = document.getElementById('agentSummary');
    const agentExcerpt = document.getElementById('agentExcerpt');
    const nextSteps = document.getElementById('nextSteps');

    const historyList = document.getElementById('historyList');
    const clearHistoryBtn = document.getElementById('clearHistory');
    const exampleChips = document.querySelectorAll('.example-chip');

    let history = JSON.parse(localStorage.getItem('newsHistory') || '[]');

    renderHistory();
    updateCount();
    checkHealth();

    input.addEventListener('input', updateCount);

    clearBtn.addEventListener('click', () => {
        input.value = '';
        updateCount();
        resultContainer.classList.add('hidden');
    });

    exampleChips.forEach(chip => {
        chip.addEventListener('click', () => {
            input.value = chip.dataset.text;
            updateCount();
            input.focus();
        });
    });

    analyzeBtn.addEventListener('click', async () => {
        const text = input.value.trim();
        if (!text) {
            input.focus();
            return;
        }

        setLoading(true);

        try {
            const response = await fetch('/agent', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                const errorPayload = await response.json().catch(() => ({ detail: 'Request failed.' }));
                throw new Error(errorPayload.detail || 'Request failed.');
            }

            const data = await response.json();
            showResult(data);
            addToHistory(data, text);
        } catch (error) {
            resultContainer.classList.remove('hidden');
            resultContainer.classList.add('result-fake');
            resultTitle.textContent = 'Agent unavailable';
            verdictText.textContent = 'No verdict';
            confidencePct.textContent = '0%';
            confidenceFill.style.width = '0%';
            agentMessage.textContent = error.message || 'The agent could not complete the request.';
            agentSummary.textContent = 'Check whether the backend is running and the model artifacts are available.';
            agentExcerpt.textContent = text.slice(0, 220) || 'No text provided.';
            nextSteps.innerHTML = '<li>Confirm the API is running.</li><li>Confirm the saved model files exist.</li>';
        } finally {
            setLoading(false);
        }
    });

    clearHistoryBtn.addEventListener('click', () => {
        history = [];
        localStorage.setItem('newsHistory', JSON.stringify(history));
        renderHistory();
    });

    function updateCount() {
        const length = input.value.length;
        charCount.textContent = `${length} character${length !== 1 ? 's' : ''}`;
    }

    async function checkHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            healthBadge.textContent = data.model_loaded ? 'Model Ready' : 'Model Missing';
            healthBadge.style.background = data.model_loaded
                ? 'rgba(255, 255, 255, 0.18)'
                : 'rgba(185, 60, 45, 0.32)';
        } catch (_) {
            healthBadge.textContent = 'Offline';
            healthBadge.style.background = 'rgba(185, 60, 45, 0.32)';
        }
    }

    function setLoading(isLoading) {
        analyzeBtn.disabled = isLoading;
        btnText.textContent = isLoading ? 'Analyzing...' : 'Analyze With Agent';
        btnLoader.style.display = isLoading ? 'inline-block' : 'none';
    }

    function showResult(data) {
        const prediction = data.prediction;
        const agent = data.agent;
        const isReal = prediction.raw_prediction === 1;

        resultContainer.classList.remove('hidden', 'result-real', 'result-fake');
        resultContainer.classList.add(isReal ? 'result-real' : 'result-fake');

        resultTitle.textContent = isReal ? 'Reliable signal detected' : 'Potential misinformation detected';
        verdictText.textContent = prediction.label;
        confidencePct.textContent = `${agent.confidence_percent}%`;
        confidenceFill.style.width = `${agent.confidence_percent}%`;
        agentMessage.textContent = agent.message;
        agentSummary.textContent = agent.summary;
        agentExcerpt.textContent = agent.excerpt;
        nextSteps.innerHTML = agent.next_steps.map(step => `<li>${step}</li>`).join('');
    }

    function addToHistory(data, text) {
        const prediction = data.prediction;
        const item = {
            id: Date.now(),
            text,
            label: prediction.raw_prediction === 1 ? 'REAL' : 'FAKE',
            confidence: data.agent.confidence_percent,
            timestamp: new Date().toLocaleTimeString()
        };

        history.unshift(item);
        history = history.slice(0, 8);
        localStorage.setItem('newsHistory', JSON.stringify(history));
        renderHistory();
    }

    function renderHistory() {
        if (!history.length) {
            historyList.innerHTML = '<p class="empty-msg">No recent activity yet.</p>';
            return;
        }

        historyList.innerHTML = history.map(item => `
            <div class="history-item" data-id="${item.id}">
                <p class="h-text">${escapeHtml(item.text.slice(0, 180))}${item.text.length > 180 ? '...' : ''}</p>
                <div class="h-meta">
                    <span class="${item.label === 'REAL' ? 'h-real' : 'h-fake'}">${item.label}</span>
                    <span>${item.confidence}% | ${item.timestamp}</span>
                </div>
            </div>
        `).join('');

        historyList.querySelectorAll('.history-item').forEach(node => {
            node.addEventListener('click', () => {
                const item = history.find(entry => String(entry.id) === node.dataset.id);
                if (item) {
                    input.value = item.text;
                    updateCount();
                    input.focus();
                }
            });
        });
    }

    function escapeHtml(value) {
        return value
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#39;');
    }
});
