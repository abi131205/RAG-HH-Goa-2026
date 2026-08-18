document.addEventListener('DOMContentLoaded', () => {
    const langSelect = document.getElementById('lang-select');
    const textInput = document.getElementById('text-query-input');
    const btnSubmit = document.getElementById('btn-submit-text');
    const btnRecord = document.getElementById('btn-record');
    const recStatus = document.getElementById('recording-status');
    
    const badgeStatus = document.getElementById('badge-status');
    const badgeGuard = document.getElementById('badge-guardrail');
    const badgeLatency = document.getElementById('badge-total-latency');
    
    const outTranscript = document.getElementById('output-transcript');
    const outAnswer = document.getElementById('output-answer');
    
    const barStt = document.getElementById('bar-stt');
    const barFaiss = document.getElementById('bar-faiss');
    const barGuard = document.getElementById('bar-guard');
    const barLlm = document.getElementById('bar-llm');
    
    const valStt = document.getElementById('val-stt');
    const valFaiss = document.getElementById('val-faiss');
    const valGuard = document.getElementById('val-guard');
    const valLlm = document.getElementById('val-llm');
    
    const chunksContainer = document.getElementById('chunks-container');
    
    const modalBench = document.getElementById('modal-bench');
    const btnOpenBench = document.getElementById('btn-open-bench');
    const btnCloseBench = document.getElementById('btn-close-bench');
    
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    // Preset Chips Handler
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            textInput.value = chip.getAttribute('data-query');
            langSelect.value = chip.getAttribute('data-lang');
            executeTextQuery();
        });
    });

    // Execute Text Query
    btnSubmit.addEventListener('click', executeTextQuery);
    textInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') executeTextQuery();
    });

    async function executeTextQuery() {
        const query = textInput.value.trim();
        if (!query) return;

        setSystemStatus('PROCESSING...', 'badge-idle');
        outAnswer.textContent = 'Generating grounded answer...';
        
        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text_query: query,
                    language_code: langSelect.value,
                    top_k: 3
                })
            });
            const data = await response.json();
            renderResponse(data);
        } catch (err) {
            console.error(err);
            setSystemStatus('ERROR', 'badge-idle');
            outAnswer.textContent = 'Pipeline execution error occurred.';
        }
    }

    // Mic Voice Recording (Web Audio API)
    btnRecord.addEventListener('click', async () => {
        if (!isRecording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    if (event.data.size > 0) audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    await handleAudioUpload(audioBlob);
                };

                mediaRecorder.start();
                isRecording = true;
                btnRecord.classList.add('recording');
                recStatus.textContent = 'Recording audio... Click again to stop.';
            } catch (err) {
                console.error("Microphone access error:", err);
                recStatus.textContent = "Mic access error. Check browser permissions.";
            }
        } else {
            mediaRecorder.stop();
            isRecording = false;
            btnRecord.classList.remove('recording');
            recStatus.textContent = 'Processing recorded speech...';
        }
    });

    async function handleAudioUpload(blob) {
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = async () => {
            const base64Audio = reader.result.split(',')[1];
            setSystemStatus('PROCESSING STT...', 'badge-idle');
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        audio_base64: base64Audio,
                        language_code: langSelect.value,
                        top_k: 3
                    })
                });
                const data = await response.json();
                renderResponse(data);
                recStatus.textContent = 'Click microphone to speak';
            } catch (err) {
                console.error(err);
                recStatus.textContent = 'Audio processing failed.';
            }
        };
    }

    // Render RAG Pipeline Output & Visualizer
    function renderResponse(data) {
        outTranscript.textContent = data.transcript || textInput.value;
        outAnswer.textContent = data.answer;
        
        // System & Guardrail Status
        if (data.status === 'SUCCESS') {
            setSystemStatus('SUCCESS', 'badge-safe');
        } else if (data.status === 'REJECTED_GUARDRAIL') {
            setSystemStatus('REFUSED (GUARDRAIL)', 'badge-idle');
        } else {
            setSystemStatus('ERROR', 'badge-idle');
        }
        
        if (data.guardrail.is_safe && data.guardrail.is_relevant) {
            badgeGuard.textContent = "GUARDRAILS PASS";
            badgeGuard.className = "status-badge badge-safe";
        } else {
            badgeGuard.textContent = data.guardrail.refusal_reason || "GUARDRAILS REFUSAL";
            badgeGuard.className = "status-badge badge-idle";
        }

        // Timing Breakdown
        const t = data.timing_ms || {};
        const total = t.total_ms || 1.0;
        badgeLatency.textContent = `TOTAL: ${total.toFixed(1)}ms`;

        valStt.textContent = `${(t.stt_ms || 0).toFixed(1)}ms`;
        valFaiss.textContent = `${(t.retrieval_ms || 0).toFixed(1)}ms`;
        valGuard.textContent = `${(t.guardrail_ms || 0).toFixed(1)}ms`;
        valLlm.textContent = `${(t.llm_ms || 0).toFixed(1)}ms`;

        barStt.style.width = `${Math.min(100, ((t.stt_ms || 0) / total) * 100)}%`;
        barFaiss.style.width = `${Math.min(100, ((t.retrieval_ms || 0) / total) * 100)}%`;
        barGuard.style.width = `${Math.min(100, ((t.guardrail_ms || 0) / total) * 100)}%`;
        barLlm.style.width = `${Math.min(100, ((t.llm_ms || 0) / total) * 100)}%`;

        // Render Chunks
        chunksContainer.innerHTML = '';
        if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
            data.retrieved_chunks.forEach(chunk => {
                const card = document.createElement('div');
                card.className = 'chunk-card';
                card.innerHTML = `
                    <div class="chunk-header">
                        <span>[LANG: ${chunk.language.toUpperCase()}] Score: ${chunk.score.toFixed(4)}</span>
                        <span>Passage ID: ${chunk.passage_id}</span>
                    </div>
                    <div class="chunk-text">${chunk.raw_text}</div>
                `;
                chunksContainer.appendChild(card);
            });
        } else {
            chunksContainer.innerHTML = '<div class="empty-state">No context passages retrieved for this query.</div>';
        }
    }

    function setSystemStatus(text, className) {
        badgeStatus.textContent = text;
        badgeStatus.className = `status-badge ${className}`;
    }

    // Modal Listeners
    btnOpenBench.addEventListener('click', async () => {
        modalBench.classList.remove('hidden');
        try {
            const resp = await fetch('/api/benchmark');
            const data = await resp.json();
            renderBenchTable(data.metrics);
        } catch (e) {
            console.error(e);
        }
    });

    btnCloseBench.addEventListener('click', () => {
        modalBench.classList.add('hidden');
    });

    function renderBenchTable(metrics) {
        if (!metrics) return;
        const tbody = document.getElementById('bench-table-body');
        tbody.innerHTML = '';
        for (const [stage, p] of Object.entries(metrics)) {
            const tr = document.createElement('tr');
            if (stage.includes("Total")) tr.className = "row-highlight";
            tr.innerHTML = `
                <td>${stage}</td>
                <td>${p['P50'].toFixed(2)} ms</td>
                <td>${p['P70'].toFixed(2)} ms</td>
                <td>${p['P100 (Max)'].toFixed(2)} ms</td>
            `;
            tbody.appendChild(tr);
        }
    }
});
