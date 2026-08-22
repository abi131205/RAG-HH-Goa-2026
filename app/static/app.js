document.addEventListener('DOMContentLoaded', () => {
    let currentLanguage = 'auto';
    let currentStrategy = 'metadata';

    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    const btnRecord = document.getElementById('btn-record');
    const recordingStatus = document.getElementById('recording-status');
    const textInput = document.getElementById('text-query-input');
    const btnSubmitText = document.getElementById('btn-submit-text');
    const btnRefuseDemo = document.getElementById('btn-refuse-demo');
    const btnOpenEval = document.getElementById('btn-open-eval');
    const btnCloseEval = document.getElementById('btn-close-eval');
    const evalModal = document.getElementById('eval-modal');

    // 1. Web Audio API Microphone Recorder
    if (btnRecord) {
        btnRecord.addEventListener('click', async () => {
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];

                    mediaRecorder.ondataavailable = e => {
                        if (e.data.size > 0) audioChunks.push(e.data);
                    };

                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        const reader = new FileReader();
                        reader.readAsDataURL(audioBlob);
                        reader.onloadend = async () => {
                            const base64Audio = reader.result.split(',')[1];
                            recordingStatus.textContent = 'Transcribing voice & executing RAG...';
                            try {
                                await executeRAGQuery(null, base64Audio);
                            } catch (e) {
                                console.error("Voice query failed:", e);
                            } finally {
                                recordingStatus.textContent = 'Click Microphone to Speak';
                            }
                        };
                        stream.getTracks().forEach(t => t.stop());
                    };

                    mediaRecorder.start();
                    isRecording = true;
                    btnRecord.classList.add('recording');
                    recordingStatus.textContent = 'LISTENING... CLICK AGAIN TO STOP';
                } catch (err) {
                    console.error("Mic access error:", err);
                    recordingStatus.textContent = 'Voice input fallback activated.';
                    await executeRAGQuery();
                }
            } else {
                mediaRecorder.stop();
                isRecording = false;
                btnRecord.classList.remove('recording');
                recordingStatus.textContent = 'PROCESSING AUDIO...';
            }
        });
    }

    // 2. Language Switcher Pills
    const langPillsContainer = document.getElementById('lang-pills-container');
    if (langPillsContainer) {
        langPillsContainer.addEventListener('click', (e) => {
            if (e.target.classList.contains('lang-chip')) {
                document.querySelectorAll('.lang-chip').forEach(c => c.classList.remove('active'));
                e.target.classList.add('active');
                currentLanguage = e.target.dataset.lang;
            }
        });
    }

    // 3. Strategy Buttons
    document.querySelectorAll('.strat-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.strat-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentStrategy = btn.dataset.strategy;
        });
    });

    // 4. Preset Query Chips
    document.querySelectorAll('.preset-btn').forEach(chip => {
        chip.addEventListener('click', () => {
            textInput.value = chip.dataset.query;
            if (chip.dataset.lang) {
                currentLanguage = chip.dataset.lang;
                document.querySelectorAll('.lang-chip').forEach(c => {
                    c.classList.toggle('active', c.dataset.lang === currentLanguage);
                });
            }
            executeRAGQuery(textInput.value);
        });
    });

    // 5. Refusal Demo
    if (btnRefuseDemo) {
        btnRefuseDemo.addEventListener('click', () => {
            const offTopicQuery = "Who won the 2026 FIFA World Cup final match?";
            textInput.value = offTopicQuery;
            executeRAGQuery(offTopicQuery);
        });
    }

    // 6. Eval Modal
    if (btnOpenEval) btnOpenEval.addEventListener('click', () => evalModal.classList.remove('hidden'));
    if (btnCloseEval) btnCloseEval.addEventListener('click', () => evalModal.classList.add('hidden'));

    // Execute RAG
    if (btnSubmitText) {
        btnSubmitText.addEventListener('click', () => executeRAGQuery(textInput.value));
    }

    async function executeRAGQuery(textQuery = null, audioBase64 = null) {
        const queryText = textQuery || textInput.value;

        document.getElementById('badge-status').textContent = 'EXECUTING...';
        document.getElementById('answer-text').textContent = 'Generating grounded answer...';

        try {
            const res = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text_query: queryText,
                    audio_base64: audioBase64,
                    language_code: currentLanguage,
                    top_k: 3
                })
            });

            const data = await res.json();
            renderResults(data);
        } catch (err) {
            console.error(err);
            document.getElementById('badge-status').textContent = 'ERROR';
        }
    }

    function renderResults(data) {
        document.getElementById('badge-status').textContent = data.status === 'SUCCESS' ? 'PASS' : 'REJECTED';
        document.getElementById('badge-status').className = data.status === 'SUCCESS' ? 'badge-mint' : 'audit-badge';

        document.getElementById('badge-total-ms').textContent = `${data.timing_ms.total_ms || 0} ms`;
        document.getElementById('transcribed-text').textContent = data.transcript || data.text_query || "Voice input transcribed.";
        document.getElementById('answer-text').textContent = data.answer;

        const maxTime = Math.max(data.timing_ms.total_ms || 1, 100);
        document.getElementById('bar-stt').style.width = `${((data.timing_ms.stt_ms || 0) / maxTime) * 100}%`;
        document.getElementById('time-stt').textContent = `${data.timing_ms.stt_ms || 0}ms`;

        document.getElementById('bar-retrieval').style.width = `${((data.timing_ms.retrieval_ms || 0) / maxTime) * 100}%`;
        document.getElementById('time-retrieval').textContent = `${data.timing_ms.retrieval_ms || 0}ms`;

        document.getElementById('bar-guardrail').style.width = `${((data.timing_ms.guardrail_ms || 0) / maxTime) * 100}%`;
        document.getElementById('time-guardrail').textContent = `${data.timing_ms.guardrail_ms || 0}ms`;

        document.getElementById('bar-llm').style.width = `${((data.timing_ms.llm_ms || 0) / maxTime) * 100}%`;
        document.getElementById('time-llm').textContent = `${data.timing_ms.llm_ms || 0}ms`;

        const g = data.guardrail || {};
        document.getElementById('guard-status-safety').textContent = g.is_safe ? 'PASS' : 'FAIL';
        document.getElementById('guard-status-relevance').textContent = g.is_relevant ? 'PASS' : 'REJECT';
        document.getElementById('guard-status-grounding').textContent = g.is_grounded ? 'PASS' : 'FAIL';

        const chunksContainer = document.getElementById('chunks-container');
        chunksContainer.innerHTML = '';

        if (!data.retrieved_chunks || data.retrieved_chunks.length === 0) {
            chunksContainer.innerHTML = '<div style="font-size: 13px; color: var(--text-muted);">No context passages retrieved.</div>';
            return;
        }

        data.retrieved_chunks.forEach(c => {
            const card = document.createElement('div');
            card.className = 'chunk-card';
            card.innerHTML = `
                <div class="chunk-meta">
                    <span>[LANG: ${c.language.toUpperCase()}] Score: ${c.score.toFixed(4)}</span>
                    <span>Passage: ${c.passage_id}</span>
                </div>
                <div class="chunk-body">${c.raw_text}</div>
            `;
            chunksContainer.appendChild(card);
        });
    }
});
