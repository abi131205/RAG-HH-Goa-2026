document.addEventListener('DOMContentLoaded', () => {
    let currentLanguage = 'auto';
    let currentStrategy = 'metadata';

    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    const btnRecord = document.getElementById('btn-record');
    const textInput = document.getElementById('text-query-input');
    const btnSubmitText = document.getElementById('btn-submit-text');
    const btnRefuseDemo = document.getElementById('btn-refuse-demo');
    const btnOpenEval = document.getElementById('btn-open-eval');
    const btnCloseEval = document.getElementById('btn-close-eval');
    const btnNewChat = document.getElementById('btn-new-chat');
    const evalModal = document.getElementById('eval-modal');
    const chatFeed = document.getElementById('chat-feed');

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
                            try {
                                await executeRAGQuery(null, base64Audio);
                            } catch (e) {
                                console.error("Voice query error:", e);
                            }
                        };
                        stream.getTracks().forEach(t => t.stop());
                    };

                    mediaRecorder.start();
                    isRecording = true;
                    btnRecord.classList.add('recording');
                } catch (err) {
                    console.error("Mic access error:", err);
                    await executeRAGQuery();
                }
            } else {
                mediaRecorder.stop();
                isRecording = false;
                btnRecord.classList.remove('recording');
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

    // 3. Strategy Switcher
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

    // 6. New Chat Thread
    if (btnNewChat) {
        btnNewChat.addEventListener('click', () => {
            chatFeed.innerHTML = `
                <div class="chat-message ai">
                    <div class="chat-avatar">🤖</div>
                    <div class="chat-bubble-card">
                        <div class="bubble-header">
                            <span class="bubble-sender">INDIC RAG ASSISTANT</span>
                            <span>Fresh Session</span>
                        </div>
                        <div class="bubble-text">New chat thread initialized. Ask any question in 14 Indic languages!</div>
                    </div>
                </div>
            `;
        });
    }

    // 7. Eval Modal
    if (btnOpenEval) btnOpenEval.addEventListener('click', () => evalModal.classList.remove('hidden'));
    if (btnCloseEval) btnCloseEval.addEventListener('click', () => evalModal.classList.add('hidden'));

    // 8. Execute RAG
    if (btnSubmitText) {
        btnSubmitText.addEventListener('click', () => executeRAGQuery(textInput.value));
    }
    if (textInput) {
        textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                executeRAGQuery(textInput.value);
            }
        });
    }

    async function executeRAGQuery(textQuery = null, audioBase64 = null) {
        const queryText = textQuery !== null ? textQuery : textInput.value;
        if (!queryText.trim() && !audioBase64) return;

        // Append User Speech/Text Message to Chat Stream
        appendUserMessage(queryText || "Voice Input");
        textInput.value = '';

        // Append Loading Indicator
        const loadingId = appendLoadingMessage();

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
            removeMessage(loadingId);
            appendAIMessage(data);
        } catch (err) {
            console.error(err);
            removeMessage(loadingId);
            appendErrorMessage();
        }
    }

    function appendUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'chat-message user';
        msgDiv.innerHTML = `
            <div class="chat-avatar">👤</div>
            <div class="chat-bubble-card">
                <div class="bubble-header">
                    <span class="bubble-sender">YOU</span>
                    <span>${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                </div>
                <div class="bubble-text">${escapeHtml(text)}</div>
            </div>
        `;
        chatFeed.appendChild(msgDiv);
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }

    function appendLoadingMessage() {
        const id = 'loading-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.id = id;
        msgDiv.className = 'chat-message ai';
        msgDiv.innerHTML = `
            <div class="chat-avatar">🤖</div>
            <div class="chat-bubble-card">
                <div class="bubble-header">
                    <span class="bubble-sender">INDIC RAG ASSISTANT</span>
                    <span>Searching FAISS 70k...</span>
                </div>
                <div class="bubble-text">Retrieving grounded context and generating answer...</div>
            </div>
        `;
        chatFeed.appendChild(msgDiv);
        chatFeed.scrollTop = chatFeed.scrollHeight;
        return id;
    }

    function removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function appendAIMessage(data) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'chat-message ai';

        const chunksHtml = (data.retrieved_chunks || []).map(c => `
            <div class="context-chunk-item">
                <span class="chunk-tag">[LANG: ${c.language.toUpperCase()}] Score: ${c.score.toFixed(4)} (ID: ${c.passage_id})</span>
                <div>${escapeHtml(c.raw_text)}</div>
            </div>
        `).join('');

        const accordionId = 'acc-' + Date.now();
        const isSuccess = data.status === 'SUCCESS';
        const safeAnswer = (data.answer || '').replace(/'/g, "\\'").replace(/\n/g, " ");

        msgDiv.innerHTML = `
            <div class="chat-avatar">🤖</div>
            <div class="chat-bubble-card">
                <div class="bubble-header">
                    <span class="bubble-sender">INDIC RAG ASSISTANT</span>
                    <div>
                        <span class="badge-pill">${isSuccess ? 'GROUNDED 100%' : 'REFUSED (GUARD)'}</span>
                        <span class="badge-pill">${data.timing_ms.total_ms || 0}ms</span>
                    </div>
                </div>
                <div class="bubble-text">${escapeHtml(data.answer || '')}</div>

                <div class="bubble-tools">
                    <button class="tool-btn" onclick="speakText('${safeAnswer}')">🔊 Listen</button>
                    <button class="tool-btn" onclick="copyText('${safeAnswer}')">📋 Copy</button>
                    ${data.retrieved_chunks && data.retrieved_chunks.length > 0 ? 
                        `<button class="tool-btn" onclick="toggleAccordion('${accordionId}')">📚 Sources (${data.retrieved_chunks.length})</button>` : ''}
                </div>

                <div id="${accordionId}" class="context-accordion hidden">
                    <div style="font-weight: 700; font-size: 11px; color: var(--mint-neon);">RETRIEVED FAISS CONTEXT CHUNKS:</div>
                    ${chunksHtml}
                </div>
            </div>
        `;

        chatFeed.appendChild(msgDiv);
        chatFeed.scrollTop = chatFeed.scrollHeight;

        // Update Guardrail Status Cards
        const g = data.guardrail || {};
        const safetyEl = document.getElementById('guard-status-safety');
        const relEl = document.getElementById('guard-status-relevance');
        const groundEl = document.getElementById('guard-status-grounding');
        if (safetyEl) safetyEl.textContent = g.is_safe ? 'PASS' : 'FAIL';
        if (relEl) relEl.textContent = g.is_relevant ? 'PASS' : 'REJECT';
        if (groundEl) groundEl.textContent = g.is_grounded ? 'PASS' : 'FAIL';
    }

    function appendErrorMessage() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'chat-message ai';
        msgDiv.innerHTML = `
            <div class="chat-avatar">🤖</div>
            <div class="chat-bubble-card">
                <div class="bubble-header">
                    <span class="bubble-sender">INDIC RAG ASSISTANT</span>
                    <span style="color: var(--amber-gold)">ERROR</span>
                </div>
                <div class="bubble-text">Pipeline execution failed. Please check network or try again.</div>
            </div>
        `;
        chatFeed.appendChild(msgDiv);
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }
});

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.innerText = text;
    return div.innerHTML;
}

// Global Helper Functions for Buttons
function toggleAccordion(id) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('hidden');
}

function copyText(text) {
    navigator.clipboard.writeText(text);
    alert('Answer copied to clipboard!');
}

function speakText(text) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
    } else {
        alert('Text-to-speech not supported in this browser.');
    }
}
