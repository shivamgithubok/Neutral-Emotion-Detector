const socket = io();
const ctx = document.getElementById('statsChart').getContext('2d');

// Chart Setup
const statsChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Tracked Persons',
            data: [],
            borderColor: '#00d2ff',
            backgroundColor: 'rgba(0, 210, 255, 0.1)',
            borderWidth: 2,
            tension: 0.4,
            fill: true
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                ticks: { color: '#94a3b8' }
            },
            x: {
                grid: { display: false },
                ticks: { display: false }
            }
        }
    }
});

// UI Elements
const numTrackedEl = document.getElementById('numTracked');
const fallRisksEl = document.getElementById('fallRisks');
const primaryExprEl = document.getElementById('primaryExpression');
const exprConfEl = document.getElementById('expressionConf');
const fpsEl = document.getElementById('fps');
const eventLogEl = document.getElementById('eventLog');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const modelSelect = document.getElementById('modelSelect');
const videoFeedEl = document.getElementById('videoFeed');

// Frame Listener
socket.on('video_frame', (data) => {
    videoFeedEl.src = `data:image/jpeg;base64,${data.image}`;
});

// WebSocket Handlers
socket.on('stats_update', (data) => {
    numTrackedEl.textContent = data.num_tracked;
    fpsEl.textContent = `FPS: ${data.fps}`;
    
    // Count fall risks (high fall score)
    const risks = data.persons ? data.persons.filter(p => p.fall_score > 0.5).length : 0;
    fallRisksEl.textContent = risks;

    // Update Expression UI
    if (data.persons && data.persons.length > 0) {
        const primary = data.persons[0];
        primaryExprEl.textContent = primary.expression;
        exprConfEl.textContent = `${Math.round(primary.expr_conf * 100)}%`;
        
        // Dynamic color based on expression
        if (primary.expression === 'HAPPY' || primary.expression === 'LAUGHING') {
            primaryExprEl.style.color = '#00ff8c';
        } else if (primary.expression === 'ANGRY' || primary.expression === 'SAD') {
            primaryExprEl.style.color = '#ff4d4d';
        } else {
            primaryExprEl.style.color = '#fff';
        }
    } else {
        primaryExprEl.textContent = 'NONE';
        exprConfEl.textContent = '0%';
        primaryExprEl.style.color = '#fff';
    }

    // Update Chart
    const now = new Date().toLocaleTimeString();
    statsChart.data.labels.push(now);
    statsChart.data.datasets[0].data.push(data.num_tracked);
    
    if (statsChart.data.labels.length > 20) {
        statsChart.data.labels.shift();
        statsChart.data.datasets[0].data.shift();
    }
    statsChart.update('none');
});

socket.on('event_update', (event) => {
    addEventToLog(event);
});

function addEventToLog(event) {
    const item = document.createElement('div');
    item.className = `event-item ${event.type}`;
    
    let msg = '';
    if (event.type === 'pose') {
        msg = `T${event.track_id} changed to ${event.state}`;
    } else if (event.type === 'expr') {
        msg = `T${event.track_id} expressed ${event.expr}`;
    } else {
        msg = `${event.state}: ${event.expr}`;
    }

    item.innerHTML = `
        <span class="event-time">${event.ts}</span>
        <span class="event-msg">${msg}</span>
    `;
    
    eventLogEl.prepend(item);
    if (eventLogEl.children.length > 50) {
        eventLogEl.lastChild.remove();
    }
}

// Button Handlers
startBtn.onclick = () => {
    fetch('/api/start')
        .then(r => r.json())
        .then(d => console.log('Engine started'));
};

stopBtn.onclick = () => {
    fetch('/api/stop')
        .then(r => r.json())
        .then(d => console.log('Engine stopped'));
};

modelSelect.onchange = () => {
    const model = modelSelect.value;
    fetch(`/api/set_model/${model}`)
        .then(r => r.json())
        .then(d => {
            console.log(`Switched to ${model} model:`, d.status);
            if (d.status === 'success') {
                addEventToLog({
                    type: 'system',
                    ts: new Date().toLocaleTimeString(),
                    track_id: 'SYS',
                    state: 'Model Switched',
                    expr: d.model
                });
            }
        });
};

socket.on('status', (data) => {
    console.log('Server Status:', data.data);
});
