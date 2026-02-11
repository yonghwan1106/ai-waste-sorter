/**
 * AI 스마트 분리수거 로봇 - 대시보드 JavaScript
 * 실시간 통계 업데이트 + Chart.js 차트
 */

const POLL_INTERVAL = 2000; // 2초마다 통계 갱신
const CHART_COLORS = [
    '#f97316', // 플라스틱 - 주황
    '#a8a29e', // 캔 - 은색
    '#a16207', // 종이 - 갈색
    '#22c55e', // 유리 - 초록
    '#facc15', // 비닐 - 노랑
    '#6b7280', // 일반 - 회색
];
const CLASS_NAMES = ['플라스틱', '캔', '종이', '유리병', '비닐', '일반쓰레기'];

let pieChart = null;
let barChart = null;

// === 차트 초기화 ===
function initCharts() {
    const pieCtx = document.getElementById('pieChart').getContext('2d');
    pieChart = new Chart(pieCtx, {
        type: 'doughnut',
        data: {
            labels: CLASS_NAMES,
            datasets: [{
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: CHART_COLORS,
                borderColor: '#1e293b',
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8', padding: 12, font: { size: 12 } },
                },
            },
        },
    });

    const barCtx = document.getElementById('barChart').getContext('2d');
    barChart = new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: CLASS_NAMES,
            datasets: [{
                label: '분류 개수',
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: CHART_COLORS.map(c => c + 'cc'),
                borderColor: CHART_COLORS,
                borderWidth: 1,
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' },
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: '#94a3b8', stepSize: 1 },
                    grid: { color: '#334155' },
                },
            },
            plugins: {
                legend: { display: false },
            },
        },
    });
}

// === 통계 업데이트 ===
async function fetchStats() {
    try {
        const res = await fetch('/api/stats');
        if (!res.ok) throw new Error('API 응답 오류');
        const data = await res.json();
        updateUI(data);
        setStatus(true);
    } catch (e) {
        setStatus(false);
    }
}

function updateUI(data) {
    // 총 개수
    document.getElementById('total-count').textContent = data.total || 0;

    // 분류별 개수
    const counts = data.counts || {};
    document.getElementById('count-plastic').textContent = counts['플라스틱'] || 0;
    document.getElementById('count-can').textContent = counts['캔'] || 0;
    document.getElementById('count-paper').textContent = counts['종이'] || 0;
    document.getElementById('count-glass').textContent = counts['유리병'] || 0;
    document.getElementById('count-vinyl').textContent = counts['비닐'] || 0;
    document.getElementById('count-general').textContent = counts['일반쓰레기'] || 0;

    // FPS
    const fps = data.fps || 0;
    document.getElementById('fps-display').textContent = `FPS: ${fps.toFixed(1)}`;

    // 최근 감지
    if (data.last_detection) {
        document.getElementById('last-class').textContent = data.last_detection.class;
        document.getElementById('last-conf').textContent =
            (data.last_detection.confidence * 100).toFixed(1) + '%';
    }

    // 차트 업데이트
    const chartData = CLASS_NAMES.map(name => counts[name] || 0);

    if (pieChart) {
        pieChart.data.datasets[0].data = chartData;
        pieChart.update('none');
    }

    if (barChart) {
        barChart.data.datasets[0].data = chartData;
        barChart.update('none');
    }
}

function setStatus(connected) {
    const dot = document.getElementById('status-indicator');
    const text = document.getElementById('status-text');
    if (connected) {
        dot.classList.add('active');
        text.textContent = '연결됨';
    } else {
        dot.classList.remove('active');
        text.textContent = '연결 끊김';
    }
}

// === 초기화 ===
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    fetchStats();
    setInterval(fetchStats, POLL_INTERVAL);
});
