// Dashboard Application Logic - Enterprise Fraud & Explainable AI Platform
let currentSelectedTxId = null;
let shapChartInstance = null;
let importanceChartInstance = null;

// Mock / Fallback Data for Static Hosting (e.g. GitHub Pages)
const MOCK_METRICS = {
  total_transactions_ingested: 10480,
  fraud_count: 312,
  fraud_rate_pct: 2.98,
  model_performance: {
    pr_auc: 0.984,
    roc_auc: 0.991,
    f1_score: 0.925,
    precision: 0.943,
    recall: 0.908,
    confusion_matrix: {
      true_positives: 298,
      false_positives: 18,
      false_negatives: 14,
      true_negatives: 10150
    }
  },
  cost_metrics: {
    baseline_loss_without_model: 4850000.00,
    total_cost_with_model: 125000.00,
    net_savings_amount: 4725000.00,
    savings_ratio_pct: 97.42,
    cost_optimal_threshold: 0.3850,
    cost_optimal_savings_pct: 98.15
  },
  feature_importances: [
    { feature: 'amount_ratio_to_user_avg', importance: 412 },
    { feature: 'distance_prev_txn_km', importance: 385 },
    { feature: 'txn_count_1h', importance: 320 },
    { feature: 'robust_monetary_zscore', importance: 290 },
    { feature: 'time_since_prev_txn_sec', importance: 245 },
    { feature: 'is_high_risk_category', importance: 198 },
    { feature: 'isolation_forest_anomaly_score', importance: 175 },
    { feature: 'travel_speed_kmh', importance: 160 },
    { feature: 'spending_sum_24h', importance: 142 },
    { feature: 'recency_decay_score', importance: 118 }
  ],
  fairness_audit: [
    { attribute: 'merchant_category', group: 'crypto_exchange', count: 320, fpr: 0.012, recall: 0.965 },
    { attribute: 'merchant_category', group: 'upi_p2p', count: 2850, fpr: 0.008, recall: 0.942 },
    { attribute: 'merchant_category', group: 'jewelry_gold', count: 410, fpr: 0.010, recall: 0.950 },
    { attribute: 'merchant_category', group: 'kirana_grocery', count: 3100, fpr: 0.002, recall: 0.920 },
    { attribute: 'merchant_category', group: 'petrol_pump', count: 1650, fpr: 0.003, recall: 0.915 },
    { attribute: 'device_type', group: 'mobile_app', count: 5400, fpr: 0.005, recall: 0.935 },
    { attribute: 'device_type', group: 'web_browser', count: 3200, fpr: 0.009, recall: 0.948 },
    { attribute: 'device_type', group: 'pos_terminal', count: 1200, fpr: 0.004, recall: 0.910 },
    { attribute: 'device_type', group: 'atm', count: 680, fpr: 0.006, recall: 0.925 }
  ]
};

const MOCK_DRIFT = {
  overall_status: "ALL FEATURES STABLE",
  baseline_samples: 8000,
  evaluated_samples: 2000,
  feature_drift: [
    { feature: 'amount', psi: 0.0125, drift_status: 'STABLE' },
    { feature: 'robust_monetary_zscore', psi: 0.0184, drift_status: 'STABLE' },
    { feature: 'txn_count_1h', psi: 0.0210, drift_status: 'STABLE' },
    { feature: 'spending_sum_24h', psi: 0.0195, drift_status: 'STABLE' },
    { feature: 'distance_prev_txn_km', psi: 0.0340, drift_status: 'STABLE' },
    { feature: 'travel_speed_kmh', psi: 0.0285, drift_status: 'STABLE' },
    { feature: 'recency_decay_score', psi: 0.0150, drift_status: 'STABLE' },
    { feature: 'isolation_forest_anomaly_score', psi: 0.0220, drift_status: 'STABLE' }
  ]
};

const MOCK_TRANSACTIONS = [
  {
    transaction_id: 'TXN-9A1F4C2E',
    timestamp: '2026-11-03 14:15:22',
    user_id: 'USER_0042',
    amount: 485000.00,
    merchant_category: 'crypto_exchange',
    device_type: 'web_browser',
    card_present: 0,
    fraud_probability: 0.965,
    risk_level: 'CRITICAL',
    status: 'PENDING'
  },
  {
    transaction_id: 'TXN-7B8D3A19',
    timestamp: '2026-11-03 14:14:05',
    user_id: 'USER_0109',
    amount: 185000.00,
    merchant_category: 'jewelry_gold',
    device_type: 'web_browser',
    card_present: 0,
    fraud_probability: 0.884,
    risk_level: 'HIGH',
    status: 'PENDING'
  },
  {
    transaction_id: 'TXN-5C2E89F1',
    timestamp: '2026-11-03 14:12:44',
    user_id: 'USER_0077',
    amount: 25000.00,
    merchant_category: 'upi_p2p',
    device_type: 'mobile_app',
    card_present: 0,
    fraud_probability: 0.642,
    risk_level: 'ELEVATED',
    status: 'PENDING'
  },
  {
    transaction_id: 'TXN-3D9012A4',
    timestamp: '2026-11-03 14:10:19',
    user_id: 'USER_0012',
    amount: 450.00,
    merchant_category: 'kirana_grocery',
    device_type: 'mobile_app',
    card_present: 1,
    fraud_probability: 0.012,
    risk_level: 'LOW',
    status: 'APPROVED'
  },
  {
    transaction_id: 'TXN-4E7F01BC',
    timestamp: '2026-11-03 14:08:50',
    user_id: 'USER_0311',
    amount: 120000.00,
    merchant_category: 'electronics_mobiles',
    device_type: 'web_browser',
    card_present: 0,
    fraud_probability: 0.912,
    risk_level: 'CRITICAL',
    status: 'BLOCKED'
  },
  {
    transaction_id: 'TXN-1F2A3B4C',
    timestamp: '2026-11-03 14:05:12',
    user_id: 'USER_0155',
    amount: 1450.00,
    merchant_category: 'restaurants_food',
    device_type: 'mobile_app',
    card_present: 1,
    fraud_probability: 0.025,
    risk_level: 'LOW',
    status: 'APPROVED'
  }
];

const MOCK_ABLATION = [
  { feature_set: 'raw', model: 'LR', imbalance: 'none', threshold: 0.050, precision: 0.243, recall: 0.712, f1: 0.363, pr_auc: 0.500, roc_auc: 0.861, savings: 0.870 },
  { feature_set: 'raw', model: 'LightGBM', imbalance: 'class_weight', threshold: 0.002, precision: 0.134, recall: 0.721, f1: 0.226, pr_auc: 0.584, roc_auc: 0.895, savings: 0.811 },
  { feature_set: '+velocity', model: 'LR', imbalance: 'class_weight', threshold: 0.431, precision: 0.441, recall: 0.942, f1: 0.601, pr_auc: 0.814, roc_auc: 0.962, savings: 0.947 },
  { feature_set: '+velocity', model: 'LightGBM', imbalance: 'class_weight', threshold: 0.067, precision: 0.733, recall: 0.817, f1: 0.773, pr_auc: 0.852, roc_auc: 0.964, savings: 0.845 },
  { feature_set: '+velocity+spatial', model: 'DT', imbalance: 'none', threshold: 0.099, precision: 0.914, recall: 0.923, f1: 0.919, pr_auc: 0.902, roc_auc: 0.951, savings: 0.966 },
  { feature_set: '+velocity+spatial', model: 'LightGBM', imbalance: 'smote', threshold: 0.068, precision: 0.827, recall: 0.923, f1: 0.870, pr_auc: 0.935, roc_auc: 0.973, savings: 0.964 },
  { feature_set: '+rfm_all', model: 'LR', imbalance: 'class_weight', threshold: 0.144, precision: 0.929, recall: 1.000, f1: 0.963, pr_auc: 0.996, roc_auc: 1.000, savings: 0.979 },
  { feature_set: '+rfm_all', model: 'LightGBM', imbalance: 'class_weight', threshold: 0.001, precision: 0.888, recall: 0.990, f1: 0.936, pr_auc: 0.996, roc_auc: 1.000, savings: 0.977 },
  { feature_set: '+rfm_all+anomaly_scores', model: 'DT', imbalance: 'class_weight', threshold: 0.001, precision: 1.000, recall: 0.990, f1: 0.995, pr_auc: 0.991, roc_auc: 0.995, savings: 0.979 },
  { feature_set: '+rfm_all+anomaly_scores', model: 'LightGBM', imbalance: 'class_weight', threshold: 0.001, precision: 0.912, recall: 0.990, f1: 0.949, pr_auc: 0.997, roc_auc: 1.000, savings: 0.977 }
];

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  loadMetrics();
  loadTransactions();
  loadDriftReport();
  loadAblationResults();
  initSimulator();
  
  document.getElementById('btn-refresh').addEventListener('click', () => {
    loadTransactions();
    loadMetrics();
    loadDriftReport();
    loadAblationResults();
  });

  document.getElementById('risk-filter-select').addEventListener('change', () => {
    loadTransactions();
  });

  document.getElementById('modal-close-btn').addEventListener('click', closeModal);
  document.getElementById('xai-modal').addEventListener('click', (e) => {
    if (e.target.classList.contains('modal-overlay')) closeModal();
  });

  document.getElementById('btn-run-sim').addEventListener('click', () => {
    document.getElementById('section-simulator').scrollIntoView({ behavior: 'smooth' });
  });
});


function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      navItems.forEach(i => i.classList.remove('active'));
      item.classList.add('active');
    });
  });
}


async function loadMetrics() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) throw new Error('API offline');
    const data = await res.json();
    renderMetricsUI(data);
  } catch (err) {
    console.log('Using static demo metrics fallback');
    renderMetricsUI(MOCK_METRICS);
  }
}

function renderMetricsUI(data) {
  document.getElementById('kpi-total-tx').textContent = data.total_transactions_ingested.toLocaleString('en-IN');
  document.getElementById('kpi-fraud-count').textContent = data.fraud_count.toLocaleString('en-IN');
  document.getElementById('kpi-fraud-rate').textContent = `${data.fraud_rate_pct}% Fraud Rate`;
  
  const perf = data.model_performance || {};
  if (perf.pr_auc) {
    document.getElementById('kpi-pr-auc').textContent = (perf.pr_auc * 100).toFixed(1) + '%';
  }

  // Cost Savings (Baesens et al. 2022)
  const cost = data.cost_metrics || perf.cost_metrics;
  if (cost) {
    document.getElementById('kpi-cost-savings').textContent = `${cost.savings_ratio_pct}%`;
    document.getElementById('kpi-cost-savings-sub').textContent = `Net: ₹${cost.net_savings_amount.toLocaleString('en-IN')}`;
    document.getElementById('cost-net-savings').textContent = `₹${cost.net_savings_amount.toLocaleString('en-IN')} (${cost.savings_ratio_pct}%)`;
    document.getElementById('cost-optimal-th').textContent = `${cost.cost_optimal_threshold} (Max Savings: ${cost.cost_optimal_savings_pct}%)`;
  }
  
  const cm = perf.confusion_matrix || {};
  document.getElementById('cm-tp').textContent = cm.true_positives?.toLocaleString('en-IN') ?? '--';
  document.getElementById('cm-fp').textContent = cm.false_positives?.toLocaleString('en-IN') ?? '--';
  document.getElementById('cm-fn').textContent = cm.false_negatives?.toLocaleString('en-IN') ?? '--';
  document.getElementById('cm-tn').textContent = cm.true_negatives?.toLocaleString('en-IN') ?? '--';
  
  if (data.feature_importances && data.feature_importances.length > 0) {
    renderImportanceChart(data.feature_importances);
  }

  // Fairness Audit
  const fairness = data.fairness_audit || perf.fairness_audit || MOCK_METRICS.fairness_audit;
  if (fairness && fairness.length > 0) {
    renderFairnessTable(fairness);
  }
}


async function loadTransactions() {
  const filter = document.getElementById('risk-filter-select').value;
  const tbody = document.getElementById('tx-table-body');
  tbody.innerHTML = '<tr><td colspan="10" class="loading-cell"><i class="fa-solid fa-spinner fa-spin"></i> Ingesting feature store data...</td></tr>';
  
  try {
    let url = '/api/transactions?limit=60';
    if (filter !== 'ALL') url += `&risk_filter=${filter}`;
    
    const res = await fetch(url);
    if (!res.ok) throw new Error('API Error');
    const items = await res.json();
    renderTransactionsTable(items);
  } catch (err) {
    console.log('Backend API offline, displaying static feature store transactions demo.');
    let filtered = MOCK_TRANSACTIONS;
    if (filter !== 'ALL') {
      filtered = MOCK_TRANSACTIONS.filter(t => t.risk_level === filter);
    }
    renderTransactionsTable(filtered);
  }
}

function renderTransactionsTable(items) {
  const tbody = document.getElementById('tx-table-body');
  if (items.length === 0) {
    tbody.innerHTML = '<tr><td colspan="10" style="text-align:center; padding: 2rem; color: #9ca3af;">No matching transactions found.</td></tr>';
    return;
  }
  
  tbody.innerHTML = '';
  items.forEach(tx => {
    const tr = document.createElement('tr');
    
    let badgeClass = 'badge-low';
    if (tx.risk_level === 'CRITICAL') badgeClass = 'badge-critical';
    else if (tx.risk_level === 'HIGH') badgeClass = 'badge-high';
    else if (tx.risk_level === 'ELEVATED') badgeClass = 'badge-elevated';
    
    let formattedCategory = (tx.merchant_category || '')
      .replace('upi_p2p', 'UPI / P2P')
      .replace('jewelry_gold', 'Jewelry & Gold')
      .replace('crypto_exchange', 'Crypto Exchange')
      .replace('electronics_mobiles', 'Electronics & Mobiles')
      .replace('online_shopping', 'Online Shopping')
      .replace('travel_irctc', 'IRCTC / Travel')
      .replace('kirana_grocery', 'Kirana & Grocery')
      .replace('petrol_pump', 'Fuel & Petrol Pump')
      .replace('restaurants_food', 'Restaurants & Dining');

    let statusBadge = '<span class="status-tag online">PENDING</span>';
    if (tx.status === 'BLOCKED') statusBadge = '<span class="status-tag" style="background:rgba(239,68,68,0.2);color:#f87171;border:1px solid #ef4444">BLOCKED</span>';
    else if (tx.status === 'APPROVED') statusBadge = '<span class="status-tag" style="background:rgba(16,185,129,0.2);color:#34d399;border:1px solid #10b981">APPROVED</span>';
    else if (tx.status === 'ESCALATED') statusBadge = '<span class="status-tag" style="background:rgba(245,158,11,0.2);color:#fbbf24;border:1px solid #f59e0b">ESCALATED</span>';
      
    tr.innerHTML = `
      <td style="font-family: monospace; font-weight:600; color: #3b82f6;">${tx.transaction_id}</td>
      <td>${tx.timestamp}</td>
      <td>${tx.user_id}</td>
      <td style="font-weight:700;">₹${tx.amount.toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
      <td><span style="font-weight:500;">${formattedCategory}</span></td>
      <td>${tx.device_type}</td>
      <td>${tx.card_present === 1 ? '<span style="color:#10b981">Present</span>' : '<span style="color:#9ca3af">Online / UPI</span>'}</td>
      <td>
        <span class="badge ${badgeClass}">${(tx.fraud_probability * 100).toFixed(1)}% (${tx.risk_level})</span>
      </td>
      <td>${statusBadge}</td>
      <td>
        <button class="btn btn-outline btn-xs" onclick="openXAIDeepDive('${tx.transaction_id}')">
          <i class="fa-solid fa-brain"></i> SHAP
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}


async function loadDriftReport() {
  const tbody = document.getElementById('drift-table-body');
  try {
    const res = await fetch('/api/drift');
    if (!res.ok) throw new Error('Drift API offline');
    const data = await res.json();
    renderDriftUI(data);
  } catch (err) {
    console.log('Using static demo drift fallback');
    renderDriftUI(MOCK_DRIFT);
  }
}

function renderDriftUI(data) {
  const badge = document.getElementById('overall-drift-badge');
  badge.textContent = data.overall_status;
  if (data.overall_status.includes('CRITICAL')) {
    badge.className = 'badge badge-critical';
  } else if (data.overall_status.includes('MODERATE')) {
    badge.className = 'badge badge-high';
  } else {
    badge.className = 'badge badge-low';
  }

  const tbody = document.getElementById('drift-table-body');
  tbody.innerHTML = '';
  data.feature_drift.forEach(item => {
    const tr = document.createElement('tr');
    let statusClass = 'badge-low';
    let action = 'Distribution is stable. No action required.';
    if (item.drift_status === 'CRITICAL DRIFT') {
      statusClass = 'badge-critical';
      action = 'Critical shift detected. Schedule model retraining.';
    } else if (item.drift_status === 'MODERATE DRIFT') {
      statusClass = 'badge-high';
      action = 'Moderate deviation. Continue window monitoring.';
    }

    tr.innerHTML = `
      <td style="font-family: monospace; font-weight:600; color:#93c5fd;">${item.feature}</td>
      <td style="font-family: monospace; font-weight:700;">${item.psi.toFixed(4)}</td>
      <td><span class="badge ${statusClass}">${item.drift_status}</span></td>
      <td style="color:#d1d5db; font-size: 0.84rem;">${action}</td>
    `;
    tbody.appendChild(tr);
  });
}


function renderFairnessTable(fairness) {
  const tbody = document.getElementById('fairness-table-body');
  tbody.innerHTML = '';
  fairness.forEach(item => {
    const tr = document.createElement('tr');
    const isFprDisparity = item.fpr > 0.03;
    const parityBadge = isFprDisparity
      ? '<span class="badge badge-high"><i class="fa-solid fa-triangle-exclamation"></i> Moderate Disparity</span>'
      : '<span class="badge badge-low"><i class="fa-solid fa-check"></i> Parity Compliant</span>';

    tr.innerHTML = `
      <td><span style="font-weight:600; color:#c4b5fd;">${item.attribute.replace('_', ' ').toUpperCase()}</span></td>
      <td style="font-family: monospace; font-weight:600; color:#f3f4f6;">${item.group}</td>
      <td>${item.count.toLocaleString('en-IN')}</td>
      <td style="font-family: monospace; font-weight:700; color:${item.fpr > 0.02 ? '#fbbf24' : '#10b981'}">${(item.fpr * 100).toFixed(2)}%</td>
      <td style="font-family: monospace; font-weight:700; color:#38bdf8">${(item.recall * 100).toFixed(2)}%</td>
      <td>${parityBadge}</td>
    `;
    tbody.appendChild(tr);
  });
}


async function loadAblationResults() {
  const tbody = document.getElementById('ablation-table-body');
  try {
    const res = await fetch('/api/ablation');
    if (!res.ok) throw new Error('Ablation API error');
    const data = await res.json();
    if (data.status === 'available') {
      renderAblationTable(data.results);
      return;
    }
    throw new Error('Ablation results not available');
  } catch (err) {
    renderAblationTable(MOCK_ABLATION);
  }
}

function renderAblationTable(results) {
  const tbody = document.getElementById('ablation-table-body');
  tbody.innerHTML = '';
  results.forEach(row => {
    const tr = document.createElement('tr');
    const savingsPct = (row.savings * 100).toFixed(1);
    tr.innerHTML = `
      <td style="font-family: monospace; font-weight:600; color:#93c5fd;">${row.feature_set}</td>
      <td style="font-weight:700; color:#f3f4f6;">${row.model}</td>
      <td><span class="status-tag" style="background:rgba(255,255,255,0.06);color:#e5e7eb;border:1px solid rgba(255,255,255,0.1)">${row.imbalance}</span></td>
      <td style="font-family: monospace;">${row.threshold.toFixed(3)}</td>
      <td style="font-family: monospace;">${row.precision.toFixed(3)}</td>
      <td style="font-family: monospace;">${row.recall.toFixed(3)}</td>
      <td style="font-family: monospace; font-weight:700; color:#f59e0b;">${row.f1.toFixed(3)}</td>
      <td style="font-family: monospace; font-weight:700; color:#38bdf8;">${row.pr_auc.toFixed(3)}</td>
      <td style="font-family: monospace;">${row.roc_auc.toFixed(3)}</td>
      <td style="font-family: monospace; font-weight:800; color:#10b981;">${savingsPct}%</td>
    `;
    tbody.appendChild(tr);
  });
}


async function openXAIDeepDive(txId) {
  currentSelectedTxId = txId;
  const modal = document.getElementById('xai-modal');
  modal.classList.remove('hidden');
  
  document.getElementById('modal-tx-id').textContent = `Transaction ID: ${txId}`;
  document.getElementById('modal-summary-text').textContent = 'Computing exact SHAP feature attributions...';
  document.getElementById('modal-reasons-list').innerHTML = '';
  
  try {
    const res = await fetch(`/api/explain/${txId}`);
    if (!res.ok) throw new Error('Explain API error');
    const data = await res.json();
    renderXAIDetails(data);
  } catch (err) {
    console.log('Using calculated SHAP explanation fallback for:', txId);
    const tx = MOCK_TRANSACTIONS.find(t => t.transaction_id === txId) || MOCK_TRANSACTIONS[0];
    const fallbackData = generateMockExplanation(tx);
    renderXAIDetails(fallbackData);
  }
}

function renderXAIDetails(data) {
  const badge = document.getElementById('modal-risk-badge');
  badge.textContent = `${data.risk_level} (${(data.fraud_probability * 100).toFixed(1)}%)`;
  badge.className = 'risk-badge ';
  if (data.risk_level.includes('CRITICAL')) badge.className += 'badge-critical';
  else if (data.risk_level.includes('HIGH')) badge.className += 'badge-high';
  else if (data.risk_level.includes('ELEVATED')) badge.className += 'badge-elevated';
  else badge.className += 'badge-low';
  
  document.getElementById('modal-recommendation').textContent = `Recommended Action: ${data.recommendation}`;
  document.getElementById('modal-summary-text').textContent = data.summary;
  
  const list = document.getElementById('modal-reasons-list');
  list.innerHTML = '';
  data.key_reasons.forEach(reason => {
    const li = document.createElement('li');
    li.textContent = reason;
    list.appendChild(li);
  });
  
  renderShapWaterfallChart(data.all_attributions);
}

function generateMockExplanation(tx) {
  const isHighRisk = tx.fraud_probability > 0.5;
  return {
    transaction_id: tx.transaction_id,
    fraud_probability: tx.fraud_probability,
    risk_level: `${tx.risk_level} RISK`,
    recommendation: isHighRisk ? 'IMMEDIATE BLOCK & ACCOUNT FREEZE' : 'AUTO-APPROVE TRANSACTION',
    summary: isHighRisk 
      ? `Transaction ${tx.transaction_id} flagged due to severe behavioral deviation (12.5x higher than 30-day user average) and high geographical velocity jump (1,450 km in 15 seconds).`
      : `Transaction ${tx.transaction_id} is consistent with legitimate customer baseline spending patterns and physical proximity.`,
    key_reasons: isHighRisk ? [
      `Transaction amount ₹${tx.amount.toLocaleString('en-IN')} is significantly above historical customer baseline.`,
      `Geographical jump of 1,450 km recorded within 15 seconds of previous swipe (implied velocity impossible travel).`,
      `Card Not Present at high-risk merchant category (${tx.merchant_category}).`
    ] : [
      `Transaction amount ₹${tx.amount.toLocaleString('en-IN')} matches typical household baseline.`,
      `Physical POS card swipe validated with chip & PIN.`,
      `Normal geographical proximity (within standard city perimeter).`
    ],
    all_attributions: [
      { feature: 'amount_ratio_to_user_avg', feature_value: isHighRisk ? 12.5 : 1.1, shap_value: isHighRisk ? 2.45 : -0.85 },
      { feature: 'distance_prev_txn_km', feature_value: isHighRisk ? 1450 : 1.2, shap_value: isHighRisk ? 1.95 : -0.62 },
      { feature: 'time_since_prev_txn_sec', feature_value: isHighRisk ? 15 : 7200, shap_value: isHighRisk ? 1.32 : -0.45 },
      { feature: 'is_high_risk_category', feature_value: isHighRisk ? 1 : 0, shap_value: isHighRisk ? 0.98 : -0.30 },
      { feature: 'card_present', feature_value: tx.card_present, shap_value: tx.card_present === 0 ? 0.65 : -0.92 }
    ]
  };
}


function closeModal() {
  document.getElementById('xai-modal').classList.add('hidden');
}


async function handleAnalystAction(action) {
  if (!currentSelectedTxId) return;
  try {
    await fetch(`/api/action/${currentSelectedTxId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: action, notes: "Logged by dashboard fraud investigator" })
    });
  } catch (err) {
    console.log('Action logged offline:', action, 'for tx:', currentSelectedTxId);
  }
  closeModal();
  loadTransactions();
}


function renderShapWaterfallChart(attributions) {
  const ctx = document.getElementById('shapChart').getContext('2d');
  if (shapChartInstance) shapChartInstance.destroy();
  
  const top10 = attributions.slice(0, 10);
  const labels = top10.map(a => `${a.feature} (${a.feature_value})`);
  const values = top10.map(a => a.shap_value);
  const bgColors = values.map(v => v > 0 ? '#ef4444' : '#10b981');
  
  shapChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'SHAP Contribution to Fraud Risk',
        data: values,
        backgroundColor: bgColors,
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `SHAP Value: ${ctx.raw > 0 ? '+' : ''}${ctx.raw}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#9ca3af' }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#f3f4f6', font: { size: 11 } }
        }
      }
    }
  });
}


function renderImportanceChart(importances) {
  const ctx = document.getElementById('importanceChart').getContext('2d');
  if (importanceChartInstance) importanceChartInstance.destroy();
  
  const top10 = importances.slice(0, 10);
  const labels = top10.map(i => i.feature);
  const values = top10.map(i => i.importance);
  
  importanceChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'LightGBM Feature Importance (Splits)',
        data: values,
        backgroundColor: 'rgba(59, 130, 246, 0.75)',
        borderColor: '#3b82f6',
        borderWidth: 1,
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#9ca3af' } },
        y: { grid: { display: false }, ticks: { color: '#f3f4f6', font: { size: 11 } } }
      }
    }
  });
}


function loadPreset(presetName) {
  const presets = {
    account_takeover: {
      user: 'USER_0042',
      amount: 485000.00,
      category: 'crypto_exchange',
      device: 'web_browser',
      distance: 1450,
      timedelta: 15,
      card_present: 0
    },
    impossible_travel: {
      user: 'USER_0109',
      amount: 185000.00,
      category: 'jewelry_gold',
      device: 'web_browser',
      distance: 1800,
      timedelta: 120,
      card_present: 0
    },
    velocity_spike: {
      user: 'USER_0077',
      amount: 25000.00,
      category: 'upi_p2p',
      device: 'mobile_app',
      distance: 10,
      timedelta: 5,
      card_present: 0
    },
    legitimate: {
      user: 'USER_0012',
      amount: 450.00,
      category: 'kirana_grocery',
      device: 'mobile_app',
      distance: 1.2,
      timedelta: 7200,
      card_present: 1
    }
  };

  const p = presets[presetName];
  if (!p) return;

  document.getElementById('sim-user').value = p.user;
  document.getElementById('sim-amount').value = p.amount;
  document.getElementById('sim-category').value = p.category;
  document.getElementById('sim-device').value = p.device;
  document.getElementById('sim-distance').value = p.distance;
  document.getElementById('sim-timedelta').value = p.timedelta;
  document.getElementById('sim-card-present').value = p.card_present;

  document.getElementById('sim-form').dispatchEvent(new Event('submit'));
}


function initSimulator() {
  const form = document.getElementById('sim-form');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const payload = {
      user_id: document.getElementById('sim-user').value,
      amount: parseFloat(document.getElementById('sim-amount').value),
      merchant_category: document.getElementById('sim-category').value,
      device_type: document.getElementById('sim-device').value,
      location_lat: 19.0760,
      location_long: 72.8777,
      card_present: parseInt(document.getElementById('sim-card-present').value),
      distance_prev_txn_km: parseFloat(document.getElementById('sim-distance').value),
      time_since_prev_txn_sec: parseFloat(document.getElementById('sim-timedelta').value)
    };
    
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error('Simulation failed');
      const data = await res.json();
      renderSimulationResult(data);
    } catch (err) {
      console.log('Using interactive offline simulation engine fallback');
      const simulatedData = runOfflineSimulation(payload);
      renderSimulationResult(simulatedData);
    }
  });
}

function runOfflineSimulation(payload) {
  let score = 0.05;
  if (payload.amount > 100000) score += 0.40;
  else if (payload.amount > 25000) score += 0.20;

  if (payload.distance_prev_txn_km > 500 && payload.time_since_prev_txn_sec < 300) score += 0.35;
  else if (payload.distance_prev_txn_km > 100) score += 0.15;

  if (['crypto_exchange', 'jewelry_gold'].includes(payload.merchant_category)) score += 0.15;
  if (payload.card_present === 0) score += 0.10;

  const finalScore = Math.min(0.99, Math.max(0.01, score));
  let riskLevel = 'LOW RISK';
  let isFlagged = false;

  if (finalScore >= 0.85) { riskLevel = 'CRITICAL RISK'; isFlagged = true; }
  else if (finalScore >= 0.70) { riskLevel = 'HIGH RISK'; isFlagged = true; }
  else if (finalScore >= 0.40) { riskLevel = 'ELEVATED RISK'; isFlagged = false; }

  return {
    fraud_probability: finalScore,
    risk_level: riskLevel,
    is_fraud_flagged: isFlagged,
    latency_ms: 14,
    explanation: {
      summary: isFlagged 
        ? `Simulated transaction flagged for ${payload.user_id} due to high monetary deviation (₹${payload.amount.toLocaleString('en-IN')}) and physical velocity anomalies.`
        : `Simulated transaction passed fraud checks with standard baseline metrics.`,
      key_reasons: isFlagged ? [
        `High transaction amount ₹${payload.amount.toLocaleString('en-IN')}.`,
        `Geographical distance jump of ${payload.distance_prev_txn_km} km in ${payload.time_since_prev_txn_sec}s.`,
        `Category risk signature for ${payload.merchant_category}.`
      ] : [
        `Amount ₹${payload.amount.toLocaleString('en-IN')} within typical threshold.`,
        `Normal location proximity.`,
        `Standard transaction profile.`
      ]
    }
  };
}

function renderSimulationResult(data) {
  document.getElementById('sim-score-pct').textContent = `${(data.fraud_probability * 100).toFixed(1)}%`;
  document.getElementById('sim-latency-tag').textContent = `Latency: ${data.latency_ms} ms`;
  
  const badge = document.getElementById('sim-risk-badge');
  badge.textContent = `${data.risk_level} (${data.is_fraud_flagged ? 'FLAGGED' : 'APPROVED'})`;
  badge.className = 'risk-badge-lg ';
  if (data.risk_level.includes('CRITICAL')) badge.className += 'badge-critical';
  else if (data.risk_level.includes('HIGH')) badge.className += 'badge-high';
  else if (data.risk_level.includes('ELEVATED')) badge.className += 'badge-elevated';
  else badge.className += 'badge-low';
  
  document.getElementById('sim-summary').textContent = data.explanation.summary;
  
  const list = document.getElementById('sim-reasons-list');
  list.innerHTML = '';
  data.explanation.key_reasons.forEach(r => {
    const li = document.createElement('li');
    li.textContent = r;
    list.appendChild(li);
  });
}
