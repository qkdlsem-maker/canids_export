import json, os

here = os.path.dirname(__file__)
data = json.load(open(os.path.join(here, "..", "models", "dashboard_data.json"), encoding="utf-8"))

template = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>CAN 버스 실시간 침입탐지 데모</title>
<style>
body{font-family:'Malgun Gothic',sans-serif;background:#f4f4f2;margin:0;padding:24px;}
.wrap{max-width:820px;margin:0 auto;}
h1{font-size:20px;margin-bottom:4px;}
.sub{color:#666;font-size:13px;margin-bottom:20px;}
.status{display:flex;align-items:center;gap:8px;margin-bottom:16px;font-weight:600;}
.dot{width:12px;height:12px;border-radius:50%;background:#4a9;}
.dot.alert{background:#d33;}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px;}
.card{background:#fff;border-radius:10px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.08);}
.card .label{font-size:12px;color:#777;}
.card .value{font-size:26px;font-weight:700;margin-top:4px;}
.card .value.danger{color:#c0392b;}
table{width:100%;background:#fff;border-collapse:collapse;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08);}
th,td{padding:8px 12px;text-align:left;font-size:13px;border-bottom:1px solid #eee;}
th{background:#fafafa;color:#666;font-weight:600;}
tr.attack{background:#fdecea;}
.badge{padding:2px 8px;border-radius:6px;font-size:12px;font-weight:600;}
.badge.attack{background:#f5c6cb;color:#721c24;}
.badge.normal{background:#d4edda;color:#155724;}
.footer{margin-top:16px;font-size:12px;color:#999;}
</style></head>
<body><div class="wrap">
<h1>차량 CAN 버스 실시간 침입탐지 데모</h1>
<div class="sub">에스엘(주) 과제 - LightGBM(알려진 공격) + Mahalanobis(미지 공격 Zero-day) 하이브리드 파이프라인 실제 추론 결과 재생</div>
<div class="status"><span class="dot" id="dot"></span><span id="statusText">정상 모니터링 중</span></div>
<div class="metrics">
  <div class="card"><div class="label">처리 메시지</div><div class="value" id="count">0</div></div>
  <div class="card"><div class="label">탐지된 공격</div><div class="value danger" id="attacks">0</div></div>
  <div class="card"><div class="label">평균 지연시간(실측)</div><div class="value" id="latency">0.000ms</div></div>
</div>
<table><thead><tr><th>실제 라벨</th><th>모델 판정</th><th>지연(ms)</th><th>Mahalanobis 거리</th></tr></thead>
<tbody id="log"></tbody></table>
<div class="footer">모델: LightGBM(F1=1.0000, 709KB) + Mahalanobis(1.3KB) · 합산 메모리 0.694MB · Zero-day 탐지율 99.9~100%, FPR 1.99%</div>
</div>
<script>
const EVENTS = __DATA__;
let i=0,count=0,attacks=0,latSum=0;
const logEl=document.getElementById('log'),dot=document.getElementById('dot'),st=document.getElementById('statusText');
function tick(){
  const e=EVENTS[i%EVENTS.length]; i++;
  count++; latSum+=e.latency_ms;
  document.getElementById('count').textContent=count.toLocaleString();
  document.getElementById('latency').textContent=(latSum/count).toFixed(3)+'ms';
  const tr=document.createElement('tr');
  let badge;
  if(e.status==='attack'){
    attacks++; document.getElementById('attacks').textContent=attacks;
    badge='<span class="badge attack">'+e.attack_name+' 탐지</span>';
    tr.className='attack';
    dot.className='dot alert'; st.textContent=e.attack_name+' 공격 탐지됨 - 경보 발령';
    setTimeout(()=>{dot.className='dot'; st.textContent='정상 모니터링 중';},700);
  } else {
    badge='<span class="badge normal">정상</span>';
  }
  tr.innerHTML='<td>'+e.true_label+'</td><td>'+badge+'</td><td>'+e.latency_ms+'</td><td>'+e.maha_dist+'</td>';
  logEl.prepend(tr);
  while(logEl.children.length>12) logEl.removeChild(logEl.lastChild);
}
setInterval(tick,600); tick();
</script>
</body></html>
"""

html = template.replace("__DATA__", json.dumps(data, ensure_ascii=False))
out_path = os.path.join(here, "can_ids_demo.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("생성됨:", out_path)
