import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

# 각지고 전문적인 엔터프라이즈 스타일 테마 커스텀 설정
st.set_page_config(page_title="LabGuardian™ 관제 콘솔", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght=400;700&family=Noto+Sans+KR:wght=400;700;900&display=swap');
        html, body, [data-testid="stWidgetLabel"] {
            font-family: 'Noto Sans KR', sans-serif !important;
        }
        .report-box {
            font-family: 'Noto Sans KR', sans-serif;
            border: 1px solid #CBD5E1;
            border-radius: 6px; 
            padding: 24px;
            background-color: #F8FAFC;
            color: #1E293B;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05);
        }
        .emergency-banner {
            border: 2px solid #EF4444;
            background-color: #FEF2F2;
            color: #991B1B;
            padding: 20px;
            border-radius: 4px;
            margin-bottom: 25px;
        }
        .platform-subtitle {
            font-size: 1.1rem;
            color: #475569;
            font-weight: 500;
            margin-top: -10px;
            margin-bottom: 20px;
        }
        /* 📋 사이드바 메뉴 가시성 극대화 커스텀 스타일 */
        div[data-testid="stSidebarUserContent"] div.stRadio > label {
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            color: #1E293B !important;
            padding: 8px 12px !important;
            background-color: #E2E8F0 !important;
            border-radius: 4px !important;
            display: block !important;
            margin-bottom: 12px !important;
        }
        div[data-testid="stSidebarUserContent"] div.stRadio div[role="radiogroup"] {
            background-color: #F8FAFC !important;
            padding: 15px !important;
            border: 2px solid #CBD5E1 !important;
            border-radius: 6px !important;
        }
        /* 📊 가스 도감용 격자형 테이블 스타일링 */
        .timeline-grid-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background-color: #FFFFFF;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05);
            border: 1px solid #E2E8F0;
        }
        .timeline-grid-table th {
            background-color: #F1F5F9;
            color: #334155;
            font-weight: 700;
            padding: 12px 14px;
            text-align: left;
            border: 1px solid #E2E8F0;
            font-size: 0.9rem;
        }
        .timeline-grid-table td {
            padding: 12px 14px;
            border: 1px solid #E2E8F0;
            color: #475569;
            font-size: 0.9rem;
            vertical-align: middle;
        }
        .gas-badge {
            display: inline-block;
            padding: 4px 8px;
            font-weight: 700;
            border-radius: 4px;
            font-size: 0.85rem;
            text-align: center;
        }
        /* ⏱️ 신규 모던 타임라인 디자인 요소 */
        .timeline-container {
            border-left: 3px solid #CBD5E1;
            padding-left: 20px;
            margin-left: 10px;
            margin-top: 20px;
        }
        .timeline-block {
            position: relative;
            margin-bottom: 25px;
            background: #FFFFFF;
            padding: 18px;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        }
        .timeline-time-badge {
            font-family: 'JetBrains Mono', sans-serif;
            font-weight: 700;
            font-size: 1.05rem;
            color: #1E293B;
            background-color: #F1F5F9;
            padding: 3px 10px;
            border-radius: 4px;
            display: inline-block;
            margin-bottom: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# 🕒 타임존 결합 상수 일원화
KST = pytz.timezone('Asia/Seoul')
live_now_kst = datetime.now(KST)

@st.cache_resource
def load_enterprise_models():
    g_model = joblib.load('gas_model.pkl')
    b_model = joblib.load('behavior_model.pkl')
    g_scaler = joblib.load('gas_scaler.pkl') 
    return g_model, b_model, g_scaler

try:
    gas_model, behavior_model, gas_scaler = load_enterprise_models()
except Exception as e:
    st.error(f"⚠️ 모델 및 스케일러 파일 로드 실패. 파일명을 확인해 주세요. 에러 내용: {e}")
    st.stop()

GAS_NAMES = ["에탄올", "에틸렌", "암모니아", "아세톤", "일산화탄소", "톨루엔"]
GAS_MASTER_DB = {
    "에탄올":      {"formula": "C₂H₅OH", "limits": [150.0, 400.0, 1000.0, 2000.0], "desc": "장기 노출 시 경미한 두통 유발."},
    "에틸렌":      {"formula": "C₂H₄",   "limits": [150.0, 400.0, 1000.0, 2000.0], "desc": "고농도 흡입 시 현기증 위험 유발."},
    "암모니아":    {"formula": "NH₃",    "limits": [25.0,  70.0,  150.0,  300.0],  "desc": "강한 자극성 가스로 점막 염증 유발."},
    "아세톤":      {"formula": "CH₃COCH₃","limits": [50.0,  150.0, 400.0,  800.0],  "desc": "지속 흡입 시 중추신경계 억제 전조 유발."},
    "일산화탄소":  {"formula": "CO",     "limits": [10.0,  30.0,  100.0,  200.0],  "desc": "헤모글로빈 결합 저산소증 위험 유발."},
    "톨루엔":      {"formula": "C₇H₈",   "limits": [15.0,  50.0,  150.0,  300.0],  "desc": "강력한 유기용매 신경독성 위험 유발."}
}

# ==============================================================================
# 💾 영구 세션 스토리지 초기화
# ==============================================================================
if 'time_series_buffer' not in st.session_state:
    st.session_state.time_series_buffer = [
        {"timestamp": live_now_kst - timedelta(minutes=10 * i), "status_vector": [0]*6} for i in range(6)
    ]

if 'daily_incident_db' not in st.session_state:
    st.session_state.daily_incident_db = {}

if 'weekly_counters' not in st.session_state:
    st.session_state.weekly_counters = {
        1: 4,  2: 2,  3: 1,  4: 1,  999: 0 
    }

if 'today_accumulated_ppm' not in st.session_state:
    st.session_state.today_accumulated_ppm = {
        "solvent": 110.0, "toxic": 85.0, "general": 42.0
    }

# ==============================================================================
# 🎛️ 사이드바 컨트롤러 및 리포트 메뉴 통합
# ==============================================================================
st.sidebar.markdown("### 🎛️ 가상 연동 제어 판넬")
input_total_ppm = st.sidebar.slider("🚨 실시간 전체 유출 농도 제어 (Total ppm)", 0.0, 2500.0, 350.0, step=10.0)

st.sidebar.markdown("---")
report_menu = st.sidebar.radio(
    "📋 LabGuardian 관제 메뉴판",
    ["🖥️ 실시간 모니터링 메인", "📅 일일 정밀 리포트 조회", "📊 주간 종합 브리핑 조회"]
)

# 물리 센서 시뮬레이션 데이터 가동
np.random.seed(live_now_kst.minute)
mock_s1 = float(np.clip(np.random.normal(2.2, 0.4), 0.1, 6.0))
mock_s2 = float(np.clip(np.random.normal(1.2, 0.3), 0.1, 6.0))
mock_s3 = float(np.clip(np.random.normal(1.7, 0.5), 0.1, 6.0))
is_night_time_flag = 1 if (live_now_kst.hour >= 22 or live_now_kst.hour < 5) else 0

sensor_features = [
    float(np.clip(mock_s1, 0.1, 6.0)), 
    float(np.clip(mock_s2, 0.1, 6.0)), 
    float(np.clip(mock_s3, 0.1, 6.0)),
    float(np.clip(mock_s1, 0.1, 6.0)),
    float(live_now_kst.hour),
    float(is_night_time_flag)
]

for i in range(6, 128):
    sensor_features.append(float(np.clip(np.random.normal(1.0, 0.2), 0.1, 5.0)))

raw_features_full = np.array([sensor_features])

# ==============================================================================
# 🤖 AI 1 차원 방어 및 수학적 ppm 정합성 고정 연산
# ==============================================================================
ai1_raw_outputs = None
try:
    scaled_features_full = gas_scaler.transform(raw_features_full)
    try:
        ai1_raw_outputs = gas_model.predict(scaled_features_full)[0]
    except Exception:
        scaled_features_6 = scaled_features_full[:, :6]
        ai1_raw_outputs = gas_model.predict(scaled_features_6)[0]
        
except Exception as e:
    np.random.seed(int(input_total_ppm) % 100 + 1)
    backup_ratios = np.abs(np.random.normal(1.0, 0.3, 6))
    ai1_raw_outputs = backup_ratios / np.sum(backup_ratios)

ai1_raw_outputs = np.clip(ai1_raw_outputs, 0, None)

if ai1_raw_outputs is not None and np.sum(ai1_raw_outputs) > 0:
    gas_proportions = ai1_raw_outputs / np.sum(ai1_raw_outputs) 
    ai1_pred_ppm = gas_proportions * input_total_ppm            
else:
    gas_proportions = np.array([1/6] * 6)
    ai1_pred_ppm = gas_proportions * input_total_ppm

# ==============================================================================
# 📊 실시간 차트 버퍼 구조
# ==============================================================================
if 'chart_history_df' not in st.session_state:
    st.session_state.chart_history_df = pd.DataFrame(
        [[10.0]*6 for _ in range(30)], 
        columns=GAS_NAMES, 
        index=[(live_now_kst - timedelta(seconds=i*2)).strftime("%H:%M:%S") for i in range(30)][::-1]
    )

current_time_str = live_now_kst.strftime("%H:%M:%S")
new_row = pd.DataFrame([ai1_pred_ppm], columns=GAS_NAMES, index=[current_time_str])
st.session_state.chart_history_df = pd.concat([st.session_state.chart_history_df, new_row]).tail(30)

# 가스 경보 매트릭스 및 개별 등급 가중치 산출 (🟢 안전, 🟡 주의, 🔴 위험, ⚫ 비상)
gas_status_matrix = {}
gas_status_scores = {} 
has_black_critical = False
is_any_warn = False
current_status_vector = []

for idx, name in enumerate(GAS_NAMES):
    val = ai1_pred_ppm[idx]
    lim = GAS_MASTER_DB[name]["limits"]
    if val < lim[0]: 
        color = "🟢 안전"; score = 0
    elif val < lim[1]: 
        color = "🟡 주의"; score = 1; is_any_warn = True
    elif val < lim[2]: 
        color = "🔴 위험"; score = 2; is_any_warn = True
    else: 
        color = "⚫ 비상"; score = 3; is_any_warn = True; has_black_critical = True
    gas_status_matrix[name] = color
    gas_status_scores[name] = score
    current_status_vector.append(score if score == 0 else 1)

# 진동 패턴 감지 연산
last_recorded_time = st.session_state.time_series_buffer[-1]["timestamp"]
if live_now_kst - last_recorded_time >= timedelta(minutes=10):
    st.session_state.time_series_buffer.append({"timestamp": live_now_kst, "status_vector": current_status_vector})
    if len(st.session_state.time_series_buffer) > 10: st.session_state.time_series_buffer.pop(0)

actual_oscillation_duration = 0.0
recent_6_nodes = st.session_state.time_series_buffer[-6:]
total_crossings = 0
for i in range(len(recent_6_nodes) - 1):
    diffs = np.abs(np.array(recent_6_nodes[i]["status_vector"]) - np.array(recent_6_nodes[i+1]["status_vector"]))
    total_crossings += np.sum(diffs)
if total_crossings >= 3:
    actual_oscillation_duration = 30.0

# ==============================================================================
# 🤖 AI 2 가동 ➡️ 하이브리드 판정
# ==============================================================================
if has_black_critical:
    predicted_group = 999 
elif not is_any_warn and actual_oscillation_duration == 0.0:
    predicted_group = 0
else:
    if gas_status_matrix["일산화탄소"] != "🟢 안전" or gas_status_matrix["톨루엔"] != "🟢 안전":
        predicted_group = 4
    else:
        ai2_vector = np.array([[
            float(ai1_pred_ppm[0]), float(ai1_pred_ppm[1]), float(ai1_pred_ppm[2]), 
            float(ai1_pred_ppm[3]), float(ai1_pred_ppm[4]), float(ai1_pred_ppm[5]), 
            float(actual_oscillation_duration), float(live_now_kst.hour), float(is_night_time_flag)
        ]])
        predicted_group = int(behavior_model.predict(ai2_vector)[0])

group_meta = {
    0: {"name": "정상 상태", "behavior": "실험실 내부 대기 인프라가 안전 범위 내에서 통제됨.", "recommend": "상시 표준 환기 시스템 가동.", "color": "🟢 안전", "badge_color": "#10B981", "bg": "#E6F4EA"},
    1: {"name": "유기용매 노출", "behavior": "폐액통 밀폐 상태 불량 혹은 유기용매 휘발 포착.", "recommend": "폐액 보관함 덮개 밀폐 여부 전수조사.", "color": "🟡 주의", "badge_color": "#F59E0B", "bg": "#FFF7ED"},
    2: {"name": "야간 잔류형", "behavior": "야간 무인 가동 장비 이상 유출 발생 후 가스 체류 중.", "recommend": "공조 배기 인프라 최고 출력 원격 시동.", "color": "🔴 위험", "badge_color": "#EF4444", "bg": "#FEF2F2"},
    3: {"name": "장기 노출형", "behavior": "초록-노랑 경계면 상에서 미세 누출 진동이 30분 이상 반복됨.", "recommend": "시약장 배관 피팅 노후화 크랙 점검 요망.", "color": "🔴 위험", "badge_color": "#DC2626", "bg": "#FEF2F2"},
    4: {"name": "독성가스 우세형", "behavior": "치명적 신경독성/질식 가스(CO, 톨루엔) 유출 감지.", "recommend": "메인 가스 실린더 솔레노이드 밸브 강제 셧다운.", "color": "🔴 위험", "badge_color": "#EF4444", "bg": "#FEF2F2"},
    999: {"name": "치명적 복합 유출", "behavior": "특정 가스가 인체 마비 임계치를 초과하여 관제판 전체에 최고 등급 비상 신호 감지됨.", "recommend": "실험실 내부 인원 즉시 대피 및 원격 솔레노이드 밸브 차단 강제 집행.", "color": "⚫ 비상", "badge_color": "#000000", "bg": "#FEF2F2"}
}

is_solvent_active = (gas_status_matrix["에탄올"] != "🟢 안전" or gas_status_matrix["아세톤"] != "🟢 안전")
is_toxic_active = (gas_status_matrix["일산화탄소"] != "🟢 안전" or gas_status_matrix["톨루엔"] != "🟢 안전")

# 메인 화면 전용 스마트 복합 안내 문구 결합 활성화
if predicted_group == 999:
    final_display_title = group_meta[999]["name"]
    final_display_behavior = group_meta[999]["behavior"]
    final_display_recommend = group_meta[999]["recommend"]
    final_display_color = group_meta[999]["color"]
    final_display_badge = group_meta[999]["badge_color"]
    final_display_bg = group_meta[999]["bg"]
elif is_toxic_active and is_solvent_active:
    final_display_title = f"⚠️ 복합 감지 ({group_meta[4]['name']} + {group_meta[1]['name']})"
    final_display_behavior = f"1. {group_meta[4]['behavior']}<br>2. {group_meta[1]['behavior']}"
    final_display_recommend = f"1. {group_meta[4]['recommend']}<br>2. {group_meta[1]['recommend']}"
    final_display_color = "🔴 위험"
    final_display_badge = "#EF4444"
    final_display_bg = "#FEF2F2"
else:
    final_display_title = group_meta[predicted_group]["name"]
    final_display_behavior = group_meta[predicted_group]["behavior"]
    final_display_recommend = group_meta[predicted_group]["recommend"]
    final_display_color = group_meta[predicted_group]["color"]
    final_display_badge = group_meta[predicted_group]["badge_color"]
    final_display_bg = group_meta[predicted_group]["bg"]

# ==============================================================================
# 🖥️ UI 메인 프레임 렌더링 시작
# ==============================================================================
st.title("🛡️ LabGuardian™ 통합 안전 관제 콘솔")
st.markdown('<p class="platform-subtitle">장기적인 화학물질 노출에 위험을 겪는 연구자를 위한 안전 관리 AI 플랫폼</p>', unsafe_allow_html=True)
st.markdown("---")

current_hour = live_now_kst.hour

if predicted_group == 999:
    st.markdown(f"""
        <div class="emergency-banner">
            <h2 style="margin:0; font-weight:900;">⚫ 최고등급 비상 상황 발생 (대피 프로토콜 활성화)</h2>
            <p style="font-size:1rem; margin:10px 0 10px 0;">특정 화학 물질 노출 수치가 안전 임계치를 초과했습니다. 즉시 현장에서 대피하십시오.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_em1, col_em2 = st.columns([2, 1])
    with col_em1:
        st.warning("⚠️ 긴급 프로토콜 작동 시 관할 소방서 상황실에 실시간 가스 농도 데이터가 즉시 무인 전송됩니다.")
    with col_em2:
        if st.button("🚨 [소방서 즉시 연동 및 솔레노이드 강제 셧다운]", use_container_width=True, type="primary"):
            st.error("🔒 [프로토콜 강제 집행] 119 소방 상황실 무인 신고가 접수되었으며, 솔레노이드 밸브가 즉시 차단되었습니다.")

# ------------------------------------------------------------------------------
# 1번 메뉴: 실시간 모니터링 메인 화면
# ------------------------------------------------------------------------------
if report_menu == "🖥️ 실시간 모니터링 메인":
    if predicted_group != 999:
        st.markdown(f"""
            <div style="background-color:{final_display_bg}; border-left:6px solid {final_display_badge}; padding:15px; border-radius:4px; margin-bottom:20px;">
                <span style="color:{final_display_badge}; font-weight:700; font-size:0.9rem;">🤖 AI 데이터 종합 진단</span>
                <h3 style="color:#0F172A; margin:4px 0 8px 0; font-weight:700;">{final_display_title} ({final_display_color})</h3>
                <p style="color:#334155; font-size:0.95rem; margin-bottom:6px;"><b>위험 패턴 분석:</b><br>{final_display_behavior}</p>
                <p style="color:{final_display_badge}; font-size:0.95rem; font-weight:700;"><b>추천 현장 조치:</b><br>{final_display_recommend}</p>
            </div>
        """, unsafe_allow_html=True)

    st.write("### 📈 실시간 화학물질별 농도 추적 그래프")
    st.line_chart(st.session_state.chart_history_df, height=240)

    st.write("#### 🧪 실시간 성분별 노출 세부 지표")
    g_cols = st.columns(6)
    for idx, name in enumerate(GAS_NAMES):
        with g_cols[idx]:
            ppm_val = ai1_pred_ppm[idx]
            pct_val = float(gas_proportions[idx] * 100)
            st.metric(
                label=f"{name} ({GAS_MASTER_DB[name]['formula']})",
                value=f"{ppm_val:.1f} ppm",
                delta=f"{gas_status_matrix[name]} ({pct_val:.1f}%)",
                delta_color="inverse" if "안전" not in gas_status_matrix[name] else "normal"
            )
            
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    st.write("### 📚 관제 화학물질별 MSDS 핵심 위험성 및 주의사항 도감")
    st.markdown("""
        <table class="timeline-grid-table">
            <thead>
                <tr>
                    <th style="width:12%; text-align:center;">물질명</th>
                    <th style="width:10%; text-align:center;">분자식</th>
                    <th style="width:13%; text-align:center;">관리 분류</th>
                    <th style="text-align:left;">인체 핵심 위험성 (MSDS 기반)</th>
                    <th style="text-align:left;">노출 시 비상 현장 조치 가이드</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="text-align:center; font-weight:700; background-color:#F8FAFC;">일산화탄소</td>
                    <td style="text-align:center; font-family:'JetBrains Mono'; font-weight:bold; color:#0284C7;">CO</td>
                    <td style="text-align:center;"><span class="gas-badge" style="background-color:#FEF2F2; color:#EF4444;">💀 치명적 질식가스</span></td>
                    <td>무색·무취로 감지가 극히 어려움. 혈액 내 헤모글로빈과 강력히 결합하여 체내 산소 운반을 차단하고 <b>급성 산소결핍증, 현기증, 의식 상실</b>을 유발.</td>
                    <td>즉시 해당 구역 흡입을 중단하고 신선한 공기가 있는 곳으로 이동. 의식 불명 시 즉시 산소호흡기 유무 확인 후 119 구급대 연동 필요.</td>
                </tr>
                <tr>
                    <td style="text-align:center; font-weight:700; background-color:#F8FAFC;">톨루엔</td>
                    <td style="text-align:center; font-family:'JetBrains Mono'; font-weight:bold; color:#0284C7;">C₇H₈</td>
                    <td style="text-align:center;"><span class="gas-badge" style="background-color:#FEF2F2; color:#EF4444;">🧠 가습성 신경독성</span></td>
                    <td>강력한 유기용매 증기. 흡입 시 중추신경계를 억제하여 <b>두통, 가슴 답답함, 보행 장애</b>를 일으키며 장기 누적 노출 시 환각 증세 및 뇌 신경계 영구 손상 유발 가능.</td>
                    <td>증기 마취성이 강하므로 즉시 송풍 처리를 진행하고 환자를 전방 안전 구역으로 대피. 피복에 묻었을 경우 오염된 옷을 탈의하고 전신 세척.</td>
                </tr>
                <tr>
                    <td style="text-align:center; font-weight:700; background-color:#F8FAFC;">암모니아</td>
                    <td style="text-align:center; font-family:'JetBrains Mono'; font-weight:bold; color:#0284C7;">NH₃</td>
                    <td style="text-align:center;"><span class="gas-badge" style="background-color:#FFF7ED; color:#C2410C;">🔥 점막 자극성</span></td>
                    <td>강한 강알칼리성 상부 기도 자극 가스. 아주 미량으로도 <b>눈물 고임, 격렬한 기침, 후두부 부종</b>을 유발하며 고농도 지속 노출 시 화학적 폐렴 발생 위험.</td>
                    <td>눈이나 피부에 접촉했을 경우 흐르는 물로 최소 15분 이상 유입 세척을 진행. 기도를 확보하고 비상 송풍 공조 체계를 즉각 기동할 것.</td>
                </tr>
                <tr>
                    <td style="text-align:center; font-weight:700; background-color:#F8FAFC;">아세톤</td>
                    <td style="text-align:center; font-family:'JetBrains Mono'; font-weight:bold; color:#0284C7;">CH₃COCH₃</td>
                    <td style="text-align:center;"><span class="gas-badge" style="background-color:#EFF6FF; color:#1D4ED8;">🧪 범용 유기용매</span></td>
                    <td>고휘발성 가연성 액체 증기. 대기 중에 쉽게 체류하며 <b>인후통, 가벼운 마취 전조, 점막 건조</b>를 일으킴. 장기 정체 시 화재 및 폭발 위험성 공존.</td>
                    <td>휘발 속도가 매우 빠르므로 인근 전열기 및 스파크 발생원을 즉시 차단하고, 실험실 내부 국소배기장치(Fume Hood) 플랩을 개방하여 확산을 방지.</td>
                </tr>
                <tr>
                    <td style="text-align:center; font-weight:700; background-color:#F8FAFC;">에탄올</td>
                    <td style="text-align:center; font-family:'JetBrains Mono'; font-weight:bold; color:#0284C7;">C₂H₅OH</td>
                    <td style="text-align:center;"><span class="gas-badge" style="background-color:#EFF6FF; color:#1D4ED8;">🧪 범용 유기용매</span></td>
                    <td>지속 흡입 시 <b>경미한 두통, 심박수 증가, 집중력 저하</b> 등 만성적인 업무 효율 저하를 유발하며 점막 유동성을 약화시킬 수 있음.</td>
                    <td>대량 휘발 구역에 장시간 잔류를 금지하고 신속하게 표준 공조 공기 순환을 유도하여 실내 적정 농도로 회복 처리.</td>
                </tr>
                <tr>
                    <td style="text-align:center; font-weight:700; background-color:#F8FAFC;">에틸렌</td>
                    <td style="text-align:center; font-family:'JetBrains Mono'; font-weight:bold; color:#0284C7;">C₂H₄</td>
                    <td style="text-align:center;"><span class="gas-badge" style="background-color:#F0FDF4; color:#166534;">🟢 단순 질식성</span></td>
                    <td>그 자체로 고유 독성은 낮으나 밀폐 공간 내 고농도 정착 시 산소 분압을 떨어뜨려 <b>단순 무산소 질식, 가벼운 현기증</b>을 유발 가능.</td>
                    <td>밀폐 보관실의 경우 산소 센서 농도를 병행 확인하고 외부 대기와의 상시 자연 환기 각도를 확보할 것.</td>
                </tr>
            </tbody>
        </table>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2번 메뉴: 일일 정밀 리포트 조회 화면 (★렌더링 버그 완전 패치 완료)
# ------------------------------------------------------------------------------
elif report_menu == "📅 일일 정밀 리포트 조회":
    st.markdown("### 📅 일일 정밀 안전 리포트 아카이브")
    st.write(f"**조회 일자:** {live_now_kst.strftime('%Y-%m-%d')} | **관제 대상:** 안산 메인 연구실")
    st.markdown("<p style='color:#64748B; font-size:0.95rem; margin-top:-5px;'>독성 가스와 유기용매 조건이 중첩될 경우, 내부 레이아웃에 복합 진단 데이터가 연동되어 아카이브됩니다.</p>", unsafe_allow_html=True)
    
    # 최상위 가스 추출 헬퍼 알고리즘
    def get_top_gas_by_rule(g_list, scores, values):
        if not g_list: return None
        sorted_g = sorted([(scores[g], values[GAS_NAMES.index(g)], g) for g in g_list], key=lambda x: (x[0], x[1]), reverse=True)
        return sorted_g[0][2]

    top_toxic = get_top_gas_by_rule(["일산화탄소", "톨루엔"], gas_status_scores, ai1_pred_ppm)
    top_solvent = get_top_gas_by_rule(["에탄올", "아세톤"], gas_status_scores, ai1_pred_ppm)
    top_general = get_top_gas_by_rule(["에틸렌", "암모니아"], gas_status_scores, ai1_pred_ppm)

    # 📝 리포트 반영하기 버튼 클릭 시 로직
    if st.button("📝 [현재 시각 데이터 일일 리포트에 반영하기]", use_container_width=True):
        if current_hour not in st.session_state.daily_incident_db:
            st.session_state.daily_incident_db[current_hour] = []
            
        if predicted_group == 0:
            st.session_state.daily_incident_db[current_hour].append({
                "is_compound": False,
                "type_name": "정상 상태",
                "gas": top_general,
                "grade": gas_status_matrix[top_general],
                "reason": "대기질 데이터 분석 결과 안정 범위 기록",
                "action": "상시 표준 공조 환기 인프라 가동 자동 유지"
            })
        elif is_toxic_active and is_solvent_active:
            st.session_state.weekly_counters[4] += 1
            st.session_state.weekly_counters[1] += 1
            
            group_id = f"COMP_{datetime.now().strftime('%M%S')}"
            st.session_state.daily_incident_db[current_hour].append({
                "is_compound": True,
                "group_id": group_id,
                "type_name": "복합 유출 : 독성가스 우세형",
                "gas": top_toxic,
                "grade": gas_status_matrix[top_toxic],
                "reason": group_meta[4]["behavior"],
                "action": group_meta[4]["recommend"]
            })
            st.session_state.daily_incident_db[current_hour].append({
                "is_compound": True,
                "group_id": group_id,
                "type_name": "복합 유출 : 유기용매 노출",
                "gas": top_solvent,
                "grade": gas_status_matrix[top_solvent],
                "reason": group_meta[1]["behavior"],
                "action": group_meta[1]["recommend"]
            })
        else:
            rep_gas = top_toxic if is_toxic_active else top_solvent if is_solvent_active else top_general
            st.session_state.daily_incident_db[current_hour].append({
                "is_compound": False,
                "type_name": group_meta[predicted_group]["name"],
                "gas": rep_gas,
                "grade": gas_status_matrix[rep_gas],
                "reason": group_meta[predicted_group]["behavior"],
                "action": group_meta[predicted_group]["recommend"]
            })
            st.session_state.weekly_counters[predicted_group] += 1
            
        st.toast("🚨 현재 시각 데이터가 리포트 저장소에 영구 추가 누적되었습니다!", icon="✅")

    # ⏱️ 타임라인 출력 프레임 기동
    st.markdown('<div class="timeline-container">', unsafe_allow_html=True)
    
    for h in range(24):
        if h > current_hour:
            continue
            
        st.markdown(f'<div class="timeline-block">', unsafe_allow_html=True)
        st.markdown(f'<span class="timeline-time-badge">⏱️ {h:02d}:00</span>', unsafe_allow_html=True)
        
        # 1. 고정(Fix) 및 누적 완료된 데이터 안전 렌더링 파트
        if h in st.session_state.daily_incident_db and len(st.session_state.daily_incident_db[h]) > 0:
            items = st.session_state.daily_incident_db[h]
            rendered_idx_set = set()
            
            for idx, item in enumerate(items):
                if idx in rendered_idx_set:
                    continue
                    
                # 복합 상황 묶음 렌더링 구역 (Streamlit 순정 컨테이너 활용)
                if item.get("is_compound", False):
                    g_id = item.get("group_id")
                    sub_comp_cards = [items[k] for k in range(len(items)) if items[k].get("group_id") == g_id]
                    
                    with st.container(border=True):
                        st.caption("🔗 **[AI 하이브리드 진단] 복합 요인 연동 검출 기록**")
                        
                        for c_idx, c_item in enumerate(sub_comp_cards):
                            emoji_badge = "⚫ 비상" if "비상" in c_item['grade'] else "🔴 위험" if "위험" in c_item['grade'] else "🟡 주의"
                            
                            st.markdown(f"### {emoji_badge} <span style='color:#4F46E5;'>[{c_item['type_name']}] {c_item['gas']}</span>", unsafe_allow_html=True)
                            st.write(f"**분석 요인:** {c_item['reason']}")
                            st.markdown(f"**현장 조치:** <span style='color:#1E3A8A;'>{c_item['action']}</span>", unsafe_allow_html=True)
                            
                            if c_idx < len(sub_comp_cards) - 1:
                                st.markdown("---")
                                
                    for k in range(len(items)):
                        if items[k].get("group_id") == g_id:
                            rendered_idx_set.add(k)
                
                # 단일 상황 안전 렌더링 구역
                else:
                    emoji_badge = "⚫ 비상" if "비상" in item['grade'] else "🔴 위험" if "위험" in item['grade'] else "🟡 주의" if "주의" in item['grade'] else "🟢 정상"
                    with st.container(border=True):
                        st.markdown(f"### {emoji_badge} [{item['type_name']}] {item['gas']}")
                        st.write(f"**분석 요인:** {item['reason']}")
                        st.markdown(f"**현장 조치:** <span style='color:#1E3A8A;'>{item['action']}</span>", unsafe_allow_html=True)
                    rendered_idx_set.add(idx)
                    
        # 2. 버튼을 누르기 전 실시간 동적 스트리밍 뷰 안전 렌더링 파트
        else:
            if h == current_hour:
                if is_toxic_active and is_solvent_active:
                    with st.container(border=True):
                        st.caption("🔗 **[실시간 동적 추론] 복합 감지 연동 레이아웃 (미저장)**")
                        
                        # 독성가스 파트
                        st.markdown(f"### 🟡 [복합 유출 : 독성가스 우세형] {top_toxic}")
                        st.write(f"**분석 요인:** {group_meta[4]['behavior']}")
                        st.markdown(f"**현장 조치:** <span style='color:#1E3A8A;'>{group_meta[4]['recommend']}</span>", unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # 유기용매 파트
                        st.markdown(f"### 🟡 [복합 유출 : 유기용매 노출] {top_solvent}")
                        st.write(f"**분석 요인:** {group_meta[1]['behavior']}")
                        st.markdown(f"**현장 조치:** <span style='color:#1E3A8A;'>{group_meta[1]['recommend']}</span>", unsafe_allow_html=True)
                else:
                    rep_gas = top_toxic if is_toxic_active else top_solvent if is_solvent_active else top_general
                    emoji_badge = "⚫ 비상" if "비상" in gas_status_matrix[rep_gas] else "🔴 위험" if "위험" in gas_status_matrix[rep_gas] else "🟡 주의" if "주의" in gas_status_matrix[rep_gas] else "🟢 정상"
                    
                    with st.container(border=True):
                        st.markdown(f"### {emoji_badge} [{group_meta[predicted_group]['name']}] {rep_gas} <span style='color:#64748B; font-size:0.8rem;'>(실시간 추론)</span>", unsafe_allow_html=True)
                        st.write(f"**분석 요인:** {group_meta[predicted_group]['behavior']}")
                        st.markdown(f"**현장 조치:** <span style='color:#1E3A8A;'>{group_meta[predicted_group]['recommend']}</span>", unsafe_allow_html=True)
            else:
                # 과거 시간대 안전 렌더링 구역
                with st.container(border=True):
                    st.markdown("### 🟢 [정상 상태] 에틸렌 <span style='color:#64748B; font-size:0.8rem;'>(AI2 백그라운드 보존 기록)</span>", unsafe_allow_html=True)
                    st.write("**분석 요인:** 실험실 내부 대기 인프라가 안전 범위 내에서 통제됨.")
                    st.write("**현장 조치:** 상시 표준 환기 시스템 가동.")
                    
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3번 메뉴: 주간 종합 브리핑 리포트 조회 화면
# ------------------------------------------------------------------------------
elif report_menu == "📊 주간 종합 브리핑 조회":
    total_live_incidents = sum(st.session_state.weekly_counters.values())
    dynamic_score = max(45, 85 - int(input_total_ppm / 45))

    st.markdown(f"""
    <div class="report-box">
        <h3 style="color:#0F172A; margin-top:0; font-weight:700;">📊 주간 연구실 안전성 종합 브리핑 통계</h3>
        <p style="color:#64748B; font-size:0.95rem;">장기 축적 노출 프로필 데이터를 실시간 연동하여 판정한 주간 최종 스코어입니다.</p>
        <div style="display:flex; align-items:baseline; margin:15px 0;">
            <span style="font-size:1.1rem; font-weight:bold; margin-right:10px;">🛡️ 연구실 주간 종합 신뢰도 점수:</span>
            <span style="color:{'#10B981' if dynamic_score >= 80 else '#F59E0B' if dynamic_score >= 60 else '#EF4444'}; font-size:2.8rem; font-weight:900;">{dynamic_score}점</span>
        </div>
        <hr style="border-color:#E2E8F0; margin:15px 0;">
        <h5 style="color:#334155; font-size:1rem; margin-bottom:8px;">⚠️ 이번 주 유출 위험 이벤트 누적 통계 (총 {total_live_incidents}건)</h5>
        <ul style="font-size:0.95rem; line-height:1.7; color:#475569;">
            <li>🧪 유기용매 노출 유출 위험: <b>{st.session_state.weekly_counters[1]}건</b></li>
            <li>🌙 야간 장비 가동 잔류 위험: <b>{st.session_state.weekly_counters[2]}건</b></li>
            <li>📈 미세 장기 지속 누출 위험: <b>{st.session_state.weekly_counters[3]}건</b></li>
            <li>🚨 독성가스 농도 우세 위험: <b>{st.session_state.weekly_counters[4]}건</b></li>
            <li>⚫ 임계 초과 복합 비상 위험: <b>{st.session_state.weekly_counters[999]}건</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("#### 📆 주간 3대 가스군 노출 종합 추이 분석")
    chart_days = [(live_now_kst - timedelta(days=i)).strftime("%m/%d") for i in range(7)][::-1]
    weekly_sim_data = pd.DataFrame({
        "유기용매군(아세톤/에탄올)": [120.0, 145.0, 310.0, 150.0, 90.0, 110.0, st.session_state.today_accumulated_ppm["solvent"]],
        "독성가스군(CO/톨루엔)":   [15.0,  12.0,  45.0,  18.0,  11.0,  85.0,  st.session_state.today_accumulated_ppm["toxic"]],
        "일반가스군(암모니아/에틸렌)": [35.0,  40.0,  22.0,  55.0,  30.0,  42.0,  st.session_state.today_accumulated_ppm["general"]]
    }, index=chart_days)
    st.area_chart(weekly_sim_data, height=240)