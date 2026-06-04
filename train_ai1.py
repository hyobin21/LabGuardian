import joblib
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler  # 🔥 성능 향상을 위한 스케일러 추가!

# 6가지 가스 이름 표준 매핑 (UCI 데이터셋 공식 ID 1~6 기준)
GAS_NAMES = [
    "에탄올 (C2H5OH)",     # ID 1 -> Index 0
    "에틸렌 (C2H4)",       # ID 2 -> Index 1
    "암모니아 (NH3)",      # ID 3 -> Index 2
    "아세톤 (CH3COCH3)",   # ID 4 -> Index 3
    "일산화탄소 (CO)",     # ID 5 -> Index 4
    "톨루엔 (C7H8)"        # ID 6 -> Index 5
]

# =================================================================
# [1단계] batch1.dat부터 batch10.dat까지 10개 파일 전부 불러오기
# =================================================================
print("🔄 다운로드받은 batch1.dat ~ batch10.dat 데이터를 모두 합쳐서 불러오는 중입니다...")

def load_all_batches():
    X_all = []
    y_all = []
    for i in range(1, 11):
        file_path = f"dataset/batch{i}.dat"  
        if not os.path.exists(file_path):
            print(f"⚠️ {file_path} 파일이 현재 폴더에 없어서 건너뜁니다.")
            continue
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.split()
                label = int(parts[0].split(';')[0])
                features = [float(part.split(':')[1]) for part in parts[1:]]
                X_all.append(features)
                y_all.append(label)
    return np.array(X_all), np.array(y_all)

X_raw, y_raw = load_all_batches()
print(f"✅ 전체 데이터 로드 완료! (통합 데이터 개수: {X_raw.shape[0]}개, 센서 개수: {X_raw.shape[1]}개)")

# =================================================================
# [2단계] 6가지 가스를 동시에! 무작위 비율로 섞어 30,000개 증강하기
# =================================================================
WANTED_NUM_MIXTURES = 30000
print(f"🧪 데이터 증강 시작: 10개 배치 데이터에서 6가지 가스를 복합 혼합하여 {WANTED_NUM_MIXTURES}개 생성 중...")

def generate_multi_gas_mixtures(X, y, num_mixtures):
    X_mixed = []
    y_mixed = [] 
    gas_groups = {i: X[y == i] for i in range(1, 7)}
    
    for _ in range(num_mixtures):
        ratios = np.random.dirichlet(np.ones(6))
        mixed_sample = np.zeros(X.shape[1])
        
        # 🔥 인덱스 꼬임 문제를 완벽하게 해결한 루프 구조
        for idx, gas_id in enumerate(range(1, 7)):
            rand_idx = np.random.randint(len(gas_groups[gas_id]))
            mixed_sample += gas_groups[gas_id][rand_idx] * ratios[idx]
            
        labels = ratios * 100  # 백분율(%) 단위로 변환
        X_mixed.append(mixed_sample)
        y_mixed.append(labels)
        
    return np.array(X_mixed), np.array(y_mixed)

X_synthetic, y_synthetic = generate_multi_gas_mixtures(X_raw, y_raw, num_mixtures=WANTED_NUM_MIXTURES)

# 🔥 [개선] 데이터 스케일링 적용 (센서 정규화로 오차율 최소화)
scaler = StandardScaler()
X_synthetic_scaled = scaler.fit_transform(X_synthetic)

# 학습용 80%, 테스트용 20% 분할
X_train, X_test, y_train, y_test = train_test_split(X_synthetic_scaled, y_synthetic, test_size=0.2, random_state=42)

# =================================================================
# [3단계] AI 모델 학습
# =================================================================
print(f"🤖 통합 혼합 데이터로 AI 모델(RandomForest Regressor) 학습 시작...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("🎯 AI 모델 학습 완료!")

# =================================================================
# [4단계] AI 성능 평가
# =================================================================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("\n==============================================")
print(f"📊 [10개 배치 통합 + 스케일링] AI 모델 예측 성적표")
print(f"👉 전체 가스 농도 평균 오차(MAE): 약 {mae:.2f}%")
print("==============================================\n")

# =================================================================
# [5단계] 실제 가스 이름 매핑 및 시뮬레이션 출력
# =================================================================
print("🔮 [실시간 웹사이트 연동 시뮬레이션 결과]")
sample_index = 0
real_ratio = y_test[sample_index]
predicted_ratio = y_pred[sample_index]

print("\n[📢 AI가 실시간으로 분석한 실험실 내부 가스 농도]")
print("--------------------------------------------------")
for i in range(6):
    print(f"☣️ {GAS_NAMES[i]} -> 실제: {real_ratio[i]:.1f}% | AI 예측: {predicted_ratio[i]:.1f}%")
print("--------------------------------------------------")

CRITICAL_THRESHOLD = 40.0  
alert_triggered = False
for i in range(6):
    if predicted_ratio[i] >= CRITICAL_THRESHOLD:
        print(f"\n🚨 [위험 경보] {GAS_NAMES[i]} 수치 위험! ({predicted_ratio[i]:.1f}% 감지 / 기준치 {CRITICAL_THRESHOLD}% 초과)")
        alert_triggered = True

if alert_triggered:
    # 사용자님이 안산에 계시므로 소방서 타겟을 안산소방서로 센스있게 변경!
    print("📢 [자동 신고 가동] 대피 사이렌을 울리고 안산소방서 및 관리자에게 즉시 GPS 위치를 송신합니다.")
else:
    print("\n🟢 현재 실험실 내부 공기 질 상태: 안전 (모든 가스 기준치 이하)")

# =================================================================
# [6단계] AI 모델 및 스케일러 파일 저장 (안전 코드 적용)
# =================================================================
try:
    # ⚠️ 중요: 웹사이트에서 예측할 때도 똑같이 스케일링을 해야 하므로 스케일러도 같이 저장합니다!
    joblib.dump(model, 'gas_model.pkl')
    joblib.dump(scaler, 'gas_scaler.pkl')
    print("\n💾 웹 연동용 AI 모델(gas_model.pkl) 및 스케일러(gas_scaler.pkl) 저장 완료!")
except Exception as e:
    print(f"\n❌ 파일 저장 실패: {e}")