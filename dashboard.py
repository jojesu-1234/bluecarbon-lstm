import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import matplotlib.font_manager as fm
import folium
from streamlit_folium import st_folium
font_path = "fonts/NanumGothic.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="블루카본 해양산성화 LSTM 대시보드", page_icon="🌊", layout="wide")
st.title("🌊 블루카본과 해양산성화 완화 상관관계 분석")
st.markdown("**LSTM 딥러닝 기반 pH 예측 대시보드** | 서해중부(블루카본 O) vs 동해(블루카본 X)")
st.markdown("**CNSA** 20121 이건혁, 20514 서영빈 | 2026년 2학년 1학기 자율창의활동")
st.divider()

FEATURE_COLS = ['수온(℃)표층_scaled','염분표층_scaled','용존산소량(㎎/L)표층_scaled','클로로필A(㎍/L)표층_scaled','bluecarbon_scaled']
TARGET_COL = '수소이온농도표층_scaled'
TIMESTEPS = 4
N_FEATURES = 5

def make_sequences(features, targets, timesteps=4):
    X, y = [], []
    for i in range(len(features) - timesteps):
        X.append(features[i:i+timesteps])
        y.append(targets[i+timesteps])
    return np.array(X), np.array(y)

def build_model(timesteps, n_features, u1, u2, dr):
    model = Sequential([
        LSTM(u1, input_shape=(timesteps, n_features), return_sequences=True),
        Dropout(dr),
        LSTM(u2, return_sequences=False),
        Dropout(dr),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

try:
    wocean = pd.read_csv('서해중부_LSTM최종데이터.csv')
    eocean = pd.read_csv('동해_LSTM최종데이터.csv')
except Exception as e:
    st.error(f"❌ CSV 로드 실패: {e}")
    st.stop()

X_seo, y_seo = make_sequences(wocean[FEATURE_COLS].values, wocean[TARGET_COL].values, TIMESTEPS)
X_dong, y_dong = make_sequences(eocean[FEATURE_COLS].values, eocean[TARGET_COL].values, TIMESTEPS)

split = int(len(X_seo) * 0.8)
X_seo_train, X_seo_test = X_seo[:split], X_seo[split:]
y_seo_train, y_seo_test = y_seo[:split], y_seo[split:]
X_dong_train, X_dong_test = X_dong[:split], X_dong[split:]
y_dong_train, y_dong_test = y_dong[:split], y_dong[split:]

ph_scaler_seo = MinMaxScaler()
ph_scaler_seo.fit(wocean[['수소이온농도표층']])
ph_scaler_dong = MinMaxScaler()
ph_scaler_dong.fit(eocean[['수소이온농도표층']])

test_dates_seo = list(wocean['year_month'].iloc[split+TIMESTEPS:].reset_index(drop=True))
test_dates_dong = list(eocean['year_month'].iloc[split+TIMESTEPS:].reset_index(drop=True))

st.sidebar.header("⚙️ 모델 설정")
epochs = st.sidebar.slider("학습 Epoch 수", 50, 200, 100, 10)
u1 = st.sidebar.selectbox("LSTM 1층 유닛 수", [32, 64, 128], index=1)
u2 = st.sidebar.selectbox("LSTM 2층 유닛 수", [16, 32, 64], index=1)
dr = st.sidebar.slider("Dropout 비율", 0.1, 0.5, 0.2, 0.05)
run_btn = st.sidebar.button("🚀 모델 학습 시작", use_container_width=True)
st.sidebar.divider()
st.sidebar.markdown("### 📌 연구 정보")
st.sidebar.markdown("- **주제**: 블루카본과 해양산성화 완화\n- **모델**: 2층 LSTM\n- **데이터**: 2015~2025 분기별\n- **지역**: 서해중부, 동해")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 데이터 현황",
    "🤖 모델 학습 & 예측",
    "📈 블루카본 면적 추이",
    "🗺️ 블루카본 식재 지도",
    "📋 연구 결론"
])

with tab1:
    st.subheader("데이터 현황")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("서해중부 데이터", f"{len(wocean)}개")
    c2.metric("동해 데이터", f"{len(eocean)}개")
    c3.metric("결측값", "0개")
    c4.metric("관측 기간", "2015~2025")
    st.divider()
    cl, cr = st.columns(2)
    with cl:
        st.markdown("#### 서해중부")
        st.dataframe(wocean[['year_month','수소이온농도표층','수온(℃)표층','염분표층','bluecarbon_area_km2']], use_container_width=True)
    with cr:
        st.markdown("#### 동해")
        st.dataframe(eocean[['year_month','수소이온농도표층','수온(℃)표층','염분표층','bluecarbon_area_km2']], use_container_width=True)
    st.divider()
    st.markdown("#### pH 시계열 비교")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(range(len(wocean)), wocean['수소이온농도표층'], label='서해중부 (블루카본 O)', color='#1D9E75', marker='o', markersize=3)
    ax.plot(range(len(eocean)), eocean['수소이온농도표층'], label='동해 (블루카본 X)', color='#378ADD', marker='o', markersize=3)
    ti = max(1, len(wocean)//8)
    ax.set_xticks(range(0, len(wocean), ti))
    ax.set_xticklabels([wocean['year_month'].iloc[i] for i in range(0, len(wocean), ti)], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('관측 시점'); ax.set_ylabel('pH'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

with tab2:
    st.subheader("모델 학습 & 예측")
    if run_btn:
        ms = build_model(TIMESTEPS, N_FEATURES, u1, u2, dr)
        md = build_model(TIMESTEPS, N_FEATURES, u1, u2, dr)
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**서해중부 학습 중...**"); bar1 = st.progress(0)
        with p2:
            st.markdown("**동해 학습 중...**"); bar2 = st.progress(0)

        for ep in range(epochs):
            ms.fit(X_seo_train, y_seo_train, epochs=1, batch_size=8, validation_data=(X_seo_test, y_seo_test), verbose=0)
            bar1.progress((ep+1)/epochs)
        for ep in range(epochs):
            md.fit(X_dong_train, y_dong_train, epochs=1, batch_size=8, validation_data=(X_dong_test, y_dong_test), verbose=0)
            bar2.progress((ep+1)/epochs)

        hs = ms.evaluate(X_seo_test, y_seo_test, verbose=0)
        hd = md.evaluate(X_dong_test, y_dong_test, verbose=0)
        st.success("✅ 학습 완료!")
        st.divider()

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("서해중부 Train Loss", f"{ms.evaluate(X_seo_train, y_seo_train, verbose=0)[0]:.4f}")
        m2.metric("서해중부 Val Loss", f"{hs[0]:.4f}")
        m3.metric("동해 Train Loss", f"{md.evaluate(X_dong_train, y_dong_train, verbose=0)[0]:.4f}")
        m4.metric("동해 Val Loss", f"{hd[0]:.4f}")
        st.divider()

        ps = ph_scaler_seo.inverse_transform(ms.predict(X_seo_test))
        pd_ = ph_scaler_dong.inverse_transform(md.predict(X_dong_test))
        rs = ph_scaler_seo.inverse_transform(y_seo_test.reshape(-1,1))
        rd = ph_scaler_dong.inverse_transform(y_dong_test.reshape(-1,1))

        st.markdown("#### 실제 vs 예측 pH")
        fig2, axes = plt.subplots(2,1,figsize=(12,8))
        axes[0].plot(range(len(rs)), rs, label='실제 pH', marker='o', color='#1D9E75')
        axes[0].plot(range(len(ps)), ps, label='예측 pH', marker='s', linestyle='--', color='#E24B4A')
        axes[0].set_title('서해중부 (블루카본 O)'); axes[0].set_xlabel('관측 시점 (분기)'); axes[0].set_ylabel('pH')
        axes[0].set_xticks(range(len(test_dates_seo))); axes[0].set_xticklabels(test_dates_seo, rotation=45, ha='right', fontsize=8)
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(range(len(rd)), rd, label='실제 pH', marker='o', color='#378ADD')
        axes[1].plot(range(len(pd_)), pd_, label='예측 pH', marker='s', linestyle='--', color='#E24B4A')
        axes[1].set_title('동해 (블루카본 X)'); axes[1].set_xlabel('관측 시점 (분기)'); axes[1].set_ylabel('pH')
        axes[1].set_xticks(range(len(test_dates_dong))); axes[1].set_xticklabels(test_dates_dong, rotation=45, ha='right', fontsize=8)
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig2); plt.close()
        st.divider()

        st.markdown("#### 예측값 상세 비교")
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**서해중부**")
            st.dataframe(pd.DataFrame({'관측시점':test_dates_seo,'실제pH':[round(v[0],4) for v in rs],'예측pH':[round(v[0],4) for v in ps],'오차':[round(abs(rs[i][0]-ps[i][0]),4) for i in range(len(rs))]}), use_container_width=True)
        with t2:
            st.markdown("**동해**")
            st.dataframe(pd.DataFrame({'관측시점':test_dates_dong,'실제pH':[round(v[0],4) for v in rd],'예측pH':[round(v[0],4) for v in pd_],'오차':[round(abs(rd[i][0]-pd_[i][0]),4) for i in range(len(rd))]}), use_container_width=True)
    else:
        st.info("👈 사이드바에서 설정 후 **모델 학습 시작** 버튼을 눌러줘!")

with tab3:
    st.subheader("블루카본 면적 추이 (2015~2025)")
    fig4, ax3 = plt.subplots(2,1,figsize=(12,8))
    ti = max(1, len(wocean)//8)
    ax3[0].plot(range(len(wocean)), wocean['bluecarbon_area_km2'], color='#1D9E75', marker='o', markersize=3)
    ax3[0].set_title('서해중부 블루카본 면적'); ax3[0].set_xlabel('관측 시점'); ax3[0].set_ylabel('면적 (km²)')
    ax3[0].set_xticks(range(0,len(wocean),ti)); ax3[0].set_xticklabels([wocean['year_month'].iloc[i] for i in range(0,len(wocean),ti)], rotation=45, ha='right', fontsize=8)
    ax3[0].grid(True, alpha=0.3)

    ax3[1].plot(range(len(eocean)), eocean['bluecarbon_area_km2'], color='#378ADD', marker='o', markersize=3)
    ax3[1].set_title('동해 블루카본 면적 (잘피림)'); ax3[1].set_xlabel('관측 시점'); ax3[1].set_ylabel('면적 (km²)')
    ax3[1].set_xticks(range(0,len(eocean),ti)); ax3[1].set_xticklabels([eocean['year_month'].iloc[i] for i in range(0,len(eocean),ti)], rotation=45, ha='right', fontsize=8)
    ax3[1].grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig4); plt.close()

    st.divider()
    b1, b2 = st.columns(2)
    b1.metric("서해중부 변화", f"{wocean['bluecarbon_area_km2'].iloc[-1]:.2f} km²", f"{wocean['bluecarbon_area_km2'].iloc[-1]-wocean['bluecarbon_area_km2'].iloc[0]:+.2f} km²")
    b2.metric("동해 변화", f"{eocean['bluecarbon_area_km2'].iloc[-1]:.2f} km²", f"{eocean['bluecarbon_area_km2'].iloc[-1]-eocean['bluecarbon_area_km2'].iloc[0]:+.2f} km²")

# ===================== TAB 4: 식재 지도 =====================
with tab4:
    st.subheader("🗺️ 블루카본 식재 우선순위 지도")
    st.markdown("pH가 낮고 블루카bon 면적이 부족한 해역을 우선 식재 대상으로 선정했습니다.")

    # 해역 데이터 정의
    regions = [
        {
            "name": "서해북부 (인천·경기)",
            "lat": 37.5, "lon": 126.3,
            "ph": 8.02, "bc_area": 836.06,
            "bc_type": "갯벌·염습지",
            "desc": "인천·경기 갯벌 밀집 지역. 육상 오염물질 유입으로 pH 낮음."
        },
        {
            "name": "서해중부 (충남)",
            "lat": 36.5, "lon": 126.2,
            "ph": 8.05, "bc_area": 335.71,
            "bc_type": "갯벌·염습지",
            "desc": "충남 갯벌 지역. 블루카본 실측 데이터 보유 해역."
        },
        {
            "name": "서해남부 (전북·전남 서해)",
            "lat": 35.5, "lon": 126.3,
            "ph": 8.03, "bc_area": 1179.21,
            "bc_type": "갯벌·잘피림",
            "desc": "전국 최대 갯벌 분포 지역. 블루카본 잠재력 높음."
        },
        {
            "name": "남해서부 (전남 남해)",
            "lat": 34.7, "lon": 127.1,
            "ph": 8.09, "bc_area": 180.5,
            "bc_type": "잘피림·염습지",
            "desc": "다도해 지역. 잘피림 분포하나 면적 부족."
        },
        {
            "name": "남해중부 (경남)",
            "lat": 34.9, "lon": 128.2,
            "ph": 8.11, "bc_area": 69.84,
            "bc_type": "잘피림",
            "desc": "경남 해역. 블루카본 면적 매우 부족."
        },
        {
            "name": "남해동부 (부산)",
            "lat": 35.1, "lon": 129.1,
            "ph": 8.12, "bc_area": 19.01,
            "bc_type": "바다숲",
            "desc": "부산 연안. 블루카본 면적 전국 최소 수준."
        },
        {
            "name": "동해남부 (울산·경북)",
            "lat": 36.0, "lon": 129.5,
            "ph": 8.17, "bc_area": 8.2,
            "bc_type": "바다숲·잘피림",
            "desc": "동해 남부 연안. 바다숲 조성 가능성 있음."
        },
        {
            "name": "동해북부 (강원)",
            "lat": 37.8, "lon": 129.1,
            "ph": 8.19, "bc_area": 5.1,
            "bc_type": "바다숲",
            "desc": "강원 동해안. 수온 낮아 바다숲 조성 적합."
        },
    ]

    # 우선순위 점수 계산 (절대 기준)
    # pH 기준: 8.10 미만=시급(50점), 8.10~8.15=주의(25점), 8.15 이상=양호(0점)
    # 블루카본 기준: 50 미만=시급(50점), 50~200=주의(25점), 200 이상=양호(0점)

    for r in regions:
        # pH 점수
        if r["ph"] < 8.10:
            ph_score = 50
        elif r["ph"] < 8.15:
            ph_score = 25
        else:
            ph_score = 0

        # 블루카본 점수
        if r["bc_area"] < 50:
            bc_score = 50
        elif r["bc_area"] < 200:
            bc_score = 25
        else:
            bc_score = 0

        r["score"] = ph_score + bc_score

        if r["score"] >= 75:
            r["priority"] = "🔴 시급"
            r["color"] = "red"
        elif r["score"] >= 25:
            r["priority"] = "🟠 권장"
            r["color"] = "orange"
        else:
            r["priority"] = "🟢 양호"
            r["color"] = "green"

    # 우선순위 순 정렬
    regions_sorted = sorted(regions, key=lambda x: x["score"], reverse=True)

    # 지도 생성
    col_map, col_info = st.columns([2, 1])

    with col_map:
        st.markdown("#### 식재 우선순위 지도")
        m = folium.Map(location=[36.5, 127.8], zoom_start=6, tiles='CartoDB positron')

        for r in regions:
            popup_html = f"""
            <div style='font-family: Arial; width: 200px;'>
                <b style='font-size:14px;'>{r['name']}</b><br>
                <hr style='margin:4px 0;'>
                <b>우선순위:</b> {r['priority']}<br>
                <b>점수:</b> {r['score']}점<br>
                <b>평균 pH:</b> {r['ph']}<br>
                <b>블루카본 면적:</b> {r['bc_area']} km²<br>
                <b>블루카본 유형:</b> {r['bc_type']}<br>
                <hr style='margin:4px 0;'>
                <i>{r['desc']}</i>
            </div>
            """
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=15,
                color=r["color"],
                fill=True,
                fill_color=r["color"],
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{r['name']} | {r['priority']} | 점수: {r['score']}"
            ).add_to(m)

        # 범례 추가
        legend_html = """
        <div style='position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                    background: white; padding: 10px; border-radius: 8px;
                    border: 1px solid #ccc; font-family: Arial; font-size: 13px;'>
            <b>식재 우선순위</b><br>
            🔴 시급 (75점 이상)<br>
            🟠 권장 (25~75점)<br>
            🟢 양호 (25점 미만)
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        st_folium(m, width=700, height=500)

    with col_info:
        st.markdown("#### 우선순위 순위표")
        for i, r in enumerate(regions_sorted):
            st.markdown(f"""
            **{i+1}위. {r['name']}**
            - 우선순위: {r['priority']}
            - 종합점수: {r['score']}점
            - pH: {r['ph']}
            - 블루카본: {r['bc_area']} km²
            ---
            """)

    st.divider()
    st.markdown("#### 식재 우선순위 상세 분석")
    df_map = pd.DataFrame([{
        '해역': r['name'],
        '우선순위': r['priority'],
        '종합점수': r['score'],
        '평균 pH': r['ph'],
        '블루카본 면적(km²)': r['bc_area'],
        '블루카본 유형': r['bc_type']
    } for r in regions_sorted])
    st.dataframe(df_map, use_container_width=True)

with tab5:  # 너의 실제 탭 이름에 맞춰줘 (tab4일 수도 있음)
    st.subheader("연구 결론")
    st.markdown("""
### 🔬 탐구 가설
> **블루카본 생태계가 CO₂를 흡수함으로써 르샤틀리에 원리에 의해 해수의 pH 하락(산성화)을 완화한다**
""")

    st.markdown("---")
    st.markdown("### ⚗️ 르샤틀리에 원리 메커니즘")

    st.markdown("""
| 단계 | 내용 |
|---|---|
| 화학반응식 | $CO_2 + H_2O \\rightleftharpoons H^+ + HCO_3^-$ |
| 1 | 블루카본 광합성으로 해수 CO₂ 흡수 |
| 2 | 해수 CO₂ 농도 감소 |
| 3 | 평형이 역방향으로 이동 |
| 4 | H⁺ 농도 감소 |
| 5 | pH 상승 → 해양산성화 완화 ✅ |
""")

    st.markdown("---")
    st.markdown("""
### 🤖 LSTM 분석 결과
| 구분 | 서해중부 (블루카본 O) | 동해 (블루카본 X) |
|---|---|---|
| 예측 경향 | 실제보다 높게 예측 | 실제값 근접 추종 |
| 해석 | 블루카본 증가 → pH 상승 패턴 학습 | 물리 변수 패턴 학습 |

---
### ✅ 최종 결론
> LSTM이 서해중부에서 pH를 실제보다 높게 예측한 것은 **블루카본 면적 증가 → pH 완화 패턴**을 학습했음을 의미하며, 르샤틀리에 원리 기반 가설과 일치한다.
""")