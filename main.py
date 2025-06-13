import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="지역별 대기오염·호흡기 질환 상관관계", layout="centered")
st.title("지역별 대기오염 물질 배출량과 호흡기 질환 진료자수 비율 상관관계 분석")

# -------------------------------------------------------------
# ⚡️ 1. 파일 업로드 & 캐싱 ------------------------------------------------
# -------------------------------------------------------------

st.sidebar.header("CSV 파일 업로드")
air_file = st.sidebar.file_uploader("전국_대기오염물질_배출량.csv", type=["csv"], key="air")
resp_file = st.sidebar.file_uploader("지역별_호흡기질환진료인원.csv", type=["csv"], key="resp")

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """CSV를 DataFrame으로 읽어오고 캐싱."""
    if uploaded_file is None:
        return pd.DataFrame()
    return pd.read_csv(uploaded_file, encoding="cp949")

# 캐싱된 데이터 불러오기
air_raw = load_csv(air_file)
resp_raw = load_csv(resp_file)

if air_raw.empty or resp_raw.empty:
    st.info("왼쪽 사이드바에서 두 CSV 파일을 모두 업로드해 주세요.")
    st.stop()

# -------------------------------------------------------------
# ⚡️ 2. 대기오염 데이터 전처리 -------------------------------------------
# -------------------------------------------------------------

pollutant_row = air_raw.iloc[0]
air_df = air_raw.drop(0).reset_index(drop=True)
region_col = air_df.columns[0]

# 연도 목록 추출 (4자리 숫자)
years = sorted({c[:4] for c in air_df.columns if c[:4].isdigit()})
sel_year = st.sidebar.selectbox("연도 선택", years, index=len(years) - 1)

# 선택 연도의 오염물질별 컬럼 집합
year_cols = [c for c in air_df.columns if c.startswith(sel_year)]
poll_map = {col: pollutant_row[col] for col in year_cols}

poll_options = ["전체(모든 물질 합계)"] + list(poll_map.values())
sel_poll = st.sidebar.selectbox("오염물질 선택", poll_options)

# 불필요 행 제거
valid_air = air_df[~air_df[region_col].isin(["전국", "바다"])]

@st.cache_data(show_spinner=False)
def parse_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].replace({",": ""}, regex=True).astype(float)

num_vals = parse_numeric(valid_air, year_cols)

if sel_poll == "전체(모든 물질 합계)":
    valid_air["배출량"] = num_vals.sum(axis=1)
else:
    sel_col = next(c for c, name in poll_map.items() if name == sel_poll)
    valid_air["배출량"] = num_vals[sel_col]

air_ready = valid_air[[region_col, "배출량"]]

# -------------------------------------------------------------
# ⚡️ 3. 호흡기 질환 데이터 전처리 ---------------------------------------
# -------------------------------------------------------------

@st.cache_data(show_spinner=False)
def prepare_resp(df: pd.DataFrame, year: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["진료년도"] = tmp["진료년도"].str.replace("년", "")
    tmp = tmp[tmp["진료년도"] == year]
    tmp["비율"] = tmp["주민등록인구별 진료실인원 비율"].str.replace("%", "").astype(float)
    return tmp[["시도", "비율"]]

resp_ready = prepare_resp(resp_raw, sel_year)

# -------------------------------------------------------------
# ⚡️ 4. 병합 & 상관계수 계산 -------------------------------------------
# -------------------------------------------------------------

merged = pd.merge(air_ready, resp_ready, left_on=region_col, right_on="시도")

if merged.empty:
    st.error("선택한 연도에 일치하는 지역 데이터가 없습니다.")
    st.stop()

corr = merged["배출량"].corr(merged["비율"])

st.subheader(
    f"📈 Pearson r: **{corr:.3f}**  (연도: {sel_year}, 물질: {sel_poll})"
)

# -------------------------------------------------------------
# ⚡️ 5. Plotly 시각화 ---------------------------------------------------
# -------------------------------------------------------------

fig = px.scatter(
    merged,
    x="배출량",
    y="비율",
    hover_name="시도",
    labels={"배출량": f"{sel_poll} 배출량 (kg)", "비율": "호흡기 질환 진료자수 비율 (%)"},
    title=f"{sel_year}년 {sel_poll} 배출량 vs. 호흡기 질환 비율",
    template="plotly_white",
    height=600,
)

# 회귀선 추가
try:
    m, b = np.polyfit(merged["배출량"], merged["비율"], 1)
    x_vals = np.linspace(merged["배출량"].min(), merged["배출량"].max(), 100)
    fig.add_scatter(x=x_vals, y=m * x_vals + b, mode="lines", name="회귀선", line=dict(dash="dash"))
except np.linalg.LinAlgError:
    st.warning("회귀선을 계산할 수 없습니다. 데이터가 충분한지 확인하세요.")

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# ⚡️ 6. 데이터 테이블 --------------------------------------------------
# -------------------------------------------------------------

with st.expander("📊 데이터 보기"):
    st.dataframe(
        merged.rename(
            columns={region_col: "지역", "배출량": "배출량(kg)", "비율": "진료자수 비율(%)"}
        ).sort_values("비율", ascending=False).reset_index(drop=True),
        use_container_width=True,
    )
