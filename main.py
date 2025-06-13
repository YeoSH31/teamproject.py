import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="지역별 대기오염·호흡기 질환 상관관계", layout="centered")
st.title("지역별 대기오염 물질 배출량과 호흡기 질환 진료자수 비율 상관관계 분석")

# -------------------------- 사이드바 입력 ---------------------------
st.sidebar.header("CSV 파일 업로드")
air_file = st.sidebar.file_uploader("전국_대기오염물질_배출량.csv", type=["csv"], key="air")
resp_file = st.sidebar.file_uploader("지역별_호흡기질환진료인원.csv", type=["csv"], key="resp")

if air_file and resp_file:
    # ---------------------- 데이터 불러오기 -------------------------
    air_raw = pd.read_csv(air_file, encoding="cp949")
    resp_raw = pd.read_csv(resp_file, encoding="cp949")

    # ---------------------- 대기오염 데이터 전처리 -------------------
    pollutant_row = air_raw.iloc[0]  # 0행이 오염물질 이름
    air_df = air_raw.drop(0).reset_index(drop=True)
    region_col = air_df.columns[0]

    # 선택 연도 & 오염물질
    years = sorted({c.split(".")[0] for c in air_df.columns if c[:4].isdigit()})
    sel_year = st.sidebar.selectbox("연도 선택", years, index=len(years) - 1)
    year_cols = [c for c in air_df.columns if c.startswith(sel_year)]
    poll_map = {c: pollutant_row[c] for c in year_cols}
    poll_options = ["전체(모든 물질 합계)"] + list(poll_map.values())
    sel_pollutant = st.sidebar.selectbox("오염물질 선택", poll_options)

    # 전국·바다 행 제외
    air_df = air_df[~air_df[region_col].isin(["전국", "바다"])]

    if sel_pollutant == "전체(모든 물질 합계)":
        air_df["배출량"] = (
            air_df[year_cols].replace(",", "", regex=True).astype(float).sum(axis=1)
        )
    else:
        sel_col = next(col for col, name in poll_map.items() if name == sel_pollutant)
        air_df["배출량"] = air_df[sel_col].replace(",", "", regex=True).astype(float)

    air_ready = air_df[[region_col, "배출량"]]

    # ---------------------- 호흡기 질환 데이터 전처리 ---------------
    resp = resp_raw.copy()
    resp["진료년도"] = resp["진료년도"].str.replace("년", "")
    resp = resp[resp["진료년도"] == sel_year]
    resp["비율"] = resp["주민등록인구별 진료실인원 비율"].str.replace("%", "").astype(float)
    resp_ready = resp[["시도", "비율"]]

    # ---------------------- 병합 및 상관계수 -----------------------
    merged = pd.merge(air_ready, resp_ready, left_on=region_col, right_on="시도")

    if merged.empty:
        st.error("선택한 연도에 일치하는 지역 데이터가 없습니다.")
        st.stop()

    corr = merged["배출량"].corr(merged["비율"])
    st.subheader(f"상관계수 (Pearson r): {corr:.3f}")

    # ---------------------- 시각화 -------------------------------
    fig, ax = plt.subplots()
    ax.scatter(merged["배출량"], merged["비율"])
    ax.set_xlabel(f"{sel_pollutant} 배출량 (kg)")
    ax.set_ylabel("호흡기 질환 진료자수 비율 (%)")
    ax.set_title(f"{sel_year}년 {sel_pollutant} 배출량 vs. 호흡기 질환 비율")

    # 회귀선 추가
    m, b = np.polyfit(merged["배출량"], merged["비율"], 1)
    x_vals = np.linspace(merged["배출량"].min(), merged["배출량"].max(), 100)
    ax.plot(x_vals, m * x_vals + b, linestyle="--")

    st.pyplot(fig)

    # ---------------------- 데이터 테이블 -------------------------
    with st.expander("📊 데이터 보기"):
        st.dataframe(
            merged.rename(
                columns={region_col: "지역", "배출량": "배출량(kg)", "비율": "진료자수 비율(%)"}
            ).sort_values("비율", ascending=False).reset_index(drop=True)
        )
else:
    st.info("왼쪽 사이드바에서 두 CSV 파일을 모두 업로드해 주세요.")
