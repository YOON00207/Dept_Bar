import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import textwrap
import matplotlib.patches as patches

st.set_page_config(layout="wide") 

# -----------------------------
# 한글 폰트 설정
# -----------------------------
font_path = "KoPubWorld Dotum_Pro Medium.otf"
font_path2 = "KoPubWorld Dotum Bold.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------
# 1. 데이터 불러오기
# ---------------------------------------
@st.cache_data
def load_data():
    file_path = "0918학과경쟁력분석전체대학데이터셋.xlsx"
    return pd.read_excel(file_path, engine="openpyxl")

df = load_data()

st.title("학과경쟁력분석")

# ---------------------------------------
# 2. 지표 선택
# ---------------------------------------
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
selected_metric = st.selectbox("지표 선택", numeric_cols)
# ---------------------------------------
# 3. 학교 + 학과 검색 (str.contains)
# ---------------------------------------
school = st.selectbox("학교 선택", df["학교"].dropna().unique())
search_keyword = st.text_input("학과 검색어 입력")

if search_keyword:
    search_results = df[(df["학교"] == school) & (df["학과"].str.contains(search_keyword, na=False))]
else:
    search_results = df[(df["학교"] == school)]

# 검색 결과 보여주기
if not search_results.empty:
    st.subheader("검색 결과")
    st.dataframe(search_results)

    majors = st.multiselect("추가할 학과 선택", options=search_results["학과"].dropna().unique())
    row_data = search_results[search_results["학과"].isin(majors)]
else:
    st.info("검색 결과가 없습니다.")
    row_data = pd.DataFrame()

# ---------------------------------------
# 4. 세션 상태 초기화
# ---------------------------------------
if "selected" not in st.session_state:
    st.session_state.selected = pd.DataFrame(columns=df.columns)
if "labels" not in st.session_state:
    st.session_state.labels = []

# ---------------------------------------
# 5. 값 수정하기 (체크박스 → 필요할 때만)
# ---------------------------------------

edited_data = row_data.copy()
if st.checkbox("값 수정하기") and not row_data.empty:
    st.subheader("값 수정")
    for idx in row_data.index:
        for col in numeric_cols:
            old_val = row_data.loc[idx, col]

            # UI에는 NaN을 0.0으로만 보여줌 (실제 데이터는 NaN 유지)
            display_val = 0.0 if pd.isna(old_val) else float(old_val)

            new_val = st.number_input(
                f"{row_data.loc[idx,'학과']} - {col}",
                value=display_val,
                key=f"edit_{idx}_{col}"
            )

            # 사용자가 값을 변경하지 않았고 원래 NaN이면 NaN 유지
            if pd.isna(old_val) and new_val == 0.0:
                edited_data.at[idx, col] = np.nan
            else:
                edited_data.at[idx, col] = new_val


# ---------------------------------------
# 6. 선택 확정 (추가 버튼)
# ---------------------------------------
if st.button("추가"):
    if not row_data.empty:
        # 수정된 경우 반영해서 추가
        if not edited_data.equals(row_data):
            st.session_state.selected = pd.concat(
                [st.session_state.selected, edited_data],
                ignore_index=True
            )
            st.session_state.labels.extend(edited_data["학교"].tolist())
            st.success(f"{len(edited_data)}개 학과 (수정된 값) 추가 완료!")
        else:
            st.session_state.selected = pd.concat(
                [st.session_state.selected, row_data],
                ignore_index=True
            )
            st.session_state.labels.extend(row_data["학교"].tolist())
            st.success(f"{len(row_data)}개 학과 추가 완료!")

# ---------------------------------------
# 6-1. 수정 반영 (이미 추가된 학과 업데이트)
# ---------------------------------------
if st.button("수정 반영"):
    if not edited_data.empty:
        for idx, row in edited_data.iterrows():
            cond = (
                (st.session_state.selected["학교"] == row["학교"]) &
                (st.session_state.selected["학과"] == row["학과"])
            )
            st.session_state.selected.loc[cond, numeric_cols] = row[numeric_cols].values

        st.success("수정 사항이 반영되었습니다!")

# ---------------------------------------
# 7. 새로운 데이터 직접 추가 (체크박스 → 필요할 때만)
# ---------------------------------------
if st.checkbox("새로운 데이터 직접 추가"):
    st.subheader("신규 데이터 입력")
    new_school = st.text_input("학교명 입력")
    new_major = st.text_input("학과명 입력")
    new_values = {}
    for col in numeric_cols:
        new_values[col] = st.number_input(f"{col} 값 입력", value=0.0, key=f"new_{col}")
    if st.button("신규 데이터 추가"):
        new_row = pd.DataFrame([{**{"학교": new_school, "학과": new_major}, **new_values}])
        st.session_state.selected = pd.concat([st.session_state.selected, new_row], ignore_index=True)
        st.session_state.labels.append(new_major)
        st.success(f"{new_school} - {new_major} (신규 데이터) 추가 완료!")

# ---------------------------------------
# 8. 라벨 수정 기능
# ---------------------------------------
if st.session_state.selected.shape[0] > 0:
    st.subheader("X축 라벨 수정")
    new_labels = []
    for i, label in enumerate(st.session_state.labels):
        new_label = st.text_area(f"막대 {i+1} 라벨", value=label, key=f"label_{i}", height=50)
        new_labels.append(new_label)
    st.session_state.labels = new_labels
# ---------------------------------------
# 9. 그래프 그리기  (줄바꿈/폰트/사이즈 강화)
# ---------------------------------------

legend_font = fm.FontProperties(fname=font_path2, size=25, weight = "bold")
font_prop_x_label = fm.FontProperties(fname=font_path2, size=30, weight="bold")
font_prop_bar_label = fm.FontProperties(fname=font_path2, size=35, weight="bold")

def wrap_label(s, width=10):
    s = str(s)
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False)) if s else s

def is_percent_metric(name, series):
    if "%" in str(name):
        return True
    s = pd.to_numeric(series, errors="coerce").dropna()
    return (not s.empty) and s.min() >= 0 and s.max() <= 100

if not st.session_state.selected.empty:
    selected_df = st.session_state.selected.copy()
    selected_df[selected_metric] = pd.to_numeric(selected_df[selected_metric], errors="coerce")
    values_raw = selected_df[selected_metric]
    labels_wrapped = [wrap_label(x) for x in st.session_state.labels]

    # === 보기 모드 ===
    view_mode = st.selectbox(
        "표시 방식",
        ["원본", "상단 확대", "로그 스케일(>0만)"]
    )

    # 평균/표준편차 계산 (NaN 제외)
    base_for_stats = values_raw.dropna()
    if base_for_stats.empty:
        mean, std = np.nan, np.nan
    else:
        mean = float(base_for_stats.mean())
        std  = float(base_for_stats.std(ddof=1))

    fig, ax = plt.subplots(figsize=(18, 10), dpi=200)


    # ==========================================
    # 연구실적(논문) → 막대 2개씩
    # ==========================================
    if selected_metric in [
        "전임교원 1인당 논문 실적 연구재단등재지(후보포함) 계",
        "전임교원 1인당 논문 실적 SCI급/SCOPUS학술지 계"
    ]:
        metrics_to_plot = [
            "전임교원 1인당 논문 실적 연구재단등재지(후보포함) 계",
            "전임교원 1인당 논문 실적 SCI급/SCOPUS학술지 계"
        ]

        bar_width = 0.35
        x = np.arange(len(st.session_state.labels))
        colors = ["#dc0000", "#00005d"]

        for i, metric in enumerate(metrics_to_plot):
                vals = pd.to_numeric(selected_df[metric], errors="coerce")
                plot_vals = vals.fillna(0)

                # --- 막대 ---
                bars = ax.bar(x + i*bar_width, plot_vals, width=bar_width, color=colors[i], label=metric)

                # --- 값 라벨 ---
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                            f"{h:.3f}", ha="center", va="bottom",
                            fontproperties = font_prop_bar_label)

                # --- 평균 & 표준편차 (해당 metric 기준) ---
                base = vals.dropna()
                if not base.empty:
                    mean_i = float(base.mean())
                    std_i  = float(base.std(ddof=1))

                    # 평균선
                    ax.axhline(mean_i, color=colors[i], linestyle="--", linewidth=2,
                            label=f"{metric} 평균: {mean_i:.2f}")

                    # ±1σ 영역
                    ax.axhspan(mean_i - std_i, mean_i + std_i, alpha=0.15, color=colors[i])

        # --- X축 라벨 ---
        ax.set_xticks(x + bar_width/2)
        # X축 라벨 적용 (줄바꿈 허용)
        ax.set_xticklabels([label.replace("\r\n", "\n").replace("\r", "\n") for label in st.session_state.labels],
                        fontproperties=font_prop_x_label)


    # ==========================================
    # 일반 지표 → 막대 1개
    # ==========================================
    else:
        plot_values = values_raw.fillna(0)
        colors = ["#dc0000"] + ["#d8d8d8"] * (len(selected_df) - 1)
        bars = ax.bar(labels_wrapped, plot_values, color=colors, width = 0.3)

        # 평균선/±1σ
        if np.isfinite(mean) and np.isfinite(std):
            ax.axhline(mean, color="black", linestyle="--", linewidth=2, label=f"평균: {mean:.2f}")
            ax.axhspan(mean - std, mean + std, alpha=0.18, color="#ffcccc", label="주요 분포 범위(±1σ)")

        # 값 라벨
        def _fmt(v):
            if pd.isna(v): return ""
            return f"{v:.1f}"
        ax.bar_label(bars, labels=[_fmt(v) for v in values_raw],
                     padding=6, fontproperties=font_prop_bar_label)

        # X축 라벨
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=22)
        # X축 라벨 적용 (줄바꿈 허용)
        ax.set_xticklabels([label.replace("\r\n", "\n").replace("\r", "\n") for label in st.session_state.labels],
                        fontproperties=font_prop_x_label)

    # === 모드별 추가 처리 (두 케이스 공통) ===
# 연구실적(2축)일 때와 일반(1축)일 때 모두에서 쓸 y값 범위 계산
    if selected_metric in [
        "전임교원 1인당 논문 실적 연구재단등재지(후보포함) 계",
        "전임교원 1인당 논문 실적 SCI급/SCOPUS학술지 계"
    ]:
        metrics_to_plot = [
            "전임교원 1인당 논문 실적 연구재단등재지(후보포함) 계",
            "전임교원 1인당 논문 실적 SCI급/SCOPUS학술지 계"
        ]
        all_vals = pd.concat([
            pd.to_numeric(selected_df[m], errors="coerce")
            for m in metrics_to_plot
        ])
    else:
        all_vals = pd.to_numeric(selected_df[selected_metric], errors="coerce")

    # 안전한 min/max 계산 (전부 NaN인 경우 대비)
    if all_vals.notna().any():
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
        if vmin == vmax:
            vmax = vmin + 1.0  # 슬라이더 에러 방지용
    else:
        vmin, vmax = 0.0, 1.0

    if view_mode == "상단 확대":
        if is_percent_metric(selected_metric, all_vals):
            # % 지표면 상한은 100 고정, 하한만 슬라이더
            default_lower = min(90.0, max(0.0, vmax - 10))
            lower = st.slider("하한(%)", 0.0, 100.0, default_lower, 0.5)
            ax.set_ylim(lower, 100.0)
        else:
            lo, hi = st.slider("표시 범위", min_value=vmin, max_value=vmax, value=(vmin, vmax))
            ax.set_ylim(lo, hi)

    elif view_mode == "로그 스케일(>0만)":
        # 로그스케일은 0 이하 값이 있으면 불가
        if (all_vals.dropna() <= 0).any():
            st.warning("로그 스케일은 0 이하 값에 적용할 수 없어요. 다른 모드를 사용해 주세요.")
        else:
            ax.set_yscale("log")

    # 공통 설정 (연구실적/일반 둘 다)
    ax.get_yaxis().set_visible(False)
    ax.legend(prop=legend_font)

    # 상단 제목 박스
    ax.add_patch(
        patches.Rectangle(
            (0, 1.02), 1, 0.15,
            transform=ax.transAxes, clip_on=False,
            facecolor="#dc0000", edgecolor="#dc0000"
        )
    )

    font_prop_title = fm.FontProperties(fname=font_path2, size=48, weight="bold")

    title_map = {
        "신입생 충원율(%)": "신입생 충원율 (2023년 기준, 단위 : %)",
        "신입생 경쟁률" : "신입생 경쟁률 (2023년 기준)",
        "재학생충원율" : "재학생 충원율 (2023년 기준, 단위 : %)",
        "중도탈락률(%)" : "중도탈락률 (2023년 기준, 단위 : %)",
        "캡스톤디자인 평균이수학생수" : "캡스톤디자인 평균 이수 학생 수 (2023년 기준, 단위 : 명)",
        "졸업생 취업률(%)" : "졸업생 취업률 (2023년 기준, 단위 : %)",
        "졸업생 진학률(%)" : "졸업생 진학률 (2023년 기준, 단위 : %)",
        "전임교원 1인당 논문 실적 연구재단등재지(후보포함) 계": "교원 1인당 연구실적 (2023년 기준)",
        "전임교원 1인당 논문 실적 SCI급/SCOPUS학술지 계": "교원 1인당 연구실적 (2023년 기준)",
        "1인당 연구비(천원)": "교원 1인당 연구비 (2023년 기준, 단위 : 천원)"
    }
    title = title_map.get(selected_metric, f"{selected_metric} (2023년 기준)")

    ax.text(
        0.01, 1.1, title,
        ha="left", va="center",
        fontproperties=font_prop_title, color="white",
        transform=ax.transAxes
    )

    # Streamlit 렌더링
    plt.subplots_adjust(bottom=0.25)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # 선택된 학과 표 출력 + 삭제 버튼
    st.subheader("선택된 학과 목록")

    if not selected_df.empty:
        for idx, row in selected_df.iterrows():
            cols = st.columns([3, 3, 3, 2, 1])  # 열 비율 조정
            cols[0].write(row["학교"])
            cols[1].write(row["학과"])
            cols[2].write(st.session_state.labels[idx] if idx < len(st.session_state.labels) else "")
            cols[3].write(row[selected_metric])

            # 삭제 버튼
            if cols[4].button("❌", key=f"del_{idx}"):
                # 데이터프레임에서 해당 행 삭제
                st.session_state.selected = st.session_state.selected.drop(idx).reset_index(drop=True)

                # 라벨 리스트에서 해당 요소 삭제 (안전하게 처리)
                if idx < len(st.session_state.labels):
                    st.session_state.labels.pop(idx)

                st.success(f"{row['학교']} - {row['학과']} 삭제 완료!")
                st.stop()  # 바로 렌더링 멈추고 새로 그림


else:
    st.info("학교와 학과를 검색하고, 선택 후 [추가] 버튼을 눌러주세요.")
