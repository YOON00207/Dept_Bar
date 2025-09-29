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

files = {
    "2023 데이터": "0918학과경쟁력분석전체대학데이터셋.xlsx",
    "2024 데이터(졸업생 취업률 및 진학률은 2023년도 기준)": "2024대학데이터셋.xlsx"
}

choice = st.selectbox("데이터 선택", list(files.keys()))

@st.cache_data
def load_data(path):
    return pd.read_excel(path, engine="openpyxl")

df = load_data(files[choice])

if "buffer" not in st.session_state: 
    st.session_state.buffer = pd.DataFrame(columns=df.columns)

st.title("학과경쟁력분석")

# ---------------------------------------
# 2. 지표 선택
# ---------------------------------------
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
selected_metric = st.selectbox("지표 선택", numeric_cols)
# ---------------------------------------
# 3. 학교 + 학과 검색 (str.contains)
# ---------------------------------------
# 학교명 변환 함수
def shorten_school(name: str) -> str:
    if isinstance(name, str):
        return name.replace("대학교", "대")
    return name

def make_label(row):
    school = shorten_school(row["학교"])
    major = row["학과"]

    # 특정 키워드가 있으면 학과는 제외
    if any(keyword in school for keyword in ["본교", "거점국립대", "강원권 사립대"]):
        return school
    else:
        return f"{school}\n{major}"

row_data = pd.DataFrame()

# -------------------------
# 학교 선택 (여러 개 가능)
# -------------------------
schools_all = df["학교"].dropna().unique().tolist()
selected_schools = st.multiselect("학교 선택 (여러 개 가능)", ["전체"] + schools_all, default=["전체"])

# -------------------------
# 학과 검색 (부분 일치)
# -------------------------
search_keyword = st.text_input("학과 검색어 입력")

# -------------------------
# 조건 결합
# -------------------------
if "전체" in selected_schools or not selected_schools:
    search_results = df.copy()
else:
    search_results = df[df["학교"].isin(selected_schools)]

if search_keyword:
    search_results = search_results[search_results["학과"].str.contains(search_keyword, na=False)]

# 검색 결과 나오기 전, 안전하게 기본값 선언(에러 방지용)
checked = pd.DataFrame(columns=df.columns)

# 검색 결과 보여주기
if not search_results.empty:
        st.subheader("검색 결과")

        # '추가' 체크박스 컬럼 붙이기
        search_results_display = search_results.copy()
        search_results_display["추가"] = False

        #추가 버튼 맨 앞으로 빼기
        cols = ["추가"] + [c for c in search_results_display.columns if c != "추가"]
        search_results_display = search_results_display[cols]

        # data_editor로 보여주기 (행 클릭해서 체크 가능)
        edited_results = st.data_editor(
            search_results_display,
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            key = "search_editor"
        )

        # 체크된 학과만 필터링
        checked = edited_results[edited_results["추가"] == True].copy()

        if st.button('선택한 학과 불러오기'):
            if "buffer" not in st.session_state:
                st.session_state.buffer = pd.DataFrame(columns = df.columns)
            st.session_state.buffer = pd.concat([st.session_state.buffer, checked], ignore_index=True)
            st.success(f"{len(checked)}개 학과 불러오기 완료")

        if not st.session_state.buffer.empty:
            st.subheader("불러온 학과 목록 (행 추가 및 수정 가능)")

            buffer_edit = st.data_editor(
                st.session_state.buffer,
                num_rows = 'dynamic',
                hide_index=True,
                use_container_width=True,
                key = "buffer_editor"
            )
            if st.button('수정 반영하기'):
                # 필수값 없는 행 제거
                buffer_edit = buffer_edit.dropna(subset=["학교", "학과"])

                # 사용자 안내 메시지
                st.warning("학교와 학과가 입력되지 않은 행은 자동으로 삭제되었습니다.")

                # 세션 상태 업데이트
                st.session_state.buffer = buffer_edit.copy()
                st.session_state.selected = buffer_edit.copy()
                st.session_state.labels = st.session_state.selected.apply(make_label, axis=1).tolist()
                
                st.success("수정 사항이 반영되었습니다")

            

# ---------------------------------------
# 4. 세션 상태 초기화
# ---------------------------------------
if "selected" not in st.session_state:
    st.session_state.selected = pd.DataFrame(columns=df.columns)
if "labels" not in st.session_state:
    st.session_state.labels = []



# ---------------------------------------
# 6. 선택 확정 (추가 버튼)
# ---------------------------------------
if st.button('추가'):
    if checked.empty:
        st.warning("추가할 학과를 먼저 선택하세요")
    else:
        st.session_state.selected = pd.concat(
            [st.session_state.selected, checked],
            ignore_index=True
        )
        combined_labels = checked.apply(make_label, axis=1)
        st.session_state.labels.extend(list(combined_labels))
        st.success(f"{len(checked)}개 학과 추가 완료!")



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

legend_font = fm.FontProperties(fname=font_path2, size=38, weight = "bold")
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
        legend_map = {
        "전임교원 1인당 논문 실적 연구재단등재지(후보포함) 계": "연구재단 등재지(후보포함)",
        "전임교원 1인당 논문 실적 SCI급/SCOPUS학술지 계": "SCI/SCOPUS 학술지"
    }


        bar_width = 0.35
        x = np.arange(len(st.session_state.labels))
        colors = ["#dc0000", "#00005d"]

        for i, metric in enumerate(metrics_to_plot):
                vals = pd.to_numeric(selected_df[metric], errors="coerce")
                plot_vals = vals

                # --- 평균/표준편차 (NaN 제외) ---
                base = vals.dropna()
                if not base.empty:
                    mean_i = float(base.mean())
                    std_i  = float(base.std(ddof=1))
                    ax.axhline(mean_i, color=colors[i], linestyle="--", linewidth=2,
                            label=f"{legend_map[metric]} 평균: {mean_i:.2f}")
                    ax.axhspan(mean_i - std_i, mean_i + std_i, alpha=0.15, color=colors[i], label = '주요 분포 범위(±1σ)')

                # --- 막대 ---
                bars = ax.bar(x + i*bar_width, plot_vals.fillna(0), width=bar_width, color=colors[i])

                # --- 값 라벨 (NaN은 빈칸) ---
                for bar, orig in zip(bars, vals):
                    h = bar.get_height()
                    label = "" if pd.isna(orig) else f"{orig:.1f}"
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                            label, ha="center", va="bottom",
                            fontproperties=font_prop_bar_label)
                    
                # --- y축 상단 여백 확보 (두 지표의 최대값 기준) ---
                all_max_candidates = max(
                    pd.to_numeric(selected_df[m], errors="coerce").max(skipna=True)
                    for m in metrics_to_plot
                )

                #유효한 값만 추출
                all_max_valid = [v for v in all_max_candidates if pd.notna(v) and np.isfinite(v)]

                if all_max_valid : # 유효한 값이 하나라도 있을 때
                    all_max = max(all_max_valid)

                else: #전부 nan값일때
                    all_max = 1.0

                ax.set_ylim(0, all_max * 1.15)


        # --- X축 라벨 ---
        ax.set_xticks(x + bar_width/2)
        # X축 라벨 적용 (줄바꿈 허용)
        ax.set_xticklabels([label.replace("\r\n", "\n").replace("\r", "\n") for label in st.session_state.labels],
                        fontproperties=font_prop_x_label)


    # ==========================================
    # 일반 지표 → 막대 1개
    # ==========================================
    else:
        plot_values = values_raw
        colors = ["#dc0000"] + ["#d8d8d8"] * (len(selected_df) - 1)

        
        # 평균선/±1σ
        if np.isfinite(mean) and np.isfinite(std):
            ax.axhline(mean, color="black", linestyle="--", linewidth=2, label=f"평균: {mean:.2f}")
            ax.axhspan(mean - std, mean + std, alpha=0.18, color="#ffcccc", label="주요 분포 범위(±1σ)")

        bars = ax.bar(labels_wrapped, plot_values.fillna(0), color=colors, width = 0.3)

        # --- y축 상단 여백 확보 ---
        if plot_values.notna().any():   # 유효한 값이 하나라도 있을 때
            ymax = float(plot_values.max())
            if not np.isfinite(ymax):   # inf 같은 값 대비
                ymax = 1.0
        else:
            ymax = 1.0   # 데이터가 전부 NaN일 때 기본값
        ax.set_ylim(0, ymax * 1.15)   # 값의 15% 여유 공간 확보


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
            # 퍼센트 지표는 100 기준, 상단 확대
            default_upper = 100.0
            default_lower = max(0.0, vmax - 5)  # 마지막 5% 구간만 보여주기
            lower = st.slider("하한(%)", 0.0, 100.0, default_lower, 0.5)
            ax.set_ylim(lower, default_upper*1.05)

        else:
            # 일반 지표는 최대값 중심으로 확대
            margin = (vmax - vmin) * 0.1 if vmax > vmin else 1
            lo = st.slider(
                "하한", min_value=vmin, max_value=vmax,
                value=max(vmin, vmax - margin), step=0.1
            )
            ax.set_ylim(lo, vmax*1.1)


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
    # title_override 변수를 사용자가 선택할 수 있도록 설정 (예: Streamlit selectbox 등과 연동 가능)
    # 제목 수정 칸 (기본값은 빈칸, placeholder로 안내 문구)
    title_override = st.text_input("차트 제목 수정", value="", placeholder="여기에 제목을 입력하세요")

    # 최종 타이틀 결정
    if title_override and len(title_override.strip()) > 0:
        title = title_override
    else:
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
                st.rerun()  # 바로 렌더링 멈추고 새로 그림
    # ---------------------------------------
    # 10. 데이터 출처 표시 (토글)
    # ---------------------------------------
    if st.toggle("데이터 출처 보기"):
        st.markdown("""
        ### 📊 데이터 출처 및 계산식

        | 지표 | 출처 | 지표 계산에 사용한 열 | 특징 | 계산식 |
        |------|------|----------------------|------|--------|
        | 신입생 충원율(%) | [대학알리미] 공시 데이터 다운로드 → 4-다. 신입생 충원 현황| 정원내 신입생 충원율(%) (D/B) × 100 | 좌측 열 그대로 반영 | B=모집인원_정원내, D=입학자_정원내 |
        | 신입생 경쟁률(%) |[대학알리미] 공시 데이터 다운로드 → 4-다. 신입생 충원 현황| 경쟁률 (C/B) | 좌측 열 그대로 반영 |  |
        | 재학생 충원율(%) | [대학알리미] 공시 데이터 다운로드 → 4-라. 학생 충원 현황(편입학 포함) 중 재학생 충원율| 정원내 재학생 충원율(%){D/(A-B)}×100 | 계열별 자료 사용 | 2024년도 파일에는 2023하반기·2024상반기 / 2023년도 파일에는 2023상반기·2022하반기 존재하여<br>2023년 상·하반기 각각 추출해서 평균<br><br>A=학생정원, B=모집정지인원, D=재학생 정원내 |
        | 중도탈락률(%) | [대학알리미] 공시 데이터 다운로드 → 4-사. 중도탈락학생현황| 중도탈락학생비율(%) (B/A)×100 | 좌측 열 그대로 반영 | A=재적학생, B=중도탈락학생 계 |
        | 캡스톤디자인 평균이수학생수(명) | [대학알리미] 개별 요청 자료 (기관 담당자 제공) | 이수 학생수(명)_해당학과 |  |학과(주간)별 1,2학기 평균|
        | 졸업생 취업률(%) | [대학알리미] 공시데이터다운로드 →  5-다. 졸업생의 취업 현황| 취업률(%) [B/{A-(C+D+E+F+G)}]×100 | 좌측 열 그대로 반영 | A=졸업자, B=취업자, C=진학자, D=입대자, E=취업불가능자, F=외국인유학생, G=제외인정자 |
        | 졸업생 진학률(%) | [대학알리미] 공시데이터다운로드 →  5-다. 졸업생의 취업 현황| 졸업자(A), 진학자(C) | 좌측 열 그대로 반영 | 진학률 = 진학자 (남+여) / 졸업자(남+여) |
        | 교원 1인당 연구(논문) 실적(건) - 연구재단등재지 | [대학알리미] 공시 데이터 다운로드 → 7-가. 전임교원의 연구 실적| 전임교원 1인당 논문 실적_연구재단등재지(후보포함) | 좌측 열 그대로 반영 |  |
        | 교원 1인당 연구(논문) 실적(건) - SCI급/SCOPUS |[대학알리미] 공시 데이터 다운로드 → 7-가. 전임교원의 연구 실적| 전임교원 1인당 논문 실적_SCI급/SCOPUS학술지 | 좌측 열 그대로 반영 |  |
        | 교원 1인당 연구비 수혜 실적(천원) | [대학알리미] 공시 데이터 다운로드 → 12-가. 연구비 수혜 실적| 전임교원수, 연구비 계, 대응자금 | 좌측 열 그대로 반영 |교원 1인당 연구비 수혜 실적 = (연구비 지원 계(교내+교외) 남 + 연구비 지원 계 여 + 대응자금 남 + 대응자금 여) / (전임교원 남 + 전임교원 여) |
        """, unsafe_allow_html=True)



else:
    st.info("학교와 학과를 검색하고, 선택 후 [추가] 버튼을 눌러주세요.")
