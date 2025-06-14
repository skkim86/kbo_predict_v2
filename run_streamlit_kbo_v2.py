import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import itertools

# 파일 경로 설정
SCHEDULE_PATH = "kbo_schedule.csv"
PITCHER_STATS_PATH = "kbo_pitcher_full_combined.csv"
PITCHER_LIST_PATH = "kbo_pitcher_list_2025.csv"

CATEGORY_LABELS = {
    "starter": "① 선발투수",
    "recent_form": "② 최근 성적",
    "head_to_head": "③ 상대 전적",
    "home_away": "④ 홈/원정"
}

# 6월 14일 기준 선발투수 디폴트 설정
DEFAULT_STARTERS = {
    "LG": "임찬규",
    "한화": "폰세",
    "롯데": "감보아",
    "SSG": "김광현",
    "키움": "알칸타라",
    "두산": "최승용",
    "KIA": "양현종",
    "NC": "신영우",
    "KT": "쿠에바스",
    "삼성": "후라도"
}

def force_rerun():
    try:
        st.rerun()
    except AttributeError:
        try:
            st._rerun()
        except:
            st.warning("⚠️ 자동 리프레시 실패: Streamlit 버전 확인 필요")

@st.cache_data
def load_data():
    pitchers = pd.read_csv(PITCHER_LIST_PATH)
    stats = pd.read_csv(PITCHER_STATS_PATH)
    schedule = pd.read_csv(SCHEDULE_PATH)
    stats["날짜"] = pd.to_datetime(stats["날짜"])
    schedule["date"] = pd.to_datetime(schedule["date"])
    return pitchers, stats, schedule

def get_recent_team_record(schedule_df, team_name, current_date):
    games = schedule_df[(schedule_df["date"] < current_date) &
                        ((schedule_df["home_team"] == team_name) | (schedule_df["away_team"] == team_name))]
    games = games.sort_values("date", ascending=False).head(10)
    win, draw, loss = 0, 0, 0
    for _, row in games.iterrows():
        hs, as_ = row["home_score"], row["away_score"]
        if pd.isna(hs) or pd.isna(as_):
            continue
        if hs == as_:
            draw += 1
        elif (row["home_team"] == team_name and hs > as_) or (row["away_team"] == team_name and as_ > hs):
            win += 1
        else:
            loss += 1
    return win, draw, loss

def show_pitcher_stats(df_stats, pitcher_name, opponent_team, selected_date):
    df = df_stats[df_stats["선수명"] == pitcher_name].copy()
    if df.empty:
        st.info("📭 기록 없음")
        return 0.0
    df["ERA"] = pd.to_numeric(df["ERA"], errors="coerce")
    recent_games = df[df["날짜"] < selected_date].sort_values("날짜", ascending=False)
    fatigue_penalty = 0.0
    if not recent_games.empty:
        last_pitch_date = recent_games.iloc[0]["날짜"]
        days_diff = (selected_date - last_pitch_date).days
        if days_diff == 1:
            st.warning(f"⚠️ {pitcher_name}은 어제({last_pitch_date.date()}) 등판")
            fatigue_penalty = 0.50
        elif days_diff == 2:
            fatigue_penalty = 0.35
        elif days_diff == 3:
            fatigue_penalty = 0.20
        elif days_diff == 4:
            fatigue_penalty = 0.10
    return fatigue_penalty

def evaluate_betting_combinations(results):
    best_combo, best_ev, best_win = None, -float("inf"), 0
    for combo in itertools.product(["홈 승", "무승부", "원정 승"], repeat=len(results)):
        total_ev, total_win = 0, 0
        for i, choice in enumerate(combo):
            item = results[i]
            prob = item["홈 승률"] if choice == "홈 승" else item["무승부"] if choice == "무승부" else item["원정 승률"]
            bet = item["베팅"]
            odds = item["배당률"]
            ev = bet * odds * prob - bet * (1 - prob)
            total_ev += ev
            total_win += bet * odds * prob
        if total_ev > best_ev:
            best_combo, best_ev, best_win = combo, total_ev, total_win
    return best_combo, best_ev, best_win

def run_app():
    st.set_page_config("KBO 예측 시스템", layout="wide")
    st.title("⚾ KBO 경기 베팅 예측")

    st.markdown("""---\n### 🔄 수동 리프레시\n투수 변경 후 화면이 반영되지 않으면 아래 버튼을 눌러주세요.""")
    if st.button("🔁 새로고침"):
        force_rerun()

    col1, col2 = st.columns(2)
    with col1:
        w1 = st.slider(CATEGORY_LABELS["starter"], 0, 100, 25)
        w2 = st.slider(CATEGORY_LABELS["recent_form"], 0, 100, 25)
    with col2:
        w3 = st.slider(CATEGORY_LABELS["head_to_head"], 0, 100, 25)
        w4 = st.slider(CATEGORY_LABELS["home_away"], 0, 100, 25)

    if w1 + w2 + w3 + w4 > 100:
        st.error("⚠️ 가중치 총합은 100% 이하여야 합니다.")
        return

    pitchers, stats, schedule = load_data()
    selected_date = st.date_input("🗓️ 날짜 선택", value=datetime(2025, 6, 14))
    selected_date = pd.to_datetime(selected_date)

    matches_today = schedule[schedule["date"] == selected_date]
    if matches_today.empty:
        st.info("해당 날짜에 경기가 없습니다.")
        return

    has_result = not matches_today[["home_score", "away_score"]].isnull().any(axis=1).all()
    if has_result:
        st.warning("⚠️ 해당 날짜는 경기 결과가 이미 존재합니다. 베팅은 불가능하며 결과만 확인할 수 있습니다.")

    results = []

    for i, row in matches_today.iterrows():
        home_team, away_team = row["home_team"], row["away_team"]
        st.subheader(f"⚔️ {home_team} vs {away_team}")

        colh, cola = st.columns(2)
        with colh:
            home_pitchers = pitchers[pitchers["팀명"].str.contains(home_team)]["선수명"].tolist()
            default_home = DEFAULT_STARTERS.get(home_team, None)
            home_index = home_pitchers.index(default_home) if default_home in home_pitchers else 0
            home_selected = st.selectbox(f"{home_team} 선발", home_pitchers, index=home_index, key=f"{home_team}_{i}")
            prev_home = st.session_state.get(f"prev_{home_team}_{i}", "")
            if home_selected != prev_home:
                st.session_state[f"prev_{home_team}_{i}"] = home_selected
                force_rerun()
            hw, _, hl = get_recent_team_record(schedule, home_team, selected_date)

        with cola:
            away_pitchers = pitchers[pitchers["팀명"].str.contains(away_team)]["선수명"].tolist()
            default_away = DEFAULT_STARTERS.get(away_team, None)
            away_index = away_pitchers.index(default_away) if default_away in away_pitchers else 0
            away_selected = st.selectbox(f"{away_team} 선발", away_pitchers, index=away_index, key=f"{away_team}_{i}")
            prev_away = st.session_state.get(f"prev_{away_team}_{i}", "")
            if away_selected != prev_away:
                st.session_state[f"prev_{away_team}_{i}"] = away_selected
                force_rerun()
            aw, _, al = get_recent_team_record(schedule, away_team, selected_date)

        fatigue_home = show_pitcher_stats(stats, home_selected, away_team, selected_date)
        fatigue_away = show_pitcher_stats(stats, away_selected, home_team, selected_date)

        home_score = w1 * (1 - fatigue_home) + w2 * hw + w3 * 0.5 + w4 * 1
        away_score = w1 * (1 - fatigue_away) + w2 * aw + w3 * 0.5 + w4 * 0
        total_score = home_score + away_score
        home_prob = round(home_score / total_score, 3)
        away_prob = round(away_score / total_score, 3)
        draw_prob = round(1 - (home_prob + away_prob), 3)

        st.markdown(f"🔮 **예상 승률**: 홈 {home_prob*100:.1f}%, 무 {draw_prob*100:.1f}%, 원정 {away_prob*100:.1f}%")

        if not has_result:
            bet = st.radio("📌 선택", ["홈 승", "무승부", "원정 승"], horizontal=True, key=f"bet_{home_team}_{away_team}_{i}")
            amount = st.number_input("💰 베팅 금액", 1000, value=10000, step=1000, key=f"amount_{i}")
            odds = st.number_input("📈 배당률", 1.01, value=1.95, step=0.01, key=f"odds_{i}")
        else:
            bet, amount, odds = None, 0, 0

        prob = home_prob if bet == "홈 승" else draw_prob if bet == "무승부" else away_prob
        expected_win = amount * odds * prob
        expected_loss = amount * (1 - prob)
        expected_value = expected_win - expected_loss
        st.markdown(f"💸 기대 수익: ₩{expected_win:,.0f} / EV: ₩{expected_value:,.0f}")

        results.append({
            "경기": f"{home_team} vs {away_team}",
            "예측": bet,
            "홈투수": home_selected,
            "원정투수": away_selected,
            "홈 승률": home_prob,
            "무승부": draw_prob,
            "원정 승률": away_prob,
            "베팅": amount,
            "배당률": odds,
            "기대수익": expected_win,
            "EV": expected_value
        })

    if not has_result and st.button("🎯 모든 경기 베팅 확정 결과 보기"):
        df = pd.DataFrame(results)
        st.dataframe(df)

        st.subheader("📊 전체 베팅 요약")
        st.metric("총 베팅", f"₩{df['베팅'].sum():,.0f}")
        st.metric("총 기대 수익", f"₩{df['기대수익'].sum():,.0f}")
        st.metric("총 기대값(EV)", f"₩{df['EV'].sum():,.0f}")

        combo, ev, win = evaluate_betting_combinations(results)
        st.subheader("🧠 최적 베팅 추천 (EV 기준)")
        for i, choice in enumerate(combo):
            st.markdown(f"- {results[i]['경기']}: **{choice}**")
        st.markdown(f"💡 최적 기대값: ₩{ev:,.0f}")
        st.markdown(f"🎯 최적 기대 수익: ₩{win:,.0f}")

if __name__ == "__main__":
    run_app()