import os
import time
import json
import re
import base64
from datetime import datetime

import dotenv
import fastf1
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# =========================
# Setup
# =========================
if not os.path.exists("cache"):
    os.makedirs("cache")

fastf1.Cache.enable_cache("cache")
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None

st.set_page_config(
    page_title="🏎️ F1Expert AI Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Styling
# =========================
def set_background(image_file: str):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(135deg, #1e1e1e, #2d2d2d, #1e1e1e);
                background-size: 400% 400%;
                animation: gradient 15s ease infinite;
            }
            @keyframes gradient {
                0% {background-position: 0% 50%;}
                50% {background-position: 100% 50%;}
                100% {background-position: 0% 50%;}
            }
            </style>
            """,
            unsafe_allow_html=True
        )

set_background("f1_background.jpg")

st.markdown(
    """
    <style>
    .block-container {
        background-color: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        margin-top: 10px;
    }

    h1, h2, h3, h4, h5 {
        color: #f2f2f2;
    }

    .stButton > button {
        background-color: #e10600;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background-color: #b30500;
        color: white;
    }

    .chat-container {
        background-color: rgba(30, 30, 30, 0.75);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
        min-height: 220px;
        max-height: 420px;
        overflow-y: auto;
    }

    .user-message {
        background-color: rgba(14, 17, 23, 0.85);
        color: white;
        padding: 8px 12px;
        border-radius: 15px 15px 0 15px;
        margin: 6px 0;
    }

    .bot-message {
        background-color: rgba(225, 6, 0, 0.85);
        color: white;
        padding: 8px 12px;
        border-radius: 15px 15px 15px 0;
        margin: 6px 0;
    }

    .driver-card {
        background: linear-gradient(145deg, rgba(30,30,30,0.85), rgba(60,60,60,0.85));
        border-radius: 8px;
        padding: 8px 10px;
        margin: 6px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .position {
        font-weight: 800;
        font-size: 1.1rem;
        color: #e10600;
    }

    .driver-name {
        font-weight: 700;
        color: white;
    }

    .driver-time {
        color: #dddddd;
        font-family: monospace;
    }

    .trivia-question {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #e10600;
    }

    .trivia-answer {
        padding: 10px;
        border-radius: 8px;
        margin-top: 8px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(60, 60, 60, 0.45);
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        color: white;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(225, 6, 0, 0.85) !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Session State Init
# =========================
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "user",
            "parts": "System prompt: You are an F1 knowledge expert and intelligent Formula 1 assistant named RaceMaster."
        }
    ]

if "chat_box_input" not in st.session_state:
    st.session_state.chat_box_input = ""

if "trivia_questions" not in st.session_state:
    st.session_state.trivia_questions = []
    st.session_state.show_answers = {}
    st.session_state.user_score = 0
    st.session_state.total_questions = 0
    st.session_state.trivia_active = False

# =========================
# Helpers
# =========================
def fetch_conversation_history():
    return st.session_state["messages"]

def response(messages):
    if model is None:
        return "Error: GEMINI_API_KEY is missing."
    try:
        return model.generate_content(messages)
    except Exception as e:
        return f"Error: {str(e)}"

def send_chat_message():
    user_input = st.session_state.chat_box_input.strip()
    if not user_input:
        return

    messages = fetch_conversation_history()
    messages.append({"role": "user", "parts": user_input})

    with st.spinner("RaceMaster is thinking..."):
        response_text = response(messages)

    if isinstance(response_text, str):
        st.error(response_text)
    else:
        messages.append({"role": "model", "parts": response_text.text})

    st.session_state.chat_box_input = ""

def generate_f1_trivia():
    trivia_prompt = """
    Generate exactly 5 unique Formula 1 trivia questions and very short direct answers.
    Answer must be very short, like a driver name, team, year, country, or number.
    Return only valid JSON inside a ```json``` block.

    Example:
    ```json
    [
      {"question":"Who won the 2021 F1 World Championship?","answer":"Max Verstappen"},
      {"question":"Which team has the most Constructors' Championships?","answer":"Ferrari"}
    ]
    ```
    """

    if model is None:
        return [
            {"question": "Who holds the most F1 World Championships?", "answer": "Lewis Hamilton"},
            {"question": "Which team is nicknamed the Silver Arrows?", "answer": "Mercedes"},
            {"question": "Which country hosts the Monaco Grand Prix?", "answer": "Monaco"},
            {"question": "Who was Ferrari's first F1 world champion?", "answer": "Alberto Ascari"},
            {"question": "What tire supplier is used in Formula 1?", "answer": "Pirelli"}
        ]

    try:
        resp = model.generate_content(trivia_prompt)
        text = resp.text
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)

        json_str = match.group(1) if match else text
        json_str = json_str.replace("\\n", "").replace("\\", "")
        questions = json.loads(json_str)

        cleaned = []
        for qa in questions[:5]:
            q = str(qa.get("question", "")).strip()
            a = str(qa.get("answer", "")).strip().split(".")[0]
            if q and a:
                cleaned.append({"question": q, "answer": a})

        if cleaned:
            return cleaned

    except Exception as e:
        st.warning(f"Trivia generation fallback used: {e}")

    return [
        {"question": "Who holds the most F1 World Championships?", "answer": "Lewis Hamilton"},
        {"question": "Which team is nicknamed the Silver Arrows?", "answer": "Mercedes"},
        {"question": "Which country hosts the Monaco Grand Prix?", "answer": "Monaco"},
        {"question": "Who was Ferrari's first F1 world champion?", "answer": "Alberto Ascari"},
        {"question": "What tire supplier is used in Formula 1?", "answer": "Pirelli"}
    ]

@st.cache_data(show_spinner=False)
def get_valid_schedule_for_year(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year)
    if schedule is None or schedule.empty:
        return pd.DataFrame()

    valid = schedule[
        schedule["EventFormat"].notna() &
        (~schedule["EventName"].astype(str).str.contains("Testing|Test", case=False, na=False))
    ].copy()

    # Keep only useful columns if they exist
    for col in ["EventName", "Location", "Country", "RoundNumber"]:
        if col not in valid.columns:
            valid[col] = None

    valid = valid.dropna(subset=["EventName"]).copy()
    return valid

@st.cache_data(show_spinner=False)
def get_latest_schedule_year():
    current_year = datetime.now().year
    for year in range(current_year, 2018, -1):
        try:
            sched = get_valid_schedule_for_year(year)
            if not sched.empty:
                return year
        except Exception:
            continue
    return 2025

def resolve_event_name(year: int, target_gp_name: str, target_location: str = None, target_country: str = None):
    try:
        schedule = get_valid_schedule_for_year(year)
        if schedule.empty:
            return None

        # Exact event name
        exact = schedule[schedule["EventName"] == target_gp_name]
        if not exact.empty:
            return exact.iloc[0]["EventName"]

        # Exact location
        if target_location:
            loc = schedule[schedule["Location"] == target_location]
            if not loc.empty:
                return loc.iloc[0]["EventName"]

        # Exact country
        if target_country:
            ctry = schedule[schedule["Country"] == target_country]
            if not ctry.empty:
                return ctry.iloc[0]["EventName"]

        # Partial name match
        words = [w for w in str(target_gp_name).split() if len(w) > 3]
        for w in words:
            partial = schedule[schedule["EventName"].astype(str).str.contains(w, case=False, na=False)]
            if not partial.empty:
                return partial.iloc[0]["EventName"]

        # Fallback to first matching grand prix token
        partial = schedule[schedule["EventName"].astype(str).str.contains("Grand Prix", case=False, na=False)]
        if not partial.empty:
            return partial.iloc[0]["EventName"]

    except Exception:
        pass

    return None

@st.cache_data(show_spinner=False)
def get_predictable_gps(year):
    try:
        schedule = get_valid_schedule_for_year(year)
        if schedule.empty:
            return [], {}

        races = schedule["EventName"].dropna().tolist()
        info_map = {}

        for _, row in schedule.iterrows():
            gp_name = row["EventName"]
            info_map[gp_name] = {
                "location": row.get("Location"),
                "country": row.get("Country")
            }

        return races, info_map

    except Exception:
        return [], {}

@st.cache_data(show_spinner=False)
def get_predictable_years():
    latest_year = get_latest_schedule_year()
    valid_years = []

    for year in range(2018, latest_year + 1):
        try:
            schedule = get_valid_schedule_for_year(year)
            if not schedule.empty:
                valid_years.append(year)
        except Exception:
            continue

    return valid_years

def predict_all_positions(gp_name, upcoming_year, gp_location=None, gp_country=None):
    historical_years = list(range(max(2018, upcoming_year - 3), upcoming_year))
    all_races = []

    for year in historical_years:
        resolved_name = resolve_event_name(year, gp_name, gp_location, gp_country)
        if not resolved_name:
            continue

        try:
            session = fastf1.get_session(year, resolved_name, "R")
            session.load()

            if session.results is None or session.results.empty:
                continue

            results = session.results.copy()

            needed_cols = ["Abbreviation", "Position", "TeamName"]
            missing = [c for c in needed_cols if c not in results.columns]
            if missing:
                continue

            results = results[needed_cols].copy()
            results["Year"] = year
            results["Position"] = pd.to_numeric(results["Position"], errors="coerce")
            results = results.dropna(subset=["Abbreviation", "Position", "TeamName"])

            if not results.empty:
                all_races.append(results)

        except Exception:
            continue

    if not all_races:
        return None, None, None

    df = pd.concat(all_races, ignore_index=True)
    if df.empty:
        return None, None, None

    df["Abbreviation"] = df["Abbreviation"].astype(str)
    df["TeamName"] = df["TeamName"].astype(str)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df = df.dropna(subset=["Year", "Position"])

    if df.empty:
        return None, None, None

    # Try getting actual results for selected year
    actual_results = None
    show_actual = False
    upcoming_drivers = None

    try:
        resolved_upcoming = resolve_event_name(upcoming_year, gp_name, gp_location, gp_country) or gp_name
        upcoming_session = fastf1.get_session(upcoming_year, resolved_upcoming, "R")
        upcoming_session.load()

        if upcoming_session.results is not None and not upcoming_session.results.empty:
            upcoming_results = upcoming_session.results.copy()

            if {"Abbreviation", "TeamName"}.issubset(upcoming_results.columns):
                upcoming_drivers = upcoming_results[["Abbreviation", "TeamName"]].drop_duplicates().copy()

            if {"Abbreviation", "Position"}.issubset(upcoming_results.columns):
                actual_results = upcoming_results[["Abbreviation", "Position"]].copy()
                actual_results["Position"] = pd.to_numeric(actual_results["Position"], errors="coerce")
                actual_results = actual_results.dropna(subset=["Position"])

            show_actual = actual_results is not None and not actual_results.empty

    except Exception:
        pass

    # If there are no actual results yet, use the latest historical lineup
    if upcoming_drivers is None or upcoming_drivers.empty:
        latest_year = int(df["Year"].max())
        upcoming_drivers = df[df["Year"] == latest_year][["Abbreviation", "TeamName"]].drop_duplicates().copy()
        show_actual = False
        actual_results = None

    if upcoming_drivers.empty:
        return None, None, None

    upcoming_drivers["Abbreviation"] = upcoming_drivers["Abbreviation"].astype(str)
    upcoming_drivers["TeamName"] = upcoming_drivers["TeamName"].astype(str)

    # Encode
    le_driver = LabelEncoder()
    le_driver.fit(pd.concat([df["Abbreviation"], upcoming_drivers["Abbreviation"]]).astype(str))

    le_team = LabelEncoder()
    le_team.fit(pd.concat([df["TeamName"], upcoming_drivers["TeamName"]]).astype(str))

    df["Driver_encoded"] = le_driver.transform(df["Abbreviation"])
    df["Team_encoded"] = le_team.transform(df["TeamName"])

    X = df[["Driver_encoded", "Team_encoded", "Year"]]
    y = df["Position"]

    if X.empty or y.empty:
        return None, None, None

    model_rf = RandomForestRegressor(n_estimators=150, random_state=42)
    model_rf.fit(X, y)

    upcoming_drivers["Year"] = upcoming_year
    upcoming_drivers["Driver_encoded"] = le_driver.transform(upcoming_drivers["Abbreviation"])
    upcoming_drivers["Team_encoded"] = le_team.transform(upcoming_drivers["TeamName"])

    X_upcoming = upcoming_drivers[["Driver_encoded", "Team_encoded", "Year"]]
    upcoming_drivers["Predicted Position"] = model_rf.predict(X_upcoming)
    upcoming_drivers = upcoming_drivers.sort_values("Predicted Position").reset_index(drop=True)

    upcoming_drivers["Time Gap (s)"] = [round(i * 2.5 + (i ** 1.1), 3) for i in range(len(upcoming_drivers))]
    upcoming_drivers["Predicted Finish Time"] = upcoming_drivers["Time Gap (s)"].apply(
        lambda t: "Leader" if t == 0 else f"+{t:.3f}s"
    )

    return upcoming_drivers, actual_results, show_actual

def simulate_live_race(drivers_df):
    if "race_lap" not in st.session_state:
        st.session_state.race_lap = 1
        st.session_state.race_data = drivers_df.copy()
        st.session_state.race_data["Current Gap"] = st.session_state.race_data["Time Gap (s)"]

    if st.session_state.race_lap > 1:
        n_drivers = len(st.session_state.race_data)
        overtake_chance = np.random.random(n_drivers)

        for i in range(1, n_drivers):
            if overtake_chance[i] < 0.15:
                temp_gap = st.session_state.race_data.iloc[i - 1]["Current Gap"]
                st.session_state.race_data.iloc[i - 1, st.session_state.race_data.columns.get_loc("Current Gap")] = (
                    st.session_state.race_data.iloc[i]["Current Gap"]
                )
                st.session_state.race_data.iloc[i, st.session_state.race_data.columns.get_loc("Current Gap")] = temp_gap

    st.session_state.race_data = st.session_state.race_data.sort_values("Current Gap").reset_index(drop=True)
    st.session_state.race_lap += 1
    return st.session_state.race_data, st.session_state.race_lap

team_colors = {
    "Red Bull": "#0600EF",
    "Mercedes": "#00D2BE",
    "Ferrari": "#DC0000",
    "McLaren": "#FF8700",
    "Alpine": "#0090FF",
    "AlphaTauri": "#2B4562",
    "Aston Martin": "#006F62",
    "Williams": "#005AFF",
    "Alfa Romeo": "#900000",
    "Haas F1 Team": "#FFFFFF",
    "Racing Point": "#F596C8",
    "Renault": "#FFF500",
    "Toro Rosso": "#469BFF",
    "Force India": "#F596C8",
    "Sauber": "#9B0000",
    "Manor": "#323232",
    "Caterham": "#10518F",
    "Lotus": "#FFB800",
    "Marussia": "#6E0000",
    "HRT": "#333333",
    "Jaguar": "#006F62",
    "Jordan": "#EFC600",
    "BAR": "#B26500",
    "Arrows": "#FF8700",
    "Brawn GP": "#B5B5B5"
}

def generate_race_commentary(lap_number, position_data, total_laps):
    try:
        leader = position_data.iloc[0]["Abbreviation"]
        second_place = position_data.iloc[1]["Abbreviation"] if len(position_data) > 1 else "N/A"
        gap = position_data.iloc[1]["Current Gap"] if len(position_data) > 1 else 0
        race_progress = (lap_number / total_laps) * 100

        position_changes = []
        if "previous_positions" in st.session_state and lap_number > 1:
            current_positions = position_data["Abbreviation"].tolist()
            prev_positions = st.session_state.previous_positions

            for i, driver in enumerate(current_positions):
                if driver in prev_positions:
                    prev_idx = prev_positions.index(driver)
                    if prev_idx != i:
                        if prev_idx > i:
                            position_changes.append(f"{driver} gained {prev_idx - i} position(s)")
                        else:
                            position_changes.append(f"{driver} lost {i - prev_idx} position(s)")

        st.session_state.previous_positions = position_data["Abbreviation"].tolist()

        prompt = f"""
        Generate a brief exciting F1 commentary for lap {lap_number} of {total_laps}.
        Race progress: {race_progress:.1f}%.
        Leader: {leader}
        Second place: {second_place}
        Gap: {gap:.2f} seconds
        {"Position changes: " + ", ".join(position_changes) if position_changes else ""}
        Keep it to 2 short sentences.
        """

        if model is None:
            return f"Lap {lap_number}/{total_laps}: {leader} leads from {second_place} with a gap of {gap:.2f}s."

        resp = model.generate_content(prompt)
        return resp.text

    except Exception:
        return f"Lap {lap_number}/{total_laps}: The race continues."

# =========================
# Main UI
# =========================
st.markdown(
    """
    <div style="text-align:center; background-color: rgba(0,0,0,0.72); padding: 24px; border-radius: 16px; margin-bottom: 20px;">
        <h1 style="color:#e10600; margin:0; font-size:3em;">🏁 F1Expert AI Dashboard</h1>
        <p style="color:#f2f2f2; font-size:1.2em;">Your ultimate Formula 1 companion powered by AI</p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 2, 1], gap="large")

# =========================
# Column 1: Chatbot
# =========================
with col1:
    st.markdown(
        """
        <div style="background-color: rgba(0,0,0,0.72); padding: 15px; border-radius: 10px; border-top: 4px solid #e10600;">
            <h3 style="color:white; margin-top:0;">💬 Ask RaceMaster</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    messages = fetch_conversation_history()

    for message in messages:
        if message["role"] == "model":
            st.markdown(
                f'<div class="bot-message"><strong>RaceMaster:</strong> {message["parts"]}</div>',
                unsafe_allow_html=True
            )
        elif message["role"] == "user" and "System prompt" not in message["parts"]:
            st.markdown(
                f'<div class="user-message"><strong>You:</strong> {message["parts"]}</div>',
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("trivia_active", False):
        st.markdown(
            """
            <div style="background-color: rgba(225, 6, 0, 0.3); padding: 10px; border-radius: 6px; margin: 10px 0;">
                <p style="color:white; margin:0;">Chat is disabled during Trivia Game. Complete or end the game to resume chatting.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.text_input("Chat disabled during trivia...", value="", disabled=True)
    else:
        st.text_input("Ask something about F1...", key="chat_box_input")
        col_send, col_clear = st.columns(2)

        with col_send:
            if st.button("Send", use_container_width=True):
                send_chat_message()
                st.rerun()

        with col_clear:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state["messages"] = [
                    {
                        "role": "user",
                        "parts": "System prompt: You are an F1 knowledge expert and intelligent Formula 1 assistant named RaceMaster."
                    }
                ]
                st.session_state.chat_box_input = ""
                st.rerun()

# =========================
# Column 2: Predictor + Trivia
# =========================
with col2:
    tab1, tab2 = st.tabs(["🏎️ Race Predictor", "❓ F1 Trivia"])

    # -------- Race Predictor --------
    with tab1:
        st.markdown(
            """
            <div style="background-color: rgba(0,0,0,0.72); padding: 15px; border-radius: 10px; border-top: 4px solid #e10600;">
                <h3 style="color:white; margin-top:0;">🔮 Predict Race Results</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_year, col_gp = st.columns(2)

        available_years = get_predictable_years()

        if not available_years:
            st.error("No valid seasons are available right now.")
            selected_year = None
            selected_gp = None
            selected_gp_location = None
            selected_gp_country = None
        else:
            with col_year:
                default_year = available_years[-1]
                selected_year = st.selectbox("Select Year", available_years, index=len(available_years) - 1)

            races, event_info = get_predictable_gps(selected_year)

            if not races:
                st.warning(f"No predictable races available for {selected_year}.")
                selected_gp = None
                selected_gp_location = None
                selected_gp_country = None
            else:
                with col_gp:
                    selected_gp = st.selectbox("Choose a Grand Prix", races)

                selected_gp_location = event_info[selected_gp]["location"] if selected_gp in event_info else None
                selected_gp_country = event_info[selected_gp]["country"] if selected_gp in event_info else None

                if selected_gp:
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(0,0,0,0.5); padding: 10px; border-radius: 6px; margin: 10px 0;">
                            <p style="color:white; margin:0;"><strong>📍 Circuit:</strong> {selected_gp_location or 'Unknown'}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        if selected_gp and st.button("Run Prediction", key="predict_button"):
            with st.spinner("Running prediction model..."):
                predictions, actual, has_real = predict_all_positions(
                    selected_gp,
                    selected_year,
                    selected_gp_location,
                    selected_gp_country
                )

            if predictions is None or predictions.empty:
                st.error(f"No usable race data found for {selected_gp} in {selected_year}.")
            else:
                st.success("✅ Prediction complete!")
                st.session_state.predictions = predictions.copy()

                st.markdown("#### 🏁 Predicted Finish Order")
                st.dataframe(
                    predictions[["Abbreviation", "TeamName", "Predicted Finish Time"]],
                    hide_index=True,
                    use_container_width=True
                )

                st.markdown("#### 📊 Predicted Results Visualization")
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor("#1e1e1e")
                ax.set_facecolor("#1e1e1e")

                team_colors_list = [team_colors.get(team, "#777777") for team in predictions["TeamName"]]
                bars = ax.barh(
                    predictions["Abbreviation"],
                    predictions["Time Gap (s)"],
                    color=team_colors_list,
                    height=0.6
                )

                for i, (bar, team) in enumerate(zip(bars, predictions["TeamName"])):
                    ax.text(
                        bar.get_width() + 0.5,
                        i,
                        team,
                        va="center",
                        color="white",
                        fontsize=8
                    )

                ax.set_title(f"Predicted Time Gaps: {selected_gp} {selected_year}", color="white", fontsize=14)
                ax.set_xlabel("Gap to Leader (seconds)", color="white")
                ax.set_ylabel("Drivers", color="white")
                ax.tick_params(colors="white")
                ax.grid(True, linestyle="--", alpha=0.3)
                for spine in ax.spines.values():
                    spine.set_color("#333333")
                plt.tight_layout()
                st.pyplot(fig)

                if has_real and actual is not None and not actual.empty:
                    st.markdown("#### 🏁 Actual Results")
                    actual_sorted = actual.sort_values("Position")
                    st.dataframe(
                        actual_sorted[["Abbreviation", "Position"]],
                        hide_index=True,
                        use_container_width=True
                    )

                    st.markdown("#### 📊 Prediction vs Actual Results")
                    comparison_df = predictions.merge(
                        actual[["Abbreviation", "Position"]],
                        on="Abbreviation",
                        how="left"
                    )
                    comparison_df["Actual Position"] = comparison_df["Position"].fillna(0).astype(int)

                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    fig2.patch.set_facecolor("#1e1e1e")
                    ax2.set_facecolor("#1e1e1e")

                    team_colors_scatter = [team_colors.get(team, "#777777") for team in comparison_df["TeamName"]]

                    ax2.scatter(
                        comparison_df["Predicted Position"],
                        comparison_df["Actual Position"],
                        s=100,
                        c=team_colors_scatter,
                        alpha=0.7,
                        edgecolors="white"
                    )

                    for i, txt in enumerate(comparison_df["Abbreviation"]):
                        ax2.annotate(
                            txt,
                            (
                                comparison_df["Predicted Position"].iloc[i],
                                comparison_df["Actual Position"].iloc[i]
                            ),
                            fontsize=9,
                            color="white",
                            ha="center",
                            va="bottom",
                            xytext=(0, 5),
                            textcoords="offset points"
                        )

                    ax2.plot([0, 20], [0, 20], "r--", alpha=0.5)
                    ax2.set_title(f"Prediction vs Actual: {selected_gp} {selected_year}", color="white", fontsize=14)
                    ax2.set_xlabel("Predicted Position", color="white")
                    ax2.set_ylabel("Actual Position", color="white")
                    ax2.tick_params(colors="white")
                    ax2.grid(True, linestyle="--", alpha=0.3)
                    for spine in ax2.spines.values():
                        spine.set_color("#333333")
                    plt.tight_layout()
                    st.pyplot(fig2)

    # -------- Trivia --------
    with tab2:
        st.markdown(
            """
            <div style="background-color: rgba(0,0,0,0.72); padding: 15px; border-radius: 10px; border-top: 4px solid #e10600;">
                <h3 style="color:white; margin-top:0;">🎯 F1 Trivia Challenge</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_trivia_btn, col_score = st.columns([2, 1])

        with col_trivia_btn:
            if st.button("Get New Trivia Questions"):
                with st.spinner("Generating trivia questions..."):
                    st.session_state.trivia_questions = generate_f1_trivia()
                    st.session_state.show_answers = {i: False for i in range(len(st.session_state.trivia_questions))}
                    st.session_state.user_score = 0
                    st.session_state.total_questions = len(st.session_state.trivia_questions)
                    st.session_state.trivia_active = True
                    st.rerun()

        with col_score:
            if st.session_state.total_questions > 0:
                st.markdown(
                    f"""
                    <div style="background-color: rgba(0,50,0,0.72); padding: 10px; border-radius: 8px; text-align: center;">
                        <p style="color:white; margin:0; font-size:1.1em;">Score: {st.session_state.user_score}/{st.session_state.total_questions}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if st.session_state.trivia_questions:
            for i, qa in enumerate(st.session_state.trivia_questions):
                st.markdown(
                    f"""
                    <div class="trivia-question">
                        <p style="color:white; font-size:1.05em; font-weight:bold; margin-bottom:5px;">Question {i + 1}:</p>
                        <p style="color:white;">{qa['question']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if not st.session_state.show_answers[i]:
                    c1, c2 = st.columns([3, 1])

                    with c1:
                        if f"user_answer_{i}" not in st.session_state:
                            st.session_state[f"user_answer_{i}"] = ""
                        user_answer = st.text_input("Your answer:", key=f"answer_input_{i}")

                    with c2:
                        if st.button("Submit", key=f"submit_{i}"):
                            st.session_state[f"user_answer_{i}"] = user_answer
                            st.session_state.show_answers[i] = True

                            correct_answer = qa["answer"].strip().lower()
                            user_input = user_answer.strip().lower()

                            if user_input == correct_answer:
                                is_correct = True
                            else:
                                key_terms = correct_answer.split()
                                significant_terms = [term for term in key_terms if len(term) > 3]
                                matches = [term for term in significant_terms if term in user_input]
                                is_correct = bool(matches)

                            if is_correct:
                                st.session_state.user_score += 1

                            st.rerun()

                if st.session_state.show_answers[i]:
                    user_response = st.session_state.get(f"user_answer_{i}", "")
                    correct_answer = qa["answer"].strip().lower()
                    user_input = user_response.strip().lower()

                    if user_input == correct_answer:
                        is_correct = True
                    else:
                        key_terms = correct_answer.split()
                        significant_terms = [term for term in key_terms if len(term) > 3]
                        matches = [term for term in significant_terms if term in user_input]
                        is_correct = bool(matches)

                    bg = "rgba(0,100,0,0.5)" if is_correct else "rgba(225,0,0,0.5)"
                    result_text = "✅ Correct!" if is_correct else "❌ Incorrect"

                    st.markdown(
                        f"""
                        <div class="trivia-answer" style="background-color:{bg};">
                            <p style="color:white;"><strong>Your answer:</strong> {user_response}</p>
                            <p style="color:white;"><strong>Correct answer:</strong> {qa['answer']}</p>
                            <p style="color:white;"><strong>Result:</strong> {result_text}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("Click the button above to get some challenging F1 trivia questions.")

        if st.session_state.trivia_questions and st.session_state.trivia_active:
            if st.button("End Trivia Game"):
                st.session_state.trivia_active = False
                st.rerun()

# =========================
# Column 3: Live Race Simulation
# =========================
with col3:
    st.markdown(
        """
        <div style="background-color: rgba(0,0,0,0.72); padding: 15px; border-radius: 10px; border-top: 4px solid #e10600;">
            <h3 style="color:white; margin-top:0;">🌀 Live Race Tracking</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "predictions" not in st.session_state or st.session_state.predictions is None or st.session_state.predictions.empty:
        st.info("Predict a race first, then start the simulation to see live race tracking.")
    else:
        if "race_active" not in st.session_state:
            st.session_state.race_active = False
            st.session_state.commentary_history = []
            st.session_state.previous_positions = []
            st.session_state.race_points = {}
            st.session_state.auto_run = False

        if not st.session_state.race_active:
            col_laps, col_auto = st.columns(2)

            with col_laps:
                total_laps = st.slider("Number of Laps", min_value=5, max_value=50, value=10, step=5)
                st.session_state.total_laps = total_laps

            with col_auto:
                auto_run = st.checkbox("Auto Run Race", value=False)
                st.session_state.auto_run = auto_run
                if auto_run:
                    st.session_state.sim_speed = st.select_slider(
                        "Simulation Speed",
                        options=["Slow", "Medium", "Fast"],
                        value="Medium"
                    )
                else:
                    st.session_state.sim_speed = "Medium"

        col_start, col_next, col_score = st.columns([1, 1, 1])

        with col_start:
            if not st.session_state.race_active:
                if st.button("Start Race Simulation"):
                    st.session_state.race_active = True
                    st.session_state.race_lap = 0
                    st.session_state.race_data = st.session_state.predictions.copy()
                    st.session_state.race_data["Current Gap"] = st.session_state.race_data["Time Gap (s)"]
                    st.session_state.commentary_history = []
                    st.session_state.previous_positions = []
                    st.session_state.race_points = {
                        driver: 0 for driver in st.session_state.race_data["Abbreviation"].tolist()
                    }
                    st.rerun()
            else:
                if st.button("Reset Simulation"):
                    st.session_state.race_active = False
                    st.session_state.commentary_history = []
                    st.session_state.auto_run = False
                    st.rerun()

        with col_next:
            if st.session_state.race_active and not st.session_state.auto_run:
                if st.button("Next Lap"):
                    if st.session_state.race_lap < st.session_state.total_laps:
                        updated_data, current_lap = simulate_live_race(st.session_state.race_data)
                        commentary = generate_race_commentary(current_lap, updated_data, st.session_state.total_laps)
                        st.session_state.commentary_history.append({"lap": current_lap, "text": commentary})

                        if current_lap == st.session_state.total_laps:
                            points_structure = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
                            for i, (_, driver) in enumerate(updated_data.iterrows()):
                                if i < len(points_structure):
                                    st.session_state.race_points[driver["Abbreviation"]] = points_structure[i]
                        st.rerun()

        with col_score:
            if st.session_state.race_active:
                if st.session_state.race_lap == st.session_state.total_laps:
                    st.markdown(
                        """
                        <div style="background-color: rgba(0,100,0,0.72); padding: 10px; border-radius: 8px; text-align: center; margin: 10px 0;">
                            <p style="color:white; margin:0; font-size:1.1em;">Race Complete!</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(225,6,0,0.72); padding: 10px; border-radius: 8px; text-align: center; margin: 10px 0;">
                            <p style="color:white; margin:0; font-size:1.1em;">Lap: {st.session_state.race_lap}/{st.session_state.total_laps}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        if st.session_state.race_active:
            if st.session_state.auto_run and st.session_state.race_lap < st.session_state.total_laps:
                speed_delays = {"Slow": 3.0, "Medium": 1.5, "Fast": 0.5}
                delay = speed_delays.get(st.session_state.sim_speed, 1.5)

                progress_bar = st.progress(0)
                for i in range(101):
                    progress_bar.progress(i)
                    time.sleep(delay / 100)

                updated_data, current_lap = simulate_live_race(st.session_state.race_data)
                commentary = generate_race_commentary(current_lap, updated_data, st.session_state.total_laps)
                st.session_state.commentary_history.append({"lap": current_lap, "text": commentary})

                if current_lap == st.session_state.total_laps:
                    points_structure = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
                    for i, (_, driver) in enumerate(updated_data.iterrows()):
                        if i < len(points_structure):
                            st.session_state.race_points[driver["Abbreviation"]] = points_structure[i]

                st.rerun()

            col_standings, col_points = st.columns(2)

            with col_standings:
                st.markdown("#### 🏁 Current Race Standings")
                for i, (_, driver) in enumerate(st.session_state.race_data.iterrows()):
                    position = i + 1
                    time_gap = driver["Current Gap"]
                    time_display = "Leader" if i == 0 else f"+{time_gap:.3f}s"
                    team_color = team_colors.get(driver["TeamName"], "#777777")

                    st.markdown(
                        f"""
                        <div class="driver-card" style="border-left: 4px solid {team_color}">
                            <span class="position">{position}</span>
                            <span class="driver-name">{driver['Abbreviation']}</span>
                            <span class="driver-time">{time_display}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with col_points:
                st.markdown("#### 🏆 Championship Points")
                sorted_points = dict(sorted(st.session_state.race_points.items(), key=lambda item: item[1], reverse=True))

                for driver, points in sorted_points.items():
                    driver_team = st.session_state.race_data[
                        st.session_state.race_data["Abbreviation"] == driver
                    ]["TeamName"].values
                    team_color = team_colors.get(driver_team[0] if len(driver_team) > 0 else "", "#777777")

                    st.markdown(
                        f"""
                        <div class="driver-card" style="border-left: 4px solid {team_color}">
                            <span class="driver-name">{driver}</span>
                            <span class="position">{points} pts</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("#### 🎙️ Race Commentary")
            if st.session_state.commentary_history:
                for comment in reversed(st.session_state.commentary_history):
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(30,30,30,0.75); padding: 8px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #e10600;">
                            <p style="color:#e10600; margin:0; font-weight:bold;">Lap {comment['lap']}/{st.session_state.total_laps}</p>
                            <p style="color:white; margin:0;">{comment['text']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("Commentary will appear once the race begins.")
