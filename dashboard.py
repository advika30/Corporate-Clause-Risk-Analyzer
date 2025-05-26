import streamlit as st
import pandas as pd
import hashlib
import plotly.express as px
from datetime import datetime

# Load user data
@st.cache_data
def load_user_data():
    try:
        df = pd.read_csv("users.csv", header=None, names=["username", "hashed_password", "timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["username", "hashed_password", "timestamp"])

# Page config
st.set_page_config("User Management Dashboard", layout="wide")

# Title & Intro
st.markdown("""
    <style>
        .gradient-text {
            font-size: 48px;
            font-weight: bold;
            background: linear-gradient(90deg, #1995AD, #A1C6EA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>

    <h1 class='gradient-text'>👥 User Management Dashboard</h1>
    <p style='color:#555;'>Manage user accounts, detect weak security, and review account activity.</p>
""", unsafe_allow_html=True)

df = load_user_data()

# Handle missing or invalid timestamps by dropping NaT rows
df = df.dropna(subset=["timestamp"])

# Sidebar search and filters
st.sidebar.header("🔍 Search & Filters")

# Username search
search_term = st.sidebar.text_input("Search username:")

# Date filters
st.sidebar.subheader("📅 Date Filters")

# Extract available years and months from the data
if not df.empty and not df["timestamp"].isnull().all():
    available_years = sorted(df["timestamp"].dt.year.unique().tolist())
    available_months = list(range(1, 13))  # 1-12 for months
else:
    available_years = [datetime.now().year]
    available_months = list(range(1, 13))

# Year filter
selected_year = st.sidebar.selectbox(
    "Filter by Year:",
    ["All Years"] + available_years,
    index=0
)

# Month filter
selected_month = st.sidebar.selectbox(
    "Filter by Month:",
    ["All Months"] + [datetime(2000, m, 1).strftime('%B') for m in available_months],
    index=0
)

# Apply filters
filtered_df = df.copy()

# Apply username search filter
if search_term:
    filtered_df = filtered_df[filtered_df["username"].str.contains(search_term, case=False, na=False)]

# Apply year filter
if selected_year != "All Years":
    filtered_df = filtered_df[filtered_df["timestamp"].dt.year == selected_year]

# Apply month filter
if selected_month != "All Months":
    # Convert month name to number (1-12)
    month_num = datetime.strptime(selected_month, '%B').month
    filtered_df = filtered_df[filtered_df["timestamp"].dt.month == month_num]

# Reset index for proper display
filtered_df = filtered_df.reset_index(drop=True)

# Summary stats
st.subheader("📊 Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("👤 Total Users", filtered_df.shape[0])
with col2:
    st.metric("🔑 Unique Passwords", filtered_df["hashed_password"].nunique())
with col3:
    reused_count = filtered_df["hashed_password"].duplicated(keep=False).sum()
    st.metric("⚠️ Passwords Reused", reused_count)

# Recent Activity Panel
st.subheader("🕒 Recent User Activity")
recent_users = filtered_df.sort_values(by="timestamp", ascending=False).head(5)[["username", "timestamp"]]
st.table(recent_users)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Users", "⚠️ Reused Passwords", "📈 Reuse Frequency", "📅 User Growth"])

# Tab 1: User Table
with tab1:
    st.subheader("📋 User List")
    st.dataframe(filtered_df[["username", "timestamp"]], use_container_width=True)

# Tab 2: Reused Passwords
with tab2:
    st.subheader("⚠️ Users with Reused Passwords")
    reused_df = filtered_df[filtered_df.duplicated(subset="hashed_password", keep=False)]
    if reused_df.empty:
        st.success("✅ All users have unique passwords!")
    else:
        st.warning("Some users are sharing the same password (hash hidden).")
        grouped = reused_df.groupby("hashed_password")["username"].apply(list).reset_index()
        grouped["User Count"] = grouped["username"].apply(len)
        grouped["Users"] = grouped["username"].apply(lambda x: ", ".join(x))
        st.dataframe(grouped[["Users", "User Count"]], use_container_width=True)

# Tab 3: Password Reuse Frequency
with tab3:
    st.subheader("🔢 Password Reuse Frequency (Anonymized)")
    if reused_df.empty:
        st.info("No reused passwords to display.")
    else:
        reuse_counts = reused_df["hashed_password"].value_counts().reset_index()
        reuse_counts.columns = ["Anonymized Password", "User Count"]
        reuse_counts["Anonymized Password"] = reuse_counts["Anonymized Password"].apply(lambda x: "hash_" + str(hash(x))[-6:])
        fig = px.bar(
            reuse_counts,
            x="Anonymized Password",
            y="User Count",
            title="Password Reuse (Anonymized)",
            color="User Count",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: User Signup Growth Over Time
with tab4:
    st.subheader("📅 User Signup Growth")
    if filtered_df["timestamp"].isnull().all():
        st.warning("Timestamps not available for user signups.")
    else:
        filtered_df["date"] = filtered_df["timestamp"].dt.date
        signup_trend = filtered_df.groupby('date').size().reset_index(name='User Count')
        
        # Display time range in title if filters are applied
        title = 'User Signups Over Time'
        if selected_year != "All Years" or selected_month != "All Months":
            filter_text = []
            if selected_month != "All Months":
                filter_text.append(selected_month)
            if selected_year != "All Years":
                filter_text.append(str(selected_year))
            title += f" ({' '.join(filter_text)})"
            
        fig = px.line(signup_trend, x='date', y='User Count', title=title, markers=True)
        st.plotly_chart(fig, use_container_width=True)

# Optional: Download current filtered view
st.sidebar.markdown("### 📥 Download Filtered Users")
st.sidebar.download_button(
    "Download CSV",
    filtered_df[["username", "timestamp"]].to_csv(index=False).encode(),
    file_name="filtered_users.csv",
    mime="text/csv"
)
