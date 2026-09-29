import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🍔",
    layout="wide"
)

# ============================================================
# SIMPLE NATIVE STREAMLIT STYLE
# No custom HTML/CSS - reliable and easy to run
# ============================================================

st.title("🍔 Food Delivery Time Predictor")

st.caption(
    "Predict estimated food delivery time using order, "
    "traffic, weather, distance and rider information."
)

st.divider()

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


try:
    model = load_model()

except Exception as e:
    st.error("❌ Could not load model.pkl")
    st.code(str(e))
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Prediction Settings")

    st.info(
        "Enter the delivery details and click "
        "'Predict Delivery Time'."
    )

    st.divider()

    st.caption("Food Delivery Time Prediction")
    st.caption("Machine Learning Project")


# ============================================================
# ORDER INFORMATION
# ============================================================

st.subheader("📦 Order Information")

col1, col2, col3 = st.columns(3)

with col1:
    order_hour = st.number_input(
        "Order Hour",
        min_value=0,
        max_value=23,
        value=13,
        step=1,
        key="order_hour"
    )

with col2:
    day_of_week = st.selectbox(
        "Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ],
        key="day_of_week"
    )

with col3:
    is_weekend = st.selectbox(
        "Is Weekend",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        key="is_weekend"
    )


# ============================================================
# WEATHER AND TRAFFIC
# ============================================================

st.divider()

st.subheader("🌦️ Weather & Traffic")

col1, col2, col3 = st.columns(3)

with col1:
    is_festival = st.selectbox(
        "Is Festival",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        key="is_festival"
    )

with col2:
    weather = st.selectbox(
        "Weather",
        [
            "Clear",
            "Rain",
            "Cloudy",
            "Storm",
            "Fog"
        ],
        key="weather"
    )

with col3:
    traffic_level = st.selectbox(
        "Traffic Level",
        [
            "Low",
            "Moderate",
            "High",
            "Severe"
        ],
        key="traffic_level"
    )


# ============================================================
# LOCATION
# ============================================================

st.divider()

st.subheader("📍 Location Information")

col1, col2, col3 = st.columns(3)

with col1:
    pickup_zone = st.selectbox(
        "Pickup Zone",
        [
            "Residential",
            "CBD",
            "Commercial",
            "Industrial",
            "Suburban"
        ],
        key="pickup_zone"
    )

with col2:
    dropoff_zone = st.selectbox(
        "Dropoff Zone",
        [
            "Residential",
            "Commercial",
            "CBD",
            "Suburban",
            "Industrial"
        ],
        key="dropoff_zone"
    )

with col3:
    delivery_distance_category = st.selectbox(
        "Delivery Distance Category",
        [
            "Short",
            "Medium",
            "Long"
        ],
        key="delivery_distance_category"
    )


# ============================================================
# RESTAURANT INFORMATION
# ============================================================

st.divider()

st.subheader("🍽️ Restaurant Information")

col1, col2, col3 = st.columns(3)

with col1:
    cuisine_type = st.selectbox(
        "Cuisine Type",
        [
            "North Indian",
            "Biryani",
            "Pizza",
            "Chinese",
            "Burger",
            "South Indian",
            "Cafe",
            "Bakery",
            "Desserts"
        ],
        key="cuisine_type"
    )

with col2:
    restaurant_load = st.selectbox(
        "Restaurant Load",
        [
            "Low",
            "Medium",
            "High"
        ],
        key="restaurant_load"
    )

with col3:
    order_items = st.number_input(
        "Number of Order Items",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
        key="order_items"
    )


# ============================================================
# RIDER INFORMATION
# ============================================================

st.divider()

st.subheader("🛵 Rider Information")

col1, col2, col3 = st.columns(3)

with col1:
    vehicle_type = st.selectbox(
        "Vehicle Type",
        [
            "Bike",
            "Scooter",
            "Electric Scooter",
            "Bicycle"
        ],
        key="vehicle_type"
    )

with col2:
    rider_experience = st.number_input(
        "Rider Experience (Years)",
        min_value=0.0,
        max_value=30.0,
        value=2.0,
        step=0.5,
        key="rider_experience"
    )

with col3:
    rider_rating = st.number_input(
        "Rider Rating",
        min_value=0.0,
        max_value=5.0,
        value=4.5,
        step=0.1,
        key="rider_rating"
    )


# ============================================================
# RATINGS AND PREPARATION
# ============================================================

st.divider()

st.subheader("⭐ Ratings & Preparation")

col1, col2, col3 = st.columns(3)

with col1:
    restaurant_rating = st.number_input(
        "Restaurant Rating",
        min_value=0.0,
        max_value=5.0,
        value=4.0,
        step=0.1,
        key="restaurant_rating"
    )

with col2:
    preparation_time = st.number_input(
        "Preparation Time (Minutes)",
        min_value=0.0,
        max_value=180.0,
        value=20.0,
        step=1.0,
        key="preparation_time"
    )

with col3:
    number_of_signals = st.number_input(
        "Number of Signals",
        min_value=0,
        max_value=100,
        value=5,
        step=1,
        key="number_of_signals"
    )


# ============================================================
# DELIVERY DISTANCE
# ============================================================

st.divider()

st.subheader("🛣️ Delivery Distance")

col1, col2 = st.columns(2)

with col1:
    road_distance = st.number_input(
        "Road Distance (km)",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.1,
        key="road_distance"
    )

with col2:
    delivery_priority = st.selectbox(
        "Delivery Priority",
        [
            "Normal",
            "Priority",
            "VIP"
        ],
        key="delivery_priority"
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

st.subheader("🚀 Get Prediction")

predict_button = st.button(
    "🚀 Predict Delivery Time",
    type="primary",
    use_container_width=True,
    key="predict_delivery_time"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_data = pd.DataFrame({
        "Order_Hour": [order_hour],
        "Day_of_Week": [day_of_week],
        "Is_Weekend": [is_weekend],
        "Is_Festival": [is_festival],
        "Weather": [weather],
        "Pickup_Zone": [pickup_zone],
        "Dropoff_Zone": [dropoff_zone],
        "Vehicle_Type": [vehicle_type],
        "Rider_Experience_Years": [rider_experience],
        "Rider_Rating": [rider_rating],
        "Restaurant_Rating": [restaurant_rating],
        "Cuisine_Type": [cuisine_type],
        "Order_Items": [order_items],
        "Restaurant_Load": [restaurant_load],
        "Preparation_Time_Min": [preparation_time],
        "Road_Distance_km": [road_distance],
        "Delivery_Distance_Category": [
            delivery_distance_category
        ],
        "Traffic_Level": [traffic_level],
        "Number_of_Signals": [number_of_signals],
        "Delivery_Priority": [delivery_priority]
    })

    try:

        prediction = model.predict(input_data)

        predicted_time = float(prediction[0])

        st.success("✅ Prediction completed successfully!")

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "⏱️ Estimated Delivery Time",
                f"{int(round(predicted_time))} min"
            )

        with col2:
            st.metric(
                "🕐 Approximately",
                f"{predicted_time / 60:.2f} hrs"
            )

        with col3:
            st.metric(
                "📍 Road Distance",
                f"{road_distance:.1f} km"
            )

        st.divider()

        st.subheader("📊 Prediction Summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.write("**Traffic Level:**", traffic_level)
            st.write("**Weather:**", weather)
            st.write("**Distance Category:**", delivery_distance_category)
            st.write("**Vehicle:**", vehicle_type)
            st.write("**Delivery Priority:**", delivery_priority)

        with summary_col2:

            st.write("**Pickup Zone:**", pickup_zone)
            st.write("**Dropoff Zone:**", dropoff_zone)
            st.write("**Restaurant Load:**", restaurant_load)
            st.write("**Preparation Time:**", f"{preparation_time:.0f} min")
            st.write("**Rider Rating:**", f"{rider_rating:.1f} / 5")

        # ----------------------------------------------------
        # INPUT DATA
        # ----------------------------------------------------

        with st.expander("🔍 View Complete Input Data"):

            st.dataframe(
                input_data,
                use_container_width=True
            )

    except Exception as e:

        st.error("❌ Prediction failed")

        st.write("Error details:")

        st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🍔 Food Delivery Time Prediction"
)
