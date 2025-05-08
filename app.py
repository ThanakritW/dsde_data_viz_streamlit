import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import plotly.figure_factory as ff

st.set_page_config(page_title="Bangkok Traffy Fondue Analysis", layout="wide")
st.title("Bangkok Traffy Fondue Analysis")

# Load data using st.cache_data
@st.cache_data
def load_data():
    data_traffy = pd.read_parquet("valid_data_with_predictions.parquet")
    data_traffy['state'] = data_traffy['state'].replace({
        'inprogress': 'กำลังดำเนินการ',
        'forward': 'รอรับเรื่อง',
        'follow': 'รอรับเรื่อง',
        'finish': 'เสร็จสิ้น'
    })
    data_traffy[['lon', 'lat']] = data_traffy['coords'].str.split(",", expand=True).astype(float)
    data_traffy['position'] = data_traffy[['lon', 'lat']].values.tolist()
    data_traffy['position'] = data_traffy['position'].apply(lambda x: [float(x[0]), float(x[1])])
    data_traffy['star'] = data_traffy['star'].fillna(0)
    data_traffy = data_traffy[data_traffy['district'] != 'NaN']
    data_traffy = data_traffy.dropna(subset=['coords', "ticket_id"])
    return data_traffy

data_traffy = load_data()

# Sidebar filters
st.sidebar.header("Filter Options")
star_range = st.sidebar.slider(
    "Rating Range (0 is no rating)", 0, 5, (0, 5), step=1, key="star_slider"
)
map_style = st.sidebar.selectbox(
    "Select Base Map Style", ['Dark', 'Light', 'Road', 'Satellite'], index=0, key="map_selectbox"
)
rating_source = st.sidebar.selectbox(
    "Rating Source", ["Both", "Only Actual", "Only Predicted"], index=0, key="rating_selectbox"
)

MAP_STYLES = {
    "Dark": "mapbox://styles/mapbox/dark-v10",
    "Light": "mapbox://styles/mapbox/light-v10",
    "Road": "mapbox://styles/mapbox/streets-v11",
    "Satellite": "mapbox://styles/mapbox/satellite-v9",
}

# Filter data based on star_range and rating_source
@st.cache_data
def filter_data(df, star_range, rating_source):
    filtered_df = df.copy()
    filtered_df = filtered_df[
        (df['star'] >= star_range[0]) & (df['star'] <= star_range[1])
    ]
    if rating_source == "Only Actual":
        filtered_df["star_predicted"] = None
    elif rating_source == "Only Predicted":
        filtered_df["star"] = None
    return filtered_df

filtered_data_traffy = filter_data(data_traffy, star_range, rating_source)

# Calculate and display metrics
@st.cache_data
def calculate_metrics(df, rating_source):
    rated_df = df[df['star'] > 0 if rating_source != "Only Predicted" else df["star_predicted"] > 0]
    total_tickets = len(df)
    completed_tickets = (df["state"] == "เสร็จสิ้น").sum()
    avg_rating = (rated_df["star"].mean() if rating_source != "Only Predicted" else rated_df["star_predicted"].mean()) if not rated_df.empty else 0
    return total_tickets, completed_tickets, avg_rating

total_tickets, completed_tickets, avg_rating = calculate_metrics(filtered_data_traffy, rating_source)

st.header("Actual Data Analysis")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Ticket", total_tickets)
with col2:
    st.metric("Total Ticket Completed", completed_tickets)
with col3:
    st.metric("Average Rating", f"⭐{avg_rating:.1f}")

# Model Evaluation Metrics
@st.cache_data  # Cache the results
def calculate_evaluation_metrics(df):
    conf_matrix_data = df.dropna(subset=["star", "star_predicted"])
    if not conf_matrix_data.empty:
        y_true = conf_matrix_data["star"].astype(int)
        y_pred = conf_matrix_data["star_predicted"].astype(int)

        total_predict = len(y_pred)
        total_correct = (y_true == y_pred).sum()
        accuracy = total_correct / total_predict if total_predict > 0 else 0  # Prevent ZeroDivisionError

        # Calculate F1-score using sklearn
        f1 = f1_score(y_true, y_pred, average='weighted')  # Use weighted for multi-class

        return total_predict, total_correct, f1, accuracy
    return 0, 0, 0, 0  # Return zeros if no data


# Display Evaluation Metrics
st.header("Model Evaluation Metrics")
total_predict, total_correct, f1, accuracy = calculate_evaluation_metrics(filtered_data_traffy)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Predicted", total_predict)
with col2:
    st.metric("Total Correct", total_correct)
with col3:
    st.metric("F1-score", f"{f1:.3f}") # Format with 3 decimal places
with col4:
    st.metric("Accuracy", f"{accuracy:.3f}") # Format with 3 decimal places

# Confusion Matrix
@st.cache_resource
def create_confusion_matrix(df):
    conf_matrix_data = df.dropna(subset=["star", "star_predicted"])
    if not conf_matrix_data.empty:
        conf_matrix = confusion_matrix(conf_matrix_data["star"].astype(int), conf_matrix_data["star_predicted"].astype(int))
        fig = ff.create_annotated_heatmap(z=conf_matrix, x=list(range(1, 6)), y=list(range(1, 6)),
                                          showscale=True, colorscale="Viridis", annotation_text=conf_matrix.astype(str))
        fig.update_layout(xaxis_title="Predicted Star Rating",
                          yaxis_title="Actual Star Rating", width=600, height=600)
        return fig
    return None  # Return None if no figure is created



st.header("Confusion Matrix (Actual vs. Predicted)")
fig_confusion = create_confusion_matrix(filtered_data_traffy)
if fig_confusion:
    st.plotly_chart(fig_confusion)
else:
    st.warning("Not enough data to generate a confusion matrix.")



# Rating Distribution Plot
@st.cache_resource
def create_rating_distribution_plot(df, rating_source):
    rated_df = df[df['star'] > 0 if rating_source != "Only Predicted" else df["star_predicted"] > 0]
    star_counts = rated_df["star"].value_counts().sort_index() if rating_source != "Only Predicted" else rated_df["star_predicted"].value_counts().sort_index()
    fig = px.bar(x=star_counts.index, y=star_counts.values,
                 labels={"x": "Star Rating", "y": "Count"},
                 title="Distribution of Star Ratings (1-5)",
                 barmode='group')
    if rating_source == "Both":
        predicted_star_counts = rated_df["star_predicted"].value_counts().sort_index()
        fig.add_bar(x=predicted_star_counts.index, y=predicted_star_counts.values, name="Predicted Star Rating")
    return fig

st.header("Rating Distribution")
fig_bar_rating = create_rating_distribution_plot(filtered_data_traffy, rating_source)
st.plotly_chart(fig_bar_rating)


# Duration Analysis Plots
@st.cache_resource
def create_duration_plots(df, rating_source):
    rated_df = df[df['star'] > 0 if rating_source != "Only Predicted" else df["star_predicted"] > 0]
    bin_size = 1000
    max_duration = rated_df['duration'].max()
    bins = list(range(0, int(max_duration) + bin_size, bin_size))
    labels = [f"{b}-{b + bin_size - 1}" for b in bins[:-1]]
    rated_df['duration_bin'] = pd.cut(rated_df['duration'], bins=bins, labels=labels, include_lowest=True)

    count_data = rated_df['duration_bin'].value_counts().sort_index().reset_index()
    count_data.columns = ['duration_bin', 'count']
    fig_count = px.bar(count_data, x='duration_bin', y='count',
                       labels={'count': 'Row Count', 'duration_bin': 'Duration Range (Hour)'},
                       title='Row Count by Duration Bin', barmode='group')


    mean_data = rated_df.groupby('duration_bin')['star'].mean().reset_index() if rating_source != "Only Predicted" else rated_df.groupby('duration_bin')['star_predicted'].mean().reset_index()
    fig_mean = px.bar(mean_data, x='duration_bin', y='star' if rating_source != "Only Predicted" else 'star_predicted',
                     labels={'star' if rating_source != "Only Predicted" else 'star_predicted': 'Average Star Rating',
                             'duration_bin': 'Duration Range (Hour)'},
                     title='Average Star Rating by Duration Bin', barmode='group')
    if rating_source == "Both":
        mean_data_predicted = rated_df.groupby('duration_bin')['star_predicted'].mean().reset_index()
        fig_mean.add_bar(x=mean_data_predicted['duration_bin'], y=mean_data_predicted['star_predicted'],
                         name="Average Predicted Star Rating")


    grouped = rated_df.groupby('duration_bin').agg(
        {'duration': 'mean', 'star' if rating_source != "Only Predicted" else 'star_predicted': 'mean'}).reset_index()
    fig_avg_corr = px.scatter(grouped, x='duration', y='star' if rating_source != "Only Predicted" else 'star_predicted',
                             trendline='ols',
                             labels={'duration': 'Average Duration',
                                     'star' if rating_source != "Only Predicted" else 'star_predicted': 'Average Star Rating'},
                             title='Correlation Between Average Duration and Average Star Rating (per Bin)')
    correlation_avg = grouped['duration'].corr(grouped['star' if rating_source != "Only Predicted" else 'star_predicted'])
    return fig_count, fig_mean, fig_avg_corr, correlation_avg

fig_count, fig_mean, fig_avg_corr, correlation_avg = create_duration_plots(filtered_data_traffy, rating_source)


st.plotly_chart(fig_count)
st.plotly_chart(fig_mean)
st.plotly_chart(fig_avg_corr)
st.write(f"**Pearson Correlation Coefficient (Average):** {correlation_avg:.3f}")



# District and Subdistrict Analysis functions (cached)
@st.cache_resource
def plot_district_analysis(df, rating_col, title_suffix=""):
    summary = df.groupby('district').agg(
        row_count=('district', 'count'),
        avg_star=(rating_col, 'mean')
    ).reset_index()
    avg_row_count = summary['row_count'].mean()
    avg_star = summary['avg_star'].mean()
    fig = px.scatter(
        summary,
        x='row_count', y='avg_star',
        text='district',
        labels={'row_count': 'Number of Tickets', 'avg_star': 'Average Star'},
        title=f"District-wise Ticket Volume vs. Average Star {title_suffix}"
    )
    fig.add_shape(type='line', x0=avg_row_count, x1=avg_row_count, y0=summary['avg_star'].min(), y1=summary['avg_star'].max(),
                 line=dict(color='Red', dash='dash'), name='Avg Ticket Count')
    fig.add_shape(type='line', x0=summary['row_count'].min(), x1=summary['row_count'].max(), y0=avg_star, y1=avg_star,
                 line=dict(color='Blue', dash='dash'), name='Avg Star Rating')
    fig.add_annotation(x=avg_row_count, y=summary['avg_star'].max(), text='Avg Tickets', showarrow=False, yanchor='bottom',
                      font=dict(color='Red'))
    fig.add_annotation(x=summary['row_count'].max(), y=avg_star, text='Avg Star', showarrow=False, xanchor='right',
                      font=dict(color='Blue'))
    fig.update_traces(textposition='top center')
    fig.update_layout(hovermode='closest', height=600)
    st.plotly_chart(fig)

@st.cache_resource
def plot_subdistrict_analysis(df, selected_district, rating_col, title_suffix=""):
    sub_data = df[df['district'] == selected_district]
    sub_summary = sub_data.groupby('subdistrict').agg(
        row_count=('subdistrict', 'count'),
        avg_star=(rating_col, 'mean')
    ).reset_index()

    # Calculate averages for reference lines
    avg_row_count_sub = sub_summary['row_count'].mean()
    avg_star_sub = sub_summary['avg_star'].mean()

    # Create scatter plot
    fig_subdistrict = px.scatter(
        sub_summary,
        x='row_count',
        y='avg_star',
        text='subdistrict',
        labels={'row_count': 'Number of Tickets', 'avg_star': 'Average Star'},
        title=f'Subdistrict-wise Ticket Volume vs. Average Star in {selected_district}'
    )

    # Add average lines
    fig_subdistrict.add_shape(
        type='line',
        x0=avg_row_count_sub, x1=avg_row_count_sub,
        y0=sub_summary['avg_star'].min(), y1=sub_summary['avg_star'].max(),
        line=dict(color='Red', dash='dash')
    )

    fig_subdistrict.add_shape(
        type='line',
        x0=sub_summary['row_count'].min(), x1=sub_summary['row_count'].max(),
        y0=avg_star_sub, y1=avg_star_sub,
        line=dict(color='Blue', dash='dash')
    )

    # Add annotations
    fig_subdistrict.add_annotation(
        x=avg_row_count_sub, y=sub_summary['avg_star'].max(),
        text='Avg Tickets',
        showarrow=False,
        yanchor='bottom',
        font=dict(color='Red')
    )

    fig_subdistrict.add_annotation(
        x=sub_summary['row_count'].max(), y=avg_star_sub,
        text='Avg Star',
        showarrow=False,
        xanchor='right',
        font=dict(color='Blue')
    )

    # Final touches
    fig_subdistrict.update_traces(textposition='top center')
    fig_subdistrict.update_layout(
        hovermode='closest',
        height=600
    )

    # Show chart
    st.plotly_chart(fig_subdistrict)


# District Analysis
st.title("District-wise Ticket Analysis")

if rating_source in ("Both", "Only Actual"):
    plot_district_analysis(filtered_data_traffy, "star", title_suffix="(Actual)")
if rating_source in ("Both", "Only Predicted"):
    plot_district_analysis(filtered_data_traffy, "star_predicted", title_suffix="(Predicted)")


# Subdistrict Analysis
selected_district = st.selectbox(
    "Select a District for Subdistrict-level Analysis",
    sorted(filtered_data_traffy['district'].unique()), key="subdistrict_selectbox"
)


if rating_source in ("Both", "Only Actual"):
    plot_subdistrict_analysis(filtered_data_traffy, selected_district, "star", title_suffix="(Actual)")
if rating_source in ("Both", "Only Predicted"):
    plot_subdistrict_analysis(filtered_data_traffy, selected_district, "star_predicted", title_suffix="(Predicted)")


# Hotspot Analysis (largely unchanged, but make sure viz_data_traffy is derived from filtered_data_traffy)
st.header('Ticket Hotspot Analysis')

try:
    # Analyze clusters
    clusters_count = filtered_data_traffy['district'].value_counts()
    clusters_count = clusters_count[clusters_count.index != -1]  # Exclude noise points
    top_clusters = clusters_count
    
    # Generate colors for clusters
    unique_clusters = filtered_data_traffy[filtered_data_traffy['district'].isin(top_clusters.index)]['district'].unique()
    colormap = plt.get_cmap('hsv')
    cluster_colors = {cluster: [int(x*255) for x in colormap(i/len(unique_clusters))[:3]] + [160] 
                     for i, cluster in enumerate(unique_clusters)}
    
    # Create visualization
    viz_data_traffy = filtered_data_traffy[filtered_data_traffy['district'].isin(top_clusters.index)].copy()
    viz_data_traffy['color'] = viz_data_traffy['district'].map(cluster_colors)
    
    # Scatter map for clusters
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=viz_data_traffy,
        get_position='position',
        get_color='color',
        get_radius=100,
        pickable=True,
        opacity=0.8,
    )
    scatter_map = pdk.Deck(
        map_style=MAP_STYLES[map_style],
        initial_view_state=pdk.ViewState(
            latitude=13.7563,
            longitude=100.5018,
            zoom=11,
            pitch=0,
        ),
        layers=[scatter_layer],
        tooltip={"text": "ID: {ticket_id}\nDistrict: {district}\nSubdistrict: ฿{subdistrict}\nRating: {star}"},
    )
    st.pydeck_chart(scatter_map)

    # Hexagon map for clusters
    hex_layer = pdk.Layer(
        "HexagonLayer",
        data=viz_data_traffy,
        get_position='position',
        radius=500,
        elevation_scale=4,
        elevation_range=[0, 1000],
        pickable=True,
        opacity=0.5,
        get_elevation_weight='star',
        elevation_aggregation='MEAN',
    )

    hex_map = pdk.Deck(
        map_style=MAP_STYLES[map_style],
        initial_view_state=pdk.ViewState(
            latitude=13.7563,
            longitude=100.5018,
            zoom=11,
            pitch=0,
        ),
        layers=[hex_layer],
        tooltip={
            "html": "<b>Average Star:</b> {elevationValue}",
            "style": {"backgroundColor": "black", "color": "white"}
        }
    )
    st.pydeck_chart(hex_map)

except Exception as e:
    st.error(f"Error in clustering analysis: {e}")