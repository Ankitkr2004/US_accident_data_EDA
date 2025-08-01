# STEP 0: Install if needed
# !pip install pandas matplotlib seaborn plotly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# STEP 1: Load only first 50,000 rows
df = pd.read_csv('accident.csv', nrows=50000, parse_dates=['Start_Time', 'End_Time'])

# STEP 2: Initial Exploration
print("Shape of data:", df.shape)
print(df.info())
print(df.describe(include='all').T)

# STEP 3: Feature Engineering
df = df.dropna(subset=['Start_Time', 'End_Time', 'State'])
df['hour'] = df['Start_Time'].dt.hour
df['date'] = df['Start_Time'].dt.date
df['day_of_week'] = df['Start_Time'].dt.day_name()
df['month'] = df['Start_Time'].dt.month

# Common figure size
FIG_SIZE = (8, 4)

# STEP 4: Accidents by Hour
plt.figure(figsize=FIG_SIZE)
sns.countplot(x='hour', data=df, palette='viridis')
plt.title('Accidents by Hour')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# STEP 5: Accidents by Day
plt.figure(figsize=FIG_SIZE)
sns.countplot(x='day_of_week', data=df, order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], palette='magma')
plt.title('Accidents by Day of Week')
plt.xlabel('Day')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# STEP 6: Severity vs Hour (Plotly interactive)
fig = px.histogram(df, x='hour', color='Severity', barmode='group',
                   title='Severity by Hour', width=800, height=300)
fig.show()

# STEP 7: Contributing Factors
factors = ['Weather_Condition', 'Temperature(F)', 'Sunrise_Sunset', 'Severity']
for col in factors:
    plt.figure(figsize=FIG_SIZE)
    if df[col].dtype == 'object':
        sns.countplot(y=col, data=df, order=df[col].value_counts().index[:10])
        plt.title(f'Most Common {col}')
    else:
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

# STEP 8: Weather vs Hour (Top 5)
plt.figure(figsize=FIG_SIZE)
weather_hour = df.groupby(['hour', 'Weather_Condition']).size().reset_index(name='counts')
top_weather = weather_hour.groupby('Weather_Condition')['counts'].sum().nlargest(5).index
subset = weather_hour[weather_hour['Weather_Condition'].isin(top_weather)]
sns.lineplot(data=subset, x='hour', y='counts', hue='Weather_Condition')
plt.title('Top Weather Conditions by Hour')
plt.xlabel('Hour')
plt.ylabel('Accident Count')
plt.tight_layout()
plt.show()

# STEP 9: Daily Accidents
daily_acc = df.groupby('date').size().reset_index(name='daily_count')
plt.figure(figsize=FIG_SIZE)
sns.lineplot(data=daily_acc, x='date', y='daily_count')
plt.title('Daily Accident Count')
plt.xlabel('Date')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# STEP 10: Monthly Pattern
plt.figure(figsize=FIG_SIZE)
sns.countplot(x='month', data=df, palette='coolwarm')
plt.title('Accidents by Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# STEP 11: Correlation Heatmap
num_cols = ['Temperature(F)', 'Humidity(%)', 'Wind_Speed(mph)', 'Visibility(mi)', 'Precipitation(in)']
corr = df[num_cols].corr()
plt.figure(figsize=FIG_SIZE)
sns.heatmap(corr, annot=True, cmap='vlag')
plt.title('Correlation of Environmental Factors')
plt.tight_layout()
plt.show()
