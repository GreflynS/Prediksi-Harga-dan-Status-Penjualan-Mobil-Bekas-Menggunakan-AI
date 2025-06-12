# Prediksi-Harga-dan-Status-Penjualan-Mobil-Bekas-Menggunakan-AI
#### Anggota: Greflyn - Reyza - Evanly

Library dan Dataset
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper
from bokeh.transform import factor_cmap
import seaborn as sns
import matplotlib.pyplot as plt

output_notebook()

df = pd.read_csv("used_car_sales.csv")
df['Car Sale Status'] = df['Car Sale Status'].str.strip().str.lower()
df = df[df['Car Sale Status'].isin(['sold', 'un sold'])]
