import os
# set openblas thread number to 1
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'

plt.style.use('shera')


# read file with spaces as delimiter
df = pd.read_csv('metrics.txt', delimiter='\s+')

df.columns

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(df['iter_num'], df['rmse_epa'], 'o-', color='blue', markersize=4)
plt.ylabel('RMSE EPA')
plt.title('Training Metrics')

plt.subplot(3, 1, 2)
plt.plot(df['iter_num'], df['rmse_f_comp'], 'o-', color='red', markersize=4)
plt.ylabel('RMSE F Comp')

plt.subplot(3, 1, 3)
plt.plot(df['iter_num'], df['loss'], 'o-', color='green', markersize=4)
plt.ylabel('Loss')
plt.xlabel('Iteration Number')

plt.tight_layout()
# plt.show()
os.makedirs('analysis', exist_ok=True)
plt.savefig('analysis/training_metrics.png')


plt.figure(figsize=(12, 8))

# Skip some initial frames
skip_frames = 1000
df_plot = df.iloc[skip_frames:]

plt.subplot(3, 1, 1)
plt.plot(df_plot['iter_num'], df_plot['rmse_epa'], 'o-', color='blue', markersize=4)
plt.ylabel('RMSE EPA')
plt.title('Training Metrics')

plt.subplot(3, 1, 2)
plt.plot(df_plot['iter_num'], df_plot['rmse_f_comp'], 'o-', color='red', markersize=4)
plt.ylabel('RMSE F Comp')

plt.subplot(3, 1, 3)
plt.plot(df_plot['iter_num'], df_plot['loss'], 'o-', color='green', markersize=4)
plt.ylabel('Loss')
plt.xlabel('Iteration Number')

plt.tight_layout()
# plt.show()
os.makedirs('analysis', exist_ok=True)
plt.savefig('analysis/training_metrics_zoom.png')
