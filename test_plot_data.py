import pandas as pd
import numpy as np

test_all_file = '/Users/gengyunxin/Documents/项目/traffic_model/TVTS_9.28/data/prediction_compare_step36/test_step36_pre72_smoothed1.csv'
df = pd.read_csv(test_all_file,header=None)
data_all = df.values
two_week = data_all[0] # (10000, 360)
for row in range(1, 119):
    two_week = np.append(two_week, data_all[row][-36:]) # (4608)

one_week = two_week[288*6:288*13]
oneweek_plus3day = two_week[288*5:288*15] # 10天 288*10=2880 头288是真实值去掉，后288是不足sample_len的去掉，多288在最后去掉一个bs以后看看最后去到2304即可
oneweek_plus3day_list = oneweek_plus3day.tolist()
Length = len(oneweek_plus3day_list)
# print(oneweek_plusaday_list)

seq_len = 288
for step in [1, 6, 12, 36, 72]:
    data = []
    print('step: ', step)
    save_path = f'/Users/gengyunxin/Documents/项目/traffic_model/TVTS_9.28/data/prediction_compare_step36/test_plot_{step}.csv'
    sample_len = seq_len + step
    N = ((Length - seq_len) / step) + 1
    print('N: ', N)
    N = int(N)
    for i in range(N):
        data.append(oneweek_plus3day_list[i*step : i*step+sample_len])
    print('Num of data: ', len(data))
    print(len(data[0]))
    print(len(data[-1]))
    data_array = np.array(data[:-1])
    print(data_array.shape)
    np.savetxt(save_path, data_array, delimiter=',')
    print(f'Saving data to {save_path}...')