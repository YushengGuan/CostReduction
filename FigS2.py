import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Arial'  # 或 'sans-serif', 'cursive', 'fantasy', 'monospace'

random_count = 100
np.random.seed(0)
df_capacity_solar = pd.read_excel('Data_raw.xlsx', sheet_name='Capacity_solar')
df_capacity_wind = pd.read_excel('Data_raw.xlsx', sheet_name='Capacity_wind')
df_capacity_solar_cum = pd.read_excel('Data_raw.xlsx', sheet_name='Scn_nat_solar')
df_capacity_wind_cum = pd.read_excel('Data_raw.xlsx', sheet_name='Scn_nat_wind')
df_cost_solar = pd.read_excel('Data_raw.xlsx', sheet_name='Cost_solar')
df_cost_wind = pd.read_excel('Data_raw.xlsx', sheet_name='Cost_wind')
result_solar = pd.read_excel('results/Reg_result_solar.xlsx')
result_wind = pd.read_excel('results/Reg_result_wind.xlsx')

country_wind = ['Global', 'Denmark', 'United States', 'Germany', 'Sweden', 'Italy', 'United Kingdom', 'India',
                'Spain', 'Canada', 'France', 'Turkey', 'Brazil', 'China', 'Others']
country_solar = ['Global', 'Australia', 'France', 'Germany', 'India', 'Italy', 'Japan', 'South Korea', 'Spain',
                 'United Kingdom', 'United States', 'China', 'Others']
cmap = {'Global': 'grey', 'Australia': 'darkorange', 'France': 'mediumpurple', 'Germany': 'saddlebrown',
        'India': 'olive', 'Italy': 'forestgreen', 'Japan': 'khaki', 'South Korea': 'pink', 'Spain': 'aqua',
                 'United Kingdom': 'lightgreen', 'United States': 'steelblue', 'China': 'firebrick',
        'Denmark': 'royalblue', 'Sweden': 'chocolate', 'Canada':  'hotpink', 'Turkey': 'bisque', 'Brazil': 'darkgreen',
        'Others': 'grey'}


x0 = np.linspace(1, 12, 12)
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.07, bottom=0.2, right=0.95, top=0.9, wspace=0.2, hspace=0.6)
cost_sim_random_solar = []  # c, m, t
cost_price_random_solar = []
cost_self_random_solar = []
d_price, d_national, d_global = [], [], []
se_price, se_national, se_global = [], [], []
ng = pd.DataFrame(index=range(2011, 2023), columns=country_solar[1:], dtype=float)
ng_label = pd.DataFrame(index=range(2011, 2023), columns=country_solar[1:], dtype=str)
for c in country_solar[1:]:
    ind = country_solar[1:].index(c)
    cap_global_cum = np.array(df_capacity_solar['Global_cum'])[10:23]  # Year 2010-2022
    price_si = np.array(df_cost_solar['price_si'])
    co = result_solar[c][:3].values
    se = result_solar[c][3:6].values
    cost_sim = np.exp(co[0] + co[1] * np.log(cap_global_cum) + co[2] * np.log(price_si))
    co_random = np.random.normal(co[1], se[1], size=random_count)
    cost_obs = np.array(df_cost_solar[c])
    qt_price = np.array(df_capacity_solar_cum['const'])[10:23]  # Year 2010-2022, constant capacity of 2010
    cost_price = np.exp(co[0] + co[1] * np.log(qt_price) + co[2] * np.log(price_si))  # only price_si matters
    qt_self = np.array(df_capacity_solar_cum[c])[10:23]  # Year 2010-2022, cumulative
    cost_self = np.exp(co[0] + co[1] * np.log(qt_self) + co[2] * np.log(price_si))
    gap = cost_self - cost_sim 
    d_price.append(cost_sim[0] - cost_price[-1])
    d_national.append(cost_price[-1] - cost_self[-1])
    d_global.append(cost_self[-1] - cost_sim[-1])
    for t in range(len(x0)):
        if (cost_price[t+1] - cost_self[t+1]) / (cost_self[t+1] - cost_sim[t+1]) >= 1:
            ng[c].iloc[t] = 1
            ng_label[c].iloc[t] = f'{round((cost_price[t+1] - cost_self[t+1]) / (cost_self[t+1] - cost_sim[t+1]), 2)}'
        else:
            ng[c].iloc[t] = 0
            ng_label[c].iloc[t] = f'{round((cost_price[t+1] - cost_self[t+1]) / (cost_self[t+1] - cost_sim[t+1]), 2)}'
heatmap = sns.heatmap(ng, vmin=0, vmax=1, annot=ng_label, fmt='s', cbar=False, cmap=['#B1D8B1', '#D9F0F8'], 
xticklabels=country_solar[1:], yticklabels=range(2011, 2023))
plt.text(-0.9, -0.5, 'a.', fontsize=20, fontweight='bold')
plt.text(-0.9, -0.5, '      Solar PV', fontsize=15)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.xlabel('Country', fontsize=15)
plt.ylabel('Year', fontsize=15)
plt.savefig('figs/FigS2_1.jpg', dpi=600)
plt.show()


country_wind2 = ['Global', 'Denmark', 'United States', 'Germany', 'Sweden', 'Italy', 'United Kingdom', 'India',
                 'Spain', 'Canada', 'France', 'Turkey', 'Brazil', 'China', 'Others']
country_wind = ['Global', 'Denmark', 'United States', 'Germany', 'Sweden', 'Italy', 'United Kingdom', 'India',
                 'Spain', 'Canada', 'France', 'Turkiye', 'Brazil', 'China', 'Others']
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.07, bottom=0.2, right=0.95, top=0.9, wspace=0.2, hspace=0.6)
cost_sim_random_wind = []  # c, m, t
cost_self_random_wind = []
d_national, d_global = [], []
se_national, se_global = [], []
ng = pd.DataFrame(index=range(2011, 2023), columns=country_wind2[1:], dtype=float)
ng_label = pd.DataFrame(index=range(2011, 2023), columns=country_wind2[1:], dtype=str)
for c in country_wind2[1:]:
    ind = country_wind2[1:].index(c)
    cap_global_cum = np.array(df_capacity_wind['Global_cum'])[10:23]  # Year 2010-2022
    co = result_wind[c][:2].values
    cost_sim = np.exp(co[0] + co[1] * np.log(cap_global_cum))
    cost_obs = np.array(df_cost_wind[c])
    qt_self = np.array(df_capacity_wind_cum[c])[10:23]  # Year 2010-2022, cumulative
    cost_self = np.exp(co[0] + co[1] * np.log(qt_self))
    gap = cost_self - cost_sim
    d_national.append(cost_sim[0] - cost_self[-1])
    d_global.append(cost_self[-1] - cost_sim[-1])
    se = result_wind[c][2:4].values
    for t in range(len(x0)):
        if (cost_sim[0] - cost_self[t+1]) / (cost_self[t+1] - cost_sim[t+1]) >= 1:
            ng[c].iloc[t] = 1
            ng_label[c].iloc[t] = f'{round((cost_sim[0] - cost_self[t+1]) / (cost_self[t+1] - cost_sim[t+1]), 2)}'
        else:
            ng[c].iloc[t] = 0
            ng_label[c].iloc[t] = f'{round((cost_sim[0] - cost_self[t+1]) / (cost_self[t+1] - cost_sim[t+1]), 2)}'
heatmap = sns.heatmap(ng, vmin=0, vmax=1, annot=ng_label, fmt='s', cbar=False, cmap=['#B1D8B1', '#D9F0F8'], 
xticklabels=country_wind[1:], yticklabels=range(2011, 2023))
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.text(-1, -0.5, 'b.', fontsize=20, fontweight='bold')
plt.text(-1, -0.5, '      Wind power', fontsize=15)
plt.xlabel('Country', fontsize=15)
plt.ylabel('Year', fontsize=15)
plt.savefig('figs/FigS2_2.jpg', dpi=600)
plt.show()

fig, ax = plt.subplots(figsize=(10, 12), frameon=False)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
grid = plt.GridSpec(2, 1)
plt.subplot(grid[0, :])
plt.imshow(plt.imread('figs/FigS2_1.jpg'))
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.subplot(grid[1, :])
plt.imshow(plt.imread('figs/FigS2_2.jpg'))
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.savefig('figs/FigS2.jpg', dpi=600)
plt.show()

