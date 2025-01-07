import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Arial'  # 或 'sans-serif', 'cursive', 'fantasy', 'monospace'


df_capacity_solar = pd.read_excel('Data_raw.xlsx', sheet_name='Capacity_solar')
df_capacity_wind = pd.read_excel('Data_raw.xlsx', sheet_name='Capacity_wind')
df_capacity_solar_cum = pd.read_excel('Data_raw.xlsx', sheet_name='Scn_nat_solar')
df_capacity_wind_cum = pd.read_excel('Data_raw.xlsx', sheet_name='Scn_nat_wind')
df_cost_solar = pd.read_excel('Data_raw.xlsx', sheet_name='Cost_solar')
df_cost_wind = pd.read_excel('Data_raw.xlsx', sheet_name='Cost_wind')
result_solar = pd.read_excel('results/Reg_result_solar.xlsx')
result_wind = pd.read_excel('results/Reg_result_wind.xlsx')

country_wind = ['Denmark', 'United States', 'Germany', 'Sweden', 'Italy', 'United Kingdom', 'India',
                'Spain', 'Canada', 'France', 'Turkey', 'Brazil', 'China', 'Others']
country_solar = ['Australia', 'France', 'Germany', 'India', 'Italy', 'Japan', 'South Korea', 'Spain',
                 'United Kingdom', 'United States', 'China', 'Others']
cmap = {'Australia': 'darkorange', 'France': 'mediumpurple', 'Germany': 'saddlebrown',
        'India': 'olive', 'Italy': 'forestgreen', 'Japan': 'khaki', 'South Korea': 'pink', 'Spain': 'aqua',
                 'United Kingdom': 'lightgreen', 'United States': 'steelblue', 'China': 'firebrick',
        'Denmark': 'royalblue', 'Sweden': 'chocolate', 'Canada':  'hotpink', 'Turkey': 'bisque', 'Brazil': 'darkgreen',
        'Others': 'grey'}

x0 = np.linspace(1, 13, 13)
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.3)
cost_obs = np.array(df_cost_solar['Global'])
ax1 = plt.subplot(2, 1, 1)
width = 0.3
capa_global = np.array(df_capacity_solar['Global'])[10:23]  # Year 2010-2022
CostSum_sim = np.array([0] * 13, dtype='float')
CostSum_price = np.array([0] * 13, dtype='float')
CostSum_self = np.array([0] * 13, dtype='float')
cap_global_cum = np.array(df_capacity_solar['Global_cum'])[10:23]  # Year 2010-2022
price_si = np.array(df_cost_solar['price_si'])
for c in country_solar:
    capa_c = np.array(df_capacity_solar[c])[10:23]
    co = result_solar[c][:3].values
    cost_sim_c = np.exp(co[0] + co[1] * np.log(cap_global_cum) + co[2] * np.log(price_si))
    CostSum_sim += cost_sim_c * capa_c
    qt_price_c = np.array(df_capacity_solar_cum['const'])[10:23]  # Year 2010-2022, constant of 2010
    cost_price_c = np.exp(co[0] + co[1] * np.log(qt_price_c) + co[2] * np.log(price_si))
    CostSum_price += cost_price_c * capa_c
    qt_self_c = np.array(df_capacity_solar_cum[c])[10:23]  # Year 2010-2022, cumulative
    cost_self_c = np.exp(co[0] + co[1] * np.log(qt_self_c) + co[2] * np.log(price_si))
    CostSum_self += cost_self_c * capa_c
cost_sim = CostSum_sim / capa_global
cost_self = CostSum_self / capa_global
cost_price = CostSum_price / capa_global

gap_national = cost_price - cost_self
gaps = []
for c in country_solar:
    capa_c = np.array(df_capacity_solar[c])[10:23]
    co = result_solar[c][:3].values
    qt_price_c = np.array(df_capacity_solar_cum['const'])[10:23]  # Year 2010-2022, constant of 2010
    cost_price_c = np.exp(co[0] + co[1] * np.log(qt_price_c) + co[2] * np.log(price_si))
    qt_self_c = np.array(df_capacity_solar_cum[c])[10:23]  # Year 2010-2022, cumulative
    cost_self_c = np.exp(co[0] + co[1] * np.log(qt_self_c) + co[2] * np.log(price_si))
    gaps.append((cost_price_c - cost_self_c)*capa_c)
gaps = np.matrix(gaps)
gaps_r = gaps.copy()
gaps_final = gaps_r.copy()
for j in range(len(x0)):
    for i in range(len(country_solar)):
        if np.sum(gaps, axis=0)[0, j] != 0:
            gaps_r[i, j] = gaps[i, j] / np.sum(gaps, axis=0)[0, j]
        else:
            gaps_r[i, j] = 0
        gaps_final[i, j] = gaps_r[i, j] * gap_national[j]
base = np.array([0]*13, dtype='float')
for i in range(len(country_solar)):
    print('national endeavor')
    print(country_solar[i], np.array(gaps_final[i])[0])
    plt.bar(x0-width/2, np.array(gaps_final[i])[0], width=width, bottom=base, label=country_solar[i], alpha=0.8, color=cmap[country_solar[i]], edgecolor='k')
    base += np.array(gaps_final[i])[0]
ne = base.copy()
plt.plot((x0-width/2)[1:], ne[1:], label='National endeavor', alpha=1, linestyle=':', color='k', marker='.')

gap_global = cost_self - cost_sim
gaps = []
for c in country_solar:
    impact_c = np.array([0]*13, dtype='float')
    capa_c = np.array(df_capacity_solar[c])[10:23]
    cap = np.array(df_capacity_solar[c])[11:23]
    cap_global = np.array(df_capacity_solar['Global'])[11:23]  # Year 2011-2022, additive
    lamda = 1
    qt_add = cap_global - lamda * cap  # additive capacity without country c
    qt = np.array([cap_global_cum[0]])  # Year 2010, world cumulative
    for i in range(len(qt_add)):
        qt = np.append(qt, qt[-1] + qt_add[i])  # Year 2010-2022, cumulative, world cumulative without country c
    for cnt in country_solar:
        if cnt != c:
            # print('cnt:', cnt)
            capa_cnt = np.array(df_capacity_solar[cnt])[10:23]
            co_cnt = result_solar[cnt][:3].values
            cost_sim_cnt = np.exp(co_cnt[0] + co_cnt[1] * np.log(cap_global_cum) + co_cnt[2] * np.log(price_si))
            cost_cnt_c = np.exp(co_cnt[0] + co_cnt[1] * np.log(qt) + co_cnt[2] * np.log(price_si))
            qt_self_cnt = np.array(df_capacity_solar_cum[cnt])[10:23]  # Year 2010-2022, cumulative
            cost_self_cnt = np.exp(co_cnt[0] + co_cnt[1] * np.log(qt_self_cnt) + co_cnt[2] * np.log(price_si))
            gap_cnt = cost_self_cnt - cost_sim_cnt
            impacts_cnt = np.array([0]*13, dtype='float')
            for i in country_solar:
                if i != cnt:
                    qt_i = np.array([cap_global_cum[0]])
                    cap_i = np.array(df_capacity_solar[i])[11:23]
                    qt_add_i = cap_global - lamda * cap_i
                    for j in range(len(qt_add)):
                        qt_i = np.append(qt_i, qt_i[-1] + qt_add_i[j])
                    cost_cnt_i = np.exp(co_cnt[0] + co_cnt[1] * np.log(qt_i) + co_cnt[2] * np.log(price_si))
                    impacts_cnt += cost_cnt_i - cost_sim_cnt
            impact_c += (cost_cnt_c - cost_sim_cnt) * capa_cnt * (gap_cnt / impacts_cnt)  # 表示权重
            impact_c[0] = 0
    gaps.append(impact_c)
gaps = np.matrix(gaps)
gaps_r = gaps.copy()
gaps_final = gaps_r.copy()
for j in range(len(x0)):
    for i in range(len(country_solar)):
        if np.sum(gaps, axis=0)[0, j] != 0:
            gaps_r[i, j] = gaps[i, j] / np.sum(gaps, axis=0)[0, j]
        else:
            gaps_r[i, j] = 0
        gaps_final[i, j] = gaps_r[i, j] * gap_global[j]
base = np.array([0]*13, dtype='float')
for i in range(len(country_solar)):
    plt.bar(x0+width/2, np.array(gaps_final[i])[0], width=width, bottom=base, alpha=0.8, color=cmap[country_solar[i]], edgecolor='k')
    base += np.array(gaps_final[i])[0]
ge = base.copy()
plt.plot((x0+width/2)[1:], ge[1:], label='Global engagement', alpha=1, linestyle='--', color='k', marker='.')
plt.legend(loc='upper left', frameon=False, ncol=4)
plt.xlim(1, 14)
plt.ylim(0, 5000)
plt.text(0, 5250, 'a.', fontweight='bold', fontsize=15)
plt.ylabel('Cost reduction ($/kW)')
plt.xlabel('Year')
plt.xticks(range(2, 14), ["NE GE\n{}".format(i) for i in range(2011, 2023)])

ax1 = plt.subplot(2, 1, 2)
country_wind = ['Denmark', 'United States', 'Germany', 'Sweden', 'Italy', 'United Kingdom', 'India',
                'Spain', 'Canada', 'France', 'Turkey', 'Brazil', 'China', 'Others']
country_solar = ['Australia', 'France', 'Germany', 'India', 'Italy', 'Japan', 'South Korea', 'Spain',
                 'United Kingdom', 'United States', 'China', 'Others']
cmap = {'Australia': 'darkorange', 'France': 'mediumpurple', 'Germany': 'saddlebrown',
        'India': 'olive', 'Italy': 'forestgreen', 'Japan': 'khaki', 'South Korea': 'pink', 'Spain': 'aqua',
                 'United Kingdom': 'lightgreen', 'United States': 'steelblue', 'China': 'firebrick',
        'Denmark': 'royalblue', 'Sweden': 'chocolate', 'Canada':  'hotpink', 'Turkey': 'bisque', 'Brazil': 'darkgreen',
        'Others': 'grey'}

cost_obs = np.array(df_cost_wind['Global'])
capa_global = np.array(df_capacity_wind['Global'])[10:23]  # Year 2010-2022
CostSum_sim = np.array([0] * 13, dtype='float')
CostSum_self = np.array([0] * 13, dtype='float')
cap_global_cum = np.array(df_capacity_wind['Global_cum'])[10:23]  # Year 2010-2022
for c in country_wind:
    capa_c = np.array(df_capacity_wind[c])[10:23]
    co = result_wind[c][:2].values
    cost_sim_c = np.exp(co[0] + co[1] * np.log(cap_global_cum))
    CostSum_sim += cost_sim_c * capa_c
    qt_self_c = np.array(df_capacity_wind_cum[c])[10:23]  # Year 2010-2022, cumulative
    cost_self_c = np.exp(co[0] + co[1] * np.log(qt_self_c))
    CostSum_self += cost_self_c * capa_c
cost_sim = CostSum_sim / capa_global
cost_self = CostSum_self / capa_global
cost_const = np.array([cost_sim[0]]*13)
gap_national = cost_const - cost_self
gaps = []
for c in country_wind:
    capa_c = np.array(df_capacity_wind[c])[10:23]
    co = result_wind[c][:2].values
    cost_sim_c = np.exp(co[0] + co[1] * np.log(cap_global_cum))
    cost_const_c = np.array([cost_sim_c[0]]*13)
    qt_self_c = np.array(df_capacity_wind_cum[c])[10:23]  # Year 2010-2022, cumulative
    cost_self_c = np.exp(co[0] + co[1] * np.log(qt_self_c))
    gaps.append((cost_const_c - cost_self_c)*capa_c)
gaps = np.matrix(gaps)
gaps_r = gaps.copy()
gaps_final = gaps_r.copy()
for j in range(len(x0)):
    for i in range(len(country_wind)):
        if np.sum(gaps, axis=0)[0, j] != 0:
            gaps_r[i, j] = gaps[i, j] / np.sum(gaps, axis=0)[0, j]
        else:
            gaps_r[i, j] = 0
        gaps_final[i, j] = gaps_r[i, j] * gap_national[j]
base = np.array([0]*13, dtype='float')
for i in range(len(country_wind)):
    print('national endeavor')
    print(country_wind[i], np.array(gaps_final[i])[0])
    plt.bar(x0-width/2, np.array(gaps_final[i])[0], width=width, bottom=base, label=country_wind[i], alpha=0.8, color=cmap[country_wind[i]], edgecolor='k')
    base += np.array(gaps_final[i])[0]
ne = base.copy()
plt.plot((x0-width/2)[1:], ne[1:], label='National endeavor', alpha=1, linestyle=':', color='k', marker='.')

gap_global = cost_self - cost_sim
gaps = []
for c in country_wind:
    impact_c = np.array([0]*13, dtype='float')
    capa_c = np.array(df_capacity_wind[c])[10:23]
    cap = np.array(df_capacity_wind[c])[11:23]
    cap_global = np.array(df_capacity_wind['Global'])[11:23]  # Year 2011-2022, additive
    lamda = 1
    qt_add = cap_global - lamda * cap  # additive capacity without country c
    qt = np.array([cap_global_cum[0]])  # Year 2010, world cumulative
    for i in range(len(qt_add)):
        qt = np.append(qt, qt[-1] + qt_add[i])  # Year 2010-2022, cumulative, world cumulative without country c
    for cnt in country_wind:
        if cnt != c:
            capa_cnt = np.array(df_capacity_wind[cnt])[10:23]
            co_cnt = result_wind[cnt][:3].values
            cost_sim_cnt = np.exp(co_cnt[0] + co_cnt[1] * np.log(cap_global_cum))
            qt_self_cnt = np.array(df_capacity_wind_cum[cnt])[10:23]  # Year 2010-2022, cumulative
            cost_self_cnt = np.exp(co_cnt[0] + co_cnt[1] * np.log(qt_self_cnt))
            cost_cnt_c = np.exp(co_cnt[0] + co_cnt[1] * np.log(qt))
            gap_cnt = cost_self_cnt - cost_sim_cnt
            impacts_cnt = np.array([0] * 13, dtype='float')
            for i in country_wind:
                if i != cnt:
                    qt_i = np.array([cap_global_cum[0]])
                    cap_i = np.array(df_capacity_wind[i])[11:23]
                    qt_add_i = cap_global - cap_i
                    for j in range(len(qt_add)):
                        qt_i = np.append(qt_i, qt_i[-1] + qt_add_i[j])
                    cost_cnt_i = np.exp(co_cnt[0] + co_cnt[1] * np.log(qt_i))
                    impacts_cnt += cost_cnt_i - cost_sim_cnt
            impact_c += (cost_cnt_c - cost_sim_cnt) * capa_cnt * (gap_cnt / impacts_cnt)
            impact_c[0] = 0
    gaps.append(impact_c)
gaps = np.matrix(gaps)
gaps_r = gaps.copy()
gaps_final = gaps_r.copy()
for j in range(len(x0)):
    for i in range(len(country_wind)):
        if np.sum(gaps, axis=0)[0, j] != 0:
            gaps_r[i, j] = gaps[i, j] / np.sum(gaps, axis=0)[0, j]
        else:
            gaps_r[i, j] = 0
        gaps_final[i, j] = gaps_r[i, j] * gap_global[j]
base = np.array([0]*13, dtype='float')
for i in range(len(country_wind)):
    plt.bar(x0+width/2, np.array(gaps_final[i])[0], width=width, bottom=base, alpha=0.8, color=cmap[country_wind[i]], edgecolor='k')
    base += np.array(gaps_final[i])[0]
ge = base.copy()
plt.plot((x0+width/2)[1:], ge[1:], label='Global engagement', alpha=1, linestyle='--', color='k', marker='.')
plt.legend(loc='upper left', frameon=False, ncol=4)

plt.xlim(1, 14)
plt.ylim(-200, 800)
plt.hlines(0, 1, 14, color='k', linewidth=1)
plt.text(0, 850, 'b.', fontweight='bold', fontsize=15)
plt.yticks(range(-200, 801, 200), range(-200, 801, 200))
plt.ylabel('Cost reduction ($/kW)')
plt.xlabel('Year')
plt.xticks(range(2, 14), ["NE GE\n{}".format(i) for i in range(2011, 2023)])
plt.legend(loc='upper left', frameon=False, ncol=5)
# ax.remove()
plt.savefig('figs/Fig2.jpg', dpi=300)
plt.show()