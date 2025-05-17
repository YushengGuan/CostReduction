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


x0 = np.linspace(1, 14, 14)
cost_obs = np.array(df_cost_solar['Global'])
capa_global = np.array(df_capacity_solar['Global'])[10:24]  # Year 2010-2023
CostSum_sim = np.array([0] * 14, dtype='float')
CostSum_price = np.array([0] * 14, dtype='float')
CostSum_self = np.array([0] * 14, dtype='float')
cap_global_cum = np.array(df_capacity_solar['Global_cum'])[10:24]  # Year 2010-2023
price_si = np.array(df_cost_solar['price_si'])[0:14]
ne_solar, ge_solar = [], []
df_ne = pd.DataFrame(index=range(2011, 2024), columns=country_solar)
df_ge = pd.DataFrame(index=range(2011, 2024), columns=country_solar)
for c in country_solar:
    capa_c = np.array(df_capacity_solar[c])[10:24]
    co = result_solar[c][:3].values
    cost_sim_c = np.exp(co[0] + co[1] * np.log(cap_global_cum) + co[2] * np.log(price_si))
    CostSum_sim += cost_sim_c * capa_c
    qt_price_c = np.array(df_capacity_solar_cum['const'])[10:24]  # Year 2010-2023, constant of 2010
    cost_price_c = np.exp(co[0] + co[1] * np.log(qt_price_c) + co[2] * np.log(price_si))
    CostSum_price += cost_price_c * capa_c
    qt_self_c = np.array(df_capacity_solar_cum[c])[10:24]  # Year 2010-2023, cumulative
    cost_self_c = np.exp(co[0] + co[1] * np.log(qt_self_c) + co[2] * np.log(price_si))
    CostSum_self += cost_self_c * capa_c
    ne_solar.append(((cost_price_c-cost_self_c) * capa_c)[1:])
    ge_solar.append(((cost_self_c-cost_sim_c) * capa_c)[1:])
    df_ne.loc[:, c] = (cost_price_c-cost_self_c)[1:]
    df_ge.loc[:, c] = (cost_self_c-cost_sim_c)[1:]
cost_sim = CostSum_sim / capa_global
cost_self = CostSum_self / capa_global
cost_price = CostSum_price / capa_global
cost_const = np.array([cost_sim[0]]*14)
cost_red_mp = (cost_const - cost_price) * capa_global  # 2010-2023, added value
cost_red_ne = (cost_price - cost_self) * capa_global
cost_red_ge = (cost_self - cost_sim) * capa_global

plt.figure(figsize=(14,12))
x = np.linspace(2011, 2030, 20)
x1 = np.append(np.linspace(2011, 2023, 13), 2030)
x2 = np.append(np.linspace(1, 13, 13), 20)
x1labels = ["N G\n'"+str(t)[2:4] if t<=2023 else "" for t in x][:13]
x1labels.append("'30E")
xlabels = ["'"+str(t)[2:4] if t<=2023 else "" for t in x][:-1]
xlabels.append("'30E")
grid = plt.GridSpec(3, 21, wspace=0.3, hspace=0.3, top=0.95, bottom=0.05)
NE, GE = 0, 0
x = np.arange(len(x)) + 1

ax1 = plt.subplot(grid[0, 0:10])
width=0.3
base = np.array([0] * 13, dtype='float')
ge_other, ne_other = 0, 0
for i in range(len(country_solar)):
    if country_solar[i] == 'Others':
        ne_other += ne_solar[i]
    plt.bar(x[:13]-width/2, ne_solar[i], bottom=base, width=width, color=cmap[country_solar[i]], alpha=0.8, edgecolor='k')
    base += ne_solar[i]
base = np.array([0] * 13, dtype='float')
for i in range(len(country_solar)):
    if country_solar[i] == 'Others':
        ge_other += ge_solar[i]
    plt.bar(x[:13] + width / 2, ge_solar[i], bottom=base, width=width, color=cmap[country_solar[i]], label=country_solar[i], alpha=0.8, edgecolor='k')
    base += ge_solar[i]
plt.xticks(x2[:-1], x1labels[:-1])
plt.ylim(0, 1200000)
plt.yticks(np.linspace(0, 1200000, 7), ['0', '2', '4', '6', '8', '10', '12'])
plt.ylabel('Annual cost savings ($ 100 billion)')
plt.xlabel('Year')
plt.text(0, 1200000*1.05, 'Solar PV')
plt.text(-1.2, 1200000*1.05, 'a.', weight='bold', fontsize=15)
plt.legend(loc='upper left', frameon=False, ncol=3)

ax1 = plt.subplot(grid[1,0:12])
y, y1, y2 = [], [], []
for i in range(1, len(cost_red_mp)+1):
    y.append(sum(cost_red_mp[0:i]))
    y1.append(sum(cost_red_ne[0:i]))
    y2.append(sum(cost_red_ge[0:i]))

saving_solar = pd.read_excel('results/CostSavings.xlsx', sheet_name='Cum_solar')
width = 0.2
lns1 = ax1.bar(x[:13], y[1:], width, alpha=0.6, label="Material price (MP)", color='grey', edgecolor='k')
lns2 = plt.bar(x[:13] + width, saving_solar['NEing'], width, alpha=0.6, label="Savings in developing countries by NE", color='skyblue', edgecolor='k', hatch='////')
lns3 = plt.bar(x[:13] + 2*width, saving_solar['GEing'], width, alpha=0.6, label="Savings in developing countries by GE", color='green', edgecolor='k', hatch='////')
lns4 = plt.bar(x[:13] + width, saving_solar['NEed'], width, alpha=0.6, label="Savings in developed countries by NE", color='skyblue', edgecolor='k', hatch='\\\\\\\\', bottom=saving_solar['NEing'])
lns5 = plt.bar(x[:13] + 2*width, saving_solar['GEed'], width, alpha=0.6, label="Savings in developed countries by GE", color='green', edgecolor='k', hatch='\\\\\\\\', bottom=saving_solar['GEing'])
plt.bar(x[-1], cost_red_mp[-1] / capa_global[-1] * (cap_global_cum[-2] * 3 - cap_global_cum[-1]) + cost_red_mp[-1], 
width, alpha=0.6, color='grey', edgecolor='k', linestyle='dashed')
plt.bar(x[-1] + width, cost_red_ne[-1] / capa_global[-1] * (cap_global_cum[-2] * 3 - cap_global_cum[-1]) + cost_red_ne[-1],
width, alpha=0.6, color='skyblue', edgecolor='k', linestyle='dashed')
NE += cost_red_ne[-1] / capa_global[-1] * cap_global_cum[-2] * 2 + cost_red_ne[-1]
GE += cost_red_ge[-1] / capa_global[-1] * cap_global_cum[-2] * 2 + cost_red_ge[-1]
plt.bar(x[-1] + width*2, cost_red_ge[-1] / capa_global[-1] * (cap_global_cum[-2] * 3 - cap_global_cum[-1]) + cost_red_ge[-1],
width, alpha=0.6, color='green', edgecolor='k', linestyle='dashed')
plt.ylim(0, 10**7)
plt.xticks(x2+width, x1labels)
plt.yticks(np.linspace(0, 10**7, 6), [str(i)[:-2] for i in np.linspace(0, 10, 6)])
plt.text(0, 1.05*10**7, 'Solar PV')
plt.text(-2, 1.05*10**7, 'c.', weight='bold', fontsize=15)
plt.ylabel('Global cumulative cost savings ($ trillion)')
ax2 = ax1.twinx()
x3 = x2+width/2
lns6, = ax2.plot(x3[:-1]+width/2, cap_global_cum[1:14], color='k', marker='.', label='Cumulative installed capacity')
ax2.plot(x3[-2:]+width/2, [cap_global_cum[-1], cap_global_cum[-2]*3], color='k', marker='.', linestyle=':')
ax2.set_ylim(0, 4*10**3)
ax2.set_yticks(np.linspace(0, 4*10**3, 5), [str(i)[:-2] for i in np.linspace(0, 4, 5)])
lns = [lns1, lns2, lns3, lns4, lns5, lns6]
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left', frameon=False)


country_wind = ['Denmark', 'United States', 'Germany', 'Sweden', 'Italy', 'United Kingdom', 'India',
                'Spain', 'Canada', 'France', 'Turkey', 'Brazil', 'China', 'Others']
country_solar = ['Australia', 'France', 'Germany', 'India', 'Italy', 'Japan', 'South Korea', 'Spain',
                 'United Kingdom', 'United States', 'China', 'Others']
cmap = {'Australia': 'darkorange', 'France': 'mediumpurple', 'Germany': 'saddlebrown',
        'India': 'olive', 'Italy': 'forestgreen', 'Japan': 'khaki', 'South Korea': 'pink', 'Spain': 'aqua',
                 'United Kingdom': 'lightgreen', 'United States': 'steelblue', 'China': 'firebrick',
        'Denmark': 'royalblue', 'Sweden': 'chocolate', 'Canada':  'hotpink', 'Turkey': 'bisque', 'Brazil': 'darkgreen',
        'Others': 'grey'}
x0 = np.linspace(1, 14, 14)
cost_obs = np.array(df_cost_wind['Global'])
capa_global = np.array(df_capacity_wind['Global'])[10:24]  # Year 2010-2023
CostSum_sim = np.array([0] * 14, dtype='float')
CostSum_self = np.array([0] * 14, dtype='float')
cap_global_cum = np.array(df_capacity_wind['Global_cum'])[10:24]  # Year 2010-2023
ne_wind, ge_wind = [], []
df_ne = pd.DataFrame(index=range(2011, 2024), columns=country_wind)
df_ge = pd.DataFrame(index=range(2011, 2024), columns=country_wind)
for c in country_wind:
    capa_c = np.array(df_capacity_wind[c])[10:24]
    co = result_wind[c][:2].values
    cost_sim_c = np.exp(co[0] + co[1] * np.log(cap_global_cum))
    CostSum_sim += cost_sim_c * capa_c
    qt_self_c = np.array(df_capacity_wind_cum[c])[10:24]  # Year 2010-2023, cumulative
    cost_self_c = np.exp(co[0] + co[1] * np.log(qt_self_c))
    CostSum_self += cost_self_c * capa_c
    ne_wind.append(((np.array([cost_sim_c[0]]*14) - cost_self_c) * capa_c)[1:])
    ge_wind.append(((cost_self_c - cost_sim_c) * capa_c)[1:])
    df_ne.loc[:, c] = (np.array([cost_sim_c[0]]*14) - cost_self_c)[1:]
    df_ge.loc[:, c] = (cost_self_c - cost_sim_c)[1:]
cost_sim = CostSum_sim / capa_global
cost_self = CostSum_self / capa_global
cost_const = np.array([cost_sim[0]]*14)
cost_red_ne = (cost_const - cost_self) * capa_global
cost_red_ge = (cost_self - cost_sim) * capa_global

ax1 = plt.subplot(grid[0, 11:21])
width=0.3
base = np.array([0] * 13, dtype='float')
ne_other, ge_other = 0, 0
for i in range(len(country_wind)):
    if country_wind[i] == 'Others':
        ne_other += ne_wind[i]
    plt.bar(x[:13]-width/2, ne_wind[i], bottom=base, width=width, color=cmap[country_wind[i]], alpha=0.8, edgecolor='k')
    base += ne_wind[i]
base = np.array([0] * 13, dtype='float')
for i in range(len(country_wind)):
    if country_wind[i] == 'Others':
        ge_other += ge_wind[i]
    plt.bar(x[:13] + width / 2, ge_wind[i], bottom=base, width=width, color=cmap[country_wind[i]], label=country_wind[i], alpha=0.8, edgecolor='k')
    base += ge_wind[i]
plt.xticks(x2[:-1], x1labels[:-1])
plt.ylim(0, 80000)
plt.yticks(np.linspace(0, 80000, 5), ['0', '2', '4', '6', '8'])
plt.ylabel('Annual cost savings ($ 10 billion)')
plt.xlabel('Year')
plt.text(0, 80000*1.05, 'Wind power')
plt.text(-1.2, 80000*1.05, 'b.', weight='bold', fontsize=15)
plt.legend(loc='upper left', frameon=False, ncol=2)

saving_wind = pd.read_excel('results/CostSavings.xlsx', sheet_name='Cum_wind')
ax1 = plt.subplot(grid[2,0:12])
y, y1 = [], []
for i in range(1, len(cost_red_ne)+1):
    y.append(sum(cost_red_ne[0:i]))
    y1.append(sum(cost_red_ge[0:i]))
x = np.arange(len(x)) + 1
width = 0.3
lns1 = plt.bar(x[:13], saving_wind['NEing'], width, alpha=0.6, label="Savings in developing countries by NE", color='skyblue', edgecolor='k', hatch='////')
lns2 = plt.bar(x[:13] + width, saving_wind['GEing'], width, alpha=0.6, label="Savings in developing countries by GE", color='green', edgecolor='k', hatch='////')
lns3 = plt.bar(x[:13], saving_wind['NEed'], width, alpha=0.6, label="Savings in developed countries by NE", color='skyblue', edgecolor='k', hatch='\\\\\\\\', bottom=saving_wind['NEing'])
lns4 = plt.bar(x[:13] + width, saving_wind['GEed'], width, alpha=0.6, label="Savings in developed countries by GE", color='green', edgecolor='k', hatch='\\\\\\\\', bottom=saving_wind['GEing'])
plt.bar(x[-1], cost_red_ne[-1] / capa_global[-1] * (cap_global_cum[-2] * 3 - cap_global_cum[-1]) + cost_red_ne[-1], width,
alpha=0.6, color='skyblue',edgecolor='k', linestyle='dashed')
plt.bar(x[-1] + width, cost_red_ge[-1] / capa_global[-1] * (cap_global_cum[-2] * 3 - cap_global_cum[-1]) + cost_red_ge[-1], width,
alpha=0.6, color='green', edgecolor='k', linestyle='dashed')
NE += cost_red_ne[-1] / capa_global[-1] * (cap_global_cum[-2] * 3 - cap_global_cum[-1]) + cost_red_ne[-1]
GE += cost_red_ge[-1] / capa_global[-1] * (cap_global_cum[-2] * 3 - cap_global_cum[-1]) + cost_red_ge[-1]
# 将坐标设置在指定位置
plt.xticks(x2+width/2, x1labels)
plt.ylabel('Global cumulative cost savings ($ trillion)')
plt.xlabel('Year')
plt.yticks(np.linspace(0, 1.5*10**6, 6), ['0', '0.3', '0.6', '0.9', '1.2', '1.5'])
plt.ylim(0, 1.5*10**6)
plt.text(0, 1.05*1.5*10**6, 'Wind power')
plt.text(-2, 1.05*1.5*10**6, 'd.', weight='bold', fontsize=15)
ax2 = ax1.twinx()
x3 = x2+width/2
lns5, = ax2.plot(x3[:-1], cap_global_cum[1:14], color='k', marker='.', label='Cumulative installed capacity')
ax2.plot(x3[-2:], [cap_global_cum[-1], cap_global_cum[-2]*3], color='k', marker='.', linestyle=':')
ax2.set_ylim(0, 3*10**3)
ax2.set_yticks(np.linspace(0, 3*10**3, 4), [str(i)[:-2] for i in np.linspace(0, 3, 4)])
ax2.set_ylabel('Cumulative installed capacity (TW)', y=1.15)
lns = [lns1, lns2, lns3, lns4, lns5]
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left', frameon=False)

plt.subplot(grid[1:,14:])
plt.ylabel('Annual investment ($ billion)')
df = pd.read_excel('results/2030target.xlsx', sheet_name='Fig')
width=0.1
start=0.3
plt.xlim(0.2, 1)
plt.ylim(0, 2000)
plt.bar(start+width, df['scn2'].values[2], width, alpha=0.8, color='brown', edgecolor='k', label='Existing level')
plt.bar(start+width, df['scn2'].values[4], width, alpha=0.6, color='chocolate', label="Investment gap",
        bottom=df['scn2'].values[2], edgecolor='k', linestyle='dashed')
plt.bar(start+width, df['scn2'].values[0], width, alpha=0.6, label="Without GE scenario", color='green',
        bottom=df['scn2'].values[1], edgecolor='k', linestyle='dashed')
plt.vlines(0.6, ymin=df['scn2'].values[1], ymax=df['scn2'].values[5], color='green', linestyles='dashed', alpha=0.6)
plt.text(0.7, 1350, '\u0394'+'gap={}%'.format(int(df['scn2'].values[3]*100)))
plt.xticks([])
plt.legend(loc='lower right', frameon=False)
plt.text(0.05, 1.025*2000, 'e.', weight='bold', fontsize=15)
plt.savefig(r'figs/Fig3.jpg', dpi=600)
plt.show()

