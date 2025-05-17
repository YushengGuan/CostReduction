import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'  # or 'sans-serif', 'cursive', 'fantasy', 'monospace'

random_count = 1000
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

x0 = np.linspace(1, 14, 14)
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.6)
cost_sim_random_solar = []  # c, m, t
cost_price_random_solar = []
cost_self_random_solar = []
d_price, d_national, d_global = [], [], []
se_price, se_national, se_global = [], [], []
df_solar_NE = pd.DataFrame(columns=country_solar)
df_solar_GE = pd.DataFrame(columns=country_solar)
df_wind_NE = pd.DataFrame(columns=country_wind)
df_wind_GE = pd.DataFrame(columns=country_wind)
for c in country_solar[1:]:
    ind = country_solar[1:].index(c)
    plt.subplot(3, 4, ind+1)
    cap_global_cum = np.array(df_capacity_solar['Global_cum'])[10:24]  # Year 2010-2023
    price_si = np.array(df_cost_solar['price_si'])[0:14]
    co = result_solar[c][:3].values
    se = result_solar[c][3:6].values
    cost_sim = np.exp(co[0] + co[1] * np.log(cap_global_cum) + co[2] * np.log(price_si))
    co_random = np.random.normal(co[1], se[1], size=random_count)
    # print(cost_sim)
    cost_obs = np.array(df_cost_solar[c])[0:14]
    qt_price = np.array(df_capacity_solar_cum['const'])[10:24]  # Year 2010-2023, constant capacity of 2010
    cost_price = np.exp(co[0] + co[1] * np.log(qt_price) + co[2] * np.log(price_si))  # only price_si matters
    qt_self = np.array(df_capacity_solar_cum[c])[10:24]  # Year 2010-2023, cumulative
    cost_self = np.exp(co[0] + co[1] * np.log(qt_self) + co[2] * np.log(price_si))
    gap = cost_self - cost_sim 
    d_price.append(cost_sim[0] - cost_price[-1])
    d_national.append(cost_price[-1] - cost_self[-1])
    d_global.append(cost_self[-1] - cost_sim[-1])
    df_solar_NE[c] = cost_price - cost_self
    df_solar_GE[c] = cost_self - cost_sim

    cost_sims_random = []
    for m in range(len(co_random)):
        cost_sims_random.append(np.exp(co[0] + co_random[m] * np.log(cap_global_cum) + co[2] * np.log(price_si)))
    cost_sim_random_solar.append(cost_sims_random)
    cost_price_random = []
    for m in range(len(co_random)):
        cost_price_random.append(np.exp(co[0] + co_random[m] * np.log(qt_price) + co[2] * np.log(price_si)))
    cost_price_random_solar.append(cost_price_random)
    cost_self_random = []
    for m in range(len(co_random)):
        cost_self_random.append(np.exp(co[0] + co_random[m] * np.log(qt_self) + co[2] * np.log(price_si)))
    cost_self_random_solar.append(cost_self_random)
    d_price_random, d_national_random, d_global_random = [], [], []
    for m in range(len(co_random)):
        d_price_random.append(cost_sims_random[m][0] - cost_price_random[m][-1])
        d_national_random.append(cost_price_random[m][-1] - cost_self_random[m][-1])
        d_global_random.append(cost_self_random[m][-1] - cost_sims_random[m][-1])
    se_price.append(np.std(d_price_random))
    se_national.append(np.std(d_national_random))
    se_global.append(np.std(d_global_random))

    plt.hlines(cost_sim[0], 1, 14, colors='k', linestyle='--')
    plt.plot(x0, cost_price, label='MP scenario', color='k', linestyle='-.')
    plt.plot(x0, cost_self, label='NE scenario', linestyle=':', color='k')
    plt.plot(x0, cost_sim, label='GE scenario', color='k')
    plt.fill_between(x0, np.percentile(cost_sims_random, 97.5, axis=0),
                     np.percentile(cost_sims_random, 2.5, axis=0),
                     color='grey', alpha=0.3, label='95% CI, GE scenario')
    plt.fill_between(x0, np.percentile(cost_price_random, 97.5, axis=0),
                     np.percentile(cost_price_random, 2.5, axis=0),
                     color='skyblue', alpha=0.3, label='95% CI, MP scenario')
    plt.fill_between(x0, np.percentile(cost_self_random, 97.5, axis=0),
                     np.percentile(cost_self_random, 2.5, axis=0),
                     color='green', alpha=0.3, label='95% CI, NE scenario')
    plt.scatter(x0, cost_obs, marker='x', color='k', label='Observations')
    plt.title(c, fontsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylim(0, 15000)
    if ind in [0, 4, 8]:
        plt.yticks([0, 5000, 10000, 15000], ['0', '5', '10', '15'], fontsize=12)
    else:
        plt.yticks([0, 5000, 10000, 15000], ['0', '5', '10', '15'], fontsize=12)
    if ind == 4:
        plt.ylabel('Total installed cost ($1000/kW)', fontsize=20, labelpad=0)
    plt.xticks([1, 3, 5, 7, 9, 11, 14], ["'10", "'12", "'14", "'16", "'18", "'20", "'23"], fontsize=12)
    if ind == 9:
        plt.xlabel('Year', fontsize=20, x=1.15)
# ax.remove()
plt.savefig('figs/Country_solar.png', dpi=600)


data = [np.array(d_price), np.array(d_national), np.array(d_global)]
data_se = np.array([0] + se_price + [0] + se_national + [0] + se_global)
print('data_solar', data)
group_names = ['MP', 'NE', 'GE']
group_colors = ['#D8D8D8', '#D9F0F8', '#B1D8B1']
fig = plt.figure(figsize=(10, 10))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
fig.patch.set_alpha(0)
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
radii = [0]
colors = ['white']
for g, c in zip(data, group_colors):
    radii.extend(g)
    colors.extend([c] * len(g))
    radii.append(0)
    colors.append('white')
radii.pop()
colors.pop()
N = len(radii)
r_lim = 180
scale_lim = 6000
scale_major = 2000
bottom = 2000
bottoms = np.array([bottom]*len(radii))
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
width = 2 * np.pi / (N + 9)
ax.bar(theta, radii, width=width, bottom=bottom, color=colors)
ax.errorbar(theta, radii+bottoms, yerr=data_se, capsize=0, color='k', elinewidth=1, fmt='none')
def scale(ax, bottom, scale_lim, theta, width):
    t = np.linspace(theta - width / 2, theta + width / 2, 4)
    for i in range(int(bottom), int(bottom + scale_lim + scale_major), scale_major):
        ax.plot(t, [i] * 4, linewidth=0.25, color='k', alpha=0.8)
def scale_value(ax, bottom, theta, scale_lim):
    for i in range(int(bottom), int(bottom + scale_lim + scale_major), scale_major):
        ax.text(theta,
                i,
                f'{int((i - bottom)/1000)}',
                fontsize=15,
                alpha=1,
                va='center',
                ha='center',
                color='k'
                )
s_list = []
g_no = 0
aa = country_solar[1:]
for t, r in zip(theta, radii):
    if r == 0:
        s_list.append(t)
        if t == 0:
            scale_value(ax, bottom, t, scale_lim)
        else:
            scale(ax, bottom, scale_lim, t, width)
    else:
        t2 = np.rad2deg(t)
        ax.text(t, r + bottom + scale_major * 0.8,
                aa[g_no],
                fontsize=15,
                rotation=90 - t2 if t < np.pi else 270 - t2,
                rotation_mode='anchor',
                va='center',
                ha='left' if t < np.pi else 'right',
                color='black',
                clip_on=False
                )
        if g_no == (len(aa) - 1):
            g_no = 0
        else:
            g_no += 1
s_list.append(2 * np.pi)
for i in range(len(s_list) - 1):
    t = np.linspace(s_list[i] + width, s_list[i + 1] - width, 50)
    ax.plot(t, [bottom - scale_major * 0.4] * 50, linewidth=0.5, color='black')
    ax.text(s_list[i] + (s_list[i + 1] - s_list[i]) / 2,
            500,
            group_names[i],
            va='center',
            ha='center',
            fontsize=12,
            )
ax.text(0.05, 8200, 'x1000 ($/kW)', fontsize=20)
ax.set_rlim(0, bottom + scale_lim + scale_major)
ax.axis('off')
plt.savefig('figs/polar_bar_solar.png', dpi=600)


country_wind2 = ['Global', 'Denmark', 'United States', 'Germany', 'Sweden', 'Italy', 'United Kingdom', 'India',
                 'Spain', 'Canada', 'France', 'Turkey', 'Brazil', 'China', 'Others']
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.6)
cost_sim_random_wind = []  # c, m, t
cost_self_random_wind = []
d_national, d_global = [], []
se_national, se_global = [], []
for c in country_wind2[1:]:
    print(c, 'wind')
    ind = country_wind2[1:].index(c)
    plt.subplot(3, 5, ind+1)
    cap_global_cum = np.array(df_capacity_wind['Global_cum'])[10:24]  # Year 2010-2023
    co = result_wind[c][:2].values
    cost_sim = np.exp(co[0] + co[1] * np.log(cap_global_cum))
    cost_obs = np.array(df_cost_wind[c])[0:14]
    qt_self = np.array(df_capacity_wind_cum[c])[10:24]  # Year 2010-2023, cumulative
    cost_self = np.exp(co[0] + co[1] * np.log(qt_self))
    gap = cost_self - cost_sim 
    d_national.append(cost_sim[0] - cost_self[-1])
    d_global.append(cost_self[-1] - cost_sim[-1])

    df_wind_NE[c] = cost_sim[0] - cost_self
    df_wind_GE[c] = cost_self - cost_sim

    print(gap, '只有自己会贵多少钱')
    print('cost_self', cost_self)
    print('cost_sim', cost_sim)
    se = result_wind[c][2:4].values
    co_random = np.random.normal(co[1], se[1], size=random_count)
    cost_sims_random = []
    for m in range(len(co_random)):
        cost_sims_random.append(np.exp(co[0] + co_random[m] * np.log(cap_global_cum)))
    cost_sim_random_wind.append(cost_sims_random)
    cost_self_random = []
    for m in range(len(co_random)):
        cost_self_random.append(np.exp(co[0] + co_random[m] * np.log(qt_self)))
    cost_self_random_wind.append(cost_self_random)
    d_national_random, d_global_random = [], []
    for m in range(len(co_random)):
        d_national_random.append(cost_sims_random[m][0] - cost_self_random[m][-1])
        d_global_random.append(cost_self_random[m][-1] - cost_sims_random[m][-1])
    se_national.append(np.std(d_national_random))
    se_global.append(np.std(d_global_random))
    print('95%', np.percentile(d_national_random, 97.5), np.percentile(d_national_random, 2.5))
    print('95%', np.percentile(d_global_random, 97.5), np.percentile(d_global_random, 2.5))

    plt.hlines(cost_sim[0], 1, 14, colors='k', linestyle='--')
    plt.plot(x0, cost_self, label='NE scenario', linestyle=':', color='k')
    plt.plot(x0, cost_sim, label='GE scenario', color='k')
    plt.scatter(x0, cost_obs, marker='x', color='k', label='Observations')
    plt.fill_between(x0, np.percentile(cost_sims_random, 95, axis=0),
                     np.percentile(cost_sims_random, 5, axis=0),
                     color='grey', alpha=0.3, label='90% CI, GE scenario')
    plt.fill_between(x0, np.percentile(cost_self_random, 95, axis=0),
                     np.percentile(cost_self_random, 5, axis=0),
                     color='green', alpha=0.3, label='90% CI, NE scenario')
    plt.title(c, fontsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylim(1000, 10000)
    if ind in [0, 5, 10]:
        plt.yticks([1000, 4000, 7000, 10000], ['1', '4', '7', '10'], fontsize=12)
    else:
        plt.yticks([1000, 4000, 7000, 10000], ['1', '4', '7', '10'], fontsize=12)
    if ind == 5:
        plt.ylabel('Total installed cost ($1000/kW)', fontsize=20, labelpad=5)
    plt.xticks([1, 5, 9, 14], ["'10", "'14", "'18", "'23"], fontsize=12)
    if ind == 12:
        plt.xlabel('Year', fontsize=20)
# ax.remove()
plt.savefig('figs/Country_wind.png', dpi=600)

data = [np.array(d_national), np.array(d_global)]
data_se = np.array([0] + se_national + [0] + se_global)
print('data_wind', data)
group_names = ['NE', 'GE']
group_colors = ['#D9F0F8', '#B1D8B1']
fig = plt.figure(figsize=(10, 10))
fig.patch.set_alpha(0)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
radii = [0]
colors = ['white']
for g, c in zip(data, group_colors):
    radii.extend(g)
    colors.extend([c] * len(g))
    radii.append(0)
    colors.append('white')
radii.pop()
colors.pop()
N = len(radii)
r_lim = 180
scale_lim = 2500
scale_major = 500
bottom = 500
bottoms = np.array([bottom]*len(radii))
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
width = 2 * np.pi / (N + 9)
ax.bar(theta, radii, width=width, bottom=bottom, color=colors)
ax.errorbar(theta, radii+bottoms, yerr=data_se, capsize=0, color='k', elinewidth=1, fmt='none')
def scale(ax, bottom, scale_lim, theta, width):
    t = np.linspace(theta - width / 2, theta + width / 2, 5)
    for i in range(int(bottom), int(bottom + scale_lim + scale_major), scale_major):
        ax.plot(t, [i] * 5, linewidth=0.25, color='k', alpha=0.8)
def scale_value(ax, bottom, theta, scale_lim):
    count_i = 0
    for i in range(int(bottom), int(bottom + scale_lim + scale_major), scale_major):
        ax.text(theta,
                i,
                ['0', '0.5', '1', '1.5', '2', '2.5'][count_i],
                fontsize=15,
                alpha=1,
                va='center',
                ha='center',
                color='k'
                )
        count_i += 1
s_list = []
g_no = 0
aa = country_wind[1:]
for t, r in zip(theta, radii):
    if r == 0:
        s_list.append(t)
        if t == 0:
            scale_value(ax, bottom, t, scale_lim)
        else:
            scale(ax, bottom, scale_lim, t, width)
    else:
        t2 = np.rad2deg(t)
        ax.text(t, r + bottom + scale_major * 1.7,
                aa[g_no],
                fontsize=15,
                rotation=90 - t2 if t < np.pi else 270 - t2,
                rotation_mode='anchor',
                va='center',
                ha='left' if t < np.pi else 'right',
                color='black',
                clip_on=False
                )
        if g_no == (len(aa) - 1):
            g_no = 0
        else:
            g_no += 1
s_list.append(2 * np.pi)
for i in range(len(s_list) - 1):
    t = np.linspace(s_list[i] + width, s_list[i + 1] - width, 50)
    ax.plot(t, [bottom - scale_major * 0.4] * 50, linewidth=0.5, color='black')
    ax.text(s_list[i] + (s_list[i + 1] - s_list[i]) / 2,
            125,
            group_names[i],
            va='center',
            ha='center',
            fontsize=12,
            )
ax.text(0.05, 3050, 'x1000 ($/kW)', fontsize=20)
ax.set_rlim(0, bottom + scale_lim + scale_major)
ax.axis('off')
plt.savefig('figs/polar_bar_wind.png', dpi=600)


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
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.6)
cost_obs = np.array(df_cost_wind['Global'])[0:14]
plt.subplot(1, 1, 1)
capa_global = np.array(df_capacity_wind['Global'])[10:24]  # Year 2010-2023
CostSum_sim = np.array([0] * 14, dtype='float')
CostSum_self = np.array([0] * 14, dtype='float')
cap_global_cum = np.array(df_capacity_wind['Global_cum'])[10:24]  # Year 2010-2023
for c in country_wind:
    capa_c = np.array(df_capacity_wind[c])[10:24]
    co = result_wind[c][:2].values
    cost_sim_c = np.exp(co[0] + co[1] * np.log(cap_global_cum))
    CostSum_sim += cost_sim_c * capa_c
    qt_self_c = np.array(df_capacity_wind_cum[c])[10:24]  # Year 2010-2022, cumulative
    cost_self_c = np.exp(co[0] + co[1] * np.log(qt_self_c))
    CostSum_self += cost_self_c * capa_c
cost_sim = CostSum_sim / capa_global
cost_self = CostSum_self / capa_global
cost_const = np.array([cost_sim[0]]*14)

df_wind_NE['Global'] = cost_const - cost_self
df_wind_GE['Global'] = cost_self - cost_sim

df_wind_GE.to_excel('results/wind_GE.xlsx', index=False)
df_wind_NE.to_excel('results/wind_NE.xlsx', index=False)

cost_sim_random = []
cost_self_random = []
gap_ne, gap_ge = [], []
for j in range(random_count):
    cost_sum_sim = np.array([0] * 14, dtype='float')
    cost_sum_self = np.array([0] * 14, dtype='float')
    for i in range(len(country_wind)):
        country = country_wind[i]
        capa = np.array(df_capacity_wind[country])[10:24]
        
        cost_sum_sim += capa * cost_sim_random_wind[i][j]
        cost_sum_self += capa * cost_self_random_wind[i][j]
    cost_sim_random.append(cost_sum_sim / capa_global)
    cost_self_random.append(cost_sum_self / capa_global)
    gap_ne_j = np.array([(cost_sum_self / capa_global)[0]]*14, dtype='float') - cost_sum_self / capa_global
    gap_ge_j = cost_sum_self / capa_global - cost_sum_sim / capa_global
    gap_ne.append(gap_ne_j[-1])
    gap_ge.append(gap_ge_j[-1])

plt.hlines(cost_sim[0], 1, 14, colors='k', linestyle='--')
plt.plot(x0, cost_self, label='NE scenario', linestyle=':', color='k')
plt.plot(x0, cost_sim, label='GE scenario', color='k')
plt.xlim(0, 16)
plt.vlines(14.5, cost_self[-1], cost_const[-1], colors='k')
plt.scatter(x0, cost_obs, marker='x', color='k', label='Observations')
plt.fill_between(x0, np.percentile(cost_sim_random, 95, axis=0),
                 np.percentile(cost_sim_random, 5, axis=0),
                 color='grey', alpha=0.3, label='90% CI, GE scenario')
plt.fill_between(x0, np.percentile(cost_self_random, 95, axis=0),
                 np.percentile(cost_self_random, 5, axis=0),
                 color='green', alpha=0.3, label='90% CI, NE scenario')
plt.text(14.8, (cost_self[-1]+cost_const[-1])/2, 'National\nEndeavor', fontsize=12, va='center')
plt.vlines(14.5, cost_sim[-1], cost_self[-1], colors='k')
plt.text(14.8, (cost_self[-1]+cost_sim[-1])/2, 'Global\nEngagement', fontsize=12, va='center')
plt.hlines(cost_const[-1], 14.3, 14.7, colors='k')
plt.hlines(cost_self[-1], 14.3, 14.7, colors='k')
plt.hlines(cost_sim[-1], 14.3, 14.7, colors='k')
plt.ylim(1000, 3500)
plt.yticks([1000, 1500, 2000, 2500, 3000, 3500], ['1', '1.5', '2', '2.5', '3', '3.5'], fontsize=12)
plt.ylabel('Total installed cost ($1000/kW)', fontsize=20, labelpad=5)
plt.xticks([1, 3, 5, 7, 9, 11, 14], ["2010", "2012", "2014", "2016", "2018", "2020", "2023"], fontsize=12)
plt.xlabel('Year', fontsize=20)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# ax.remove()
plt.legend(loc='lower left', fontsize=12, frameon=False, ncol=2)
plt.savefig('figs/Global_wind.png', dpi=600)


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
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.6)
cost_obs = np.array(df_cost_solar['Global'])[0:14]
plt.subplot(1, 1, 1)
capa_global = np.array(df_capacity_solar['Global'])[10:]  # Year 2010-2022
CostSum_sim = np.array([0] * 14, dtype='float')
CostSum_price = np.array([0] * 14, dtype='float')
CostSum_self = np.array([0] * 14, dtype='float')
cap_global_cum = np.array(df_capacity_solar['Global_cum'])[10:]  # Year 2010-2022
price_si = np.array(df_cost_solar['price_si'])[0:14]
for c in country_solar:
    capa_c = np.array(df_capacity_solar[c])[10:]
    co = result_solar[c][:3].values
    cost_sim_c = np.exp(co[0] + co[1] * np.log(cap_global_cum) + co[2] * np.log(price_si))
    CostSum_sim += cost_sim_c * capa_c
    qt_price_c = np.array(df_capacity_solar_cum['const'])[10:]  # Year 2010-2022, constant of 2010
    cost_price_c = np.exp(co[0] + co[1] * np.log(qt_price_c) + co[2] * np.log(price_si))
    CostSum_price += cost_price_c * capa_c
    qt_self_c = np.array(df_capacity_solar_cum[c])[10:]  # Year 2010-2022, cumulative
    cost_self_c = np.exp(co[0] + co[1] * np.log(qt_self_c) + co[2] * np.log(price_si))
    CostSum_self += cost_self_c * capa_c
cost_sim = CostSum_sim / capa_global
cost_self = CostSum_self / capa_global
cost_price = CostSum_price / capa_global

df_solar_NE['Global'] = cost_price - cost_self
df_solar_GE['Global'] = cost_self - cost_sim

df_solar_GE.to_excel('results/solar_GE.xlsx', index=False)
df_solar_NE.to_excel('results/solar_NE.xlsx', index=False)

cost_sim_random = []
cost_price_random = []
cost_self_random = []
gap_ne, gap_ge = [], []
for j in range(random_count):
    cost_sum_sim = np.array([0] * 14, dtype='float')
    cost_sum_price = np.array([0] * 14, dtype='float')
    cost_sum_self = np.array([0] * 14, dtype='float')
    for i in range(len(country_solar)):
        country = country_solar[i]
        capa = np.array(df_capacity_solar[country])[10:]
        cost_sum_sim += capa * cost_sim_random_solar[i][j]
        cost_sum_price += capa * cost_price_random_solar[i][j]
        cost_sum_self += capa * cost_self_random_solar[i][j]
    cost_sim_random.append(cost_sum_sim / capa_global)
    cost_price_random.append(cost_sum_price / capa_global)
    cost_self_random.append(cost_sum_self / capa_global)
    gap_ne_j = cost_sum_price / capa_global - cost_sum_self / capa_global
    gap_ge_j = cost_sum_self / capa_global - cost_sum_sim / capa_global
    gap_ne.append(gap_ne_j[-1])
    gap_ge.append(gap_ge_j[-1])
plt.hlines(cost_sim[0], 1, 14, colors='k', linestyle='--')
plt.plot(x0, cost_price, label='MP scenario', color='k', linestyle='-.')
plt.plot(x0, cost_self, label='NE scenario', linestyle=':', color='k')
plt.plot(x0, cost_sim, label='GE scenario', color='k')
plt.scatter(x0, cost_obs, marker='x', color='k', label='Observations')
plt.fill_between(x0, np.percentile(cost_sim_random, 97.5, axis=0),
                 np.percentile(cost_sim_random, 2.5, axis=0),
                 color='grey', alpha=0.3, label='95% CI, GE scenario')
plt.fill_between(x0, np.percentile(cost_price_random, 97.5, axis=0),
                 np.percentile(cost_price_random, 2.5, axis=0),
                 color='skyblue', alpha=0.3, label='95% CI, MP scenario')
plt.fill_between(x0, np.percentile(cost_self_random, 97.5, axis=0),
                 np.percentile(cost_self_random, 2.5, axis=0),
                 color='green', alpha=0.3, label='95% CI, NE scenario')
plt.xlim(0, 16)
plt.vlines(14.5, cost_self[-1], cost_price[-1], colors='k')
plt.text(14.8, (cost_self[-1]+cost_price[-1])/2, 'National\nEndeavor', fontsize=12, va='center')
plt.vlines(14.5, cost_sim[-1], cost_self[-1], colors='k')
plt.text(14.8, (cost_self[-1]+cost_sim[-1])/2, 'Global\nEngagement', fontsize=12, va='center')
plt.vlines(14.5, cost_price[-1], cost_sim[0], colors='k')
plt.text(14.8, (cost_price[-1]+cost_sim[0])/2, 'PolySilicon\nPrice', fontsize=12, va='center')
plt.hlines(cost_price[-1], 14.3, 14.7, colors='k')
plt.hlines(cost_self[-1], 14.3, 14.7, colors='k')
plt.hlines(cost_sim[-1], 14.3, 14.7, colors='k')
plt.hlines(cost_sim[0], 14.3, 14.7, colors='k')
plt.ylim(0, 8000)
plt.yticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], ['0', '1', '2', '3', '4', '5', '6', '7', '8'], fontsize=12)
plt.ylabel('Total installed cost ($1000/kW)', fontsize=20, labelpad=5)
plt.xticks([1, 3, 5, 7, 9, 11, 14], ["2010", "2012", "2014", "2016", "2018", "2020", "2023"], fontsize=12)
plt.xlabel('Year', fontsize=20)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# ax.remove()
plt.legend(loc='lower left', fontsize=12, frameon=False, ncol=2)
plt.savefig('figs/Global_solar.png', dpi=600)


fig, ax = plt.subplots(figsize=(20, 16), frameon=False)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)
grid = plt.GridSpec(16, 2)

plt.subplot(grid[6:16, 0])
plt.imshow(plt.imread('figs/polar_bar_solar.png'))
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.text(0, 140, 'c.', weight='bold', size=20)
plt.subplot(grid[6:16, 1])
plt.imshow(plt.imread('figs/polar_bar_wind.png'))
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.text(0, 140, 'd.', weight='bold', size=20)

plt.subplot(grid[0:6, 0])
plt.imshow(plt.imread('figs/Global_solar.png'))
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.text(0, 1, 'a.', weight='bold', size=20)
plt.subplot(grid[0:6, 1])
plt.imshow(plt.imread('figs/Global_wind.png'))
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.text(0, 1, 'b.', weight='bold', size=20)
# ax.remove()
plt.savefig('figs/Fig1.jpg', dpi=600)
# plt.show()

fig, ax = plt.subplots(figsize=(16, 20), frameon=False)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)
grid = plt.GridSpec(2, 1)
plt.subplot(grid[0, :])
plt.imshow(plt.imread('figs/Country_solar.png'))
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.text(0, 1, 'a.', weight='bold', size=20)
plt.subplot(grid[1, :])
plt.imshow(plt.imread('figs/Country_wind.png'))
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.text(0, 1, 'b.', weight='bold', size=20)
plt.savefig('figs/FigS1.jpg', dpi=600)
# plt.show()
