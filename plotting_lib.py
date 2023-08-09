## Authors:
## github Repo: https://github.com/clejae

def plot_map(shp, out_pth, col, vmin=None, vmax=None, shp2_pth=None, highlight_extremes=False):
    import geopandas as gpd
    import matplotlib.pyplot as plt

    print(f"Plot {col} values.")
    if not vmin:
        # vmin = shp[col].min()
        vmin = shp[col].quantile(q=0.02)

    if not vmax:
        # vmax = shp[col].max()
        vmax = shp[col].quantile(q=0.98)

    print(f'\tSaving at {out_pth}')
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])

    shp.plot(
        column=col,
        ax=ax,
        legend=True,
        legend_kwds={'label': f"{col}", 'orientation': "horizontal", "fraction": 0.04, "anchor": (0.5, 1.5)},
        vmin=vmin,
        vmax=vmax,
        edgecolor='none'
    )

    if highlight_extremes:
        tr = shp[col].mean() + (2 * shp[col].std())
        shp.loc[shp[col] > tr].plot(edgecolor='red', facecolor="none", ax=ax, lw=0.5, zorder=2)

    if shp2_pth:
        shp2 = gpd.read_file(shp2_pth)
        shp2.plot(edgecolor='black', facecolor="none", ax=ax, lw=0.3, zorder=3)

    ax.axis("off")
    ax.margins(0)

    plt.tight_layout()
    plt.savefig(out_pth)
    plt.close()


def plot_maps_in_grid(shp, out_pth, cols, nrow, ncol, figsize, shp2_pth=None, titles=None, highlight_extremes=False, dpi=300):

    import numpy as np
    import matplotlib.pyplot as plt
    import geopandas as gpd

    # ncol = len(cols)
    print(f'Plotting {cols} to {out_pth}')
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)

    for i, col in enumerate(cols):
        ix = np.unravel_index(i, axs.shape)

        vmin = shp[col].quantile(q=0.02)
        vmax = shp[col].quantile(q=0.98)

        p = shp.plot(
            column=col,
            ax=axs[ix],
            legend=True,
            legend_kwds={'label': f"{col}", 'orientation': "horizontal", "fraction": 0.04, "anchor": (0.5, 1.5)},
            vmin=vmin,
            vmax=vmax,
            edgecolor=None,
        )

        axs[ix].axis("off")
        axs[ix].margins(0)

        if titles:
            title = titles[i]
            axs[ix].set_title(title, size=14)
        axs[ix].axis("off")

        if highlight_extremes:
            tr = shp[col].mean() + (2 * shp[col].std())
            shp.loc[shp[col] > tr].plot(edgecolor='red', facecolor="none", ax=axs[ix], lw=0.5, zorder=2)

        if shp2_pth:
            shp2 = gpd.read_file(shp2_pth)
            shp2.plot(edgecolor='black', facecolor="none", ax=axs[ix], lw=0.3, zorder=2)

    fig.tight_layout()
    plt.savefig(out_pth, dpi=dpi)
    plt.close()


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def scatterplot_two_columns(df, col1, col2, out_pth, x_label=None, y_label=None, title=None, hue=None, log=False):
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f'Plotting scatterplot of {col1} against {col2}.\n Saving at {out_pth}')

    if hue:
        fig, ax = plt.subplots(1, 1, figsize=cm2inch(15, 10))
        sns.scatterplot(x=col1, y=col2, data=df, hue=hue, s=2, ax=ax)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 8}, title=hue)
    else:
        fig, ax = plt.subplots(1, 1, figsize=cm2inch(10, 10))
        # sns.scatterplot(x=col1, y=col2, data=df, s=4, ax=ax)

        sns.regplot(x=col1, y=col2, data=df, ax=ax, robust=True, n_boot=100,
                    scatter_kws=dict(s=4, linewidths=.7, edgecolors='none'))

        # Decorations
        # gridobj.set(xlim=(0.5, 7.5), ylim=(0, 50))

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{col1} vs. {col2}")

    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    if log:
        ax.set(xscale="log", yscale="log")
        # ax.set(xscale="log")

    fig.tight_layout()
    plt.savefig(out_pth, dpi=600)
    plt.close()


def kernel_density_plot(df, col, out_pth):
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f'Plotting kernel density of {col} a.\n Saving at {out_pth}')

    fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=cm2inch(10, 10))
    sns.kdeplot(data=df, x=col, ax=ax, legend=False, linestyle='-', linewidth=1,
                color='black') #log_scale=True, cut=0,
    ax.set_title(col, x=.05, y=.995, pad=-14, fontdict={'size': 10})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()
    plt.savefig(out_pth, dpi=300)


def boxplot_by_categories(df, category_col, value_col, out_pth, ylims=None, x_label = None, y_label = None, title = None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=cm2inch((20, 10)))
    sns.boxplot(x=category_col, y=value_col, data=df, ax=ax, linewidth=0.5, showfliers=False)
    if ylims:
        ax.set_ylim(ylims)

    if title:
        ax.set_title(title)

    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    fig.tight_layout()
    plt.savefig(out_pth, dpi=300)
    plt.close()


def boxplots_in_grid(df, out_pth, value_cols, category_col, nrow, ncol, figsize, x_labels=None, y_labels=None, titles=None, dpi=300):
    import matplotlib.pyplot as plt
    import seaborn as sns

    import numpy as np
    print(f'Plotting {value_cols} to {out_pth}')
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)

    for i, col in enumerate(value_cols):
        ix = np.unravel_index(i, axs.shape)

        sns.boxplot(x=category_col, y=col, data=df, ax=axs[ix], linewidth=0.5, showfliers=False)

        if x_labels:
            x_label= x_labels[i]
            axs[ix].set_xlabel(x_label)

        if y_labels:
            y_label = y_labels[i]
            axs[ix].ylabel(y_label)

        if titles:
            title = titles[i]
            axs[ix].set_title(title, size=14)

    fig.tight_layout()
    plt.savefig(out_pth, dpi=dpi)
    plt.close()


def plot_farm_vs_field_sizes_from_iacs(df, farm_id_col, area_col, crop_class_col, bins, crop_classes, out_pth,
                                       col_wrap=4, x_lim=(-5, 50)):

    import pandas as pd
    import seaborn as sns
    # farms = df.groupby(farm_id_col).agg(
    #      farm_area=pd.NamedAgg(column=area_col, aggfunc="sum"),
    #      field_size=pd.NamedAgg(column=area_col, aggfunc="mean"))
    # farms.sort_values(by="farm_area", inplace=True)
    #
    # # farms["farm_quant"] = pd.qcut(farms["farm_area"], [0, .2, .4, .6, .8, 1])
    # farms["farm_area_class"] = pd.cut(farms["farm_area"], bins)
    #
    # unique = farms["farm_area_class"].unique()
    # palette = dict(zip(unique, sns.color_palette("flare", n_colors=len(unique))))
    # palette.update({"Total": "k"})
    # # p1 = sns.displot(farms, x="field_size", hue="farm_area_class", kind="kde", palette=palette)

    farms2 = df.groupby([farm_id_col, crop_class_col]).agg(
        farm_area=pd.NamedAgg(column=area_col, aggfunc="sum"),
        field_size=pd.NamedAgg(column=area_col, aggfunc="mean")).reset_index()
    farms2.sort_values(by="farm_area", inplace=True)
    # farms2["farm_quant"] = pd.qcut(farms2["farm_area"], [0,.2,.4,.6,.8,1])
    farms2["farm_area_class"] = pd.cut(farms2["farm_area"], bins)

    unique = farms2["farm_area_class"].unique()
    palette = dict(zip(unique, sns.color_palette("flare", n_colors=len(unique))))
    palette.update({"Total": "k"})

    farms2.sort_values(by=crop_class_col, inplace=True)

    g = sns.FacetGrid(farms2.loc[farms2[crop_class_col].isin(crop_classes)],
                      col=crop_class_col,
                      col_wrap=col_wrap,
                      hue="farm_area_class",
                      palette=palette,
                      legend_out=False)
    g.map(sns.kdeplot, "field_size")
    g.add_legend()
    g.set(xlim=x_lim)
    # g.set(yscale='log')
    # g.set(xscale='log')

    g.savefig(out_pth)
