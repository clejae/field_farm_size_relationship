import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotting_lib
import numpy as np
import warnings

def create_summary_statistics():
    shp = gpd.read_file(r"Q:\FORLand\Field_farm_size_relationship\data\vector\IACS\IACS_ALL_2018_with_grassland_recl_3035.shp")

    shp["field_area"] = shp["geometry"].area

    ## Data prep
    shp["federal_state"] = shp["field_id"].apply(lambda x: x[:2])
    shp["FormerDDR"] = shp["federal_state"].map({"BB": 1, "SA": 1, "TH": 1, "LS": 0, "BV": 0})
    shp["cover_type"] = shp["ID_KTYP"].map({
            "MA": "Cropland",  # maize --> maize
            "WW": "Cropland",  # winter wheat --> cereals
            "SU": "Cropland",  # sugar beet --> sugar beet
            "OR": "Cropland",  # oilseed rape --> oilseed rape
            "PO": "Cropland",  # potato --> potatoes
            "SC": "Cropland",  # summer cereals --> cereals
            "TR": "Cropland",  # triticale --> cereals
            "WB": "Cropland",  # winter barely --> cereals
            "WR": "Cropland",  # winter rye --> cereals
            "LE": "Cropland",  # legumes  --> legumes
            "AR": "Grassland",  # arable grass --> grass
            "GR": "Grassland",  # permanent grassland --> grass
            "FA": "Other",  # unkown --> others
            "PE": "Other",  # 40-mehrjährige Kulturen und Dauerkulturen --> others
            "UN": "Other",  # unkown --> others
            "GA": "Other",  # garden flowers --> others
            "MI": "Other",  # mix --> others
        })

    ## Examples of farm_identifiers
    t = shp.drop_duplicates(subset="federal_state")

    ## Mean field sizes
    avg_field_sizes_ddr = shp.groupby(by="FormerDDR").agg(
        num_fields=pd.NamedAgg("field_id", "nunique"),
        mean_field_size=pd.NamedAgg("field_size", "mean")
    ).reset_index()

    avg_field_sizes_states = shp.groupby(by=["federal_state"]).agg(
        num_fields=pd.NamedAgg("field_id", "nunique"),
        mean_field_size=pd.NamedAgg("field_size", "mean"),
        total_area=pd.NamedAgg("field_size", "sum")
    ).reset_index()

    avg_field_sizes_states_detailed = shp.groupby(by=["federal_state", "cover_type"]).agg(
        num_fields=pd.NamedAgg("field_id", "nunique"),
        mean_field_size=pd.NamedAgg("field_size", "mean"),
        total_area=pd.NamedAgg("field_size", "sum")
    ).reset_index()

    avg_field_sizes_cover_type= shp.groupby(by=["cover_type"]).agg(
        num_fields=pd.NamedAgg("field_id", "nunique"),
        mean_field_size=pd.NamedAgg("field_size", "mean"),
        total_area=pd.NamedAgg("field_size", "sum")
    ).reset_index()

    ## Mean farm size
    farms = shp.drop_duplicates(subset="farm_id").copy()
    farms.drop(columns=["field_size", "field_id"], inplace=True)

    avg_farm_sizes_ddr = farms.groupby(by="FormerDDR").agg(
        num_farms=pd.NamedAgg("farm_id", "nunique"),
        mean_farm_size=pd.NamedAgg("farm_size", "mean")
    ).reset_index()

    avg_farm_sizes_states = farms.groupby(by=["federal_state"]).agg(
        num_farms=pd.NamedAgg("farm_id", "nunique"),
        mean_farm_size=pd.NamedAgg("farm_size", "mean")
    ).reset_index()

    summary_v1 = pd.merge(avg_farm_sizes_states, avg_field_sizes_states, on="federal_state", how="left")
    summary_v1.columns = ["State", "N farms", "Average farm size", "N fields", "Average field size", "Total area"]
    summary_v1 = summary_v1[["State", "N farms", "N fields", "Average farm size", "Average field size", "Total area"]]
    summary_v1["Average farm size"] = round(summary_v1["Average farm size"], 0)
    summary_v1["Average field size"] = round(summary_v1["Average field size"], 1)

    avg_field_sizes_states_detailed.columns = ["State", "Cover type", "N fields", "Average field size", "Total area"]
    avg_field_sizes_states_detailed["Average field size"] = round(avg_field_sizes_states_detailed["Average field size"], 0)

    summary_v1.to_csv(r"Q:\FORLand\Field_farm_size_relationship\data\tables\summary_stats_fields_farms\summary_stats_fields_farms.csv", index=False)
    avg_field_sizes_states_detailed.to_csv(r"Q:\FORLand\Field_farm_size_relationship\data\tables\summary_stats_fields_farms\avg_field_sizes_states_detailed.csv", index=False)


    farms["fstate_id"] = farms["farm_id"].apply(lambda x: x[:2])


    df_agg = shp.groupby(["farm_id"]).agg(
            num_states=pd.NamedAgg("federal_state", pd.Series.nunique)
        ).reset_index()

    df_agg = df_agg.loc[df_agg["num_states"] > 1].copy()

    iacs_sub = shp.loc[shp["farm_id"].isin(list(df_agg["farm_id"]))].copy()

    def calc_state_area(group, state):
        state_area = group.loc[group["federal_state"] == state, "field_size"].sum()
        return state_area

    bb = iacs_sub.groupby(["farm_id"]).apply(calc_state_area, state='BB').reset_index(name="bb_area")
    sa = iacs_sub.groupby(["farm_id"]).apply(calc_state_area, state='SA').reset_index(name="sa_area")
    th = iacs_sub.groupby(["farm_id"]).apply(calc_state_area, state='TH').reset_index(name="th_area")
    ls = iacs_sub.groupby(["farm_id"]).apply(calc_state_area, state='LS').reset_index(name="ls_area")
    bv = iacs_sub.groupby(["farm_id"]).apply(calc_state_area, state='BV').reset_index(name="bv_area")

    from functools import reduce
    df_out1 = reduce(lambda left, right: pd.merge(left, right, on='farm_id'), [bb, sa, th, ls, bv])

    def count_fields(group, state):
        num_fields = len(group.loc[group["federal_state"] == state, "field_id"].unique())
        return num_fields

    bb = iacs_sub.groupby(["farm_id"]).apply(count_fields, state='BB').reset_index(name="bb_num_fields")
    sa = iacs_sub.groupby(["farm_id"]).apply(count_fields, state='SA').reset_index(name="sa_num_fields")
    th = iacs_sub.groupby(["farm_id"]).apply(count_fields, state='TH').reset_index(name="th_num_fields")
    ls = iacs_sub.groupby(["farm_id"]).apply(count_fields, state='LS').reset_index(name="ls_num_fields")
    bv = iacs_sub.groupby(["farm_id"]).apply(count_fields, state='BV').reset_index(name="bv_num_fields")

    df_out2 = reduce(lambda left, right: pd.merge(left, right, on='farm_id'), [bb, sa, th, ls, bv])

    df_out = pd.merge(df_out1, df_out2, "left", "farm_id")

    cols = ["bb_area", "sa_area", "th_area", "ls_area", "bv_area"]
    df_out["main_state"] = df_out[cols].idxmax(axis=1)
    df_out["farm_size"] = df_out[cols].sum(axis=1)
    df_out["main_st_area"] = df_out[cols].max(axis=1)
    df_out["out_st_area"] = df_out["farm_size"] - df_out["main_st_area"]
    df_out["share_out"] = round(df_out["out_st_area"] / df_out["farm_size"] * 100, 1)

    cols = ["bb_num_fields", "sa_num_fields", "th_num_fields", "ls_num_fields", "bv_num_fields"]
    df_out["num_fields"] = df_out[cols].sum(axis=1)
    df_out["main_st_fields"] = df_out[cols].max(axis=1)
    df_out["out_st_fields"] = df_out["num_fields"] - df_out["main_st_fields"]

    df_out["fstate_id"] = df_out["farm_id"].apply(lambda x: x[:2])
    df_out = df_out.loc[df_out["fstate_id"].isin(["12", "15", "16"])].copy()


    t = farms.loc[farms["federal_state"].isin(["BB", "SA", "TH"])].copy()
    t2 = t.groupby(["federal_state", "fstate_id"]).count()

    ## Manual calculation: Farms registerd in respective federal state minus farms active in the other two states
    bb_farms_from_bb = (5506 - 3 - 0)
    sa_farms_from_sa = (4208 - 59 - 1)
    th_farms_from_th = (4461 - 0 - 78)
    farms_only_in_registered_state = bb_farms_from_bb + sa_farms_from_sa + th_farms_from_th
    farms_operating_in_more_states = 1 - (farms_only_in_registered_state / (6034 + 4898 + 5038))
    num_farms = len(shp["farm_id"].unique())
    num_fields = len(shp)
    avg_field_size = shp["field_size"].mean()
    avg_farm_size = farms["farm_size"].mean()
    total_area = farms["farm_size"].sum()
    mean_share_outside_main_state = df_out["share_out"].mean()
    std_share_outside_main_stata = df_out["share_out"].std()

    with open(r"Q:\FORLand\Field_farm_size_relationship\data\tables\summary_stats_fields_farms\summary_stats_overall.csv", "w") as file:
        file.write(f"num_farms, {num_farms}\n")
        file.write(f"num_fields, {num_fields}\n")
        file.write(f"avg_field_size, {avg_field_size}\n")
        file.write(f"avg_farm_size, {avg_farm_size}\n")
        file.write(f"total_area, {total_area}\n")
        file.write(f"farms_operating_in_more_states, {farms_operating_in_more_states}\n")
        file.write(f"mean_share_outside_main_state, {mean_share_outside_main_state}\n")
        file.write(f"std_share_outside_main_stata, {std_share_outside_main_stata}\n")

def explore_farm_distribution_across_states(iacs, shp_out_pth, csv_out_pth, plt_out_pth):


    print("Explore farm distribution across states borders.")
    iacs["state"] = iacs["field_id"].apply(lambda x: x.split("_")[0])

    df_agg = iacs.groupby(["farm_id"]).agg(
        num_states=pd.NamedAgg("state", pd.Series.nunique)
    ).reset_index()

    df_agg = df_agg.loc[df_agg["num_states"] > 1].copy()

    iacs_sub = iacs.loc[iacs["farm_id"].isin(list(df_agg["farm_id"]))].copy()
    iacs_sub.to_file(shp_out_pth)

    def calc_state_area(group, state):
        state_area = group.loc[group["state"] == state, "field_size"].sum()
        return state_area

    bb = iacs_sub.groupby(["farm_id"]).apply(calc_state_area, state='BB').reset_index(name="bb_area")
    sa = iacs_sub.groupby(["farm_id"]).apply(calc_state_area, state='SA').reset_index(name="sa_area")
    th = iacs_sub.groupby(["farm_id"]).apply(calc_state_area, state='TH').reset_index(name="th_area")
    ls = iacs_sub.groupby(["farm_id"]).apply(calc_state_area, state='LS').reset_index(name="ls_area")
    bv = iacs_sub.groupby(["farm_id"]).apply(calc_state_area, state='BV').reset_index(name="bv_area")

    from functools import reduce
    df_out1 = reduce(lambda left, right: pd.merge(left, right, on='farm_id'), [bb, sa, th, ls, bv])

    def count_fields(group, state):
        num_fields = len(group.loc[group["state"] == state, "field_id"].unique())
        return num_fields

    bb = iacs_sub.groupby(["farm_id"]).apply(count_fields, state='BB').reset_index(name="bb_num_fields")
    sa = iacs_sub.groupby(["farm_id"]).apply(count_fields, state='SA').reset_index(name="sa_num_fields")
    th = iacs_sub.groupby(["farm_id"]).apply(count_fields, state='TH').reset_index(name="th_num_fields")
    ls = iacs_sub.groupby(["farm_id"]).apply(count_fields, state='LS').reset_index(name="ls_num_fields")
    bv = iacs_sub.groupby(["farm_id"]).apply(count_fields, state='BV').reset_index(name="bv_num_fields")

    df_out2 = reduce(lambda left, right: pd.merge(left, right, on='farm_id'), [bb, sa, th, ls, bv])

    df_out = pd.merge(df_out1, df_out2, "left", "farm_id")

    cols = ["bb_area", "sa_area", "th_area", "ls_area", "bv_area"]
    df_out["main_state"] = df_out[cols].idxmax(axis=1)
    df_out["farm_size"] = df_out[cols].sum(axis=1)
    df_out["main_st_area"] = df_out[cols].max(axis=1)
    df_out["out_st_area"] = df_out["farm_size"] - df_out["main_st_area"]
    df_out["share_out"] = round(df_out["out_st_area"] / df_out["farm_size"] * 100, 1)

    cols = ["bb_num_fields", "sa_num_fields", "th_num_fields", "ls_num_fields", "bv_num_fields"]
    df_out["num_fields"] = df_out[cols].sum(axis=1)
    df_out["main_st_fields"] = df_out[cols].max(axis=1)
    df_out["out_st_fields"] = df_out["num_fields"] - df_out["main_st_fields"]

    fig, ax = plt.subplots()
    sns.histplot(data=df_out, x="share_out", bins=30, ax=ax)
    ax.annotate(s=f"Only for BB, SA, TH, as LS and BV\nuse different farm identifiers.", xy=(29, 50))
    ax.annotate(s=f"Total affected area: {round(df_out['out_st_area'].sum(), 0)} ha", xy=(29, 43))
    ax.annotate(s=f"No. affected fields: {round(df_out['out_st_fields'].sum(), 0)}", xy=(29, 40))
    ax.annotate(s=f"No. affected farms: {len(df_out['farm_id'].unique())}", xy=(29, 37))
    ax.set_xlabel("Share of farm outside 'main state' [%]")

    plt.savefig(plt_out_pth)

    df_out.to_csv(csv_out_pth, index=False)

def explore_grasslands(iacs, field_id_col, field_size_col, farm_id_col, farm_size_col,  out_pth):
    print("Explore field and farm sizes of grasslands.")
    ## Derive federal state from field id
    federal_state_col = "federal_state"
    iacs[federal_state_col] = iacs[field_id_col].apply(lambda x: x.split("_")[0])

    iacs["grassland"] = 0
    iacs.loc[iacs["ID_KTYP"] == "GR", "grassland"] = 1
    iacs.loc[iacs["ID_KTYP"].isin(["FA", "MI", "PE", "GA"]), "grassland"] = 2

    ## Calculate field statistice for arable fields, grassland, and fields that did not fall into the CSTypology
    field_stats = iacs.groupby(["grassland"]).agg(
        area=pd.NamedAgg(field_size_col, sum),
        mean_field_size=pd.NamedAgg(field_size_col, np.mean),
        std_field_size=pd.NamedAgg(field_size_col, np.std)
    )

    ## Calculate farm statistics for farms with grassland and without
    def calc_grass_share(group):
        total_area = group[field_size_col].sum()
        grass_area = group.loc[group["grassland"] == 1, field_size_col].sum()
        return round(grass_area / total_area, 2)

    grass_share = iacs.groupby([farm_id_col]).apply(calc_grass_share).reset_index()
    grass_share.columns = [farm_id_col, "grass_share"]

    farms = iacs.groupby(farm_id_col).agg(
        farm_size=pd.NamedAgg(farm_size_col, "first"),
        federal_state=pd.NamedAgg(federal_state_col, "first"),
        n_farm_sizes=pd.NamedAgg(farm_size_col, "nunique"),  # to control for consistency
        mean_field_size=pd.NamedAgg(field_size_col, "mean"),
        num_fields=pd.NamedAgg(field_id_col, "nunique")
    ).reset_index()

    farms = pd.merge(farms, grass_share, how="left", on=farm_id_col)
    farms["with_grassland"] = 0
    farms.loc[farms["grass_share"] > 0, "with_grassland"] = 1

    farm_stats = farms.groupby(["with_grassland"]).agg(
        mean_farm_area=pd.NamedAgg("farm_size", "mean"),
        mean_field_size=pd.NamedAgg("mean_field_size", "mean"),
        mean_num_fields=pd.NamedAgg("num_fields", "mean"),
        total_area=pd.NamedAgg("farm_size", "sum"),
        num_farms=pd.NamedAgg(farm_id_col, "nunique")
    ).reset_index()

    farm_stats_fs = farms.groupby(["with_grassland", "federal_state"]).agg(
        mean_farm_area=pd.NamedAgg("farm_size", "mean"),
        mean_field_size=pd.NamedAgg("mean_field_size", "mean"),
        mean_num_fields=pd.NamedAgg("num_fields", "mean"),
        total_area=pd.NamedAgg("farm_size", "sum"),
        num_farms=pd.NamedAgg(farm_id_col, "nunique")
    ).reset_index()

    ## Write out
    with pd.ExcelWriter(out_pth) as writer:
        field_stats.to_excel(writer, sheet_name='field_stats')
        farm_stats.to_excel(writer, sheet_name='farm_stats')
        farm_stats_fs.to_excel(writer, sheet_name='farm_stats_fs')

    print("Done!")


def explore_field_sizes(iacs, farm_id_col, field_id_col, field_size_col, farm_size_col, crop_class_col):
    print("Explore field sizes.")
    ## Derive federal state from field id
    federal_state_col = "federal_state"
    iacs[federal_state_col] = iacs[field_id_col].apply(lambda x: x.split("_")[0])

    ## Derive general statistics about each farm
    farms = iacs.groupby(farm_id_col).agg(
        farm_size=pd.NamedAgg(farm_size_col, "first"),
        n_farm_sizes=pd.NamedAgg(farm_size_col, "nunique"), # to control for consistency
        mean_field_size=pd.NamedAgg(field_size_col, "mean"),
        max_field_size=pd.NamedAgg(field_size_col, "max"),
        min_field_size=pd.NamedAgg(field_size_col, "min"),
        num_fields=pd.NamedAgg(field_id_col, "nunique")
    )

    consistency_test = farms.loc[farms["n_farm_sizes"] > 1].copy()

    if not consistency_test.empty:
        warnings.warn("IACS dataframe has multiple farm sizes for a single farm!")
    else:
        print("Consistency test passed.")

    ## Generate some ratios based on the maximum and minimum field sizes
    farms["range_field_sizes"] = farms["max_field_size"] - farms["min_field_size"]
    farms["range_field_ratio"] = farms["range_field_sizes"] / farms["farm_size"]
    farms["max_field_ratio"] = farms["max_field_size"] / farms["farm_size"]
    farms["min_field_ratio"] = farms["min_field_size"] / farms["farm_size"]
    farms["diff_ratios"] = farms["max_field_ratio"] - farms["min_field_ratio"]

    farms["log_max_field_ratio"] = np.log(farms["max_field_ratio"])
    farms["log_min_field_ratio"] = np.log(farms["min_field_ratio"])
    farms["log_farm_size"] = np.log(farms["farm_size"])

    ## Classify farms based on their total farm size
    farms["farm_size_class"] = pd.cut(farms["farm_size"], [0, 10, 25, 100, 250, 1000, 2500], labels=[10, 25, 100, 250, 1000, 2500])
    farms["farm_size_class"] = pd.qcut(x=farms["farm_size"], q=10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    ## Plot farm size vs max and minimum field size ratios in scatterplot
    # fig, axs = plt.subplots(nrows=3, ncols=1)
    # sns.scatterplot(data=farms, x="farm_size", y="max_field_ratio", s=3, hue="num_fields", ax=axs[0])
    # sns.scatterplot(data=farms, x="farm_size", y="min_field_ratio", s=3, ax=axs[1])
    # sns.scatterplot(data=farms, x="farm_size", y="max_field_ratio", s=3, ax=axs[2])
    # fig.tight_layout()

    ## Plot farm size vs difference of maximum and minimum field size
    # fig, axs = plt.subplots()
    # sns.scatterplot(data=farms, x="farm_size", y="diff_ratios", s=3, ax=axs)
    # fig.tight_layout()

    ## Plot farm size vs range of maximum and minimum field size
    # fig, axs = plt.subplots()
    # sns.scatterplot(data=farms, x="farm_size", y="range_field_ratio", s=3, ax=axs)
    # fig.tight_layout()

    ## Plot farm size vs maximum and minimum field size ratio
    fig, axs = plt.subplots()
    sns.scatterplot(data=farms, x="farm_size", y="max_field_ratio", s=3, ax=axs)
    sns.scatterplot(data=farms, x="farm_size", y="min_field_ratio", s=3, ax=axs)
    axs.set_xlabel("Farm size [ha]")
    axs.set_ylabel("ratio of max. (blue) and min. (orange) field sizes")
    fig.tight_layout()
    plt.savefig(fr"figures\scatterplot_max+min_field_ratios_with_grassland.png")
    plt.close()

    ## Plot minimum and maximum field sizes per farm size class in boxplots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=plotting_lib.cm2inch((16, 6)))
    sns.boxplot(data=farms, x="farm_size_class", y="min_field_size", showfliers=False, ax=axs[1])
    axs[1].set_xlabel("Deciles of farm sizes")
    axs[1].set_ylabel("Min. field size [ha]")
    axs[1].set_title("b)", loc="left")
    axs[1].grid(visible=True, which="major", axis="y", zorder=0)
    axs[1].set_axisbelow(True)
    sns.boxplot(data=farms, x="farm_size_class", y="max_field_size", showfliers=False, ax=axs[0])
    axs[0].set_xlabel("Deciles of farm sizes")
    axs[0].set_ylabel("Max. field size [ha]")
    axs[0].set_title("a)", loc="left")
    axs[0].grid(visible=True, which="major", axis="y", zorder=0)
    axs[0].set_axisbelow(True)
    fig.tight_layout()
    plt.savefig(fr"figures\boxplot_min_max_field_sizes_per_farm_decile_with_grassland.png")

    ## Calculate area of crop classes in total and per federal state
    crop_classes = iacs[[crop_class_col, federal_state_col, field_size_col]].groupby([crop_class_col, federal_state_col]).agg(
        total_area=pd.NamedAgg(field_size_col, "sum"),
        mean_area=pd.NamedAgg(field_size_col, "mean"),
        max_area=pd.NamedAgg(field_size_col, "max"),
        min_area=pd.NamedAgg(field_size_col, "min"),
        median_area=pd.NamedAgg(field_size_col, "median")
    ).reset_index()
    crop_classes["number_fields"] = crop_classes["total_area"] / crop_classes["mean_area"]
    crop_classes["share"] = round(crop_classes["total_area"] / crop_classes["total_area"].sum() * 100, 2)
    crop_classes.to_csv(fr"data\tables\statistics_about_crop_classes_per_federal_state.csv", index=False, sep=";")

    crop_classes_total = iacs[[crop_class_col, field_size_col]].groupby([crop_class_col]).agg(
        total_area=pd.NamedAgg(field_size_col, "sum"),
        mean_area=pd.NamedAgg(field_size_col, "mean"),
        max_area=pd.NamedAgg(field_size_col, "max"),
        min_area=pd.NamedAgg(field_size_col, "min"),
        median_area=pd.NamedAgg(field_size_col, "median")
    ).reset_index()
    crop_classes_total["number_fields"] = crop_classes_total["total_area"] / crop_classes_total["mean_area"]
    crop_classes_total["share"] = round(crop_classes_total["total_area"] / crop_classes_total["total_area"].sum() * 100, 2)
    crop_classes_total.to_csv(fr"data\tables\statistics_about_crop_classes.csv", index=False, sep=";")

    fig, axs = plt.subplots(nrows=2, figsize=plotting_lib.cm2inch((16, 12)))
    sns.boxplot(data=iacs, x=crop_class_col, y=field_size_col, showfliers=False, ax=axs[0])
    axs[0].set_xlabel("Crop class")
    axs[0].set_ylabel("Field size")
    axs[0].set_title("a) Field sizes", loc="left")
    axs[0].grid(visible=True, which="major", axis="y", zorder=0)
    axs[0].set_axisbelow(True)
    sns.boxplot(data=iacs, x=crop_class_col, y=farm_size_col, showfliers=False, ax=axs[1])
    axs[1].set_xlabel("Crop class")
    axs[1].set_ylabel("Farm size")
    axs[1].set_title("a) Farm sizes", loc="left")
    axs[1].grid(visible=True, which="major", axis="y", zorder=0)
    axs[1].set_axisbelow(True)
    fig.tight_layout()
    plt.savefig(fr"figures\boxplots_field_and_farm_sizes_per_crop_class.png")

    crop_names = list(iacs[crop_class_col].unique())
    iacs["farm_size_r"] = iacs["farm_size"] - iacs["field_size"]
    iacs["log_farm_size"] = np.log(iacs[farm_size_col])
    iacs["log_field_size"] = np.log(iacs[field_size_col])
    iacs["log_farm_size_r"] = np.log(iacs["farm_size_r"])

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=plotting_lib.cm2inch((32, 32)))
    for cn, ax in zip(crop_names, axs.ravel()):
        sub = iacs.loc[iacs[crop_class_col] == cn].copy()
        sns.scatterplot(data=sub, x="log_field_size", y="log_farm_size", s=3, ax=ax)
        ax.set_title(cn.upper())
        ax.set_xlabel("Log field size")
        ax.set_ylabel("Log farm size")
    fig.tight_layout()
    plt.savefig(fr"figures\scatter_log_field_vs_log_farm_sizes_per_crop_class.png")

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=plotting_lib.cm2inch((32, 32)))
    for cn, ax in zip(crop_names, axs.ravel()):
        sub = iacs.loc[iacs[crop_class_col] == cn].copy()
        sns.scatterplot(data=sub, x="log_field_size", y="log_farm_size_r", s=3, ax=ax)
        ax.set_title(cn.upper())
        ax.set_xlabel("Log field size")
        ax.set_ylabel("Log farm size r")
    fig.tight_layout()
    plt.savefig(fr"figures\scatter_log_field_vs_log_rest_farm_sizes_per_crop_class.png")

    print("done!")


def main():
    create_summary_statistics()

    # pth = fr"data\vector\IACS\IACS_{key}_2018_new.shp"
    # iacs = gpd.read_file(pth)
    # explore_farm_distribution_across_states(
    #     iacs=iacs,
    #     shp_out_pth=r"data\vector\IACS\IACS_2018_farms_in_multiple_states_no_grassland.shp",
    #     csv_out_pth=fr"data\tables\farm_distribution_across_states_no_grassland.csv",
    #     plt_out_pth=r"figures\share_of_farm_outside_main_state_no_grassland.png")

    iacs = gpd.read_file(fr"data\vector\IACS\IACS_ALL_2018_with_grassland_recl.shp")
    explore_grasslands(
        iacs=iacs,
        farm_id_col="farm_id",
        field_id_col="field_id",
        field_size_col="field_size",
        farm_size_col="farm_size",
        out_pth=r"data\tables\grassland_statistics.xlsx"
    )

    explore_farm_distribution_across_states(
        iacs=iacs,
        shp_out_pth=r"data\vector\IACS\IACS_2018_farms_in_multiple_states.shp",
        csv_out_pth=fr"data\tables\farm_distribution_across_states.csv",
        plt_out_pth=r"figures\share_of_farm_outside_main_state_png"
    )


    explore_field_sizes(
        iacs=iacs,
        farm_id_col="farm_id",
        field_id_col="field_id",
        field_size_col="field_size",
        farm_size_col="farm_size",
        crop_class_col="ID_KTYP"
    )

    # Get some basic statistics
    iacs["fstate"] = iacs["field_id"].apply(lambda x: x.split('_')[0])
    df_stat = iacs.groupby("fstate").agg(
        area=pd.NamedAgg(column="field_size", aggfunc="sum"),
        num_fields=pd.NamedAgg(column="field_id", aggfunc="count"),
        num_farms=pd.NamedAgg(column="farm_id", aggfunc=list)
    ).reset_index()
    df_stat["num_farms"] = df_stat["num_farms"].apply(lambda x: len(set(x)))
    df_stat["mean_field_size"] = df_stat["area"] / df_stat["num_fields"]
    df_stat["mean_farm_size"] = df_stat["area"] / df_stat["num_farms"]

    df_stat.to_csv(r"data\tables\general_stats.csv", index=False)
    print("Done calculating general statistics.")

if __name__ == '__main__':
    main()