# Authors:
# github Repo: https://github.com/clejae

## Plots mean field sizes vs farm sizes per federal state.
## It's a bit messy.


import geopandas as gpd
import pandas as pd
import time
import numpy as np

# project library for plotting
import plotting_lib
import os

WD = r"Q:\FORLand\Field_farm_size_relationship"

def field_vs_farm_size(df, farm_id_col, area_col, out_pth):

    farms = df.groupby(farm_id_col).agg(
         farm_area=pd.NamedAgg(column=area_col, aggfunc="sum"),
         field_size=pd.NamedAgg(column=area_col, aggfunc="mean"),
         max_field_size=pd.NamedAgg(column=area_col, aggfunc="max"),
         min_field_size=pd.NamedAgg(column=area_col, aggfunc="min"),
         field_num=pd.NamedAgg(column="ID", aggfunc="count")).reset_index()
    farms.sort_values(by="farm_area", inplace=True)

    one_fs = farms.loc[farms["field_num"] == 1].copy()

    # farms2 = farms.loc[~farms["BTNR"].isin(one_fs["BTNR"])].copy()
    farms2 = farms
    farms2["farm_area_log"] = np.log2(farms2["farm_area"])
    farms2["field_size_log"] = np.log2(farms2["field_size"])
    farms2["mp2"] = farms2["farm_area_log"] / farms2["field_size_log"]
    farms2["mp1"] = farms2["farm_area"] / farms2["field_size"]
    bins = [0, 1, 2, 3,4,5,6,7,8,9 ,10, 20, 2000]
    farms2["field_num_bins"] = pd.cut(farms2.field_num, bins)

    bins = [0, 1, 10, 25, 50, 100, 250, 2000]
    farms2["mp1"] = farms2["farm_area"] / farms2["field_size"]
    farms2["mp1_bins"] = pd.cut(farms2.mp1, bins)

    df2 = df.loc[df["BTNR"].isin(farms2.loc[farms2["mp1"] == 2, "BTNR"])].copy()

    df = pd.merge(df, farms[[farm_id_col, "farm_area"]], how="left", on=farm_id_col)
    bins = [0, 10, 100, 200, 1000, 20000]
    df["farm_area_class"] = pd.cut(df.farm_area, bins)
    df["farm_area_class"] = df["farm_area_class"].astype(str)

    plotting_lib.scatterplot_two_columns(df=farms2, col1="field_size", col2="farm_area", out_pth=out_pth,
                            x_label=None, y_label=None, title=None, hue="field_num_bins", log=True)

    out_pth2 = r'figures\all_field_vs_farm_size.png'
    plotting_lib.scatterplot_two_columns(df=df, col1="area", col2="farm_area", out_pth=out_pth2,
                            x_label=None, y_label=None, title=None, hue=None, log=True)

    out_pth2 = r'figures\all_field_vs_farm_size2.png'
    plotting_lib.scatterplot_two_columns(df=df, col1="area", col2="farm_area", out_pth=out_pth2,
                            x_label=None, y_label=None, title=None, hue="farm_area_class", log=False)

    for cl in enumerate(list(df["farm_area_class"].unique())):
        sub = df.loc[df["farm_area_class"].astype(str) == cl].copy()
        out_pth2 = fr'figures\field_vs_farm_size2_{cl}.png'
        plotting_lib.scatterplot_two_columns(df=sub, col1="area", col2="farm_area", out_pth=out_pth2,
                                x_label=None, y_label=None, title=None, hue=None, log=False)


    out_pth2 = r'figures\field_vs_farm_size2.png'
    farms3 = farms2.loc[farms2["mp1"] < 25].copy()
    plotting_lib.scatterplot_two_columns(df=farms3, col1="field_size", col2="farm_area", out_pth=out_pth2,
                            x_label=None, y_label=None, title=None, hue="field_num_bins", log=False)

    out_pth3 = r'figures\bp_field_vs_field_size_bins.png'
    plotting_lib.boxplot_by_categories(df=farms3, category_col="field_num_bins", value_col="field_size", out_pth=out_pth3,
                          ylims=None, x_label=None, y_label=None, title=None)

    out_pth3 = r'figures\bp_farms_size_vs_field_size_bins.png'
    plotting_lib.boxplot_by_categories(df=farms3, category_col="field_num_bins", value_col="farm_area", out_pth=out_pth3,
                          ylims=None, x_label=None, y_label=None, title=None)

    # data = np.random.randint(1, 50, size=1000)
    # dfr = pd.DataFrame(data, columns=['area'])
    # dfr["BTNR"] = np.random.randint(1, 250, size=1000)
    # dfr["ID"] = range(1, len(dfr)+1)
    #
    # farms = dfr.groupby(farm_id_col).agg(
    #      farm_area=pd.NamedAgg(column=area_col, aggfunc="sum"),
    #      field_size=pd.NamedAgg(column=area_col, aggfunc="mean"),
    #      field_num=pd.NamedAgg(column="ID", aggfunc="count")).reset_index()
    # farms.sort_values(by="farm_area", inplace=True)
    # bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 2000]
    # farms["field_num_bins"] = pd.cut(farms.field_num, bins)
    # out_pth2 = r'figures\field_vs_farm_size2_random.png'
    # scatterplot_two_columns(df=farms, col1="field_size", col2="farm_area", out_pth=out_pth2,
    #                         x_label="field_size_random", y_label="farm_size_random", title=None, hue="field_num_bins", log=False)
    # out_pth2 = r'figures\field_vs_farm_size2_random2.png'
    # plotting_lib.scatterplot_two_columns(df=farms, col1="field_size", col2="farm_area", out_pth=out_pth2,
    #                         x_label="field_size_random", y_label="farm_size_random", title=None, hue="field_num_bins", log=True)




def main():
    stime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + stime)
    os.chdir(WD)

    pth = r"data\vector\IACS\IACS_BB_2020_klassifiziert_25832.shp"
    df = gpd.read_file(pth)
    df["area"] = df.geometry.area / 10000
    df = df.loc[df["BTNR"].str.slice(0, 2) == '12'].copy()
    df.loc[df["ID_KTYP"].isna(), "ID_KTYP"] = 99
    df["ID_KTYP"] = df["ID_KTYP"].astype(int).astype(str)

    field_vs_farm_size(
        df=df,
        farm_id_col="BTNR",
        area_col="area",
        out_pth=r"figures\field_vs_farm_size.png")

    # crop_class_dict = {
    #     "1": "Maize",
    #     "2": "Winter wheat",
    #     "3": "Sugar beet",
    #     "4": "Winter rapeseed",
    #     "5": "Potato",
    #     "6": "Spring cereals",
    #     "7": "Triticale",
    #     "9": "Winter barley",
    #     "10": "Winter rye",
    #     "12": "Legumes",
    #     "13": "Arable Grass",
    #     "14": "Legumes",
    #     "60": "Vegetables",
    #     "80": "Set-aside",
    #     "99": "Unkown"
    # }
    #
    # df["crop_class"] = df["ID_KTYP"].map(crop_class_dict)
    # # bins = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 200, 20000] ## BMEL Statistik, Größenklassen
    # bins = [0, 10, 100, 200, 1000, 20000]
    # crop_classes = ["Maize", "Winter wheat", "Sugar beet", "Winter rapeseed", "Potato", "Spring cereals" "Triticale",
    #                 "Winter barley", "Winter rye", "Legumes", "Arable Grass", "Vegetables"]
    # out_pth=r"figures\field_sizes_per_farm_classes_and_crop_type.png"
    # plotting_lib.plot_farm_vs_field_sizes_from_iacs(df=df, farm_id_col='BTNR', area_col='area', crop_class_col='crop_class',
    #                                    bins=bins, crop_classes=crop_classes, out_pth=out_pth)
    #
    # crop_class_dict2 = {
    #     "1": "Spring cereal crop",
    #     "2": "Winter cereal crop",
    #     "3": "Spring leaf crop",
    #     "4": "Winter leaf crop",
    #     "5": "Spring leaf crop",
    #     "6": "Spring cereal crop",
    #     "7": "Winter cereal crop",
    #     "9": "Winter cereal crop",
    #     "10": "Winter cereal crop",
    #     "12": "Spring leaf crop",
    #     "13": "Arable Grass",
    #     "14": "Winter leaf crop",
    #     "60": "Vegetables",
    #     "80": "Set-aside",
    #     "99": "Unkown"
    # }
    #
    # df["crop_class2"] = df["ID_KTYP"].map(crop_class_dict2)
    # # bins = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 200, 20000] ## BMEL Statistik, Größenklassen
    # bins = [0, 10, 100, 200, 1000, 20000]
    # crop_classes = ["Winter leaf crop", "Winter cereal crop", "Spring leaf crop", "Spring cereal crop"]
    # out_pth = r"figures\field_sizes_per_farm_classes_and_broader_crop_type.png"
    # plotting_lib.plot_farm_vs_field_sizes_from_iacs(df=df, farm_id_col='BTNR', area_col='area', crop_class_col='crop_class2',
    #                                    bins=bins, crop_classes=crop_classes, out_pth=out_pth)


    # pth = r"C:\Users\IAMO\Documents\work_data\cst_paper\data\tables\FarmSize-CSTs\BB_2012-2018_sequences_farm-size.csv"
    # df = pd.read_csv(pth, dtype={"BNR":str, "CST":str})
    # df = df.loc[df["BNR"].str.slice(0, 2) == '12'].copy()
    # df = df.loc[df["CST"] != "255"].copy()
    # df["struct_seq_div"] = df["CST"].str.slice(0, 1)
    # df["funct_seq_div"] = df["CST"].str.slice(1, 2)
    # bins = [0, 10, 100, 200, 1000, 20000]
    #
    #
    # struct_div_dict = {
    #         "1": "A",
    #         "2": "B",
    #         "3": "C",
    #         "4": "D",
    #         "5": "E",
    #         "6": "F",
    #         "7": "G",
    #         "8": "H",
    #         "9": "I"
    #     }
    # df["struct_seq_div"] = df["struct_seq_div"].map(struct_div_dict)
    # crop_classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    # out_pth = r"C:\Users\IAMO\OneDrive - IAMO\2022_12 - Field vs farm sizes\figures\field_sizes_per_farm_classes_and_struc_seq_div.png"
    # plotting_lib.plot_farm_vs_field_sizes_from_iacs(df=df, farm_id_col='BNR', area_col='field size', crop_class_col='struct_seq_div',
    #                                    bins=bins, crop_classes=crop_classes, out_pth=out_pth, col_wrap=3, x_lim=(-5, 100))
    #
    # crop_classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    # out_pth = r"C:\Users\IAMO\OneDrive - IAMO\2022_12 - Field vs farm sizes\figures\field_sizes_per_farm_classes_and_funct_seq_div.png"
    # plotting_lib.plot_farm_vs_field_sizes_from_iacs(df=df, farm_id_col='BNR', area_col='field size',
    #                                    crop_class_col='funct_seq_div',
    #                                    bins=bins, crop_classes=crop_classes, out_pth=out_pth, col_wrap=3, x_lim=(-5, 100))

    df1 = gpd.read_file(r"data\vector\IACS\IACS_BB_2018.shp")
    df2 = gpd.read_file(r"data\vector\IACS\IACS_BV_2018.shp")
    df3 = gpd.read_file(r"data\vector\IACS\IACS_SA_2018.shp")
    df4 = gpd.read_file(r"data\vector\IACS\IACS_LS_2018.shp")

    df1["area"] = df1["geometry"].area
    df2["area"] = df2["geometry"].area
    df3["area"] = df3["geometry"].area
    df4["area"] = df4["geometry"].area

    farm_id_col1 = "BNR_ZD"
    farm_id_col2 = "bnrhash"
    farm_id_col3 = "btnr"
    farm_id_col4 = "REGISTRIER"

    area_col = "area"

    farms1 = df1.groupby(farm_id_col1).agg(
        farm_area=pd.NamedAgg(column=area_col, aggfunc="sum"),
        field_size=pd.NamedAgg(column=area_col, aggfunc="mean"),
        max_field_size=pd.NamedAgg(column=area_col, aggfunc="max"),
        min_field_size=pd.NamedAgg(column=area_col, aggfunc="min"),
        field_num=pd.NamedAgg(column="ID", aggfunc="count")).reset_index()
    farms1.sort_values(by="farm_area", inplace=True)

    farms2 = df2.groupby(farm_id_col2).agg(
        farm_area=pd.NamedAgg(column=area_col, aggfunc="sum"),
        field_size=pd.NamedAgg(column=area_col, aggfunc="mean"),
        max_field_size=pd.NamedAgg(column=area_col, aggfunc="max"),
        min_field_size=pd.NamedAgg(column=area_col, aggfunc="min"),
        field_num=pd.NamedAgg(column="ID", aggfunc="count")).reset_index()
    farms2.sort_values(by="farm_area", inplace=True)

    farms3 = df3.groupby(farm_id_col3).agg(
        farm_area=pd.NamedAgg(column=area_col, aggfunc="sum"),
        field_size=pd.NamedAgg(column=area_col, aggfunc="mean"),
        max_field_size=pd.NamedAgg(column=area_col, aggfunc="max"),
        min_field_size=pd.NamedAgg(column=area_col, aggfunc="min"),
        field_num=pd.NamedAgg(column="ID", aggfunc="count")).reset_index()
    farms3.sort_values(by="farm_area", inplace=True)

    farms4 = df4.groupby(farm_id_col4).agg(
        farm_area=pd.NamedAgg(column=area_col, aggfunc="sum"),
        field_size=pd.NamedAgg(column=area_col, aggfunc="mean"),
        max_field_size=pd.NamedAgg(column=area_col, aggfunc="max"),
        min_field_size=pd.NamedAgg(column=area_col, aggfunc="min"),
        field_num=pd.NamedAgg(column="ID", aggfunc="count")).reset_index()
    farms4.sort_values(by="farm_area", inplace=True)

    sample1 = df1[[farm_id_col1, "ID", "area"]].sample(n=2000, random_state=1)
    sample1 = pd.merge(sample1, farms1, how="left", on=farm_id_col1)
    sample1["state"] = "BB"
    sample1.rename(columns={farm_id_col1: "BTNR"}, inplace=True)

    sample2 = df2[[farm_id_col2, "ID", "area"]].sample(n=2000, random_state=1)
    sample2 = pd.merge(sample2, farms2, how="left", on=farm_id_col2)
    sample2["state"] = "BV"
    sample2.rename(columns={farm_id_col2: "BTNR"}, inplace=True)

    sample3 = df3[[farm_id_col3, "ID", "area"]].sample(n=2000, random_state=1)
    sample3 = pd.merge(sample3, farms3, how="left", on=farm_id_col3)
    sample3["state"] = "SA"
    sample3.rename(columns={farm_id_col3: "BTNR"}, inplace=True)

    sample4 = df4[[farm_id_col4, "ID", "area"]].sample(n=2000, random_state=1)
    sample4 = pd.merge(sample4, farms4, how="left", on=farm_id_col4)
    sample4["state"] = "LS"
    sample4.rename(columns={farm_id_col4: "BTNR"}, inplace=True)

    df = pd.concat([sample1, sample2, sample3, sample4], axis=0)

    bins = [0, 10, 100, 200, 1000, 20000]
    df["farm_area_class"] = pd.cut(df.farm_area, bins)
    df["farm_area_class"] = df["farm_area_class"].astype(str)

    out_pth = r"figures\field_vs_farm_size_by_state_xlogscaled.png"
    plotting_lib.scatterplot_two_columns(df=df, col1="area", col2="farm_area", out_pth=out_pth,
                            x_label=None, y_label=None, title=None, hue="state", log=True)

    out_pth = r"figures\field_vs_farm_size_by_state.png"
    plotting_lib.scatterplot_two_columns(df=df, col1="area", col2="farm_area", out_pth=out_pth,
                            x_label=None, y_label=None, title=None, hue="state", log=False)


    etime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + stime)
    print("end: " + etime)




if __name__ == '__main__':
    main()
