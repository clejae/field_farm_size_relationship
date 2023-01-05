## Authors: Clemens Jänicke
## github Repo: https://github.com/clejae

## Calculates linear regressions to explain farm sizes with field sizes in hexagon grids with varying sizes from IACS data.

## ------------------------------------------ LOAD PACKAGES ---------------------------------------------------#
import os
import time
import geopandas as gpd
import pandas as pd
import statsmodels.api as sm
import numpy as np

# project library for plotting
import plotting_lib
## ------------------------------------------ USER INPUT ------------------------------------------------------#
WD = r"Q:\FORLand\Field_farm_size_relationship"


## ------------------------------------------ DEFINE FUNCTIONS ------------------------------------------------#
def get_field_names(shp):
    """
    :param shp: Shapefile to get the field names from
    :return: List of all field names
    """
    from osgeo import ogr

    lyr = shp.GetLayer()
    lyr_def = lyr.GetLayerDefn()

    fname_lst = []
    for i in range(lyr_def.GetFieldCount()):
        fname = lyr_def.GetFieldDefn(i).GetName()
        fname_lst.append(fname)

    return fname_lst


def classify_crops(shp_pth, kart_fname, kartk_fname):
    from osgeo import ogr

    ## open reference table
    ## ToDo: Replace path. move file
    df_m = pd.read_excel(r"Daten\vector\InVekos\Tables\UniqueCropCodes_AllYrsAndBundeslaender.xlsx",
                         sheet_name='UniqueCodes')

    ## loop over shapefiles, add new column and fill it with code for the kulturtyp
    shp = ogr.Open(shp_pth, 1)
    lyr = shp.GetLayer()

    ## get list of field names
    fname_lst = get_field_names(shp)

    ## column name of Kulturtypen
    fname_ktyp = "ID_KTYP"
    fname_ws = "ID_WiSo"
    fname_cl = "ID_HaBl"

    ## check if this column name already exists
    ## if yes, then no new column will be created
    ## if not, then the column will be created and the field name list will be updated
    for col_name in [fname_ktyp, fname_ws, fname_cl]:
        if fname_ktyp in fname_lst:
            print("The field {0} exists already in the layer.".format(col_name))
        else:
            lyr.CreateField(ogr.FieldDefn(col_name, ogr.OFTInteger))
            fname_lst = get_field_names(shp)

    ## loop over features and set kulturtyp and WinterSummer-code depending on the k_art code,
    ## set CerealLeaf-Code depending on kulturtyp
    for f, feat in enumerate(lyr):
        fid = feat.GetField("ID")

        ## get kulturart code
        kart = feat.GetField(kart_fname)  ##
        if not kart:
            continue
        kart = int(kart)  # convert string to int

        ## get kulturart name
        ## Although all umlaute were replaced, there are some encoding issues.
        ## After every replaced Umlaut, there is still a character, that can't be decoded by utf-8
        ## By encoding it again and replacing it with '?', I can remove this character
        kart_k = feat.GetField(kartk_fname)
        # print(kart_k)
        if kart_k != None:
            # print(kart_k)

            kart_k = kart_k.encode('ISO-8859-1', 'replace')  # utf8| 'windows-1252' turns string to bytes representation
            # print(kart_k)
            kart_k = kart_k.replace(b'\xc2\x9d', b'')  # this byte representations got somehow into some strings
            kart_k = kart_k.replace(b'\xc2\x81', b'')  # this byte representations got somehow into some strings
            kart_k = kart_k.decode('ISO-8859-1', 'replace')  # turns bytes representation to string
            kart_k = kart_k.replace('?', '')
            kart_k = kart_k.replace('ä', 'ae')
            kart_k = kart_k.replace('ö', 'oe')
            kart_k = kart_k.replace('ü', 'ue')
            kart_k = kart_k.replace('ß', 'ss')
            kart_k = kart_k.replace('Ä', 'Ae')
            kart_k = kart_k.replace('Ö', 'Oe')
            kart_k = kart_k.replace('Ü', 'Ue')
        else:
            pass

        identifier = '{}_{}'.format(kart, kart_k)
        error_lst = []

        if identifier in df_m['K_ART_UNIQUE_noUmlaute']:
            ktyp = df_m['ID_KULTURTYP4_FL'].loc[df_m['K_ART_UNIQUE_noUmlaute'] == identifier]  # returns a pd Series
            ktyp = ktyp.iloc[0]  # extracts value from pd Series

            ws = df_m['ID_WinterSommer'].loc[df_m['K_ART_UNIQUE_noUmlaute'] == identifier]  # returns a pd Series
            ws = ws.iloc[0]  # extracts value from pd Series

            cl = df_m['ID_HalmfruchtBlattfrucht'].loc[
                df_m['K_ART_UNIQUE_noUmlaute'] == identifier]  # returns a pd Series
            cl = cl.iloc[0]

            length = len(fname_lst)

            feat.SetField(length - 3, int(ktyp))
            feat.SetField(length - 2, int(ws))
            feat.SetField(length - 1, int(cl))
            lyr.SetFeature(feat)
        else:
            error_lst.append(identifier)
    lyr.ResetReading()

def prepare_large_iacs_shp(input_dict, out_pth):
    print("Join all IACS files together.")

    shp_lst = []
    for key in input_dict:
        print(f"\tProcessing IACS data of {key}")
        iacs_pth = input_dict[key]["iacs_pth"]
        farm_id_col = input_dict[key]["farm_id_col"]

        shp = gpd.read_file(iacs_pth)
        shp["field_id"] = shp.index
        shp["field_id"] = shp["field_id"].apply(lambda x: f"{key}_{x}")
        if "ID_KTYP" in list(shp.columns):
            shp = shp.loc[shp["ID_KTYP"].isin([1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 60])].copy()

        shp["field_size"] = shp["geometry"].area / 10000

        farm_sizes = shp.groupby(farm_id_col).agg(
            farm_size=pd.NamedAgg(column="field_size", aggfunc="sum")
        ).reset_index()
        shp = pd.merge(shp, farm_sizes, on=farm_id_col, how="left")

        shp.rename(columns={farm_id_col: "farm_id"}, inplace=True)
        shp = shp[["field_id", "farm_id", "field_size", "farm_size", "geometry"]].copy()

        shp_lst.append(shp)

    print("\tConcatenating all shapefiles and writing out.")
    out_shp = pd.concat(shp_lst, axis=0)
    out_shp.to_file(out_pth)

    return out_shp


def farm_field_regression_in_hexagons(hex_shp, iacs, hexa_id_col, farm_id_col, field_id_col=None, field_size_col=None,
                                      farm_size_col=None, log_x=False, log_y=False):
    print("Farm vs. field size regression.")

    ## Calculate centroids of fields to assign each field to a hexagon based on the centroid location
    print("\tAssign fields to hexagons.")
    iacs["centroids"] = iacs["geometry"].centroid
    if field_id_col:
        iacs.rename(columns={field_id_col: "field_id"}, inplace=True)
    else:
        iacs["field_id"] = iacs.index + 1
    centroids = iacs[["field_id", "centroids"]].copy()
    centroids.rename(columns={"centroids": "geometry"}, inplace=True)
    ## Create a centroids layer and add the hex id to the centroids
    centroids = gpd.GeoDataFrame(centroids, crs=25832)
    centroids = gpd.sjoin(centroids, hex_shp, how="left")

    ## Add hexagon id to iacs data by joining the centroids values with the
    iacs = pd.merge(iacs, centroids[["field_id", hexa_id_col]], how="left", on="field_id")

    ## Calculate field and farm sizes if not already provided
    print("\tDetermine field and farm sizes.")
    if field_size_col:
        iacs.rename(columns={field_size_col: "field_size"}, inplace=True)
    else:
        iacs["field_size"] = iacs["geometry"].area / 10000

    if farm_size_col:
        iacs.rename(columns={farm_size_col: "farm_size"}, inplace=True)
    else:
        farm_sizes = iacs.groupby(farm_id_col).agg(
            farm_size=pd.NamedAgg(column="field_size", aggfunc="sum")
        ).reset_index()
        iacs = pd.merge(iacs, farm_sizes, on=farm_id_col, how="left")

    ## Calculate regression explaining farm sizes with field sizes
    print("\tCalculate the regression.")
    def calc_regression(group, log_x=log_x, log_y=log_y):
        group = pd.DataFrame(group)
        if log_x:
            group['field_size'] = np.log(group['field_size'])
        if log_y:
            group['farm_size'] = np.log(group['farm_size'])

        # group.to_csv(r"C:\Users\IAMO\OneDrive - IAMO\2022_12 - Field vs farm sizes\data\tables\test2.csv")

        x = np.array(group["field_size"])
        y = np.array(group["farm_size"]).T

        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        estimation = model.fit()

        if len(estimation.params) > 1:
            slope = estimation.params[1]
            intercept = estimation.params[0]
            rsquared = estimation.rsquared
        else:
            slope = 0
            intercept = 1
            rsquared = 0

        return (intercept, slope, len(x), rsquared)

    model_results = iacs.groupby(hexa_id_col).apply(calc_regression).reset_index()
    model_results.columns = [hexa_id_col, "model_results"]
    model_results["intercept"] = model_results["model_results"].apply(lambda x: x[0])
    model_results["slope"] = model_results["model_results"].apply(lambda x: x[1])
    model_results["num_fields"] = model_results["model_results"].apply(lambda x: x[2])
    model_results["rsquared"] = model_results["model_results"].apply(lambda x: x[3])
    model_results.drop(columns=["model_results"], inplace=True)

    mean_sizes = iacs.groupby(hexa_id_col).agg(
        avgfarm_s=pd.NamedAgg(column="farm_size", aggfunc="mean"),
        avgfield_s=pd.NamedAgg(column="field_size", aggfunc="mean"),
        num_farms=pd.NamedAgg(column=farm_id_col, aggfunc=list)
    ).reset_index()
    mean_sizes["num_farms"] = mean_sizes["num_farms"].apply(lambda x: len(set(x)))

    ## Add hexagon geometries to model results
    model_results = pd.merge(model_results, mean_sizes, how="left", on=hexa_id_col)
    model_results = pd.merge(model_results, hex_shp, how="left", on=hexa_id_col)
    model_results = gpd.GeoDataFrame(model_results)

    return model_results





## ------------------------------------------ RUN PROCESSES ---------------------------------------------------#
def main():
    s_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    os.chdir(WD)

    ## Only locally available
    input_dict = {
        "BB": {
            "iacs_pth": r"data\vector\IACS\IACS_BB_2018.shp",
            "farm_id_col": "BNR_ZD",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None
        },
        "SA": {
            "iacs_pth": r"data\vector\IACS\IACS_SA_2018.shp",
            "farm_id_col": "btnr",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None
        },
        "TH": {
            ## ToDo: The crops in Thuringia are not yet classified into crop classes. This should be done for further analysis.
            "iacs_pth": r"data\vector\IACS\IACS_TH_2019_red.shp",
            "farm_id_col": "BDF_PI",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None
        },
        "LS": {
            "iacs_pth": r"data\vector\IACS\IACS_LS_2018.shp",
            "farm_id_col": "REGISTRIER",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None
        },
        "BV": {
            "iacs_pth": r"data\vector\IACS\IACS_BV_2018.shp",
            "farm_id_col": "bnrhash",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None
        }
    }

    key = "ALL"
    pth = fr"data\vector\IACS\IACS_{key}_2018.shp"
    # iacs = prepare_large_iacs_shp(
    #     input_dict=input_dict,
    #     out_pth=pth
    # )
    iacs = gpd.read_file(pth)

    for hexasize in [30, 15, 5, 1]:
        hexagon_pth = fr"data\vector\grid\hexagon_grid_germany_{hexasize}km.shp"
        hex_shp = gpd.read_file(hexagon_pth)
        hex_shp.rename(columns={"id": "hexa_id"}, inplace=True)
        model_results_shp = farm_field_regression_in_hexagons(
            hex_shp=hex_shp,
            iacs=iacs,
            hexa_id_col="hexa_id",
            farm_id_col="farm_id",
            field_id_col="field_id",
            field_size_col="field_size",
            farm_size_col="farm_size",
            log_x=True,
            log_y=True
        )
        pth = fr"data\vector\grid\hexagon_grid_{key}_{hexasize}km_with_values.shp"
        model_results_shp.to_file(pth)

    for hexasize in [1, 5, 15, 30]:
        pth = fr"data\vector\grid\hexagon_grid_{key}_{hexasize}km_with_values.shp"
        model_results_shp = gpd.read_file(pth)
        model_results_shp.replace([np.inf, -np.inf], np.nan, inplace=True)
        model_results_shp.dropna(subset=["rsquared"], how="all", inplace=True)
        model_results_shp = model_results_shp[model_results_shp["num_farms"] > 1].copy()
        if hexasize == 1:
            dpi = 600
            figsize = (40, 24)
        else:
            dpi = 300
            figsize = (20, 12)

        plotting_lib.plot_maps_in_grid(
            shp=model_results_shp,
            out_pth=fr"figures\maps\{key}_model_results_logscaled_{hexasize}km.png",
            cols=["intercept", "slope", "rsquared", "num_farms", "num_fields", "avgfield_s", "avgfarm_s"],
            nrow=2,
            ncol=4,
            figsize=figsize,
            dpi=dpi,
            shp2_pth=fr"data\vector\administrative\GER_bundeslaender.shp",
            titles=["intercept", "slope", "rsquared", "num farms", "num fields", "mean field size [ha]",
                    "mean farm size [ha]"],
            highlight_extremes=False
        )

    pth1 = fr"data\vector\grid\hexagon_grid_{key}_1km_with_values.shp"
    pth5 = fr"data\vector\grid\hexagon_grid_{key}_5km_with_values.shp"
    pth15 = fr"data\vector\grid\hexagon_grid_{key}_15km_with_values.shp"
    pth30 = fr"data\vector\grid\hexagon_grid_{key}_30km_with_values.shp"
    model_results_shp1 = gpd.read_file(pth1)
    model_results_shp5 = gpd.read_file(pth5)
    model_results_shp15 = gpd.read_file(pth15)
    model_results_shp30 = gpd.read_file(pth30)

    model_results_shp1["df"] = "1"
    model_results_shp5["df"] = "5"
    model_results_shp15["df"] = "15"
    model_results_shp30["df"] = "30"

    df = pd.concat([model_results_shp1, model_results_shp5, model_results_shp15, model_results_shp30], axis=0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["rsquared"], how="all", inplace=True)
    df = df[df["num_farms"] > 1].copy()
    df["num_farms_bins"] = pd.cut(df["num_farms"], [0, 10, 25, 100, 250, 1000, 2500])

    # fig, ax = plt.subplots(1, 1, figsize=cm2inch(15, 10))
    # sns.jointplot(data=df,  x="num_farms", y="rsquared", kind="kde")
    # fig.tight_layout()
    # plt.savefig(fr"figures\num_farms_vs_rsquared.png", dpi=300)
    # plt.close()

    # plotting_lib.scatterplot_two_columns(
    #     df=df,
    #     out_pth=fr"figures\num_farms_vs_rsquared.png",
    #     col1="num_farms",
    #     col2="rsquared",
    #     hue="df"
    # )

    plotting_lib.boxplots_in_grid(
        df=df,
        out_pth=fr"figures\boxplots_hexasize_rsquared2.png",
        value_cols=["intercept", "slope", "rsquared", "num_farms", "num_fields", "avgfield_s", "avgfarm_s"],
        category_col="df",
        nrow=7,
        ncol=1,
        figsize=(7, 21),
        x_labels=("hexagon_size [km]", "hexagon_size [km]", "hexagon_size [km]", "hexagon_size [km]", "hexagon_size [km]", "hexagon_size [km]", "hexagon_size [km]"),
        dpi=300)

    plotting_lib.boxplots_in_grid(
        df=df,
        out_pth=fr"figures\boxplots_num_farms_bins_rsquared2.png",
        value_cols=["intercept", "slope", "rsquared", "num_farms", "num_fields", "avgfield_s", "avgfarm_s"],
        category_col="num_farms_bins",
        nrow=7,
        ncol=1,
        figsize=(7, 21),
        x_labels=("Number farms bin", "Number farms bin", "Number farms bin", "Number farms bin", "Number farms bin", "Number farms bin", "Number farms bin"),
        dpi=300)


    # for var in ["slope", "intercept", "num_fields", "rsquared", "mean_field_size", "mean_farm_size"]:
    #     plotting_lib.plot_map(
    #         shp=model_results_shp,
    #         out_pth=fr"figures\maps\{key}_{var}.png",
    #         col=var,
    #         shp2_pth=fr"data\vector\administrative\GER_bundeslaender.shp",
    #         highlight_extremes=False
    #     )


    # for key in input_dict:
    #     print(f"Current federeal state: {key}")
    #
    #     hexagon_pth = fr"data\vector\grid\hexagon_grid_germany_15km.shp"
    #     hex_shp = gpd.read_file(hexagon_pth)
    #     hex_shp.rename(columns={"id": "hexa_id"}, inplace=True)
    #
    #     # iacs_pth = r"vector\IACS\IACS_BB_2018.shp"
    #     iacs_pth = input_dict[key]["iacs_pth"]
    #     farm_id_col = input_dict[key]["farm_id_col"]
    #     field_size_col = input_dict[key]["field_size_col"]
    #     farm_size_col = input_dict[key]["farm_size_col"]
    #     field_id_col = input_dict[key]["field_id_col"]
    #
    #     iacs = gpd.read_file(iacs_pth)
    #
    #     model_results_shp = farm_field_regression_in_hexagons(
    #         hex_shp=hex_shp,
    #         iacs=iacs,
    #         hexa_id_col="hexa_id",
    #         farm_id_col=farm_id_col,
    #         field_id_col=field_id_col,
    #         field_size_col=field_size_col,
    #         farm_size_col=farm_size_col
    #     )
    #     pth = fr"data\vector\grid\hexagon_grid_{key}_15km_with_values.shp"
        # model_results_shp.to_file(pth)
    #
    #     model_results_shp = gpd.read_file(pth)
    #     plotting_lib.plot_map(
    #         shp=model_results_shp,
    #         out_pth=fr"figures\maps\{key}_slope.png",
    #         col="slope",
    #         shp2_pth=fr"data\vector\administrative\{key}_bundesland.shp",
    #         highlight_extremes=False
    #     )
    #
    #     plotting_lib.plot_map(
    #         shp=model_results_shp,
    #         out_pth=fr"figures\maps\{key}_intercept.png",
    #         col="intercept",
    #         shp2_pth=fr"data\vector\administrative\{key}_bundesland.shp",
    #         highlight_extremes=False
    #     )
    #
    #     plotting_lib.plot_map(
    #         shp=model_results_shp,
    #         out_pth=fr"figures\maps\{key}_num_fields.png",
    #         col="num_fields",
    #         shp2_pth=fr"data\vector\administrative\{key}_bundesland.shp",
    #         highlight_extremes=False
    #     )

    e_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    print("end: " + e_time)


if __name__ == '__main__':
    main()
