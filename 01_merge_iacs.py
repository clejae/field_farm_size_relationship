## Authors: 
## github Repo: https://github.com/clejae

## Calculates linear regressions to explain farm sizes with field sizes in hexagon grids with varying sizes from IACS data.

## ------------------------------------------ LOAD PACKAGES ---------------------------------------------------#
import os
import time
import geopandas as gpd
import pandas as pd

## ------------------------------------------ USER INPUT ------------------------------------------------------#
WD = r"Q:\FORLand\Field_farm_size_relationship"


## ------------------------------------------ DEFINE FUNCTIONS ------------------------------------------------#


def prepare_large_iacs_shp(input_dict, out_pth, grassland=True):
    print("Join all IACS files together.")

    shp_lst = []
    for key in input_dict:
        print(f"\tProcessing IACS data of {key}")
        iacs_pth = input_dict[key]["iacs_pth"]
        farm_id_col = input_dict[key]["farm_id_col"]
        crop_name_col = input_dict[key]["crop_name_col"]

        shp = gpd.read_file(iacs_pth)
        shp["field_id"] = shp.index
        shp["field_id"] = shp["field_id"].apply(lambda x: f"{key}_{x}")
        shp["state"] = key
        if grassland == False:
            if "ID_KTYP" in list(shp.columns):
                shp["ID_KTYP"] = shp["ID_KTYP"].astype(int)
                shp = shp.loc[shp["ID_KTYP"].isin([1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 60])].copy()
        shp = shp.loc[shp["geometry"] != None].copy()

        shp["field_size"] = shp["geometry"].area / 10000

        shp.rename(columns={farm_id_col: "farm_id"}, inplace=True)
        shp.rename(columns={crop_name_col: "crop_name"}, inplace=True)
        shp = shp[["field_id", "farm_id", "field_size", "crop_name", "ID_KTYP", "state", "geometry"]].copy()

        shp_lst.append(shp)

    print("\tConcatenating all shapefiles and writing out.")
    out_shp = pd.concat(shp_lst, axis=0)
    del shp_lst

    ## Calculate farm sizes
    farm_sizes = out_shp.groupby("farm_id").agg(
        farm_size=pd.NamedAgg(column="field_size", aggfunc="sum")
    ).reset_index()
    out_shp = pd.merge(out_shp, farm_sizes, on="farm_id", how="left")
    out_shp = out_shp[["field_id", "farm_id", "field_size", "farm_size", "crop_name", "ID_KTYP",  "state", "geometry"]].copy()
    # t = out_shp.loc[out_shp["geometry"] != None].copy()
    out_shp = out_shp.to_crs(3035)
    out_shp.to_file(out_pth, driver="GPKG")

    ## Temporary:
    if grassland == True:
        out_shp["state"] = out_shp["field_id"].apply(lambda x: x.split("_")[0])
        unique_crop_names = out_shp.loc[out_shp["ID_KTYP"].isin([30, 99])].copy()
        unique_crop_names = unique_crop_names[["state", "crop_name"]].drop_duplicates()
        unique_crop_names.to_csv(r"data\tables\unique_crop_names.csv")

    print("done!")
    return out_shp


def get_unqiue_crop_names_and_crop_classes_relations(iacs, crop_name_col, crop_class_col, out_pth):
    print("Get unique crop name - crop class relations for manual reclassification.")
    iacs.drop_duplicates(subset=[crop_name_col, crop_class_col], inplace=True)
    out_df = iacs[[crop_name_col, crop_class_col]]
    out_df.to_csv(out_pth, sep=";", index=False)


def new_crop_names_classification(iacs, crop_name_col, crop_class_col, classification_df, new_crop_class_col, out_pth,
                                  validity_check_pth=None):
    print("Assgin new crop classes.")
    iacs.drop(columns=[crop_class_col], inplace=True)

    classification = dict(zip(classification_df[crop_name_col], classification_df[new_crop_class_col]))
    iacs[crop_class_col] = iacs[crop_name_col].map(classification)

    ## Translate crop codes to crop class names
    t_dict = {
        1: "MA", #maize
        2: "WW", #winter wheat
        3: "SU", #sugar beet
        4: "OR", #oilseed rape
        5: "PO", #potato
        6: "SC", #summer cereals
        7: "TR", #triticale
        9: "WB", #winter barely
        10: "WR", #winter rye
        12: "LE", #legumes
        13: "AR", #arable grass
        14: "LE", #legumes
        20: "GR", #permanent grassland
        30: "UN", #unkown
        40: "PE", #40-mehrjährige Kulturen und Dauerkulturen
        60: "LE", #legumes
        70: "HE", #hedges, tree rows, shore stripes
        80: "UN", #unkown
        90: "GA", #garden flowers
        95: "OT", #mix
        99: "FA", #fallow
    }

    ## Reclassify
    iacs[crop_class_col] = iacs[crop_class_col].map(t_dict)
    iacs[crop_class_col].unique()

    ## Drop unkown and hedges
    iacs = iacs[~iacs[crop_class_col].isin(["UN", "HE"])].copy()

    ## Reclassify ID_KTYP
    t_dict = {
        "MA": "MA",  # maize --> maize
        "WW": "CE",  # winter wheat --> cereals
        "SU": "SU",  # sugar beet --> sugar beet
        "OR": "OR",  # oilseed rape --> oilseed rape
        "PO": "PO",  # potato --> potatoes
        "SC": "CE",  # summer cereals --> cereals
        "TR": "CE",  # triticale --> cereals
        "WB": "CE",  # winter barely --> cereals
        "WR": "CE",  # winter rye --> cereals
        "LE": "LE",  # legumes  --> legumes
        "AR": "GR",  # arable grass --> grass
        "GR": "GR",  # permanent grassland --> grass
        "FA": "OT",  # unkown --> others
        "PE": "OT",  # 40-mehrjährige Kulturen und Dauerkulturen --> others
        "UN": "OT",  # unkown --> others
        "GA": "OT",  # garden flowers --> others
        "MI": "OT",  # mix --> others
    }

    iacs[new_crop_class_col] = iacs[crop_class_col].map(t_dict)

    ## Drop fields < 100m²
    t = iacs.loc[iacs["field_size"] < 0.005]
    t2 = iacs.loc[iacs["field_size"] < 0.01]
    print("Fields <50 m²:", len(t), "Fields <100 m²:", len(t2))

    drop_stats = t2.groupby("state").agg(num_fields=pd.NamedAgg("field_id", "count")).reset_index()
    drop_stats.to_csv(out_pth[:-4] + "_drop_stats.csv", index=False)
    print(drop_stats)

    iacs = iacs.loc[iacs["field_size"] > 0.01]

    ## Recalculate farm sizes
    farm_sizes = iacs.groupby("farm_id").agg(
        farm_size=pd.NamedAgg(column="field_size", aggfunc="sum")
    ).reset_index()
    cols = ["field_id", "farm_id", "field_size", "crop_name", crop_class_col, new_crop_class_col,  "state", "geometry"] # without farm size
    iacs = pd.merge(iacs[cols], farm_sizes, on="farm_id", how="left")

    ## Reorder columns
    cols = ["field_id", "farm_id", "field_size", "farm_size", "crop_name", crop_class_col, new_crop_class_col, "state", "geometry"]
    iacs = iacs[cols]

    iacs.to_file(out_pth, driver="GPKG")

    if validity_check_pth:
        out_df = iacs[[crop_name_col, crop_class_col, new_crop_class_col]].copy()
        out_df.drop_duplicates(inplace=True)
        out_df.to_csv(validity_check_pth, sep=";", index=False)

    return iacs



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
            "farm_size_col": None,
            "crop_name_col": "K_ART_K"
        },
        "SA": {
            "iacs_pth": r"data\vector\IACS\IACS_SA_2018.shp",
            "farm_id_col": "btnr",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None,
            "crop_name_col": "NU_BEZ"
        },
        "TH": {
            "iacs_pth": r"data\vector\IACS\IACS_TH_2018.shp",
            "farm_id_col": "BDF_PI",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None,
            "crop_name_col": "BDF_KLA_"
        },
        "LS": {
            "iacs_pth": r"data\vector\IACS\IACS_LS_2018.shp",
            "farm_id_col": "REGISTRIER",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None,
            "crop_name_col": "K_ART_UNI"
        },
        "BV": {
            "iacs_pth": r"data\vector\IACS\IACS_BV_2018.shp",
            "farm_id_col": "bnrhash",
            "field_id_col": None,
            "field_size_col": None,
            "farm_size_col": None,
            "crop_name_col": "beschreibu"
        }
    }

    key = "ALL"

    ## Without grasslands
    # pth = fr"data\vector\IACS\IACS_{key}_2018_new.shp"
    # iacs = prepare_large_iacs_shp(
    #     input_dict=input_dict,
    #     out_pth=pth,
    #     grassland=False
    # )

    ## With grasslands
    pth = fr"data\vector\IACS\IACS_{key}_2018_all_areas.gpkg"
    iacs = prepare_large_iacs_shp(
        input_dict=input_dict,
        out_pth=pth,
        grassland=True
    )
    # iacs = gpd.read_file(pth)

    ## Reclassify grassland class
    # get_unqiue_crop_names_and_crop_classes_relations(
    #     iacs=iacs,
    #     crop_name_col="crop_name",
    #     crop_class_col="ID_KTYP",
    #     out_pth=rf"data\tables\unique_crop_name_class_relations.csv")

    classification_df = pd.read_excel(r"data\tables\crop_names_to_new_classes.xlsx")
    iacs = new_crop_names_classification(
        iacs=iacs,
        crop_name_col="crop_name",
        crop_class_col="ID_KTYP",
        classification_df=classification_df,
        new_crop_class_col="new_ID_KTYP",
        out_pth=fr"data\vector\IACS\IACS_{key}_2018_cleaned.gpkg",
        validity_check_pth=fr"data\tables\IACS_{key}_2018_cleaned_validitiy_check.csv")

    e_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    print("end: " + e_time)


if __name__ == '__main__':
    main()
