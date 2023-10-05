## Authors:
## github Repo: https://github.com/clejae

## Calculates linear regressions to explain farm sizes with field sizes in hexagon grids with varying sizes from IACS data.

## ------------------------------------------ LOAD PACKAGES ---------------------------------------------------#
import time
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

## ------------------------------------------ USER INPUT ------------------------------------------------------#
WD = r"Q:\FORLand\Field_farm_size_relationship"


def log_and_scale_data(input_pth, output_pth):
    print("Read Input data.")
    iacs = pd.read_csv(input_pth, dtype={"farm_id": str})

    print("Prepare data.")

    ## Subtract field from farm size
    iacs["farm_size_r"] = iacs["farm_size"] - iacs["field_size"]

    ## Due to some different calculations of the field sizes in prior steps (R-pyhton), sometimes the field sizes are
    ## only slightly smaller than the farm sizes, causing trouble in later steps
    iacs.loc[iacs["fieldCount"] == 1, "farm_size_r"] = 0
    iacs.loc[iacs["fieldCount"] == 1, "farm_size"] = iacs.loc[iacs["fieldCount"] == 1, "field_size"]

    ## Calculate values to m²
    cols = ["field_size", "surrf_mean", "surrf_medi", "surrf_std", "surrf_min", "surrf_max"]
    # cols = [
    #     ["field_size", "surrf_mean_100", "surrf_median_100", "surrf_std_100", "surrf_min_100", "surrf_max_100",
    #      "surrf_no_fields_100",
    #      "surrf_mean_500", "surrf_median_500", "surrf_std_500", "surrf_min_500", "surrf_max_500", "surrf_no_fields_500",
    #      "surrf_mean_1000", "surrf_median_1000", "surrf_std_1000", "surrf_min_1000", "surrf_max_1000",
    #      "surrf_no_fields_1000"]]
    for col in cols:
        iacs[col] = iacs[col] * 10000

    ## Remove all fields smaller 1m²
    iacs = iacs.loc[iacs["field_size"] >= 2].copy()

    ## Drop zeros in the surrounding field statistics
    cols = ["surrf_mean", "surrf_medi", "surrf_std", "surrf_min", "surrf_max"]
    # cols = ["surrf_mean_100", "surrf_median_100", "surrf_std_100", "surrf_min_100", "surrf_max_100",
    #         "surrf_mean_500", "surrf_median_500", "surrf_std_500", "surrf_min_500", "surrf_max_500",
    #         "surrf_mean_1000", "surrf_median_1000", "surrf_std_1000", "surrf_min_1000", "surrf_max_1000"]
    for col in cols:
        iacs = iacs.loc[iacs[col] > 0].copy()

    ## Calculate natural logarithm
    # cols = ["field_size", "farm_size", "farm_size_r", "surrf_mean_100", "surrf_median_100", "surrf_std_100",
    #         "surrf_min_100", "surrf_max_100",
    #         "surrf_mean_500", "surrf_median_500", "surrf_std_500", "surrf_min_500", "surrf_max_500",
    #         "surrf_mean_1000", "surrf_median_1000", "surrf_std_1000", "surrf_min_1000", "surrf_max_1000"]
    cols = ["field_size", "farm_size", "farm_size_r", "surrf_mean", "surrf_medi", "surrf_std", "surrf_min", "surrf_max"]
    for col in cols:
        iacs[f"log_{col}"] = np.log(iacs[col])
    log_cols = [f"log_{col}" for col in cols]

    ## Create interaction terms
    iacs["inter_cr_st"] = iacs["new_IDKTYP"] + iacs["federal_st"]
    iacs["inter_sfm_prag"] = iacs["log_surrf_mean"] + iacs["propAg1000"]

    ## Scale variables
    scale_cols = log_cols + ["inter_sfm_prag", "propAg1000", "ElevationA", "sdElev0", "avgTRI0", "avgTRI500",
                  "avgTRI1000", "SQRAvrg"]
    scale_cols.remove("log_farm_size")
    scale_cols.remove("log_farm_size_r")
    scale = StandardScaler()
    for col in scale_cols:
        iacs[col] = scale.fit_transform(iacs[[col]])

    ## Drop farms that have likely most of their areas outside our study area
    iacs["fstate_id"] = iacs["farm_id"].apply(lambda x: x[:2])
    drop_fstate_ids = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "13", "14"]
    iacs = iacs.loc[~(iacs["fstate_id"].isin(drop_fstate_ids) & iacs["federal_st"].isin(["BB", "SA", "TH"]))].copy()
    print("Remaining observations:", len(iacs))

    print("Write out")
    iacs.to_csv(output_pth, index=False)


## ------------------------------------------ RUN PROCESSES ---------------------------------------------------#
def main():
    s_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    os.chdir(WD)

    log_and_scale_data(
        input_pth=rf'data\test_run2\all_predictors_w_grassland_.csv',
        output_pth=rf'models\all_predictors_w_grassland_w_sqr_log_and_scaled.csv'
    )

    e_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    print("end: " + e_time)


if __name__ == '__main__':
    main()
