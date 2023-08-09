## Authors: Clemens Jänicke
## github Repo: https://github.com/clejae

## Calculates linear regressions to explain farm sizes with field sizes in hexagon grids with varying sizes from IACS data.

## ------------------------------------------ LOAD PACKAGES ---------------------------------------------------#
import time
import os
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression as lr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

from prop_match_functions import *
import numpy as np

# project library for plotting
import plotting_lib
## ------------------------------------------ USER INPUT ------------------------------------------------------#
WD = r"Q:\FORLand\Field_farm_size_relationship"


## ------------------------------------------ DEFINE FUNCTIONS ------------------------------------------------#
def compare_distributions_of_variables(df, hue_col, out_pth, hue_dict=None):

    print("Plot densities of covariates.")

    ## Do this to not change original df
    df_plt = df.copy()

    if hue_dict:
        df_plt[hue_col] = df_plt[hue_col].map(hue_dict)

    # csts = df_plt.groupby(["ft", hue_col]).agg(
    #     field_count=pd.NamedAgg("field_id", "count")
    # ).reset_index()
    # csts["share"] = csts['field_count'] / csts.groupby(hue_col)['field_count'].transform('sum') * 100

    cts = df_plt.groupby(["ID_KTYP", hue_col]).agg(
        field_count=pd.NamedAgg("field_id", "count")
    ).reset_index()
    cts["share"] = cts['field_count'] / cts.groupby(hue_col)['field_count'].transform('sum') * 100

    st = df_plt.groupby(["federal_st", hue_col]).agg(
        field_count=pd.NamedAgg("field_id", "count")
    ).reset_index()
    st["share"] = st['field_count'] / st.groupby(hue_col)['field_count'].transform('sum') * 100

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=plotting_lib.cm2inch(30, 30))
    sns.kdeplot(data=df_plt, x="farm_size", hue=hue_col, ax=axs[0, 0])
    axs[0, 0].set(xscale="log")
    axs[0, 0].set_xlabel("log(farm size)")
    sns.kdeplot(data=df_plt, x="fieldSizeM", hue=hue_col, ax=axs[0, 1])
    axs[0, 1].set(xscale="log")
    axs[0, 1].set_xlabel("log(field size)")
    sns.kdeplot(data=df_plt, x="propAg1000", hue=hue_col, ax=axs[0, 2])
    axs[0, 2].set_xlabel("proportion Agric.")
    sns.kdeplot(data=df_plt, x="ElevationA", hue=hue_col, ax=axs[1, 0])
    axs[1, 0].set_xlabel("Elevation of field")
    sns.kdeplot(data=df_plt, x="avgTRI1000", hue=hue_col, ax=axs[1, 1])
    axs[1, 1].set_xlabel("Avg. TRI 1000m")
    sns.kdeplot(data=df_plt, x="avgTRI0", hue=hue_col, ax=axs[1, 2])
    axs[1, 2].set_xlabel("Avg. TRI field")
    sns.kdeplot(data=df_plt, x="fieldCount", hue=hue_col, ax=axs[2, 0])
    axs[2, 0].set_xlabel("Field count of farm")
    sns.kdeplot(data=df_plt, x="fieldCount", hue=hue_col, ax=axs[2, 1])
    axs[2, 1].set_xlabel("Field count")
    sns.kdeplot(data=df_plt, x="surrf_mean", hue=hue_col, ax=axs[2, 1])
    axs[2, 1].set_xlabel("surr. fields avg. size")
    sns.kdeplot(data=df_plt, x="surrf_no_fields", hue=hue_col, ax=axs[2, 2])
    axs[2, 2].set_xlabel("number surr. fields")
    sns.barplot(data=cts, x="ID_KTYP", y="share", hue=hue_col, ax=axs[3, 0])
    axs[3, 0].set_xlabel("Crop class")
    sns.barplot(data=st, x="federal_st", y="share", hue=hue_col, ax=axs[3, 1])
    axs[3, 1].set_xlabel("Federal state")
    fig.tight_layout()
    plt.savefig(out_pth)
    print("Plotting distribution comparison done!")
    plt.close()


def draw_sample_with_matched_distributions(iacs_pth, out_pth):
    """
    ## 1. Draw a random sample of all data including NAs
    ## 2. Remove NAs from complete dataset --> reduced dataset
    ## 3. Match reduced dataset with random sample to get a sample with similar distributions in all variables

    :param iacs_pth:
    :param out_pth:
    :return:
    """

    print("Read IACS data.")
    iacs = gpd.read_file(iacs_pth)
    surr = pd.read_csv(r"Q:\FORLand\Field_farm_size_relationship\data\tables\surrounding_fields_stats_ALL.csv")
    for col in surr.columns:
        surr.loc[surr[col].isna(), col] = 0
    print(len(surr), len(surr.loc[surr.isnull().any(axis=1)]))
    iacs = pd.merge(iacs, surr, "left", "field_id")
    for col in surr.columns:
        iacs.loc[iacs[col].isna(), col] = 0
    # iacs["ft"] = iacs["CstMaj"].apply(lambda x: str(int(x))[1] if x != 0 else 0)
    # iacs["state"] = iacs["field_id"].apply(lambda x: x[:2])

    sqr_col = "SQRAvrg"
    pred_col = "farm_size"
    ## temp:
    iacs.rename(columns={"fieldSizeM2": "fieldSizeM", "federal_state": "federal_st"}, inplace=True)

    iacs["na"] = 0
    iacs.loc[iacs.isnull().any(axis=1), "na"] = 1
    # compare_distributions_of_variables(
    #     df=iacs,
    #     hue_col="na",
    #     hue_dict={0: "rows wo NAN", 1: "rows w NAN"},
    #     out_pth=r"Q:\FORLand\Field_farm_size_relationship\data\vector\final\matching\01_comparison_sqr_existent_vs_nan.png"
    # )

    ## 1. Draw a random sample of all data including NAs
    n = int(len(iacs) * 0.01)
    iacs_rsample = iacs.sample(n=n, random_state=1)
    # iacs_rsample.to_file(r"Q:\FORLand\Field_farm_size_relationship\data\vector\final\all_predictors_sqr_sub.shp")
    iacs_rsample["treatment"] = 1

    ## Remove rows where
    print("Length of random sample including NAs:", len(iacs_rsample))
    print("No. of NAs in random sample:", len(iacs_rsample.loc[iacs_rsample[sqr_col].isna()]))
    print("No. complete observations in random sample:", len(iacs_rsample.loc[iacs_rsample[sqr_col].notna()]))

    ## 2. Remove NAs from complete dataset --> reduced dataset
    iacs_red = iacs.loc[~iacs.isnull().any(axis=1)].copy()
    # iacs_red = iacs_red.loc[iacs_red["ft"] != 0].copy()
    iacs_red["treatment"] = 0

    # for col in iacs_red.columns:
    #     t = iacs_red.loc[iacs_red[col].isna()].copy()
    #     print(col, len(t))
    #
    # for col in iacs_rsample.columns:
    #     t = iacs_rsample.loc[iacs_rsample[col].isna()].copy()
    #     print(col, len(t))

    ## Combine reduced df and sample into 1 df for further processing
    ## treatment == 0 --> comes from reduced dataset
    ## treatment == 1 --> comes from sample
    df = pd.concat([iacs_rsample, iacs_red])
    print("Length of combined df:", len(df))
    # compare_distributions_of_variables(
    #     df=df,
    #     hue_col="treatment",
    #     hue_dict={0: "reduced dataset", 1: "random sample"},
    #     out_pth=r"Q:\FORLand\Field_farm_size_relationship\data\vector\final\matching\02_comparison_random_sample_with_reduced_df.png"
    # )

    ## 3. Match reduced dataset with random sample to get a sample with similar distributions in all variables
    txt_pth = rf"Q:\FORLand\Field_farm_size_relationship\data\vector\final\matching\matching.txt"
    with open(txt_pth, 'w') as f:
        f.write(f"##### Matching #####\n")
    f.close()

    df.index = range(len(df))
    y = df[[pred_col]]

    ## Exclude unecessary columns
    match_cols = ['treatment', 'ID_KTYP', 'fieldSizeM', 'fieldCount', 'propAg1000', 'avgTRI1000', 'avgTRI0',
                 'surrf_mean', 'surrf_no_fields', 'ElevationA', 'federal_st']
    df_data = df[match_cols].copy()

    ## Separate treatment from other variables
    T = df_data.treatment
    X = df_data.loc[:, df_data.columns != 'treatment']

    ## Convert categorical variables into dummy variables
    X_encoded = pd.get_dummies(
        data=X,
        columns=["federal_st", "ID_KTYP"],
        prefix={"federal_st": "state", "ID_KTYP": "crop"}, drop_first=False)

    ## Design pipeline to build the treatment estimator. It standardizes the data and applies a logistic classifier
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic_classifier', lr())
    ])

    for col in X_encoded.columns:
        t = X_encoded.loc[X_encoded[col].isna()].copy()
        print(col, len(t))

    ## Fit the classifier to the data
    pipe.fit(X_encoded, T)
    predictions = pipe.predict_proba(X_encoded)
    predictions_binary = pipe.predict(X_encoded)

    ## Get some accuracy measurements
    with open(txt_pth, 'a') as f:
        f.write('\nAccuracy: {:.4f}\n'.format(metrics.accuracy_score(T, predictions_binary)))
        f.write('Confusion matrix:\n{}\n'.format(metrics.confusion_matrix(T, predictions_binary)))
        f.write('F1 score is: {:.4f}'.format(metrics.f1_score(T, predictions_binary)))
    f.close()
    predictions_logit = np.array([logit(xi) for xi in predictions[:, 1]])

    df_data.loc[:, 'propensity_score'] = predictions[:, 1]
    df_data.loc[:, 'propensity_score_logit'] = predictions_logit
    df_data.loc[:, 'outcome'] = y[pred_col]

    X_encoded.loc[:, 'propensity_score'] = predictions[:, 1]
    X_encoded.loc[:, 'propensity_score_logit'] = predictions_logit
    X_encoded.loc[:, 'outcome'] = y[pred_col]
    X_encoded.loc[:, 'treatment'] = df_data.treatment

    ## Use Nearest Neighbors to identify matching candidates.
    ## Then perform 1-to-1 matching by isolating/identifying groups of (T=1,T=0).
    caliper = np.std(df_data.propensity_score) * 0.3

    with open(txt_pth, 'a') as f:
        f.write('\nCaliper (radius) is: {:.4f}\n'.format(caliper))
    f.close()

    df_data = X_encoded

    ## caliper reduces the space from which the neighbors are searched
    ## p defines how the distance is calculated. P=2 --> euclidean distance
    knn = NearestNeighbors(n_neighbors=10, p=2, radius=caliper)
    knn.fit(df_data[['propensity_score_logit']].to_numpy())

    distances, indexes = knn.kneighbors(
        df_data[['propensity_score_logit']].to_numpy(), n_neighbors=10)

    with open(txt_pth, 'a') as f:
        f.write('\nFor item 0, the 4 closest distances are (first item is self):')
        for ds in distances[0, 0:4]:
            f.write('\nElement distance: {:4f}'.format(ds))
        f.write('\n...')
        f.write('\nFor item 0, the 4 closest indexes are (first item is self):')
        for idx in indexes[0, 0:4]:
            f.write('\nElement index: {}'.format(idx))
        f.write('\n...')
    f.close()

    def perfom_matching_v2(row, indexes, df_data):
        current_index = int(row['index'])  # Obtain value from index-named column, not the actual DF index.
        prop_score_logit = row['propensity_score_logit']
        for idx in indexes[current_index, :]:
            if (current_index != idx) and (row.treatment == 1) and (df_data.loc[idx].treatment == 0):
                return int(idx)

    df_data['matched_element'] = df_data.reset_index().apply(perfom_matching_v2, axis=1, args=(indexes, df_data))
    treated_with_match = ~df_data.matched_element.isna()
    treated_matched_data = df_data[treated_with_match][df_data.columns]
    print("No. of treated data points with a match:", len(treated_with_match[treated_with_match == True]))

    ## for all untreated matched observations, retrieve the co-variates
    def obtain_match_details(row, all_data, attribute):
        return all_data.loc[row.matched_element][attribute]

    untreated_matched_data = pd.DataFrame(data=treated_matched_data.matched_element)

    ## Combine untreated matched data with IACS data to retrieve all information
    ## Do combination based on several attributes that uniquely describe each field as the unique ID was deleted earlier
    attributes = ["fieldSizeM", "fieldCount", "propAg1000"]
    for attr in attributes:
        untreated_matched_data[attr] = untreated_matched_data.apply(obtain_match_details, axis=1, all_data=df_data,
                                                                    attribute=attr)
    untreated_matched_data = untreated_matched_data.set_index('matched_element')

    matched_sample = pd.merge(
        untreated_matched_data[attributes],
        iacs_red[attributes + ["field_id"]],
        on=attributes,
        how="left"
    )

    iacs_out = iacs_red.loc[iacs_red["field_id"].isin(matched_sample["field_id"].tolist())].copy()
    iacs_out.to_file(out_pth)

    print("No. of matched samples", len(matched_sample))

    ## compare matched sample with random sample
    # iacs["treatment"] = 2
    # iacs = iacs[iacs_out.columns]
    df_plt = pd.concat([iacs_rsample, iacs_out])
    compare_distributions_of_variables(
        df=df_plt,
        hue_col="treatment",
        hue_dict={1: "random sample", 0: "matched sample", 2: "full dataset"},
        out_pth=r"Q:\FORLand\Field_farm_size_relationship\data\vector\final\matching\03_comparison_random_sample_with_matched_sample.png"
    )

    ## Add information on matched sample to final dataset
    iacs["matched_sample"] = 0
    iacs.loc[iacs["field_id"].isin(matched_sample["field_id"].tolist()), "matched_sample"] = 1

    ## Temp:
    # print(len(iacs.loc[(iacs["matched_sample"] == 1) & (iacs["na"] == 1)]))
    # print(len(iacs.loc[(iacs["matched_sample"] == 1) & (iacs["na"] == 0)]))

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

    iacs["new_IDKTYP"] = iacs["ID_KTYP"].map(t_dict)
    iacs.drop(columns=["na"], inplace=True)

    ## Reorder columns
    iacs = iacs[['field_id', 'farm_id', 'field_size', 'fieldSizeM', 'farm_size', 'fieldCount', 'crop_name',
       'ID_KTYP', 'new_IDKTYP', 'federal_st', 'FormerDDRm',
       'hexa_id', 'sumAgr1000', 'propAg1000', 'ElevationA', 'sdElev0',
       'avgTRI0', 'avgTRI500',  'avgTRI1000', 'SQRAvrg',
       'surrf_mean', 'surrf_median', 'surrf_std', 'surrf_min', 'surrf_max',
       'surrf_no_fields', 'matched_sample', 'geometry']]

    iacs.to_file(r"Q:\FORLand\Field_farm_size_relationship\data\test_run2\all_predictors_w_grassland.shp")

    ## Create smaller version of shapefile
    iacs = iacs[['field_id', 'farm_id', 'fieldSizeM', 'farm_size', 'fieldCount', 'ID_KTYP', 'new_IDKTYP', 'federal_st',
                 'FormerDDRm', 'propAg1000', 'ElevationA', 'sdElev0',  'avgTRI0', 'avgTRI500', 'avgTRI1000', 'SQRAvrg',
                 'surrf_mean', 'surrf_median', 'surrf_std', 'surrf_min', 'surrf_max', 'surrf_no_fields',
                 'matched_sample', 'geometry']]
    iacs.to_file(r"Q:\FORLand\Field_farm_size_relationship\data\test_run2\all_predictors_w_grassland_red.shp")

## ------------------------------------------ RUN PROCESSES ---------------------------------------------------#
def main():
    s_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    os.chdir(WD)

    draw_sample_with_matched_distributions(
        iacs_pth=r"Q:\FORLand\Field_farm_size_relationship\data\test_run2\all_predictors_w_grassland\all_predictors_w_grassland.shp",
        out_pth=r"Q:\FORLand\Field_farm_size_relationship\data\test_run2\matched_sample_w_grassland.shp"
    )


    e_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + s_time)
    print("end: " + e_time)


if __name__ == '__main__':
    main()
