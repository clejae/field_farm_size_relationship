import geopandas as gpd
import pandas as pd

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