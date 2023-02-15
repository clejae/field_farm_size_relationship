# field_farm_size_relationships
Exploration and analysis of the field vs farm size relationship across Germany

An empirical assessment of the field size ~ farm size relationship (FFSR)

## Objectives
- 1.) Map linear relationship on hexagon level (farm size ~ intercept + slope*field size) [x]
- 2.) Model farm size based on field size and set of other predictors [ ]

## Data:
- crop rotations (IACS)
- organic agriculture (IACS)
- livestock at farm (IACS)
- agricultural suitabitity (Max?)
- fragmentation indices (?)
- Landscape diversity (?)
- Terrain (DEM)

## Methods (and ToDo):
- 1.)
  - [x] bring all IACS data into one shapefile
  - [x] create hexagon grids with different resolutions
  - [x] calculate farm sizes, add information to each field
  - [x] assign fields to hexagons based on location of centroid of field
  - [x] calculate regression per hexagon

- 2.)
  - [x] Create classification table for TH ("Kulturart" --> crop classes)
  - [x] Assing crop classes to "Kulturarten" in TH
  - [x] Determine Crop Sequence Types for TH
  - [x] Aggregate predictors per field
  - [ ] Add share of UAA per heaxgon to each field 
  - [ ] Add Ruggedness per hexagon to each field
  
- 3.)
  - [ ] Bayesian modelling of relationship
