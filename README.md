# README

This repository contains solver for the Vehicle Routing Problem.


## Dependencies

To avoid problems with dependencies I recommend using a python virtual environment. It can be created with the following command:

`python -m venv vrp_venv`

Then you can activate it with: `source vrp_venv/bin/activate` and deactivate with `deactivate`.

After creating a virtual environment install all dependencies: `pip install -r requirements.txt`.

## Running D-Wave solver

To use D-Wave computer, you need to have a `sapi_token`. You can obtain it here: https://cloud.dwavesys.com/qubist/apikey/ and then store it in `src/dwave_credentials.txt`.

## Using Google Maps service

Obtain Google Maps API key (`https://cloud.google.com/maps-platform/`) and save it in `src/solver/gmaps_credentials.txt`.

## Warning

Some sensitive parts of code and data have been removed before making this repository public. Therefore, even though we've tried to make sure this version works, some parts of code might not work.

## Input/output

### Data format

Data format will be csv files.
Since some of the values in these files might include commas (e.g. address), the delimiter we use will be a semicolon (";").

### Input
Solver per single run will use three input csv files:

1. List of outposts
2. List of vehicles (which can be replaced by vehicles list generated on the fly)
3. Graph representing costs
4. Constraints (optional)

#### List of outposts

File will have the following columns:
- outpost_id - the outpost with id=0 is by default a starting point
- outpost_name (optional) - human readable name of the outpost
- address (optional) - address of the outpost might be useful for scraping the data.
- load - how many packages must be taken from this outpost. The units are tons. 
- latitude
- longitude

`outpost_name` and address are optional fields.  
`load` field is approximated for each week day and time shift (day - 5AM-5PM, night - 5PM-5AM) based on original routes stored in XLS file.

##### Creating lists of outposts

Use `outposts.py` script: `./src/solver/outposts.py raw_data/2.xls raw_data/outposts-geo.kml raw_data/outposts-helper.csv --savedir data/outposts_full_wgs84`  

Script will create single `outposts_??.csv` file for each day and time shift in directory given as `--savedir` flag argument.  

This script generates also original routes descriptions and saves them as `orig_routes_??.csv` files in the same directory.

#### List of vehicles

File will have the following columns:
- vehicle_id
- capacity - how many packages can be taken by this vehicle. The units are unknown yet.

#### Graph

File will have the following columns:
- node_a - id of the first node
- node_b - id of the second node
- cost

cost field will be a float value. In future it may change to e.g. functions or lists of values, to make modelling time dependencies easier.

Graph is assumed to be undirected, though there directed graphs may also be implemented in future.

##### Generating graph description

Graph description is created along with `outposts_??.csv` and `orig_routes_??.csv` files in same directory.

Pass `--gmaps` flag when running `./src/solver/outposts.py` to use Google Maps Directions API.

If `--gmaps` flag is not set, distance between `(latitude, longitude)` points is calculated as following:  

1) project geographic coordinates to XY plane (see: WGS84 system), using one of two points as reference  
2) calculate distance between points on XY plane using Manhattan metrics.  

Manhattan metrics should be more reliable when calculating distance between points of a city.

##### Google Maps - Directions API

Google changed their pricing method. Nevertheles, we can't just ask for every edge in the graph and some clever tricks are needed.

#### Constraints

The format of constraints file will be specified later.

Now, single constraint is hardcoded in ortools solver. It's max distance of single route.
This constraint value can be changed using `--max_distance` flag when using `./src/solver/solver.py`.  
Default value is `200km`.

### Output
Solver will create the following output files:

1. Routes
2. Plot (optional)

Now `./src/solver/solver.py` generates single `routes_??.csv` file for each week day and time shift.

#### Routes

- route_id
- vehicle_id
- cost
- list_of_outposts - in the following format: `[1, 2, 3, 4]`, where each element is some outpost_id

#### Plot

A `png` file with all the outposts and routes plotted on the map. 

Use `./src/solver/summarize.py [flags] <outposts dir> <routes dir>` to create:  
- plots with routes visualizaiton  
- bar charts with comparsion of total routes length and number of used vehicles
- textual summary

#### Solution verification

If `--verify` flag is set when running `./src/solver/summarize.py`, script does sanity check of result
and writes results into `check_visits.txt` and `check_load.txt` in given `--savedir`.

- `check_load.txt` - outposts load calculated for each timeshift for BORH and original solution
- `check_visits.txt` - how many times each outpost was assigned to any route, data splited into timeshifts for BOHR and original solution

### Running the solver

Solver can be run using following command: `./src/solver/solver.py <data directory> [flags]`

Solver requires following `*.csv` data files:

- `outposts_??.csv`
- `graph.csv`
- `vehicles.csv` (can be generated on the fly using `--gen_vehicles k:load` flag)

Additional options are:

- `--savedir SAVEDIR`: Output directory for `routes_??.csv` files
- `--calc_time CALC_TIME`: Suggested calculation time. Default: 5s
- `--gen_vehicles k:load`: Generate vehicles instead of loading them from `vehicle.csv`. Each use of this flag will generate `k` new 
vehicles of load `load` tons, where `k` is integer and `load` is float value.
- `--limit_outposts N`: Try to solve the problem only for first `N` outposts in the `outposts_??.csv` files.
- `--max_distance L`: Set max length of single route to float value `L`


