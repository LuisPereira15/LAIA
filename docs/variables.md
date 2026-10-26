# Dataset Variables — BTS Airline On-Time Performance Data

## Features (used for model development)

| Variable | Type | Description |
|---|---|---|
| YEAR | int | Year of the flight record (2022 or 2023) |
| QUARTER | int | Quarter of the year (1–4) |
| MONTH | int | Month of the year (1–12) |
| DAY_OF_WEEK | int | Day of the week (1=Monday through 7=Sunday, 9=Unknown) |
| OP_CARRIER_AIRLINE_ID | int | Unique identifier for the operating airline |
| OP_CARRIER_FL_NUM | int | Flight number assigned by the operating carrier |
| ORIGIN | str | Origin airport code (e.g., "ATL", "LAX") |
| ORIGIN_STATE_ABR | str | US state abbreviation of the origin airport (e.g., "GA", "CA"). Redundant with ORIGIN but kept for regional analysis. |
| DEST | str | Destination airport code |
| DEST_STATE_ABR | str | US state abbreviation of the destination airport. Redundant with DEST but kept for regional analysis. |
| CRS_DEP_TIME | int | Scheduled departure time (local, HHMM format, e.g., 1430 = 2:30 PM) |
| CRS_ELAPSED_TIME | float | Scheduled flight duration in minutes (gate to gate) |
| DISTANCE | float | Distance between origin and destination in miles |

## Target Variable

| Variable | Type | Description |
|---|---|---|
| ARR_DELAY_CATEGORY | str | Derived from ARR_DELAY_NEW during cleaning |

| Category | Condition |
|---|---|
| `on_time` | ARR_DELAY_NEW < 15 minutes |
| `moderate_delay` | 15 <= ARR_DELAY_NEW < 60 minutes |
| `severe_delay` | ARR_DELAY_NEW >= 60 minutes |

## Columns dropped during cleaning

| Variable | Reason |
|---|---|
| DEP_DELAY | Post-departure info — leaky feature |
| ARR_DELAY_NEW | Used to derive target variable, then dropped |
| CANCELLED | Used to filter out cancelled flights, then dropped |
| DIVERTED | Used to filter out diverted flights, then dropped |
| CRS_ARR_TIME | Redundant — derivable from CRS_DEP_TIME + CRS_ELAPSED_TIME |
| DISTANCE_GROUP | Redundant — binned version of DISTANCE (each group = 250 miles) |

## Cleaning steps applied

1. Removed 513,402 exact duplicate rows
2. Removed 286,573 cancelled or diverted flights
3. Removed 2 rows with missing ARR_DELAY_NEW
4. Removed 0 rows with DAY_OF_WEEK = 9 (none found)
5. Created ARR_DELAY_CATEGORY target variable
6. Dropped leaky/redundant columns
7. Reset index

**Final dataset shape: 12,777,047 rows x 14 columns**