# Dataset Variables — BTS Airline On-Time Performance Data

## Features (after cleaning, before feature selection)

| Variable | Type | Description |
|---|---|---|
| YEAR | int | Year of the flight record (2022 or 2023) |
| QUARTER | int | Quarter of the year (1–4) |
| MONTH | int | Month of the year (1–12) |
| DAY_OF_WEEK | int | Day of the week (1=Monday through 7=Sunday) |
| OP_CARRIER_AIRLINE_ID | int | Unique identifier for the operating airline |
| OP_CARRIER_FL_NUM | int | Flight number assigned by the operating carrier |
| ORIGIN | str | Origin airport code (e.g., "ATL", "LAX") |
| ORIGIN_STATE_ABR | str | US state abbreviation of the origin airport (e.g., "GA", "CA") |
| DEST | str | Destination airport code |
| DEST_STATE_ABR | str | US state abbreviation of the destination airport |
| CRS_DEP_TIME | int | Scheduled departure time (local, HHMM format, e.g., 1430 = 2:30 PM) |
| CRS_ARR_TIME | float | Scheduled arrival time (local, HHMM format) |
| CRS_ELAPSED_TIME | float | Scheduled flight duration in minutes (gate to gate) |
| DISTANCE | float | Distance between origin and destination in miles |

## Target Variable

| Variable | Type | Description |
|---|---|---|
| ARR_DELAY_CATEGORY | str | Derived from ARR_DELAY_NEW during cleaning |

| Category | Condition |
|---|---|
| `on_time` | ARR_DELAY_NEW < 15 minutes |
| `moderate_delay` | 15 ≤ ARR_DELAY_NEW < 60 minutes |
| `severe_delay` | ARR_DELAY_NEW ≥ 60 minutes |

## Columns removed during cleaning (before EDA)

| Variable | Reason |
|---|---|
| DEP_DELAY | Post-departure info — leaky feature |
| ARR_DELAY_NEW | Used to derive target variable, then dropped |
| CANCELLED | Used to filter out cancelled flights, then dropped |
| DIVERTED | Used to filter out diverted flights, then dropped |
| DISTANCE_GROUP | Redundant — binned version of DISTANCE (each group = 250 miles) |

## Columns removed after EDA (feature selection)

| Variable | Reason |
|---|---|
| QUARTER | Redundant with MONTH — same info at coarser granularity |
| OP_CARRIER_FL_NUM | Too many unique values (~thousands), essentially a flight identifier (noise) |
| CRS_ELAPSED_TIME | Pearson r = 0.983 with DISTANCE — near-perfectly correlated, keep the simpler one |